#!/usr/bin/env python3
"""Leakage-safe NeuroToxPredictor training with adaptive multi-modal gating."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch import nn

import train_compare as core

PATIENCE = 10
MIN_DELTA = 1e-6
MIN_MODALITY_WEIGHT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chemberta-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resample-seed", type=int, default=0)
    return parser.parse_args()


def train_with_early_stopping(batches, records, model_config, config, output_dir, device):
    """Train NeuroToxPredictor and reload the minimum-validation-loss state."""
    core.seed_everything(config.seed)
    variant_config = dict(model_config)
    variant_config.update({"anti_collapse": True, "min_gate_weight": MIN_MODALITY_WEIGHT})
    model = core.GatedNeuroToxPredictor(**variant_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    best_loss, best_epoch, best_state, no_improvement, history = float("inf"), 0, None, 0, []

    for epoch in range(1, config.epochs + 1):
        train_result = core.run_batches(model, batches["train"], criterion, device, True, config, epoch, optimizer)
        val_result = core.run_batches(model, batches["val"], criterion, device, True, config, epoch)
        row = {"epoch": epoch}
        for split, result in (("train", train_result), ("val", val_result)):
            for metric_name, value in core.scalar_metrics(result).items():
                row[f"{split}_{metric_name}"] = value
            for index, modality in enumerate(("gat", "maccs", "chemberta")):
                row[f"{split}_gate_{modality}_mean"] = result["weights"][:, index].mean()
        history.append(row)
        if val_result["loss"] < best_loss - MIN_DELTA:
            best_loss, best_epoch, best_state, no_improvement = val_result["loss"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            no_improvement += 1
        print(f"model=NeuroToxPredictor epoch={epoch:03d} train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} val_auc={val_result['roc_auc']:.4f} no_loss_improve={no_improvement}/{PATIENCE}", flush=True)
        if no_improvement >= PATIENCE:
            print(f"early_stopping=triggered epoch={epoch} best_epoch={best_epoch} best_val_loss={best_loss:.6f} patience={PATIENCE}", flush=True)
            break

    run_dir = output_dir / "neurotoxpredictor"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
    model.load_state_dict(best_state)
    final = {split: core.run_batches(model, split_batches, criterion, device, True, config, best_epoch) for split, split_batches in batches.items()}
    core.save_run_outputs(run_dir, model, best_state, best_epoch, final, records, variant_config, config)
    metrics_path = run_dir / "best_model_metrics.txt"
    metrics_path.write_text(metrics_path.read_text().replace("Selection Metric: val_roc_auc", f"Selection Metric: validation loss (patience={PATIENCE})"))
    return best_epoch, best_loss, final, len(history)


def main() -> None:
    args = parse_args()
    config = core.Config(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        split_seed=args.split_seed,
        resample_seed=args.resample_seed,
        selection_metric="val_loss",
    )
    core.seed_everything(config.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, excluded = core.prepare_records(args.data)
    splits = core.grouped_split(records, config)
    labels = records["label"].to_numpy(dtype="int64")
    train_draws = core.resample_training_indices(splits["train"], labels, config.resample_seed)
    split_labels = [""] * len(records)
    for split, indices in splits.items():
        for index in indices:
            split_labels[index] = split
    manifest = records[["source_index", "DTXSID", "smiles", "structure_key", "label"]].copy()
    manifest.insert(0, "record_index", range(len(records)))
    manifest["split"] = split_labels
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    resample = manifest.iloc[train_draws].copy()
    resample.insert(0, "draw_index", range(len(resample)))
    resample.to_csv(output_dir / "training_resample_manifest.csv", index=False)
    excluded[["source_index", "DTXSID", "smiles", "structure_key", "label"]].to_csv(output_dir / "excluded_conflicting_structure_labels.csv", index=False)
    print(f"structure_group_split train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} train_resampled={len(train_draws)} excluded_conflicting_rows={len(excluded)}", flush=True)

    all_maccs = core.maccs_matrix(records["smiles"])
    selected_bits = core.fit_maccs_selector(all_maccs[splits["train"]])
    fingerprints = all_maccs[:, selected_bits].astype("float32")
    (output_dir / "selected_maccs_bits.json").write_text(json.dumps(selected_bits, indent=2) + "\n")
    graphs = core.get_graph_date(records[["smiles"]]).iloc[:, 0].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chemberta = core.get_chemberta_date(records, smiles_col="smiles", model_dir=str(args.chemberta_model), batch_size=64, max_length=128, device=device).to_numpy(dtype="float32")
    torch.cuda.empty_cache()
    batches = {"train": core.make_batches(train_draws, graphs, fingerprints, chemberta, labels, config.batch_size), "val": core.make_batches(splits["val"], graphs, fingerprints, chemberta, labels, config.batch_size), "test": core.make_batches(splits["test"], graphs, fingerprints, chemberta, labels, config.batch_size)}
    model_config = {"graph_in_dim": 26, "graph_hidden_dims": list(config.graph_hidden_dims), "graph_out_dim": config.graph_out_dim, "finger_in_dim": int(fingerprints.shape[1]), "finger_hidden_dim": config.finger_hidden_dim, "finger_out_dim": config.finger_out_dim, "chem_in_dim": int(chemberta.shape[1]), "chem_hidden_dim": config.chem_hidden_dim, "chem_out_dim": config.chem_out_dim, "heads": config.heads, "dropout": config.dropout}
    best_epoch, best_loss, final, epochs_ran = train_with_early_stopping(batches, records, model_config, config, output_dir, device)
    summary = {"model": "NeuroToxPredictor", "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_ran": epochs_ran, "early_stopping_patience": PATIENCE}
    for split, result in final.items():
        for metric in ("roc_auc", "pr_auc", "accuracy", "f1", "recall", "precision"):
            summary[f"{split}_{metric}"] = result[metric]
    pd.DataFrame([summary]).to_csv(output_dir / "comparison.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": "NeuroToxPredictor", "data_sha256": core.file_sha256(args.data), "raw_samples": len(records) + len(excluded), "usable_samples": len(records), "excluded_conflicting_structure_label_rows": len(excluded), "structure_groups": int(records.structure_key.nunique()), "split_sizes_unique_records": {key: int(len(value)) for key, value in splits.items()}, "train_resampled_size": int(len(train_draws)), "gating_regularization": {"uniform_warmup_epochs": config.warmup_epochs, "branch_auxiliary_loss_weight": config.auxiliary_loss_weight, "mean_gate_balance_loss_weight": config.balance_loss_weight, "minimum_modality_weight": MIN_MODALITY_WEIGHT}, "early_stopping": {"monitor": "validation loss", "patience": PATIENCE, "min_delta": MIN_DELTA, "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_ran": epochs_ran}, "feature_selection": {"method": "MACCS fit on training split only", "selected_bit_count": len(selected_bits)}, "config": asdict(config), "model_config": model_config}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

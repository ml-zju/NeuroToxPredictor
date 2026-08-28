"""Shared leakage-safe utilities for NeuroToxPredictor training.

The split is made on standardized chemical structures before any resampling.
Only training records are upsampled; validation and test records remain unique.
MACCS feature selection is fit on training records only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

if os.environ.get("OMP_NUM_THREADS") in (None, "", "0"):
    os.environ["OMP_NUM_THREADS"] = "4"

import dgl
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from data_func import get_chemberta_date, get_graph_date
from model import GatedNeuroToxPredictor


@dataclass(frozen=True)
class Config:
    seed: int = 3407
    split_seed: int = 42
    resample_seed: int = 0
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    graph_hidden_dims: tuple[int, ...] = (64, 32)
    graph_out_dim: int = 64
    finger_hidden_dim: int = 128
    finger_out_dim: int = 32
    chem_hidden_dim: int = 256
    chem_out_dim: int = 64
    heads: int = 4
    dropout: float = 0.5
    warmup_epochs: int = 15
    temperature_start: float = 3.0
    temperature_end: float = 1.0
    temperature_end_epoch: int = 50
    min_gate_weight: float = 0.05
    auxiliary_loss_weight: float = 0.2
    balance_loss_weight: float = 0.05
    selection_metric: str = "val_roc_auc"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_structure_key(smiles: object) -> str:
    """Return an InChIKey structure group, with a canonical-SMILES fallback."""
    value = str(smiles).strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return f"unparseable::{value}"
    try:
        molecule = rdMolStandardize.Cleanup(molecule)
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    except Exception:
        pass
    try:
        inchikey = Chem.MolToInchiKey(molecule)
        if inchikey:
            return f"inchikey::{inchikey}"
    except Exception:
        pass
    return f"canonical_smiles::{Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)}"


def prepare_records(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path).dropna(subset=["smiles"]).reset_index(drop=False)
    raw = raw.rename(columns={"index": "source_index"})
    raw["smiles"] = raw["smiles"].astype(str)
    if "DTXSID" not in raw.columns:
        raw["DTXSID"] = ""
    raw["structure_key"] = raw["smiles"].map(standardized_structure_key)
    label_counts = raw.groupby("structure_key")["label"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    excluded = raw[raw["structure_key"].isin(conflicting_keys)].copy()
    records = raw[~raw["structure_key"].isin(conflicting_keys)].copy()
    records = records.reset_index(drop=True)
    return records, excluded


def grouped_split(records: pd.DataFrame, config: Config) -> dict[str, np.ndarray]:
    """Use two held-out folds as validation/test, keeping each structure in one split."""
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=config.split_seed)
    folds = list(splitter.split(records, records["label"], groups=records["structure_key"]))
    test_indices = np.sort(folds[0][1])
    val_indices = np.sort(folds[1][1])
    train_indices = np.sort(np.setdiff1d(np.arange(len(records)), np.r_[val_indices, test_indices]))
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def resample_training_indices(indices: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Upsample only the smaller training class, never validation/test records."""
    selected_labels = labels[indices]
    classes, counts = np.unique(selected_labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Training split must contain two classes, found {classes.tolist()}")
    target = int(counts.max())
    generator = np.random.default_rng(seed)
    draws = []
    for class_value in classes:
        class_indices = indices[selected_labels == class_value]
        draws.append(generator.choice(class_indices, size=target, replace=len(class_indices) < target))
    result = np.concatenate(draws)
    generator.shuffle(result)
    return result.astype(np.int64)


def maccs_matrix(smiles: pd.Series) -> np.ndarray:
    matrices = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        fingerprint = np.zeros(167, dtype=np.float32)
        if molecule is not None:
            DataStructs.ConvertToNumpyArray(
                rdMolDescriptors.GetMACCSKeysFingerprint(molecule), fingerprint
            )
        matrices.append(fingerprint)
    return np.vstack(matrices).astype(np.float32)


def fit_maccs_selector(training_matrix: np.ndarray) -> list[int]:
    frame = pd.DataFrame(training_matrix)
    zero_or_constant = set(frame.columns[frame.mean() == 0]) | set(frame.columns[frame.std() == 0])
    reduced = frame.drop(columns=list(zero_or_constant))
    correlation = reduced.corr(method="pearson")
    correlated = set()
    for col in range(len(correlation.columns)):
        for row in range(col):
            if abs(correlation.iloc[row, col]) >= 0.95:
                correlated.add(correlation.columns[col])
    selected = [int(column) for column in reduced.columns if column not in correlated]
    if not selected:
        raise RuntimeError("MACCS selector removed every feature")
    return selected


def make_batches(indices, graphs, fingerprints, chemberta, labels, batch_size):
    batches = []
    for index in range(ceil(len(indices) / batch_size)):
        selected = indices[index * batch_size : (index + 1) * batch_size]
        batches.append(
            (
                dgl.batch([graphs[i] for i in selected]),
                torch.as_tensor(fingerprints[selected], dtype=torch.float32),
                torch.as_tensor(chemberta[selected], dtype=torch.float32),
                torch.as_tensor(labels[selected], dtype=torch.long),
                selected,
            )
        )
    return batches


def metrics(labels, predictions, probabilities):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
    }


def temperature_for_epoch(epoch, config):
    if epoch <= config.warmup_epochs:
        return config.temperature_start
    span = max(config.temperature_end_epoch - config.warmup_epochs, 1)
    progress = min(max((epoch - config.warmup_epochs) / span, 0.0), 1.0)
    return config.temperature_start + progress * (config.temperature_end - config.temperature_start)


def run_batches(model, batches, criterion, device, anti_collapse, config, epoch, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_losses, main_losses, aux_losses, balance_losses = [], [], [], []
    all_labels, all_predictions, all_probabilities, all_weights, all_indices = [], [], [], [], []
    temperature = temperature_for_epoch(epoch, config) if anti_collapse else 1.0
    force_uniform = anti_collapse and training and epoch <= config.warmup_epochs
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for graph, fingerprints, chemberta, labels, indices in batches:
            graph, fingerprints, chemberta, labels = (
                graph.to(device), fingerprints.to(device), chemberta.to(device), labels.to(device)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(graph, fingerprints, chemberta, temperature=temperature, force_uniform=force_uniform)
            main_loss = criterion(output["logits"], labels)
            aux_loss = torch.zeros((), device=device)
            balance_loss = torch.zeros((), device=device)
            loss = main_loss
            if anti_collapse:
                aux_loss = sum(criterion(branch_logits, labels) for branch_logits in output["aux_logits"]) / 3.0
                mean_weights = output["weights"].mean(dim=0)
                balance_loss = ((mean_weights - 1.0 / 3.0) ** 2).sum()
                loss = main_loss + config.auxiliary_loss_weight * aux_loss + config.balance_loss_weight * balance_loss
            if training:
                loss.backward()
                optimizer.step()
            probabilities = torch.softmax(output["logits"], dim=1)[:, 1]
            predictions = torch.argmax(output["logits"], dim=1)
            total_losses.append(float(loss.detach().cpu()))
            main_losses.append(float(main_loss.detach().cpu()))
            aux_losses.append(float(aux_loss.detach().cpu()))
            balance_losses.append(float(balance_loss.detach().cpu()))
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_probabilities.extend(probabilities.detach().cpu().numpy().tolist())
            all_weights.append(output["weights"].detach().cpu().numpy())
            all_indices.extend(indices.tolist())
    result = metrics(all_labels, all_predictions, all_probabilities)
    result.update({
        "loss": float(np.mean(total_losses)), "main_loss": float(np.mean(main_losses)),
        "aux_loss": float(np.mean(aux_losses)), "balance_loss": float(np.mean(balance_losses)),
        "temperature": temperature, "labels": np.asarray(all_labels, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
        "probabilities": np.asarray(all_probabilities, dtype=np.float64),
        "weights": np.concatenate(all_weights, axis=0), "indices": np.asarray(all_indices, dtype=np.int64),
    })
    return result


def scalar_metrics(result):
    names = ("loss", "main_loss", "aux_loss", "balance_loss", "temperature", "roc_auc", "pr_auc", "accuracy", "f1", "recall", "precision")
    return {name: result[name] for name in names}


def save_run_outputs(run_dir, model, best_state, best_epoch, final, records, model_config, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "model_config": model_config, "training_config": asdict(config), "best_epoch": best_epoch, "best_metrics": {split: scalar_metrics(result) for split, result in final.items()}}, run_dir / "best_model.pth")
    rows, gate_rows = [], []
    for split, result in final.items():
        row = {"split": split, **scalar_metrics(result)}
        row.update({"gate_gat_mean": result["weights"][:, 0].mean(), "gate_maccs_mean": result["weights"][:, 1].mean(), "gate_chemberta_mean": result["weights"][:, 2].mean()})
        rows.append(row)
        selected = records.iloc[result["indices"]].reset_index(drop=True)
        predictions = selected[["source_index", "structure_key", "DTXSID", "smiles"]].copy()
        predictions.insert(0, "record_index", result["indices"])
        predictions["label"] = result["labels"]
        predictions["pred"] = result["predictions"]
        predictions["prob"] = result["probabilities"]
        predictions["gate_gat"] = result["weights"][:, 0]
        predictions["gate_maccs"] = result["weights"][:, 1]
        predictions["gate_chemberta"] = result["weights"][:, 2]
        predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
        for modality_index, modality in enumerate(("GAT", "MACCS", "ChemBERTa")):
            values = result["weights"][:, modality_index]
            gate_rows.append({"split": split, "modality": modality, "mean": values.mean(), "std": values.std(), "min": values.min(), "q05": np.quantile(values, .05), "q25": np.quantile(values, .25), "median": np.median(values), "q75": np.quantile(values, .75), "q95": np.quantile(values, .95), "max": values.max(), "argmax_fraction": np.mean(np.argmax(result["weights"], axis=1) == modality_index)})
    pd.DataFrame(rows).to_csv(run_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run_dir / "gate_summary.csv", index=False)
    with (run_dir / "best_model_metrics.txt").open("w") as handle:
        handle.write(f"Best Epoch: {best_epoch}\nSelection Metric: val_roc_auc\n")
        for split in ("train", "val", "test"):
            for name, value in scalar_metrics(final[split]).items():
                handle.write(f"{split}_{name}: {value:.6f}\n")
            for index, modality in enumerate(("gat", "maccs", "chemberta")):
                handle.write(f"{split}_gate_{modality}_mean: {final[split]['weights'][:, index].mean():.6f}\n")


def train_variant(name, anti_collapse, batches, records, model_config, config, output_dir, device):
    seed_everything(config.seed)
    variant_config = dict(model_config)
    variant_config.update({"anti_collapse": anti_collapse, "min_gate_weight": config.min_gate_weight})
    model = GatedNeuroToxPredictor(**variant_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, config.epochs + 1):
        train_result = run_batches(model, batches["train"], criterion, device, anti_collapse, config, epoch, optimizer)
        val_result = run_batches(model, batches["val"], criterion, device, anti_collapse, config, epoch)
        row = {"epoch": epoch}
        for split, result in (("train", train_result), ("val", val_result)):
            for metric_name, value in scalar_metrics(result).items(): row[f"{split}_{metric_name}"] = value
            for index, modality in enumerate(("gat", "maccs", "chemberta")): row[f"{split}_gate_{modality}_mean"] = result["weights"][:, index].mean()
        history.append(row)
        if val_result["roc_auc"] > best_auc:
            best_auc, best_epoch, best_state = val_result["roc_auc"], epoch, copy.deepcopy(model.state_dict())
        print(f"variant={name} epoch={epoch:03d} train_loss={train_result['loss']:.4f} val_auc={val_result['roc_auc']:.4f} val_acc={val_result['accuracy']:.4f} gate={val_result['weights'].mean(axis=0).round(4).tolist()}", flush=True)
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
    model.load_state_dict(best_state)
    final = {split: run_batches(model, split_batches, criterion, device, anti_collapse, config, best_epoch) for split, split_batches in batches.items()}
    save_run_outputs(run_dir, model, best_state, best_epoch, final, records, variant_config, config)
    return best_epoch, final


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/lastneurodb_clean_3.csv"))
    parser.add_argument("--chemberta-model", type=Path, default=Path("/root/autodl-tmp/neuro/ChemBERTa-zinc-base-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(epochs=args.epochs)
    seed_everything(config.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, excluded = prepare_records(args.data)
    splits = grouped_split(records, config)
    labels = records["label"].to_numpy(dtype=np.int64)
    train_draws = resample_training_indices(splits["train"], labels, config.resample_seed)
    split_labels = np.full(len(records), "", dtype=object)
    for split, indices in splits.items(): split_labels[indices] = split
    manifest = records[["source_index", "DTXSID", "smiles", "structure_key", "label"]].copy()
    manifest.insert(0, "record_index", np.arange(len(records)))
    manifest["split"] = split_labels
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    training_manifest = manifest.iloc[train_draws].copy()
    training_manifest.insert(0, "draw_index", np.arange(len(training_manifest)))
    training_manifest.to_csv(output_dir / "training_resample_manifest.csv", index=False)
    excluded[["source_index", "DTXSID", "smiles", "structure_key", "label"]].to_csv(output_dir / "excluded_conflicting_structure_labels.csv", index=False)
    print(f"structure_group_split train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} train_resampled={len(train_draws)} excluded_conflicting_rows={len(excluded)}", flush=True)
    print("feature_construction_start", flush=True)
    all_maccs = maccs_matrix(records["smiles"])
    selected_bits = fit_maccs_selector(all_maccs[splits["train"]])
    fingerprints = all_maccs[:, selected_bits].astype(np.float32)
    (output_dir / "selected_maccs_bits.json").write_text(json.dumps(selected_bits, indent=2) + "\n")
    graph_frame = get_graph_date(records[["smiles"]])
    graphs = graph_frame.iloc[:, 0].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chemberta = get_chemberta_date(records, smiles_col="smiles", model_dir=str(args.chemberta_model), batch_size=64, max_length=128, device=device).to_numpy(dtype=np.float32)
    torch.cuda.empty_cache()
    batches = {"train": make_batches(train_draws, graphs, fingerprints, chemberta, labels, config.batch_size), "val": make_batches(splits["val"], graphs, fingerprints, chemberta, labels, config.batch_size), "test": make_batches(splits["test"], graphs, fingerprints, chemberta, labels, config.batch_size)}
    model_config = {"graph_in_dim": 26, "graph_hidden_dims": list(config.graph_hidden_dims), "graph_out_dim": config.graph_out_dim, "finger_in_dim": int(fingerprints.shape[1]), "finger_hidden_dim": config.finger_hidden_dim, "finger_out_dim": config.finger_out_dim, "chem_in_dim": int(chemberta.shape[1]), "chem_hidden_dim": config.chem_hidden_dim, "chem_out_dim": config.chem_out_dim, "heads": config.heads, "dropout": config.dropout}
    summaries = {}
    for name, anti_collapse in (("baseline_gate", False), ("anticollapse_gate", True)):
        best_epoch, final = train_variant(name, anti_collapse, batches, records, model_config, config, output_dir, device)
        summaries[name] = {"best_epoch": best_epoch, **{f"{split}_{metric}": result[metric] for split, result in final.items() for metric in ("roc_auc", "accuracy", "f1", "recall", "precision")}, **{f"{split}_gate_{modality}": result["weights"][:, index].mean() for split, result in final.items() for index, modality in enumerate(("gat", "maccs", "chemberta"))}}
    pd.DataFrame.from_dict(summaries, orient="index").rename_axis("variant").reset_index().to_csv(output_dir / "comparison.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "data_sha256": file_sha256(args.data), "raw_samples": len(records) + len(excluded), "usable_samples": len(records), "excluded_conflicting_structure_label_rows": len(excluded), "structure_groups": int(records["structure_key"].nunique()), "split_sizes_unique_records": {key: int(len(value)) for key, value in splits.items()}, "train_resampled_size": int(len(train_draws)), "train_resampled_label_counts": {str(key): int(value) for key, value in pd.Series(labels[train_draws]).value_counts().items()}, "feature_selection": {"method": "MACCS zero/constant and correlation filtering fit on training split only", "selected_bit_count": len(selected_bits)}, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__, "rdkit": rdkit.__version__, "sklearn": sklearn.__version__, "config": asdict(config), "model_config": model_config}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.read_csv(output_dir / "comparison.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

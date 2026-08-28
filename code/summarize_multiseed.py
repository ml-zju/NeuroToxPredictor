#!/usr/bin/env python3
"""Aggregate fixed-split, multi-seed NeuroToxPredictor results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import matthews_corrcoef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--run-name", type=str, default="neurotoxpredictor")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        path = args.result_root / f"seed_{seed}" / "comparison.csv"
        row = pd.read_csv(path).iloc[0].to_dict()
        for split in ("train", "val", "test"):
            predictions = pd.read_csv(
                args.result_root / f"seed_{seed}" / args.run_name / f"{split}_predictions.csv"
            )
            row[f"{split}_mcc"] = matthews_corrcoef(
                predictions["label"], predictions["pred"]
            )
        row["seed"] = seed
        rows.append(row)
    detail = pd.DataFrame(rows).sort_values("seed")
    detail.to_csv(args.result_root / "five_seed_metrics.csv", index=False)

    metric_columns = [
        column for column in detail.columns
        if column.startswith(("train_", "val_", "test_"))
        and column not in {"train_resampled_size"}
    ]
    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [detail[column].mean() for column in metric_columns],
        "std": [detail[column].std(ddof=1) for column in metric_columns],
        "min": [detail[column].min() for column in metric_columns],
        "max": [detail[column].max() for column in metric_columns],
    })
    summary.to_csv(args.result_root / "five_seed_summary.csv", index=False)
    print(detail.to_string(index=False), flush=True)
    print(summary[summary.metric.str.startswith("test_")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

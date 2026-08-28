#!/usr/bin/env python3
"""Summarize five-seed validation/test reproducibility with paired test bootstrap CIs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


SEEDS = (42, 123, 3407, 2024, 2025)
METRICS = ("ROC-AUC", "PR-AUC", "Accuracy", "F1-score", "Recall", "Specificity", "Precision", "MCC")


def scores(labels, predictions, probabilities):
    return {
        "ROC-AUC": roc_auc_score(labels, probabilities),
        "PR-AUC": average_precision_score(labels, probabilities),
        "Accuracy": accuracy_score(labels, predictions),
        "F1-score": f1_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "Specificity": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def load_split(root: Path, split: str):
    result = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "neurotoxpredictor" / f"{split}_predictions.csv"
        result.append(pd.read_csv(path).sort_values("record_index").reset_index(drop=True))
    reference = result[0][["record_index", "label"]]
    for frame in result[1:]:
        if not reference.equals(frame[["record_index", "label"]]):
            raise RuntimeError(f"{split} records differ across seeds")
    labels = reference["label"].to_numpy()
    predictions = np.stack([frame["pred"].to_numpy() for frame in result])
    probabilities = np.stack([frame["prob"].to_numpy() for frame in result])
    return labels, predictions, probabilities


def bootstrap_ci(labels, predictions, probabilities, draws=2000):
    rng = np.random.default_rng(20260818)
    estimates = {metric: [] for metric in METRICS}
    while len(estimates["ROC-AUC"]) < draws:
        index = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[index]).size != 2:
            continue
        per_seed = [scores(labels[index], predictions[row, index], probabilities[row, index]) for row in range(len(SEEDS))]
        for metric in METRICS:
            estimates[metric].append(float(np.mean([row[metric] for row in per_seed])))
    return {metric: tuple(np.quantile(values, (0.025, 0.975))) for metric, values in estimates.items()}


def main():
    root = Path("results_multiseed")
    val_labels, val_predictions, val_probabilities = load_split(root, "val")
    test_labels, test_predictions, test_probabilities = load_split(root, "test")
    val = [scores(val_labels, val_predictions[row], val_probabilities[row]) for row in range(len(SEEDS))]
    test = [scores(test_labels, test_predictions[row], test_probabilities[row]) for row in range(len(SEEDS))]
    intervals = bootstrap_ci(test_labels, test_predictions, test_probabilities)
    rows = []
    for metric in METRICS:
        rows.append({
            "Metric": metric,
            "Validation set, mean ± SD": f"{np.mean([row[metric] for row in val]):.3f} ± {np.std([row[metric] for row in val], ddof=1):.3f}",
            "Independent test set, mean ± SD": f"{np.mean([row[metric] for row in test]):.3f} ± {np.std([row[metric] for row in test], ddof=1):.3f}",
            "Test-set 95% bootstrap CI": f"[{intervals[metric][0]:.3f}, {intervals[metric][1]:.3f}]",
        })
    table = pd.DataFrame(rows)
    table.to_csv(root / "reproducibility_table_R1.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

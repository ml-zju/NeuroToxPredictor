# NeuroToxPredictor: Five-Seed Reproducibility Study

Code, data and results for the five-seed NeuroToxPredictor experiment.

## Model

The GAT branch uses **4 attention heads** in every GAT layer.

NeuroToxPredictor combines graph-based molecular features (GAT), MACCS
fingerprints and ChemBERTa embeddings through a sample-adaptive gating module.
Training uses a 15-epoch uniform-gate warm-up, branch-level auxiliary
classification losses, a mean-gate balance penalty and a minimum modality
weight of 0.05.

## Evaluation protocol

- Input data: `data/neuro.csv`
- Structure-disjoint 80/10/10 train/validation/test partition
- Structure-label conflicts are excluded before splitting
- Resampling is applied to the training partition only
- MACCS feature selection is fit on training data only
- Fixed split seed: `42`; fixed resampling seed: `0`
- Training seeds: `42`, `123`, `3407`, `2024`, `2025`
- Validation-loss early stopping with patience 10

## Package contents

- `code/train_neurotoxpredictor.py`: four-head model training.
- `code/train_compare.py`: shared preprocessing, batching, loss and output utilities.
  The standalone comparison routine is not part of the five-seed experiment.
- `code/model.py` and `code/data_func.py`: model and molecular feature construction.
- `code/summarize_multiseed.py`: per-seed metrics and mean/sample-standard-deviation summary.
- `code/summarize_reproducibility.py`: validation/test table with paired test bootstrap intervals.
- `data/neuro.csv`: input dataset (4,926 records; 15 conflicting-label records excluded).
- `results_multiseed/`: results, checkpoints, predictions and split manifests.

## Requirements

Python 3.10 or newer, compatible PyTorch and DGL versions, and the packages
listed in `requirements.txt` are required.

```bash
python3 -m pip install -r requirements.txt
```

Exact dependency versions for the training environment were not recorded.
Results may vary across library versions and hardware.

The pretrained **ChemBERTa-zinc-base-v1** model is required separately. Its local
directory must contain the model configuration, tokenizer files and pretrained
weights. Automatic model download is not implemented.

## Training

Command syntax from the package directory:

```bash
bash run_5seeds.sh /absolute/path/to/ChemBERTa-zinc-base-v1
```

The model path refers to the local pretrained model directory. The default
output directory is `results_multiseed_rerun/`, separate from the included
results. The script also supports invocation by full path from other working
directories.

An alternative Python interpreter and output directory can be specified:

```bash
PYTHON_BIN=/absolute/path/to/python bash run_5seeds.sh \
  /absolute/path/to/ChemBERTa-zinc-base-v1 /absolute/path/to/new_results
```

The output directory must be new or empty. Existing results are not overwritten.

## Outputs

Results are stored in `results_multiseed/seed_<seed>/`. Each seed includes a
model checkpoint, epoch history, train/validation/test predictions, metric and
gate summaries, split and training-resampling manifests, selected MACCS bits,
excluded records and run metadata. The fixed partition contains 3,927 training
records, 492 validation records and 492 test records; training resampling produces
4,098 draws. Training-set metrics therefore refer to the resampled training set.

Aggregate results are `five_seed_metrics.csv`, `five_seed_summary.csv` and
`reproducibility_table_R1.csv` inside `results_multiseed/`.
Standard deviations are computed across the five training seeds on the same
fixed split. The reproducibility table uses 2,000 paired test-record bootstrap
draws, averaging each metric across seeds within each draw; these intervals do
not measure variation across independently sampled train/test splits.

Summary tables can be calculated from the saved predictions without training.
Commands from the package directory:

```bash
python3 code/summarize_multiseed.py --result-root results_multiseed \
  --run-name neurotoxpredictor --seeds 42 123 3407 2024 2025
python3 code/summarize_reproducibility.py
```

The summary commands overwrite the corresponding summary CSV files.
`run_5seeds.sh` generates `five_seed_metrics.csv` and `five_seed_summary.csv`
in its output directory. `summarize_reproducibility.py` reads `results_multiseed/`
relative to the working directory and writes `reproducibility_table_R1.csv`.

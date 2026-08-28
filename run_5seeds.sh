#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash run_5seeds.sh CHEMBERTA_MODEL_DIR [RESULT_ROOT]' \
    'CHEMBERTA_MODEL_DIR: local ChemBERTa-zinc-base-v1 model directory.' \
    'RESULT_ROOT: new or empty output directory (default: results_multiseed_rerun beside this script).' \
    'Optional environment variables: PYTHON_BIN (default: python3), OMP_NUM_THREADS (default: 4).'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHEMBERTA_MODEL_DIR="$1"
RESULT_ROOT="${2:-${SCRIPT_DIR}/results_multiseed_rerun}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SEEDS=(42 123 3407 2024 2025)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$CHEMBERTA_MODEL_DIR" ]]; then
  printf 'ChemBERTa model directory not found: %s\n' "$CHEMBERTA_MODEL_DIR" >&2
  exit 1
fi

# Validate dependencies and the output directory.
"$PYTHON_BIN" - "$RESULT_ROOT" <<'PY'
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
output = Path(sys.argv[1])
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit(f"Refusing to overwrite a non-empty or non-directory output path: {output}")
import dgl, numpy, pandas, rdkit, sklearn, torch, tqdm, transformers
PY

for seed in "${SEEDS[@]}"; do
  "$PYTHON_BIN" "${SCRIPT_DIR}/code/train_neurotoxpredictor.py" \
    --data "${SCRIPT_DIR}/data/neuro.csv" \
    --chemberta-model "$CHEMBERTA_MODEL_DIR" \
    --output-dir "${RESULT_ROOT}/seed_${seed}" \
    --epochs 200 \
    --seed "${seed}" \
    --split-seed 42 \
    --resample-seed 0
done

"$PYTHON_BIN" "${SCRIPT_DIR}/code/summarize_multiseed.py" \
  --result-root "${RESULT_ROOT}" \
  --run-name neurotoxpredictor \
  --seeds "${SEEDS[@]}"

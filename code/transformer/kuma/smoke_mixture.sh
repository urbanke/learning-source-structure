#!/bin/bash
# Small CPU integration check; uses a fresh temporary directory, no training data.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
python_bin=${MIXTURE_PYTHON:-python}
smoke_root=$(mktemp -d "${TMPDIR:-/tmp}/multifamily-smoke.XXXXXX")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$python_bin" code/transformer/train_mixture.py \
  --config code/transformer/configs/multifamily_smoke.json \
  --out "$smoke_root/train" --device cpu --workers 0
"$python_bin" code/transformer/eval_mixture.py sources \
  --out "$smoke_root/heldout" --n-prior 2 --n-conditional 2 \
  --n-ctx 16 --prefixes 1 4 16 --seed 2026091602
"$python_bin" code/transformer/eval_mixture.py model \
  --sources "$smoke_root/heldout" --checkpoint "$smoke_root/train/model_6.pt" \
  --out "$smoke_root/eval" --name smoke --device cpu --batch-size 2
"$python_bin" code/transformer/eval_mixture.py summary \
  --sources "$smoke_root/heldout" --evaluations "$smoke_root/eval" \
  --out "$smoke_root/comparison.json"
printf 'CPU integration smoke completed: %s\n' "$smoke_root"

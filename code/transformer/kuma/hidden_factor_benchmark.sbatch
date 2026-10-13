#!/bin/bash
# Benchmark exact Bayes before choosing a training alphabet.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
# Kuma h100 caps RAM at 5900 MiB per CPU; 32G / 8 = 4096 MiB.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$run_root/capacity"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
srun --unbuffered python code/transformer/hidden_factor_gpu.py benchmark \
  --device cuda --out "$run_root/capacity/benchmark-${SLURM_JOB_ID}.json" \
  --dimensions 2x5 2x6 3x4 2x7 3x5 2x8 2x9 --length 1025 --n-eval 4096 \
  --chunk-size 65536 --max-candidate-seconds 60 --repeats 2 --max-reference-hours 24 \
  --cache-root "$run_root/encoding_cache" --max-cache-gib 8 --max-cache-build-seconds 300

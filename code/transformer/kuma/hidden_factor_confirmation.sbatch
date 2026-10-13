#!/bin/bash
# Reuse completed Bayesian references; no training or Bayesian inference.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Separate confirmation output directory required}
baseline_root=${2:?Original 2x8 output directory required}
scaling_root=${3:?Completed scaling output directory required}
plan=${4:?Frozen original scaling plan required}
batch_size=${5:?Inference batch size required}
[[ $batch_size =~ ^[1-9][0-9]*$ ]] || { echo 'Batch size must be a positive integer.' >&2; exit 2; }
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hf-confirmation-mpl-${SLURM_JOB_ID}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
# The evaluator locks its output and reuses only identity-checked completed
# model scores. If interrupted, repeat the same launcher to finish remaining models.
srun --unbuffered python code/transformer/hidden_factor_confirmation_eval.py \
  --stage all --plan "$plan" --baseline-root "$baseline_root" \
  --scaling-root "$scaling_root" --out "$run_root" --device cuda --batch-size "$batch_size"

#!/bin/bash
# Inference on immutable saved weights; training and Bayesian work are excluded.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
out=${1:?}; plan=${2:?}; campaign=${3:?}; baseline=${4:?}; scaling=${5:?}; comparison=${6:?}
arm=${7:?}; step=${8:?}; batch=${9:?}
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/transformer/hidden_factor_campaign_comparison_eval.py \
  --plan "$plan" --campaign-root "$campaign" --baseline-root "$baseline" \
  --scaling-root "$scaling" --comparison-root "$comparison" --out "$out" \
  --arm "$arm" --step "$step" --device cuda --batch-size "$batch"

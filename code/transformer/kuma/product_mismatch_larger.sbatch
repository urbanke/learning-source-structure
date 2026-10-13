#!/bin/bash
#SBATCH --account=lthc
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
echo "Larger product mismatch evaluation starting: job=$SLURM_JOB_ID; two frozen checkpoints; no training."
cd "${SLURM_SUBMIT_DIR:?}"
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/scripts/product_mismatch_larger.py run --out "$1" --models "$2" --batch 8

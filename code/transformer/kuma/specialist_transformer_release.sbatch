#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
set -euo pipefail
echo "Specialist release allocation: job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
cd "${SLURM_SUBMIT_DIR:?}"
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/scripts/specialist_transformer_release.py run --out "$1" --plan "$2" --task "$SLURM_ARRAY_TASK_ID"

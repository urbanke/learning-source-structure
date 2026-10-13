#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --signal=USR1@180
set -euo pipefail
echo "Larger pooled model allocation: job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID:-0} mode=$1"
cd "${SLURM_SUBMIT_DIR:?}"
mode=${1:?}
out=${2:?}
plan=${3:?}
sources=${4:?}
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/scripts/pooled_transformer_large.py "$mode" \
 --plan "$plan" --out "$out" --seed "${SLURM_ARRAY_TASK_ID:-0}" --sources "$sources"

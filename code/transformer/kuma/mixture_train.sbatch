#!/bin/bash
# Submit through submit_mixture.sh so output folders exist before Slurm starts.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
run_seed=${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}
case "$run_seed" in 0|1) ;; *) echo 'Unsupported training seed' >&2; exit 2 ;; esac
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/train-${SLURM_JOB_ID}-${run_seed}.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
# Exit75 means a resumable stop, not a completed scientific run. No automatic
# requeue or submission loop: repeat the launcher to continue the same run.
resume_args=()
if [[ -f "$run_root/train/seed${run_seed}/latest.pt" ]]; then resume_args+=(--resume); fi
srun --unbuffered python code/transformer/train_mixture.py \
  --config "code/transformer/configs/multifamily_seed${run_seed}.json" \
  --out "$run_root/train/seed${run_seed}" --device cuda --workers 6 "${resume_args[@]}"

#!/bin/bash
# Frozen ordered-cohort evaluation; all generation/inference stays on compute node.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Shuffled-control output root required}
ordered_root=${2:?Original larger-model training root required}
manifest=${3:?Frozen evaluation cohort required}
module load gcc python 2>/dev/null || true
source "${SHARED_BASE_SHUFFLE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python code/transformer/shared_base_shuffle_eval.py prepare --manifest "$manifest" --out "$run_root/cohorts"
for run_seed in 0 1; do
  for role in ordered shuffled; do
    if [[ $role == ordered ]]; then
      checkpoint="$ordered_root/seed$run_seed/model_100000.pt"
    else
      checkpoint="$run_root/train/seed$run_seed/model_100000.pt"
    fi
    srun --unbuffered python code/transformer/shared_base_shuffle_eval.py evaluate \
      --manifest "$manifest" --cohorts "$run_root/cohorts" --checkpoint "$checkpoint" \
      --role "$role" --seed "$run_seed" --out "$run_root/eval/${role}_seed$run_seed" \
      --device cuda --dtype float64 --batch 4 --threads 4 --shuffle-diagnostic 16
  done
done
python code/transformer/shared_base_shuffle_eval.py summary --manifest "$manifest" \
  --evaluations "$run_root/eval" --out "$run_root/comparison"

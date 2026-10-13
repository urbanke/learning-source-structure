#!/bin/bash
# Run after both production training jobs finish; no training is done here.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Kuma h100 caps RAM at 5900 MiB per CPU; 32G / 8 = 4096 MiB.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
specialist_root="$PWD/results/transformer/runs"
for checkpoint in \
  "$run_root/train/seed0/model_300000.pt" \
  "$run_root/train/seed1/model_300000.pt" \
  "$specialist_root/d64/model_100000.pt" \
  "$specialist_root/d64_seed1/model_100000.pt" \
  "$specialist_root/prod_8x8/model_100000.pt" \
  "$specialist_root/lsa_a0_d64/model_100000.pt"; do
  [[ -f "$checkpoint" ]] || { echo "Missing final checkpoint: $checkpoint" >&2; exit 2; }
done
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/eval-${SLURM_JOB_ID}.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
srun --unbuffered python code/transformer/eval_mixture.py sources \
  --out "$run_root/heldout" --n-prior 1024 --n-conditional 256 \
  --n-ctx 1024 --prefixes 1 4 16 64 256 1024 --seed 2026091602
for run_seed in 0 1; do
  srun --unbuffered python code/transformer/eval_mixture.py model \
    --sources "$run_root/heldout" --out "$run_root/eval" \
    --checkpoint "$run_root/train/seed${run_seed}/model_300000.pt" \
    --name "pooled_seed${run_seed}" --device cuda --batch-size 16
done
for run_seed in 0 1; do
  specialist=d64
  if [[ $run_seed -eq 1 ]]; then specialist=d64_seed1; fi
  srun --unbuffered python code/transformer/eval_mixture.py model \
    --sources "$run_root/heldout" --out "$run_root/eval" \
    --checkpoint "$specialist_root/$specialist/model_100000.pt" \
    --name "shared_specialist_seed${run_seed}" --family shared_base --device cuda --batch-size 16
done
srun --unbuffered python code/transformer/eval_mixture.py model \
  --sources "$run_root/heldout" --out "$run_root/eval" \
  --checkpoint "$specialist_root/prod_8x8/model_100000.pt" \
  --name product_specialist --family product --device cuda --batch-size 16
srun --unbuffered python code/transformer/eval_mixture.py model \
  --sources "$run_root/heldout" --out "$run_root/eval" \
  --checkpoint "$specialist_root/lsa_a0_d64/model_100000.pt" \
  --name uniform_specialist --family uniform_base --device cuda --batch-size 16
python code/transformer/eval_mixture.py summary \
  --sources "$run_root/heldout" --evaluations "$run_root/eval" --out "$run_root/comparison.json" \
  --pair pooled_seed0:shared_specialist_seed0 --pair pooled_seed1:shared_specialist_seed1 \
  --pair pooled_seed0:product_specialist --pair pooled_seed1:product_specialist \
  --pair pooled_seed0:uniform_specialist --pair pooled_seed1:uniform_specialist

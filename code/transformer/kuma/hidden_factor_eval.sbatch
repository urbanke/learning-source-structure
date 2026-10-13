#!/bin/bash
# Resume evaluation of completed training runs; submit through its launcher.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Kuma h100 caps RAM at 5900 MiB per CPU; 32G / 8 = 4096 MiB.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
run_shape=${2:?Selected dimensions required}
run_stage=${3:-full}
cap_args=()
if [[ $run_stage == pilot ]]; then
  cap_args=(--max-sequences 256)
  run_summary=summary_prefix_000256.json
  run_figures=pilot_figures
elif [[ $run_stage == full ]]; then
  run_summary=summary.json
  run_figures=figures
else
  echo 'Expected pilot or full evaluation stage.' >&2; exit 2
fi
if [[ ! $run_shape =~ ^([1-9][0-9]{0,2})x([1-9][0-9]{0,2})$ ]]; then
  echo 'Invalid source dimensions' >&2; exit 2
fi
run_r=${BASH_REMATCH[1]}
run_s=${BASH_REMATCH[2]}
if (( run_r < 2 || run_s < 2 || run_r * run_s > 255 )); then
  echo 'Each dimension must be >=2 and the token alphabet <=255.' >&2; exit 2
fi
for run_seed in 0 1; do
  if [[ ! -f "$run_root/$run_shape/train/seed${run_seed}/model_100000.pt" ]]; then
    echo "Missing final checkpoint for training seed $run_seed; finish training first." >&2
    exit 2
  fi
done
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$run_root/$run_shape/eval" "$run_root/environment"
# One Slurm evaluator per output directory; a later manual job resumes it.
exec 9>"$run_root/$run_shape/eval/.slurm-evaluation.lock"
flock -n 9 || { echo 'Another evaluator holds this directory.' >&2; exit 75; }
python -m pip freeze > "$run_root/environment/eval-${run_shape}-${SLURM_JOB_ID}.txt"
python -c 'import matplotlib, torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
srun --unbuffered python code/transformer/eval_hidden_factor.py \
  --stage all --dimensions "$run_shape" --out "$run_root/$run_shape/eval" \
  --device cuda --cache-root "$run_root/encoding_cache" \
  --n-seq 4096 --length 1025 --times 1 4 16 64 256 1024 \
  "${cap_args[@]}" \
  --checkpoint "$run_root/$run_shape/train/seed0/model_100000.pt" \
  --checkpoint "$run_root/$run_shape/train/seed1/model_100000.pt"
python code/scripts/hidden_factor_figures.py \
  --summary "$run_root/$run_shape/eval/$run_summary" \
  --out "$run_root/$run_shape/eval/$run_figures"

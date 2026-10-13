#!/bin/bash
# Exact Bayes can run before either transformer finishes training.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
# Kuma normal QoS allows three days; 48h includes margin over the 36h estimate.
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
  run_figures=reference_figures
else
  echo 'Expected pilot or full reference stage.' >&2; exit 2
fi
if [[ ! $run_shape =~ ^([1-9][0-9]{0,2})x([1-9][0-9]{0,2})$ ]]; then
  echo 'Invalid source dimensions' >&2; exit 2
fi
run_r=${BASH_REMATCH[1]}
run_s=${BASH_REMATCH[2]}
if (( run_r < 2 || run_s < 2 || run_r * run_s > 255 )); then
  echo 'Each dimension must be >=2 and the token alphabet <=255.' >&2; exit 2
fi
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$run_root/$run_shape/eval" "$run_root/environment"
# Shared with the later full evaluation; do not write both stages concurrently.
exec 9>"$run_root/$run_shape/eval/.slurm-evaluation.lock"
flock -n 9 || { echo 'Another evaluator holds this directory.' >&2; exit 75; }
python -m pip freeze > "$run_root/environment/references-${run_shape}-${SLURM_JOB_ID}.txt"
python -c 'import matplotlib, torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
echo "Starting $run_stage exact Bayesian references for $run_shape; cohort size 4096, pilot size 256."
eval_args=(--dimensions "$run_shape" --out "$run_root/$run_shape/eval"
  --device cuda --cache-root "$run_root/encoding_cache" --chunk-size 65536
  --n-seq 4096 --length 1025 --times 1 4 16 64 256 1024)
srun --unbuffered python code/transformer/eval_hidden_factor.py \
  --stage all --bayes-only "${eval_args[@]}" "${cap_args[@]}"
# The pilot is most useful with transformer curves. Include any completed
# final checkpoint, but do not wait for training or load a partial checkpoint.
if [[ $run_stage == pilot ]]; then
  model_args=()
  for run_seed in 0 1; do
    run_checkpoint="$run_root/$run_shape/train/seed${run_seed}/model_100000.pt"
    if [[ -f "$run_checkpoint" ]]; then model_args+=(--checkpoint "$run_checkpoint"); fi
  done
  if [[ ${#model_args[@]} -gt 0 ]]; then
    srun --unbuffered python code/transformer/eval_hidden_factor.py \
      --stage models "${eval_args[@]}" "${cap_args[@]}" "${model_args[@]}"
    srun --unbuffered python code/transformer/eval_hidden_factor.py \
      --stage summary "${eval_args[@]}" "${cap_args[@]}"
  else
    echo 'Final transformer checkpoints are not available; pilot plots contain Bayes references only.'
  fi
fi
python code/scripts/hidden_factor_figures.py \
  --summary "$run_root/$run_shape/eval/$run_summary" \
  --out "$run_root/$run_shape/eval/$run_figures"
echo "Completed $run_stage reference stage; summary: $run_root/$run_shape/eval/$run_summary"
if [[ $run_stage == pilot ]]; then
  echo 'Stopped after 256 sequences for review; the remaining reference computation has not been submitted.'
fi

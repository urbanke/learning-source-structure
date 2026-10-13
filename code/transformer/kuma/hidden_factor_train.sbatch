#!/bin/bash
# Separate capped throughput check and selected-size production runs.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
run_shape=${2:?Selected dimensions required}
run_kind=${3:?train or throughput required}
if [[ ! $run_shape =~ ^([1-9][0-9]{0,2})x([1-9][0-9]{0,2})$ ]]; then
  echo 'Invalid source dimensions' >&2; exit 2
fi
run_r=${BASH_REMATCH[1]}
run_s=${BASH_REMATCH[2]}
if (( run_r < 2 || run_s < 2 || run_r * run_s > 255 )); then
  echo 'Each dimension must be >=2 and the token alphabet <=255.' >&2; exit 2
fi
if [[ $run_kind == throughput ]]; then
  run_seed=0
  run_out="$run_root/$run_shape/throughput/seed0"
  cap_args=(--max-steps 1000)
elif [[ $run_kind == train ]]; then
  run_seed=${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}
  run_out="$run_root/$run_shape/train/seed${run_seed}"
  cap_args=()
else
  echo 'Unsupported run kind' >&2; exit 2
fi
case "$run_seed" in 0|1) ;; *) echo 'Unsupported training seed' >&2; exit 2 ;; esac
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/${run_kind}-${run_shape}-${SLURM_JOB_ID}-${run_seed}.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
resume_args=()
if [[ -f "$run_out/latest.pt" ]]; then resume_args+=(--resume); fi
set +e
srun --unbuffered python code/transformer/train_hidden_factor.py \
  --config "code/transformer/configs/hidden_factor_seed${run_seed}.json" \
  --dimensions "$run_r" "$run_s" \
  --out "$run_out" --device cuda --workers 6 "${resume_args[@]}" "${cap_args[@]}"
run_status=$?
set -e
# A capped benchmark succeeds only at its intended cap; signal stops stay 75.
if [[ $run_kind == throughput && $run_status -eq 75 ]]; then
  python - "$run_out/status.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    state = json.load(handle)
sys.exit(0 if state['step'] == 1000 and state['reason'] == 'max_steps' else 75)
PY
  exit $?
fi
# A production exit 75 is resumable, not complete. No automatic resubmission.
exit "$run_status"

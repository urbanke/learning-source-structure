#!/bin/bash
# Campaign training and development-only milestone evaluation on a compute node.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
run_root=${1:?Output required}; parent_root=${2:?Parent runs required}; baseline_eval=${3:?Development inputs required}
plan=${4:?Plan required}; mode=${5:?Mode required}
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hf-campaign-${SLURM_JOB_ID}"
if [[ $mode == benchmark ]]; then
  arm=xlarge; seed=0; run_out="$run_root/benchmark/xlarge/seed0"
  [[ ! -f "$run_out/latest.pt" ]] || { echo 'Benchmark output exists; use a new --out for an uninterrupted benchmark.' >&2; exit 2; }
  targets=(1000)
elif [[ $mode == train ]]; then
  case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 through 3}" in
    0) arm=continue_large; seed=0 ;; 1) arm=continue_large; seed=1 ;;
    2) arm=xlarge; seed=0 ;; 3) arm=xlarge; seed=1 ;; *) echo 'Unexpected array task.' >&2; exit 2 ;;
  esac
  run_out="$run_root/train/$arm/seed$seed"
  if [[ $arm == xlarge ]]; then
    python code/scripts/hidden_factor_campaign_benchmark.py --plan "$plan" --run "$run_root/benchmark/xlarge/seed0" --check-only
    targets=(100000 200000 400000)
  else targets=(200000 400000); fi
else echo 'Unexpected run mode.' >&2; exit 2; fi
claim=$(python code/scripts/campaign_job_guard.py acquire --family hidden_factor --root "$run_root" --mode "$mode" --arm "$arm" --seed "$seed")
# EXIT only: Slurm USR1 still reaches the existing checkpoint handlers unchanged.
trap 'campaign_exit=$?; python code/scripts/campaign_job_guard.py finish --claim "$claim" --exit-code "$campaign_exit" || true; exit "$campaign_exit"' EXIT
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__,torch.cuda.get_device_name(0))'
for target in "${targets[@]}"; do
  step=$(python - "$run_out/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print(json.loads(p.read_text())['step'] if p.exists() else 0)
PY
)
  if [[ $step -lt $target ]]; then
    command=(srun --unbuffered python code/transformer/hidden_factor_campaign_train.py --plan "$plan"
      --arm "$arm" --seed "$seed" --out "$run_out" --device cuda --workers 6 --max-steps "$target")
    if [[ $arm == continue_large ]]; then command+=(--parent-checkpoint "$parent_root/seed$seed/latest.pt" --parent-weights "$parent_root/seed$seed/model_100000.pt"); fi
    if [[ -f "$run_out/latest.pt" ]]; then command+=(--resume); fi
    set +e; "${command[@]}"; status=$?; set -e
    [[ $status -eq 0 || $status -eq 75 ]] || exit "$status"
    # Only the requested milestone cap permits immediate evaluation/continuation.
    # A saved signal stop returns75 to the scheduler and waits for an explicit resume.
    python - "$run_out/status.json" "$target" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));target=int(sys.argv[2])
raise SystemExit(0 if s['step']==target and s['reason'] in ('completed','max_steps') else 75)
PY
  fi
  if [[ $mode == benchmark ]]; then
    python code/scripts/hidden_factor_campaign_benchmark.py --plan "$plan" --run "$run_out"
  else
    srun --unbuffered python code/transformer/hidden_factor_campaign_eval.py --plan "$plan" \
      --baseline-eval "$baseline_eval" --arm "$arm" --seed "$seed" \
      --checkpoint "$run_out/model_${target}.pt" --out "$run_root/eval/$arm/seed$seed/step$target" --device cuda --batch-size 8
  fi
done

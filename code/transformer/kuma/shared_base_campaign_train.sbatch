#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?}; run_mode=${2:?}; arm=${3:?}; plan=${4:?}; parent_root=${5:?}; cohorts=${6:?}; resume_mode=${7:?}
case "$arm" in continue_large|xlarge) ;; *) exit 2 ;; esac
case "$run_mode" in benchmark) seed=0; run_out="$run_root/benchmark/seed0" ;;
 train|eval) seed=${SLURM_ARRAY_TASK_ID:?}; run_out="$run_root/$arm/seed$seed" ;; *) exit 2 ;; esac
case "$seed" in 0|1) ;; *) exit 2 ;; esac
module load gcc python 2>/dev/null || true
source "${SHARED_BASE_CAMPAIGN_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
claim=$(python code/scripts/campaign_job_guard.py acquire --family shared_base --root "$run_root" --mode "$run_mode" --arm "$arm" --seed "$seed")
# Keep the claim across all milestones; do not intercept the trainer's USR1.
trap 'campaign_exit=$?; python code/scripts/campaign_job_guard.py finish --claim "$claim" --exit-code "$campaign_exit" || true; exit "$campaign_exit"' EXIT
if [[ $run_mode == eval ]]; then
  srun --unbuffered python code/transformer/shared_base_campaign_eval.py --plan "$plan" score --run "$run_out" --cohorts "$cohorts" --out "$run_root/eval/$arm/seed$seed"
  exit $?
fi
# The tested4-layer continuation is independent of this new-model benchmark.
if [[ $run_mode == train && $arm == xlarge ]]; then
  python code/scripts/shared_base_campaign_benchmark.py --plan "$plan" --run "$run_root/benchmark/seed0" --check-only
fi
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/${run_mode}-${arm}-${SLURM_JOB_ID}-${seed}.txt"
if [[ $run_mode == benchmark ]]; then targets=(1000)
elif [[ $arm == continue_large ]]; then targets=(200000 400000)
else targets=(100000 200000 400000); fi
next_resume=$resume_mode
for target in "${targets[@]}"; do
  step=$(python - "$run_out/status.json" <<'PYCODE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print(json.loads(p.read_text())['step'] if p.exists() else 0)
PYCODE
)
  if [[ $step -lt $target ]]; then
    command=(srun --unbuffered python code/transformer/shared_base_campaign_train.py --plan "$plan" --arm "$arm" --seed "$seed" --out "$run_out" --device cuda --workers 6 --max-steps "$target")
    if [[ $arm == continue_large ]]; then command+=(--parent-root "$parent_root"); fi
    if [[ $next_resume == resume ]]; then command+=(--resume); fi
    set +e
    "${command[@]}"
    code=$?
    set -e
    [[ $code -eq 0 || $code -eq 75 ]] || exit "$code"
    # A programmed milestone may continue within this allocation. A signal stop
    # must end the job and wait for the user's explicit --resume submission.
    python - "$run_out/status.json" "$target" <<'PYCODE'
import json,sys
s=json.load(open(sys.argv[1]));target=int(sys.argv[2])
raise SystemExit(0 if s['step']==target and s['reason'] in ('completed','max_steps') else 75)
PYCODE
    next_resume=resume
  fi
  if [[ $run_mode == benchmark ]]; then
    python code/scripts/shared_base_campaign_benchmark.py --plan "$plan" --run "$run_out"
  elif [[ -f $cohorts/manifest.json ]]; then
    srun --unbuffered python code/transformer/shared_base_campaign_eval.py --plan "$plan" score --run "$run_out" --cohorts "$cohorts" --out "$run_root/eval/$arm/seed$seed"
  else
    echo "Checkpoint $target saved; evaluation inputs absent. Transfer compact cohorts, then use --evaluate."
  fi
done

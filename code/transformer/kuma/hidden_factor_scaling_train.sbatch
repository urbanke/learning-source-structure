#!/bin/bash
# A benchmark or one independently checkpointed arm/seed; no Bayesian inference.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Scaling output directory required}
baseline_root=${2:?Original 2x8 output directory required}
run_kind=${3:?benchmark or train required}
plan=${4:?Frozen plan required}
if [[ $run_kind == benchmark ]]; then
  run_arm=large
  run_seed=0
  run_out="$run_root/benchmark/large/seed0"
elif [[ $run_kind == train ]]; then
  case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 through 3}" in
    0) run_arm=longsmall; run_seed=0 ;;
    1) run_arm=longsmall; run_seed=1 ;;
    2) run_arm=large; run_seed=0 ;;
    3) run_arm=large; run_seed=1 ;;
    *) echo 'Expected array task 0 through 3.' >&2; exit 2 ;;
  esac
  run_out="$run_root/train/$run_arm/seed$run_seed"
else
  echo 'Expected benchmark or train.' >&2; exit 2
fi
if [[ $run_arm == longsmall ]]; then
  final_step=400000
else
  final_step=100000
fi
module load gcc python 2>/dev/null || true
source "${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hf-scaling-mpl-${SLURM_JOB_ID}-${run_seed}"
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/${run_kind}-${run_arm}-${SLURM_JOB_ID}-${run_seed}.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
train_command=(srun --unbuffered python code/transformer/train_hidden_factor_scaling.py
  --plan "$plan" --arm "$run_arm" --seed "$run_seed" --out "$run_out"
  --device cuda --workers 6)
if [[ $run_arm == longsmall ]]; then
  train_command+=(--parent-checkpoint "$baseline_root/train/seed$run_seed/latest.pt")
fi
if [[ -f "$run_out/latest.pt" ]]; then train_command+=(--resume); fi
if [[ $run_kind == benchmark ]]; then train_command+=(--max-steps 1000); fi
set +e
"${train_command[@]}"
run_status=$?
set -e
if [[ $run_kind == benchmark ]]; then
  # Only the intended cap is a successful throughput benchmark.
  if [[ $run_status -ne 75 ]]; then
    echo "Benchmark did not stop at its intended cap (exit $run_status)." >&2
    exit 2
  fi
  python code/scripts/hidden_factor_scaling_benchmark.py --plan "$plan" --run "$run_out"
  exit $?
fi
if [[ $run_status -ne 0 ]]; then
  # A signalled or capped run is resumable, never declared complete.
  exit "$run_status"
fi
# Score only the final planned model, using the original frozen pilot/reference.
srun --unbuffered python code/transformer/hidden_factor_scaling_eval.py model \
  --plan "$plan" --baseline-eval "$baseline_root/eval" --arm "$run_arm" \
  --checkpoint "$run_out/model_${final_step}.pt" --out "$run_root/eval" \
  --device cuda --batch-size 16

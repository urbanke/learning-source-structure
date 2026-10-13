#!/bin/bash
# Separate larger shared-base training; no new Bayesian calculation or scoring.
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@180
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Scaling output directory required}
run_kind=${2:?benchmark or train required}
plan=${3:?Frozen plan required}
if [[ $run_kind == benchmark ]]; then
  run_seed=0
  run_out="$run_root/benchmark/seed0"
elif [[ $run_kind == train ]]; then
  case "${SLURM_ARRAY_TASK_ID:?Expected array task 0 or 1}" in
    0|1) run_seed=$SLURM_ARRAY_TASK_ID ;;
    *) echo 'Expected array task 0 or 1.' >&2; exit 2 ;;
  esac
  run_out="$run_root/train/seed$run_seed"
else
  echo 'Expected benchmark or train.' >&2; exit 2
fi
module load gcc python 2>/dev/null || true
source "${SHARED_BASE_SCALING_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ $run_kind == train ]]; then
  python code/scripts/shared_base_scaling_benchmark.py --plan "$plan" --run "$run_root/benchmark/seed0" --check-only
fi
mkdir -p "$run_root/environment"
python -m pip freeze > "$run_root/environment/${run_kind}-${SLURM_JOB_ID}-${run_seed}.txt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
train_command=(srun --unbuffered python code/transformer/train_shared_base_scaling.py
  --plan "$plan" --seed "$run_seed" --out "$run_out" --device cuda --workers 6)
if [[ -f "$run_out/latest.pt" ]]; then train_command+=(--resume); fi
if [[ $run_kind == benchmark ]]; then train_command+=(--max-steps 1000); fi
set +e
"${train_command[@]}"
run_status=$?
set -e
if [[ $run_kind == benchmark ]]; then
  # A signal/early exit is never accepted as the intended throughput cap.
  if [[ $run_status -ne 75 ]]; then
    echo "Benchmark did not stop at its intended cap (exit $run_status)." >&2
    exit 2
  fi
  python code/scripts/shared_base_scaling_benchmark.py --plan "$plan" --run "$run_out"
  exit $?
fi
exit "$run_status"

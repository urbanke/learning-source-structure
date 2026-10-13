#!/bin/bash
# Production-sized throughput/resume check in a separate directory.
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_root=${1:?Absolute experiment output directory required}
module load gcc python 2>/dev/null || true
source "${MIXTURE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
resume_args=()
if [[ -f "$run_root/benchmark/seed0/latest.pt" ]]; then resume_args+=(--resume); fi
set +e
srun --unbuffered python code/transformer/train_mixture.py \
  --config code/transformer/configs/multifamily_seed0.json \
  --out "$run_root/benchmark/seed0" --device cuda --workers 6 "${resume_args[@]}" --max-steps 1000
run_status=$?
set -e
# The deliberate 1,000-step cap is a successful benchmark, not production completion.
if [[ $run_status -eq 75 ]]; then
  python - "$run_root/benchmark/seed0/status.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
sys.exit(0 if state['step'] == 1000 and state['reason'] == 'max_steps' else 75)
PY
  exit $?
fi
exit "$run_status"

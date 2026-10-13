#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
echo "Expanded copying evaluation starting: job=${SLURM_JOB_ID}; frozen model pair, 16384 histories per depth."
cd "${SLURM_SUBMIT_DIR:?}"
out=${1:?New output required}
expected_runner=${2:?Frozen runner hash required}
expected_plan=${3:?Frozen plan hash required}
python3 - "$expected_runner" "$expected_plan" <<'PY'
import hashlib,pathlib,sys
for file,want in zip(('code/scripts/extend_shared_base_shuffle_large.py','code/transformer/configs/shared_base_shuffle_extension_2026-09-23.json'),sys.argv[1:]):
    assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==want, 'Launch code or plan changed'
PY
module load gcc python 2>/dev/null || true
source "${SHARED_BASE_SHUFFLE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/scripts/extend_shared_base_shuffle_large.py --out "$out" \
  --parent output/shared_base_shuffle_extension_recovery_2026-09-22 \
  --ordered-root output/shared_base_scaling_2026-09-17/train \
  --shuffled-root output/shared_base_shuffle_2026-09-17/train

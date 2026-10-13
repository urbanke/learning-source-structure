#!/bin/bash
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
echo "Paired shuffle extension starting: job=${SLURM_JOB_ID}; no training or Bayesian computation."
cd "${SLURM_SUBMIT_DIR:?}"
out=${1:?New output required}
expected_runner=${2:?Frozen runner hash required}
expected_plan=${3:?Frozen plan hash required}
python3 - "$expected_runner" "$expected_plan" <<'PY'
import hashlib,pathlib,sys
for file,want in zip(('code/scripts/extend_shared_base_shuffle.py','code/transformer/configs/shared_base_shuffle_extension_2026-09-22.json'),sys.argv[1:]):
    assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==want, 'Launch code/plan changed'
PY
module load gcc python 2>/dev/null || true
source "${SHARED_BASE_SHUFFLE_VENV:-$HOME/venv-tf}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --unbuffered python code/scripts/extend_shared_base_shuffle.py --out "$out" \
  --ordered-root output/shared_base_scaling_2026-09-17/train --parent output/shared_base_shuffle_2026-09-17

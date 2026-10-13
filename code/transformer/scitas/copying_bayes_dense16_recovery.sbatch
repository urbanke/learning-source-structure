#!/bin/bash
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --no-requeue
set -euo pipefail
module load gcc python
source "$HOME/venv-tf/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
stage=$1
out=$2
expected=$3
expected_wrapper=$4
driver=code/scripts/copying_bayes_dense16_recovery.py
wrapper=code/transformer/scitas/copying_bayes_dense16_recovery.sbatch
python - "$driver" "$expected" "$wrapper" "$expected_wrapper" <<'PY'
import hashlib,sys
for path,expected in zip(sys.argv[1::2],sys.argv[2::2]):
    assert hashlib.sha256(open(path,'rb').read()).hexdigest()==expected,'Frozen code changed'
PY
args=("$stage" --out "$out")
if [[ "$stage" == prepare ]]; then args+=(--parent "$5" --terminal-receipt "$out/terminal.json"); fi
if [[ "$stage" == dispatch ]]; then args+=(--index "$SLURM_ARRAY_TASK_ID"); fi
exec python -u "$driver" "${args[@]}"

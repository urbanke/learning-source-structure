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
driver=code/scripts/copying_bayes_dense16_final.py
wrapper=code/transformer/scitas/copying_bayes_dense16_final.sbatch
python - "$driver" "$3" "$wrapper" "$4" <<'PY'
import hashlib,sys
for path,expected in zip(sys.argv[1::2],sys.argv[2::2]):
    assert hashlib.sha256(open(path,'rb').read()).hexdigest()==expected,'Frozen wrapper changed'
PY
args=("$stage" --out "$out")
if [[ "$stage" == prepare ]]; then args+=(--first "$5" --terminal "$out/terminal.json"); fi
exec python -u "$driver" "${args[@]}"

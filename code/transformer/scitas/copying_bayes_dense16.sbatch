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
expected_driver=$3
expected_wrapper=$4
expected_plan=$5
driver=code/scripts/copying_bayes_dense16.py
wrapper=code/transformer/scitas/copying_bayes_dense16.sbatch
python - "$driver" "$expected_driver" "$wrapper" "$expected_wrapper" "$out/plan.json" "$expected_plan" <<'PY'
import hashlib,sys
for path,expected in zip(sys.argv[1::2],sys.argv[2::2]):
    assert hashlib.sha256(open(path,'rb').read()).hexdigest()==expected, 'Frozen launch input changed: '+path
PY
args=("$stage" --out "$out")
if [[ "$stage" == prepare ]]; then args+=(--dependencies "$HOME/Projects/shared_base_dependencies_2026-09-16"); fi
if [[ "$stage" == shard || "$stage" == bank ]]; then args+=(--index "$SLURM_ARRAY_TASK_ID"); fi
exec python -u "$driver" "${args[@]}"

#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
plan=${1:?}; raw=${2:?}; dependencies=${3:?}; recovery=${4:?}; out=${5:?}; stage=${6:?}
case "$stage" in prepare|merge) ;; *) exit 2 ;; esac
python3 - "$out/submission.json" "$PWD" "$plan" <<'PY'
import hashlib, json, sys
from pathlib import Path
identity = json.loads(Path(sys.argv[1]).read_text())['identity']
root, plan = Path(sys.argv[2]), Path(sys.argv[3])
assert hashlib.sha256(plan.read_bytes()).hexdigest() == identity['plan_sha256']
for name, expected in identity['code_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
PY
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_mean_population_resume.py "$stage" \
    --plan "$plan" --raw "$raw" --dependencies "$dependencies" \
    --recovery "$recovery" --out "$out"

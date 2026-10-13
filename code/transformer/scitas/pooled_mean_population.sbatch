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
echo "Pooled mean t16 allocation starting: job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:--} stage=${5:-missing}."
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
plan=${1:?}; raw=${2:?}; dependencies=${3:?}; out=${4:?}; stage=${5:?}; merge_stage=${6:?}; offset=${7:?}
case "$stage" in prepare|calibration|population|merge) ;; *) echo 'Unknown stage' >&2; exit 2 ;; esac
[[ $offset =~ ^[0-9]+$ ]] || { echo 'Invalid array offset' >&2; exit 2; }
source code/transformer/scitas/environment.sh
# Small source hashes are checked on the compute node before importing numerics.
python - "$out/submission.json" "$PWD" "$plan" <<'PY'
import hashlib, json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())['identity']
root, plan = Path(sys.argv[2]), Path(sys.argv[3])
assert record['plan_sha256'] == hashlib.sha256(plan.read_bytes()).hexdigest(), 'Plan changed after submission'
for name, expected in record['code_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'Code changed after submission: ' + name
print('Frozen launch code and plan verified.', flush=True)
PY
command=(srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_mean_population.py "$stage"
    --plan "$plan" --raw "$raw" --dependencies "$dependencies" --out "$out")
if [[ $stage == calibration || $stage == population ]]; then
    task=${SLURM_ARRAY_TASK_ID:?Array task missing}
    [[ $task =~ ^[0-9]+$ ]] || { echo 'Invalid array task' >&2; exit 2; }
    command+=(--task-index "$((offset + task))")
elif [[ $stage == merge ]]; then
    case "$merge_stage" in calibration|population) ;; *) echo 'Unknown merge stage' >&2; exit 2 ;; esac
    command+=(--stage "$merge_stage")
fi
"${command[@]}"

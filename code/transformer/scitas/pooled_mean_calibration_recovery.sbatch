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
echo "Pooled mean t16 recovery allocation starting: job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:--} stage=${6:-missing}."
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
plan=${1:?}; raw=${2:?}; dependencies=${3:?}; parent=${4:?}; out=${5:?}; stage=${6:?}
case "$stage" in prepare|recover|merge) ;; *) echo 'Unknown recovery stage' >&2; exit 2 ;; esac
python3 - "$out/submission.json" "$PWD" "$plan" "$raw" "$dependencies" "$parent" "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())['identity']
root, plan = Path(sys.argv[2]), Path(sys.argv[3])
assert receipt['original_plan_sha256'] == hashlib.sha256(plan.read_bytes()).hexdigest(), 'Original plan changed'
for key, value in zip(('original_plan', 'raw', 'dependencies', 'parent', 'out'), sys.argv[3:8]):
    assert Path(receipt[key]).resolve() == Path(value).resolve(), 'Launch path changed: ' + key
for name, expected in receipt['code_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'Recovery code changed: ' + name
print('Frozen recovery launch code and original plan verified.', flush=True)
PY
source code/transformer/scitas/environment.sh
command=(srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/recover_pooled_mean_calibration.py "$stage"
    --plan "$plan" --raw "$raw" --dependencies "$dependencies" --parent "$parent" --out "$out")
if [[ $stage == recover ]]; then
    task=${SLURM_ARRAY_TASK_ID:?Recovery array task missing}
    [[ $task =~ ^[0-9]+$ ]] || { echo 'Invalid recovery array task' >&2; exit 2; }
    command+=(--recovery-index "$task")
fi
"${command[@]}"

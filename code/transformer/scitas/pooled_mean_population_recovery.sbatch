#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from checkout}"
launch=${1:?}; expected=${2:?}; stage=${3:?}; offset=${4:-0}
echo "Pooled t16 missing-repetition recovery: job=${SLURM_JOB_ID} stage=${stage} task=${SLURM_ARRAY_TASK_ID:--}"
python3 - "$launch" "$expected" "$PWD" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
r=json.loads(p.read_text()); root=pathlib.Path(sys.argv[3])
for f,h in r['code_sha256'].items(): assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h,f
PY
source code/transformer/scitas/environment.sh
args=("$stage" --launch "$launch")
if [[ "$stage" == recover ]]; then args+=(--index "$(( ${SLURM_ARRAY_TASK_ID:?} + offset ))"); fi
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/recover_pooled_mean_population.py "${args[@]}"

#!/bin/bash -l
#SBATCH --account=lthc
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
echo "Pooled context64 starting: job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:--} stage=${1:?}."
cd "${SLURM_SUBMIT_DIR:?}"
stage=$1
shift
source code/transformer/scitas/environment.sh
cmd=(python code/scripts/pooled_mean_context64.py "$stage" "$@")
if [[ $stage == population ]]; then cmd+=(--index "${SLURM_ARRAY_TASK_ID:?}"); fi
srun --ntasks=1 --cpu-bind=cores --unbuffered "${cmd[@]}"

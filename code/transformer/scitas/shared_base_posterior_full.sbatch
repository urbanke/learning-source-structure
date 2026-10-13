#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
launch=${1:?Launch receipt required}
slot=${2:?Worker slot required}
generation=${3:?Generation required}
export SHARED_BASE_VENV=${4:?Prepared environment required}
if [[ $slot == array ]]; then slot=${SLURM_ARRAY_TASK_ID:?Array slot missing}; fi
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/shared_base_posterior_full.py worker --launch "$launch" --slot "$slot" --generation "$generation"

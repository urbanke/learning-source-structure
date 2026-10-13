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
source code/transformer/scitas/environment.sh
plan=${1:?}; raw=${2:?}; dependencies=${3:?}; out=${4:?}; calibration=${5:?}; original_out=${6:?}; after_job=${7:?}; mode=${8:?}
command=(srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_predictive_timing_extension.py "$mode"
 --plan "$plan" --raw "$raw" --dependencies "$dependencies" --out "$out" --calibration "$calibration"
 --original-out "$original_out" --after-job "$after_job")
if [[ $mode == run ]]; then
 tasks=(2 10 13 18 21 24 27 30 33)
 index=${SLURM_ARRAY_TASK_ID:?}
 [[ $index =~ ^[0-9]+$ && $index -lt ${#tasks[@]} ]] || { echo 'Invalid timing task index' >&2; exit 2; }
 command+=(--task-index "${tasks[$index]}")
fi
"${command[@]}"

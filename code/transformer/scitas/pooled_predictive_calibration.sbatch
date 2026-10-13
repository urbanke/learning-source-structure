#!/bin/bash
# Exact short-reference validation only; all hashes and calculations use a CPU allocation.
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source code/transformer/scitas/environment.sh
plan=${1:?}; dependencies=${2:?}; run_root=${3:?}; attempt=${4:?}; mode=${5:?}
command=(srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_predictive_calibration.py
  "$mode" --plan "$plan" --attempt "$attempt" --store "$dependencies/tables/anchors_prod"
  --src "$dependencies/src" --out "$run_root")
if [[ $mode == run ]]; then command+=(--task-index "${SLURM_ARRAY_TASK_ID:?}")
elif [[ $mode != merge ]]; then echo 'Unexpected mode' >&2; exit 2; fi
"${command[@]}"

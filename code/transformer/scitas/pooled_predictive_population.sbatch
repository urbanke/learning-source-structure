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
plan=${1:?}; raw=${2:?}; dependencies=${3:?}; out=${4:?}; calibration=${5:?}; stage=${6:?}; mode=${7:?}; observed=${8:?}; task_map=${9:?}
command=(srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_predictive_population.py "$mode"
 --plan "$plan" --raw "$raw" --dependencies "$dependencies" --out "$out" --calibration "$calibration" --stage "$stage")
IFS=',' read -r -a observed_times <<< "$observed"
for t in "${observed_times[@]}"; do command+=(--observed "$t"); done
if [[ $mode == run ]]; then
 IFS=',' read -r -a global_tasks <<< "$task_map"
 array_index=${SLURM_ARRAY_TASK_ID:?}
 [[ $array_index =~ ^[0-9]+$ && $array_index -lt ${#global_tasks[@]} ]] || { echo 'Invalid array mapping' >&2; exit 2; }
 command+=(--task-index "${global_tasks[$array_index]}")
fi
"${command[@]}"

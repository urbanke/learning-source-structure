#!/bin/bash
# Feasibility only: one existing development history from each generating family.
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
source code/transformer/scitas/environment.sh
raw=${1:?}; dependencies=${2:?}; run_root=${3:?}; observed=${4:?}
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) family=product ;; 1) family=shared_base ;; 2) family=uniform_base ;; *) exit 2 ;;
esac
case "$observed" in 16|64) ;; *) exit 2 ;; esac
case_root="$run_root/t$observed/$family"
mkdir -p "$case_root"
started=$SECONDS
if [[ ! -f "$case_root/selection.json" ]]; then
  srun --ntasks=1 --unbuffered python code/scripts/pooled_predictive_pilot.py freeze \
    --raw "$raw" --out "$case_root/selection.json"
fi
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/pooled_predictive_pilot.py case \
  --raw "$raw" --plan "$case_root/selection.json" --out "$case_root/results" \
  --family "$family" --index 1 --development --observed "$observed" \
  --store "$dependencies/tables/anchors_prod" --src "$dependencies/src" \
  --particles 32 --temperatures 8 --moves 1 --replicates 2 \
  --alpha-order 16 --seconds-per-replica 300
elapsed=$((SECONDS-started))
printf 'Completed predictive-output smoke test for %s at %s observations in %s seconds including input preparation. This is not an accuracy approval.\n' "$family" "$observed" "$elapsed"

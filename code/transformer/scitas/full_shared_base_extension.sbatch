#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
# Jed standard limits memory to 7000 MB/core. These cores provide 48 GiB
# for doubled saved-chain arrays and their independent readback audit.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
plan=${1:?Extension plan required}
parent=${2:?Parent launch required}
out=${3:?Separate extension output required}
export SHARED_BASE_VENV=${4:?Prepared environment required}
mode=${5:?run or merge required}
source code/transformer/scitas/environment.sh
args=("$mode" --plan "$plan" --parent-launch "$parent" --out "$out")
if [[ $mode == run ]]; then
  args+=(--index "${SLURM_ARRAY_TASK_ID:?Fixed task index required}")
elif [[ $mode != merge ]]; then
  echo "Invalid mode: $mode" >&2
  exit 2
fi
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/full_shared_base_extension.py "${args[@]}"

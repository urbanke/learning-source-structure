#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
plan=${1:?Plan required}
parent=${2:?Original launch required}
previous=${3:?Previous extension output required}
out=${4:?Separate continuation output required}
export SHARED_BASE_VENV=${5:?Prepared environment required}
mode=${6:?run or merge required}
[[ $mode == run || $mode == merge ]]
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/full_shared_base_continuation.py \
  "$mode" --plan "$plan" --parent-launch "$parent" --previous-output "$previous" --out "$out"

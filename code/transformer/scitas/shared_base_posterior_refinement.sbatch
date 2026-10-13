#!/bin/bash -l
# Extra CPUs provide the memory allowance; numerical work stays single-threaded.
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
export SHARED_BASE_VENV="${2:?Prepared Python environment required}"
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/refine_shared_base_posterior.py run --launch "${1:?Refinement launch required}"

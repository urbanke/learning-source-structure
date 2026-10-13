#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
export SHARED_BASE_VENV="${2:?Prepared Python environment required}"
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/revalidate_shared_base_depth.py run --launch "${1:?Revalidation launch required}"

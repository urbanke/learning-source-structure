#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
export SHARED_BASE_VENV="${2:?Prepared Python environment required}"
source code/transformer/scitas/environment.sh
# One numerical thread; the second allocated core provides the memory allowance.
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/shared_base_posterior_cluster.py run --launch "${1:?Launch manifest required}"

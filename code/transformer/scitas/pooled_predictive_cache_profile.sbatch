#!/bin/bash
# Two sequential diagnostic repetitions, never a population or accuracy run.
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source code/transformer/scitas/environment.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/scripts/profile_pooled_predictive_cache.py run --out "${1:?}"

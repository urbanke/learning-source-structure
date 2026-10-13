#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
# 32 GiB / 8 CPUs = 4096 MiB per CPU, below Kuma's 5900 MiB cap.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project checkout}"
source code/transformer/scitas/environment.sh
srun --ntasks=1 --cpu-bind=cores --unbuffered python code/transformer/scitas/dispatch_shared_base.py --launch "${1:?Launch manifest required}"

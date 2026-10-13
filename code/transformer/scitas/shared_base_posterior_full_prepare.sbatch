#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Input preparation must run in a compute allocation}"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
inputs=${1:?Completed input directory required}
scratch_root=${2:?Shared scratch root required}
export SHARED_BASE_VENV=${3:?Prepared environment required}
source code/transformer/scitas/environment.sh
output="$scratch_root/shared_base_posterior_full_2026-09-17"

# Prevent concurrent preparation jobs from checking or submitting twice.
# The numerical launch itself also retains its existing durable submit receipt.
exec 9>"$scratch_root/shared_base_posterior_full_preparation_2026-09-18/prepare.lock"
flock -n 9 || { echo 'Another preparation job holds the lock.' >&2; exit 2; }
[[ -d "$inputs" ]] || { echo 'Input extraction has not finished; wait for its verification report.' >&2; exit 2; }
echo 'Verifying frozen inputs on a compute node, then submitting all remaining shared-base calculations.'
echo 'This reuses the unpacked files. Checksums may take several minutes.'
srun --ntasks=1 python code/scripts/shared_base_posterior_full.py launch \
  --inputs "$inputs" --out "$output" --venv "$SHARED_BASE_VENV" --submit
echo 'Full shared-base calculation submitted; inspect its worker job ID above.'

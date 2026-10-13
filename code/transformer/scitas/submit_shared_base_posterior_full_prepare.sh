#!/bin/bash
# Login-node work is limited to directory setup and one Slurm submission.
set -euo pipefail
if [[ $# != 1 || $1 != --submit ]]; then
  echo "Usage: bash $0 --submit" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch_root=/scratch/urbanke
log_root="$scratch_root/shared_base_posterior_full_preparation_2026-09-18"
mkdir -p "$log_root"
cd "$repo_root"
sbatch --parsable --chdir="$repo_root" --job-name=sb-full-prepare \
  --output="$log_root/prepare-%j.out" \
  "$repo_root/code/transformer/scitas/shared_base_posterior_full_prepare.sbatch" \
  "$HOME/Projects/shared_base_posterior_full_inputs_2026-09-17" "$scratch_root" "$HOME/venv-tf"

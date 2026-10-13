#!/bin/bash
# Prepare/submit the two pooled-model runs, or the separate GPU benchmark.
# Default: print the exact command. --submit is required to contact Slurm.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
do_benchmark=0
run_root="$repo_root/output/multifamily_2026-09-16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --benchmark) do_benchmark=1; shift ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
submit_args=(sbatch --parsable --chdir="$repo_root")
if [[ -n ${MIXTURE_ACCOUNT:-} ]]; then submit_args+=(--account="$MIXTURE_ACCOUNT"); fi
if [[ $do_benchmark -eq 1 ]]; then
  submit_args+=(--job-name=tf-mixture-benchmark --output="$run_root/logs/benchmark-%j.out"
    "$repo_root/code/transformer/kuma/mixture_benchmark.sbatch" "$run_root")
else
  submit_args+=(--job-name=tf-mixture --array=0-1%2 --output="$run_root/logs/train-%A_%a.out"
    "$repo_root/code/transformer/kuma/mixture_train.sbatch" "$run_root")
fi
printf '%q ' "${submit_args[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  mkdir -p "$run_root/logs"
  "${submit_args[@]}"
fi

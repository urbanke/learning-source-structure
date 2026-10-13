#!/bin/bash
# Default prints the command. --submit contacts Slurm after training completes.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_root="$repo_root/output/multifamily_2026-09-16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
submit_args=(sbatch --parsable --chdir="$repo_root" --job-name=tf-mixture-eval
  --output="$run_root/logs/eval-%j.out")
if [[ -n ${MIXTURE_ACCOUNT:-} ]]; then submit_args+=(--account="$MIXTURE_ACCOUNT"); fi
submit_args+=("$repo_root/code/transformer/kuma/mixture_eval.sbatch" "$run_root")
printf '%q ' "${submit_args[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  for run_seed in 0 1; do
    [[ -f "$run_root/train/seed${run_seed}/model_300000.pt" ]] || {
      echo "Production seed $run_seed has not produced its final checkpoint." >&2; exit 2;
    }
  done
  mkdir -p "$run_root/logs"
  "${submit_args[@]}"
fi

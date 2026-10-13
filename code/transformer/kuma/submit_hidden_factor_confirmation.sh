#!/bin/bash
# Print-only by default. All scientific reads and inference run on a GPU node.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_root="$repo_root/output/hidden_factor_confirmation_2026-09-18"
baseline_root="$repo_root/output/hidden_factor_2026-09-16/2x8"
scaling_root="$repo_root/output/hidden_factor_scaling_2026-09-16"
plan="$repo_root/code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json"
after_job=''
batch_size=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    --baseline-root) baseline_root=${2:?--baseline-root requires the original 2x8 directory}; shift 2 ;;
    --scaling-root) scaling_root=${2:?--scaling-root requires the completed scaling directory}; shift 2 ;;
    --after-job) after_job=${2:?--after-job requires a numeric Slurm job id}; shift 2 ;;
    --batch-size) batch_size=${2:?--batch-size requires a positive integer}; shift 2 ;;
    --help)
      echo 'Usage: submit_hidden_factor_confirmation.sh [--submit] [--out DIR] [--baseline-root DIR] [--scaling-root DIR] [--after-job JOB_ID] [--batch-size N]'
      echo 'Evaluate the six existing final models on the untouched 3840 sequences; reuse all Bayesian references.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -z $after_job || $after_job =~ ^[0-9]+$ ]] || { echo 'Expected a numeric Slurm dependency job id.' >&2; exit 2; }
[[ $batch_size =~ ^[1-9][0-9]*$ ]] || { echo 'Batch size must be a positive integer.' >&2; exit 2; }
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
case "$baseline_root" in /*) ;; *) baseline_root="$repo_root/$baseline_root" ;; esac
case "$scaling_root" in /*) ;; *) scaling_root="$repo_root/$scaling_root" ;; esac
# Only path resolution and a handful of existence checks occur on the login node.
# Bulk checksums, reference validation, model loading and reduction are in sbatch.
python3 - "$run_root" "$baseline_root" "$scaling_root" "$repo_root" <<'PY'
from pathlib import Path
import sys
out, base, scaling, repo = (Path(p).resolve() for p in sys.argv[1:])
protected = [base, scaling, *(repo/name for name in ('code', 'results', 'notes', 'figures', '.git'))]
if out == repo or out in repo.parents or any(out == p or out in p.parents or p in out.parents for p in protected):
    raise SystemExit('Confirmation output must be separate from the original runs and source files.')
PY
command=(sbatch --parsable --chdir="$repo_root" --job-name=hf-confirmation
  --output="$run_root/logs/confirmation-%j.out")
if [[ -n ${HIDDEN_FACTOR_ACCOUNT:-} ]]; then command+=(--account="$HIDDEN_FACTOR_ACCOUNT"); fi
if [[ -n $after_job ]]; then command+=(--dependency="afterok:$after_job" --kill-on-invalid-dep=yes); fi
command+=("$repo_root/code/transformer/kuma/hidden_factor_confirmation.sbatch"
  "$run_root" "$baseline_root" "$scaling_root" "$plan" "$batch_size")
printf '%q ' "${command[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  [[ -f $plan ]] || { echo 'Frozen scaling plan is missing.' >&2; exit 2; }
  for required in cohort.json cohort.npz references/manifest.json; do
    [[ -f "$baseline_root/eval/$required" ]] || { echo "Missing original evaluation input: $required" >&2; exit 2; }
  done
  for seed in 0 1; do
    for checkpoint in "$baseline_root/train/seed$seed/model_100000.pt" \
      "$scaling_root/train/longsmall/seed$seed/model_400000.pt" \
      "$scaling_root/train/large/seed$seed/model_100000.pt"; do
      [[ -f $checkpoint ]] || { echo "Missing final checkpoint: $checkpoint" >&2; exit 2; }
    done
  done
  mkdir -p "$run_root/logs"
  "${command[@]}"
fi

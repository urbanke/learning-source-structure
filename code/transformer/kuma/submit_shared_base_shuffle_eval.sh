#!/bin/bash
# Dry-run by default. Optional dependency permits queuing while controls train.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_root="$repo_root/output/shared_base_shuffle_2026-09-17"
ordered_root="$repo_root/output/shared_base_scaling_2026-09-17/train"
manifest="$repo_root/code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json"
after_job=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    --ordered-root) ordered_root=${2:?--ordered-root requires a directory}; shift 2 ;;
    --after-job) after_job=${2:?--after-job requires a numeric Slurm job id}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -z $after_job || $after_job =~ ^[0-9]+$ ]] || { echo 'Expected a numeric Slurm dependency job id.' >&2; exit 2; }
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
case "$ordered_root" in /*) ;; *) ordered_root="$repo_root/$ordered_root" ;; esac
python3 - "$run_root" "$repo_root" <<'PY'
from pathlib import Path
import sys
out, repo = (Path(p).resolve() for p in sys.argv[1:])
if out == repo or out in repo.parents or any(out == p or out in p.parents or p in out.parents for p in (repo/'code',repo/'results',repo/'notes',repo/'figures')):
    raise SystemExit('Evaluation output must be separate from source files and scientific results.')
PY
command=(sbatch --parsable --chdir="$repo_root" --job-name=lsa-shuffle-eval
  --output="$run_root/logs/eval-%j.out")
if [[ -n ${SHARED_BASE_SHUFFLE_ACCOUNT:-} ]]; then command+=(--account="$SHARED_BASE_SHUFFLE_ACCOUNT"); fi
if [[ -n $after_job ]]; then command+=(--dependency="afterok:$after_job" --kill-on-invalid-dep=yes); fi
command+=("$repo_root/code/transformer/kuma/shared_base_shuffle_eval.sbatch" "$run_root" "$ordered_root" "$manifest")
printf '%q ' "${command[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  [[ -f $manifest ]] || { echo 'Frozen evaluation cohort manifest missing.' >&2; exit 2; }
  for run_seed in 0 1; do
    [[ -f "$ordered_root/seed$run_seed/model_100000.pt" ]] || { echo "Original larger checkpoint missing for run $run_seed." >&2; exit 2; }
    if [[ -z $after_job ]]; then
      [[ -f "$run_root/train/seed$run_seed/model_100000.pt" ]] || { echo "Shuffled checkpoint missing for run $run_seed; use --after-job while training." >&2; exit 2; }
    fi
  done
  mkdir -p "$run_root/logs"
  "${command[@]}"
fi

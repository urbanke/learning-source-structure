#!/bin/bash
# Print commands by default. Reuse the completed 2x8 baseline read-only.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_kind=train
run_arm=all
run_root="$repo_root/output/hidden_factor_scaling_2026-09-16"
baseline_root="$repo_root/output/hidden_factor_2026-09-16/2x8"
plan="$repo_root/code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --benchmark|--summary)
      [[ $run_kind == train ]] || { echo 'Choose only one run mode.' >&2; exit 2; }
      run_kind=${1#--}; shift ;;
    --arm) run_arm=${2:?--arm requires all, longsmall or large}; shift 2 ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    --baseline-root) baseline_root=${2:?--baseline-root requires the completed 2x8 directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$run_arm" in all|longsmall|large) ;; *) echo 'Unknown arm.' >&2; exit 2 ;; esac
if [[ $run_kind != train && $run_arm != all ]]; then
  echo '--arm applies only to production training.' >&2; exit 2
fi
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
case "$baseline_root" in /*) ;; *) baseline_root="$repo_root/$baseline_root" ;; esac
# Reject aliases and nested output paths before creating any files.
python3 - "$run_root" "$baseline_root" <<'PY'
from pathlib import Path
import sys
out, base = (Path(p).resolve() for p in sys.argv[1:])
if out == base or out in base.parents or base in out.parents:
    raise SystemExit('Scaling output must be separate from the original baseline directory.')
PY
if [[ $run_kind == summary ]]; then
  run_python="${HIDDEN_FACTOR_VENV:-$HOME/venv-tf}/bin/python"
  command=("$run_python" "$repo_root/code/transformer/hidden_factor_scaling_eval.py" summary
    --plan "$plan" --baseline-eval "$baseline_root/eval"
    --evaluations "$run_root/eval" --out "$run_root/comparison.json"
    --figure-out "$run_root/figures")
else
  command=(sbatch --parsable --chdir="$repo_root")
  if [[ -n ${HIDDEN_FACTOR_ACCOUNT:-} ]]; then command+=(--account="$HIDDEN_FACTOR_ACCOUNT"); fi
  if [[ $run_kind == benchmark ]]; then
    command+=(--job-name=hf-scaling-benchmark --qos=debug --time=00:30:00
      --output="$run_root/logs/benchmark-%j.out")
  else
    case "$run_arm" in
      all) run_array='0-3%4' ;;
      longsmall) run_array='0-1%2' ;;
      large) run_array='2-3%2' ;;
    esac
    command+=(--job-name=hf-scaling --array="$run_array"
      --output="$run_root/logs/train-%A_%a.out")
  fi
  command+=("$repo_root/code/transformer/kuma/hidden_factor_scaling_train.sbatch"
    "$run_root" "$baseline_root" "$run_kind" "$plan")
fi
printf '%q ' "${command[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  if [[ $run_kind == train ]]; then
    python3 "$repo_root/code/scripts/hidden_factor_scaling_benchmark.py" \
      --plan "$plan" --run "$run_root/benchmark/large/seed0" --check-only
    for required in cohort.json cohort.npz references/manifest.json models/seed0/manifest.json models/seed1/manifest.json; do
      [[ -f "$baseline_root/eval/$required" ]] || { echo "Missing baseline evidence: $required" >&2; exit 2; }
    done
    if [[ $run_arm != large ]]; then
      for seed in 0 1; do
        for checkpoint in latest.pt model_100000.pt; do
          [[ -f "$baseline_root/train/seed$seed/$checkpoint" ]] || {
            echo "Missing baseline checkpoint for continuation: seed$seed/$checkpoint" >&2; exit 2;
          }
        done
      done
    fi
  fi
  if [[ $run_kind != summary ]]; then mkdir -p "$run_root/logs"; fi
  "${command[@]}"
fi

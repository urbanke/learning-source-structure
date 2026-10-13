#!/bin/bash
# Print by default; --submit is required to contact Slurm.
# First size exact Bayes with --benchmark. Select --shape only after reviewing it.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_kind=train
run_shape=
run_pilot=0
run_root="$repo_root/output/hidden_factor_2026-09-16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --benchmark) run_kind=capacity; shift ;;
    --training-benchmark) run_kind=throughput; shift ;;
    --references) run_kind=references; shift ;;
    --evaluate) run_kind=evaluate; shift ;;
    --pilot) run_pilot=1; shift ;;
    --shape) run_shape=${2:?--shape requires dimensions such as 3x4}; shift 2 ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $run_pilot -eq 1 && $run_kind != references && $run_kind != evaluate ]]; then
  echo '--pilot requires --references or --evaluate.' >&2; exit 2
fi
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
submit_args=(sbatch --parsable --chdir="$repo_root")
if [[ -n ${HIDDEN_FACTOR_ACCOUNT:-} ]]; then submit_args+=(--account="$HIDDEN_FACTOR_ACCOUNT"); fi
if [[ $run_kind == capacity ]]; then
  submit_args+=(--job-name=hf-bayes-capacity --output="$run_root/logs/capacity-%j.out"
    "$repo_root/code/transformer/kuma/hidden_factor_benchmark.sbatch" "$run_root")
else
  if [[ ! $run_shape =~ ^([1-9][0-9]{0,2})x([1-9][0-9]{0,2})$ ]]; then
    echo 'Select --shape RxS after the exact-Bayes capacity benchmark.' >&2; exit 2
  fi
  run_r=${BASH_REMATCH[1]}
  run_s=${BASH_REMATCH[2]}
  if (( run_r < 2 || run_s < 2 || run_r * run_s > 255 )); then
    echo 'Each dimension must be >=2 and the token alphabet <=255.' >&2; exit 2
  fi
  if [[ $run_kind == references ]]; then
    run_stage=full
    run_label=references
    if [[ $run_pilot -eq 1 ]]; then
      run_stage=pilot
      run_label=references-pilot
      submit_args+=(--time=04:00:00)
    fi
    submit_args+=(--job-name="hf-$run_label-$run_shape"
      --output="$run_root/logs/$run_label-$run_shape-%j.out"
      "$repo_root/code/transformer/kuma/hidden_factor_references.sbatch" "$run_root" "$run_shape" "$run_stage")
  elif [[ $run_kind == evaluate ]]; then
    run_stage=full
    run_label=eval
    if [[ $run_pilot -eq 1 ]]; then
      run_stage=pilot
      run_label=eval-pilot
      submit_args+=(--time=01:00:00)
    fi
    submit_args+=(--job-name="hf-$run_label-$run_shape"
      --output="$run_root/logs/$run_label-$run_shape-%j.out"
      "$repo_root/code/transformer/kuma/hidden_factor_eval.sbatch" "$run_root" "$run_shape" "$run_stage")
  elif [[ $run_kind == throughput ]]; then
    submit_args+=(--job-name="hf-throughput-$run_shape" --qos=debug --time=00:30:00
      --output="$run_root/logs/throughput-$run_shape-%j.out"
      "$repo_root/code/transformer/kuma/hidden_factor_train.sbatch"
      "$run_root" "$run_shape" "$run_kind")
  else
    submit_args+=(--job-name="hf-train-$run_shape" --array=0-1%2
      --output="$run_root/logs/train-$run_shape-%A_%a.out"
      "$repo_root/code/transformer/kuma/hidden_factor_train.sbatch"
      "$run_root" "$run_shape" "$run_kind")
  fi
fi
printf '%q ' "${submit_args[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  mkdir -p "$run_root/logs"
  "${submit_args[@]}"
fi

#!/bin/bash
# Independent ABACA follow-up; reuse the original launcher and numerical code.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
plan="$repo_root/code/transformer/configs/shared_base_abaca_plan_2026-09-16.json"
out="$repo_root/output/scitas_shared_base_abaca_2026-09-16"
original="$repo_root/output/scitas_shared_base_2026-09-16/merged/calibrate.json"
mode=run
execute=false
action=--dry-run
action_set=false
options=(--plan "$plan" --stage abaca_check --out "$out" --concurrency 32 --replicas-per-shard 4)
has_resource_option=false
while (($#)); do
  case "$1" in
    --submit|--dry-run)
      [[ "$action_set" == false ]] || { echo 'Choose only one action flag.' >&2; exit 2; }
      action_set=true; action=$1
      if [[ "$1" == --submit ]]; then execute=true; fi
      shift ;;
    --merge|--review)
      [[ "$mode" == run ]] || { echo 'Choose only one of --merge and --review.' >&2; exit 2; }
      mode=${1#--}; shift ;;
    --only-task|--wall-minutes)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      options+=("$1" "$2"); has_resource_option=true; shift 2 ;;
    --help|-h)
      echo 'Usage: submit_shared_base_abaca.sh [--merge | --review] [--submit]'
      echo 'Default: print 32 CPU tasks on Jed for 128 independent ABACA repetitions.'
      echo 'Sampling resume options: --only-task INDEX, --wall-minutes MINUTES.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$mode" != run && "$has_resource_option" == true ]]; then
  echo 'Resource/task options apply only to sampling jobs.' >&2; exit 2
fi
if [[ "$mode" == review ]]; then
  command=("${SHARED_BASE_VENV:-$HOME/venv-tf}/bin/python"
    "$repo_root/code/scripts/review_shared_base_abaca.py"
    --original-plan "$repo_root/code/transformer/configs/shared_base_cluster_plan_2026-09-16.json"
    --followup-plan "$plan" --original-receipt "$original"
    --followup-receipt "$out/merged/abaca_check.json" --out "$out/comparison.json")
  printf '%q ' "${command[@]}"; printf '\n'
  if [[ "$execute" == true ]]; then exec "${command[@]}"; fi
  exit 0
fi
if [[ "$mode" == merge ]]; then options+=(--merge); fi
exec bash "$repo_root/code/transformer/scitas/submit_shared_base.sh" "${options[@]}" "$action"

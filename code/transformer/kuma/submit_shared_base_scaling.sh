#!/bin/bash
# Print by default; submit only the separate benchmark or two planned seeds.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
do_submit=0
run_kind=train
run_root="$repo_root/output/shared_base_scaling_2026-09-17"
plan="$repo_root/code/transformer/configs/shared_base_scaling_plan_2026-09-17.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) do_submit=1; shift ;;
    --dry-run) do_submit=0; shift ;;
    --benchmark)
      [[ $run_kind == train ]] || { echo 'Choose the benchmark only once.' >&2; exit 2; }
      run_kind=benchmark; shift ;;
    --out) run_root=${2:?--out requires a directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
# Check real paths before writing; never use an existing scientific directory.
python3 - "$run_root" "$repo_root" <<'PY'
from pathlib import Path
import sys
out, repo = (Path(p).resolve() for p in sys.argv[1:])
protected = [repo/'code', repo/'results', repo/'figures', repo/'notes']
if out == repo or out in repo.parents or any(out == p or out in p.parents or p in out.parents for p in protected):
    raise SystemExit('Scaling output must be separate from source files and existing scientific results.')
if out.exists():
    if not out.is_dir() or any(p.name not in {'benchmark','train','logs','environment'} for p in out.iterdir()):
        raise SystemExit('Scaling output contains unrelated existing artifacts; choose a separate directory.')
PY
command=(sbatch --parsable --chdir="$repo_root")
if [[ -n ${SHARED_BASE_SCALING_ACCOUNT:-} ]]; then command+=(--account="$SHARED_BASE_SCALING_ACCOUNT"); fi
if [[ $run_kind == benchmark ]]; then
  command+=(--job-name=lsa-scaling-benchmark --qos=debug --time=00:30:00
    --output="$run_root/logs/benchmark-%j.out")
else
  command+=(--job-name=lsa-scaling --array=0-1%2 --output="$run_root/logs/train-%A_%a.out")
fi
command+=("$repo_root/code/transformer/kuma/shared_base_scaling_train.sbatch" "$run_root" "$run_kind" "$plan")
printf '%q ' "${command[@]}"
printf '\n'
if [[ $do_submit -eq 1 ]]; then
  [[ -f "$plan" ]] || { echo 'Frozen shared-base scaling plan is missing.' >&2; exit 2; }
  if [[ $run_kind == train ]]; then
    python3 "$repo_root/code/scripts/shared_base_scaling_benchmark.py" \
      --plan "$plan" --run "$run_root/benchmark/seed0" --check-only
  fi
  mkdir -p "$run_root/logs"
  "${command[@]}"
fi

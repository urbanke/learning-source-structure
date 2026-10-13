#!/bin/bash
# Print-only by default. Heavy validation/training/evaluation runs on compute nodes.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
run_root="$repo_root/output/hidden_factor_campaign_2026-09-18"
parent_root="$repo_root/output/hidden_factor_scaling_2026-09-16/train/large"
baseline_root="$repo_root/output/hidden_factor_2026-09-16/2x8/eval"
plan="$repo_root/code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json"
submit=0; mode=train; arm=all; seed=both
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) submit=1; shift ;;
    --dry-run) submit=0; shift ;;
    --benchmark) [[ $mode == train ]] || { echo 'Benchmark mode repeated.' >&2; exit 2; }; mode=benchmark; shift ;;
    --arm) arm=${2:?--arm requires continue_large, xlarge or all}; shift 2 ;;
    --seed) seed=${2:?--seed requires 0, 1 or both}; shift 2 ;;
    --out) run_root=${2:?--out requires directory}; shift 2 ;;
    --parent-root) parent_root=${2:?--parent-root requires original large-model training directory}; shift 2 ;;
    --baseline-eval) baseline_root=${2:?--baseline-eval requires original reference directory}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$arm" in all|continue_large|xlarge) ;; *) echo 'Unknown campaign arm.' >&2; exit 2 ;; esac
case "$seed" in both|0|1) ;; *) echo 'Unknown training seed.' >&2; exit 2 ;; esac
if [[ $mode == benchmark && ( $arm == continue_large || $seed == 1 ) ]]; then
  echo 'Benchmark always uses xlarge seed 0.' >&2; exit 2
fi
case "$run_root" in /*) ;; *) run_root="$repo_root/$run_root" ;; esac
case "$parent_root" in /*) ;; *) parent_root="$repo_root/$parent_root" ;; esac
case "$baseline_root" in /*) ;; *) baseline_root="$repo_root/$baseline_root" ;; esac
python3 - "$repo_root" "$run_root" "$parent_root" "$baseline_root" <<'PY'
import sys
from pathlib import Path
repo,out,parent,baseline=map(lambda p:Path(p).resolve(),sys.argv[1:])
protected=[parent,baseline,*[repo/n for n in ('code','results','notes','figures','.git')]]
if out==repo or out in repo.parents or any(out==p or out in p.parents or p in out.parents for p in protected):
    raise SystemExit('Campaign output must be separate from original runs, references and source files.')
PY
command=(sbatch --parsable --chdir="$repo_root")
if [[ -n ${HIDDEN_FACTOR_ACCOUNT:-} ]]; then command+=(--account="$HIDDEN_FACTOR_ACCOUNT"); fi
if [[ $mode == benchmark ]]; then
  command+=(--job-name=hf-campaign-benchmark --qos=debug --time=00:30:00 --output="$run_root/logs/benchmark-%j.out")
else
  # Only tiny saved status/summary files are inspected here. No checkpoint reads.
  run_array=$(python3 - "$run_root" "$plan" "$arm" "$seed" <<'PY'
import json,hashlib,sys
from pathlib import Path
out,plan=map(Path,sys.argv[1:3]);arm,seed=sys.argv[3:];digest=hashlib.sha256(plan.read_bytes()).hexdigest();tasks=[]
for i,(a,s) in enumerate([('continue_large',0),('continue_large',1),('xlarge',0),('xlarge',1)]):
    if arm not in ('all',a) or seed not in ('both',str(s)):continue
    status=out/f'train/{a}/seed{s}/status.json';summary=out/f'eval/{a}/seed{s}/step400000/summary.json'
    if status.exists() and summary.exists():
        r=json.loads(status.read_text());e=json.loads(summary.read_text())
        if r.get('complete') is True and r.get('campaign_plan_sha256')==digest and e.get('complete') is True and e.get('identity',{}).get('plan_sha256')==digest:
            continue
    tasks.append(str(i))
print(','.join(tasks))
PY
)
  if [[ -z $run_array ]]; then echo 'All selected final training and development evaluations are already complete.'; exit 0; fi
  command+=(--job-name=hf-campaign --array="${run_array}%4" --output="$run_root/logs/train-%A_%a.out")
fi
command+=("$repo_root/code/transformer/kuma/hidden_factor_campaign.sbatch" "$run_root" "$parent_root" "$baseline_root" "$plan" "$mode")
printf '%q ' "${command[@]}"; printf '\n'
if [[ $submit -eq 1 ]]; then
  [[ -f $plan ]] || { echo 'Campaign plan missing.' >&2; exit 2; }
  if [[ $mode == train ]]; then
    if [[ $arm != continue_large ]]; then
      [[ -f "$run_root/benchmark/xlarge/seed0/benchmark.json" ]] || { echo 'Run and inspect the xlarge benchmark first; continue_large can run independently.' >&2; exit 2; }
    fi
    for f in cohort.json cohort.npz references/manifest.json; do
      [[ -f "$baseline_root/$f" ]] || { echo "Missing development input: $f" >&2; exit 2; }
    done
    if [[ $arm != xlarge ]]; then
      for s in 0 1; do
        [[ $seed == both || $seed == "$s" ]] || continue
        for f in latest.pt model_100000.pt; do
          [[ -f "$parent_root/seed$s/$f" ]] || { echo "Missing parent checkpoint: seed$s/$f" >&2; exit 2; }
        done
      done
    fi
  fi
  mkdir -p "$run_root/logs"
  guard=(python3 "$repo_root/code/scripts/campaign_job_guard.py" submit --family hidden_factor --root "$run_root")
  if [[ $mode == benchmark ]]; then guard+=(--task benchmark:xlarge:0)
  else
    IFS=',' read -r -a selected <<< "$run_array"
    for index in "${selected[@]}"; do
      case "$index" in
        0) key=train:continue_large:0 ;; 1) key=train:continue_large:1 ;;
        2) key=train:xlarge:0 ;; 3) key=train:xlarge:1 ;;
      esac
      guard+=(--task "$key")
    done
  fi
  "${guard[@]}" -- "${command[@]}"
fi

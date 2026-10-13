#!/bin/bash
# Only standard-library planning and scheduler submission run on the login node.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)
exec python3 - "$project_root" "$@" <<'PY'
import argparse, json, shlex, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); p=argparse.ArgumentParser(description='Shared-base campaign; dry run unless --submit.')
p.add_argument('--plan',type=Path,default=root/'code/transformer/configs/shared_base_campaign_plan_2026-09-18.json')
p.add_argument('--out',type=Path,default=root/'output/shared_base_campaign_2026-09-18')
p.add_argument('--parent-root',type=Path,default=root/'output/shared_base_scaling_2026-09-17/train')
p.add_argument('--cohorts',type=Path,default=root/'output/shared_base_campaign_inputs_2026-09-18')
g=p.add_mutually_exclusive_group(); g.add_argument('--continue',dest='continuation',action='store_true'); g.add_argument('--xlarge',action='store_true'); g.add_argument('--benchmark',action='store_true')
p.add_argument('--evaluate',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--seed',type=int,choices=[0,1])
p.add_argument('--after-job'); p.add_argument('--submit',action='store_true'); a=p.parse_args(sys.argv[2:])
if a.benchmark and (a.evaluate or a.resume or a.seed not in (None,0)): p.error('benchmark is a fresh seed0 run')
if a.evaluate and a.resume: p.error('--resume only applies to training')
if a.after_job and not a.after_job.isdigit(): p.error('dependency must be a numeric job ID')
for value in (a.out,a.parent_root,a.cohorts,a.plan):
    if any(c in str(value) for c in '\n\r\t'): p.error('paths cannot contain control characters')
if a.out.resolve()==root or any(a.out.resolve()==(root/q).resolve() or (root/q).resolve() in a.out.resolve().parents for q in ('code','results','.git','notes','figures')): p.error('output must be a separate experiment directory')
if a.out.resolve()==a.parent_root.resolve() or a.out.resolve() in a.parent_root.resolve().parents or a.parent_root.resolve() in a.out.resolve().parents: p.error('output overlaps parent')
if a.out.resolve()==a.cohorts.resolve() or a.out.resolve() in a.cohorts.resolve().parents or a.cohorts.resolve() in a.out.resolve().parents: p.error('output overlaps frozen cohorts')
arm='xlarge' if a.xlarge or a.benchmark else 'continue_large'; mode='benchmark' if a.benchmark else 'eval' if a.evaluate else 'train'
plan=json.loads(a.plan.read_text())
if plan.get('schema')!='shared-base-campaign-plan-v1': p.error('wrong campaign plan')
name='lsa-campaign-'+('benchmark' if a.benchmark else arm+('-eval' if a.evaluate else ''))
cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name='+name,'--output='+str(a.out/'logs'/f'{mode}-{arm}-%A_%a.out')]
if a.benchmark: cmd+=['--qos=debug','--time=00:30:00']
elif a.evaluate: cmd+=['--time=01:00:00']
if not a.benchmark: cmd+=['--array='+('0-1%2' if a.seed is None else str(a.seed))]
if a.after_job: cmd+=['--dependency=afterok:'+a.after_job,'--kill-on-invalid-dep=yes']
cmd+=[str(root/'code/transformer/kuma/shared_base_campaign_train.sbatch'),str(a.out),mode,arm,str(a.plan),str(a.parent_root),str(a.cohorts),'resume' if a.resume else 'fresh']
print(shlex.join(cmd),flush=True)
if a.submit:
    (a.out/'logs').mkdir(parents=True,exist_ok=True)
    guard=[sys.executable,str(root/'code/scripts/campaign_job_guard.py'),'submit','--family','shared_base','--root',str(a.out)]
    for seed in ([0] if a.benchmark else [a.seed] if a.seed is not None else [0,1]):
        guard+=['--task',f'{mode}:{arm}:{seed}']
    subprocess.run(guard+['--']+cmd,check=True,cwd=root)
PY

#!/bin/bash
# Login-node work is limited to a small plan, path checks and sbatch submission.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 - "$repo_root" "$@" <<'PY'
import argparse, json, shlex, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
p=argparse.ArgumentParser(description='Exact-short full-prior predictive calibration; no population submission.')
p.add_argument('--plan',type=Path,default=root/'code/transformer/configs/pooled_predictive_calibration_plan_2026-09-18.json')
p.add_argument('--dependencies',type=Path,default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
p.add_argument('--out',type=Path,default=root/'output/pooled_predictive_calibration_2026-09-18')
p.add_argument('--attempt',choices=('initial','stronger'),default='initial')
p.add_argument('--only-task',type=int,action='append')
p.add_argument('--merge',action='store_true')
p.add_argument('--after-job')
p.add_argument('--submit',action='store_true')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args(sys.argv[2:])
if a.submit and a.dry_run:p.error('--submit and --dry-run are mutually exclusive')
if a.merge and a.only_task:p.error('merge requires all planned tasks')
if a.after_job and (not a.merge or not a.after_job.isdigit()):p.error('--after-job is a numeric array ID for merge only')
plan_path,dep,out=(q.expanduser().resolve() for q in (a.plan,a.dependencies,a.out))
protected=[dep,*[root/n for n in ('code','results','notes','figures','.git')]]
if out==root or out in root.parents or any(out==q or out in q.parents or q in out.parents for q in protected):
    p.error('output must be separate from dependencies and repository sources')
plan=json.loads(plan_path.read_text())
if plan.get('schema')!='pooled-predictive-exact-short-plan-v1':p.error('unexpected plan')
count=len(plan['cases'])*plan['attempts'][a.attempt]['replicates']//plan['replicas_per_task']
indices=sorted(set(a.only_task)) if a.only_task else list(range(count))
if not indices or any(i<0 or i>=count for i in indices):p.error('task index outside plan')
mode='merge' if a.merge else 'run'
cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name=pooled-cal-'+a.attempt+('-merge' if a.merge else ''),
     '--output='+str(out/a.attempt/'logs'/f'{mode}-%A_%a.out')]
if a.merge:cmd+=['--time=00:30:00']
else:cmd+=['--array='+(','.join(map(str,indices)) if a.only_task else f'0-{count-1}')+'%'+str(plan['concurrency'])]
if a.after_job:cmd+=['--dependency=afterany:'+a.after_job,'--kill-on-invalid-dep=yes']
cmd+=[str(root/'code/transformer/scitas/pooled_predictive_calibration.sbatch'),str(plan_path),str(dep),str(out),a.attempt,mode]
print(json.dumps(dict(attempt=a.attempt,mode=mode,tasks=0 if a.merge else len(indices),concurrency=plan['concurrency'],
                     sampling_requested=a.submit and not a.merge,scientific_reference_validated=False)),flush=True)
print(shlex.join(cmd),flush=True)
if a.submit:
    if not (dep/'tables/anchors_prod/manifest.json').is_file():p.error('verified dependency bundle not found')
    (out/a.attempt/'logs').mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,check=True,cwd=root)
PY

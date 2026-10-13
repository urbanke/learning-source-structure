#!/bin/bash
# Login-side work is limited to fixed bounds, paths, hashes and Slurm submission.
set -euo pipefail
echo 'Pooled mean t16: immutable 141-repetition calibration recovery; no population submission.'
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec python3 - "$repo_root" "$@" <<'PY'
import argparse, fcntl, hashlib, json, math, os, re, shlex, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
p = argparse.ArgumentParser(description='Submit one bounded calibration recovery and merge; never population.')
p.add_argument('--plan', type=Path, default=root/'code/transformer/configs/pooled_mean_population_plan_2026-09-20.json')
p.add_argument('--raw', type=Path, default=Path.home()/'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
p.add_argument('--dependencies', type=Path, default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
p.add_argument('--parent', type=Path, default=root/'output/pooled_mean_population_memory_recovery_2026-09-20')
p.add_argument('--out', type=Path, default=root/'output/pooled_mean_calibration_recovery_2026-09-21')
p.add_argument('--max-cpu-hours', type=float)
p.add_argument('--max-cost-chf', type=float)
mode = p.add_mutually_exclusive_group(); mode.add_argument('--submit', action='store_true'); mode.add_argument('--dry-run', action='store_true')
a = p.parse_args(sys.argv[2:])
for name in ('plan','raw','dependencies','parent','out'):
    value = getattr(a,name).expanduser().resolve()
    if any(c in str(value) for c in '\n\r\t'): p.error('paths cannot contain control characters')
    setattr(a,name,value)
if a.out == a.parent or a.out in a.parent.parents or a.parent in a.out.parents: p.error('recovery output must be separate from parent')
protected=[a.raw,a.dependencies,a.parent,*[root/n for n in ('code','results','notes','figures','.git')]]
if a.out == root or root not in a.out.parents or any(a.out==q or a.out in q.parents or q in a.out.parents for q in protected):
    p.error('recovery output must be a separate location under this checkout')
plan=json.loads(a.plan.read_text())
if plan.get('schema')!='pooled-mean-population-plan-v1' or plan.get('observed')!=16: p.error('wrong original plan')
rate=plan['scheduler']['cpu_hour_rate_chf']
missing=141; prepare_minutes=60; recovery_minutes=130; merge_minutes=60; concurrency=16; cpus=1
cpu_hours=cpus*(prepare_minutes+missing*recovery_minutes+merge_minutes)/60
cost=cpu_hours*rate
for name in ('max_cpu_hours','max_cost_chf'):
    value=getattr(a,name)
    if value is not None and (not math.isfinite(value) or value<=0): p.error('budget caps must be finite and positive')
if a.submit and (a.max_cpu_hours is None or a.max_cost_chf is None): p.error('--submit requires both reviewed budget caps')
if a.max_cpu_hours is not None and cpu_hours>a.max_cpu_hours+1e-9: p.error('CPU-hour ceiling exceeds reviewed cap')
if a.max_cost_chf is not None and cost>a.max_cost_chf+1e-9: p.error('cost ceiling exceeds reviewed cap')
script=root/'code/transformer/scitas/pooled_mean_calibration_recovery.sbatch'
worker=root/'code/scripts/recover_pooled_mean_calibration.py'
launcher=root/'code/transformer/scitas/submit_pooled_mean_calibration_recovery.sh'
directives=dict(re.findall(r'^#SBATCH --([a-z-]+)=(\S+)\s*$',script.read_text(),re.M))
if directives.get('cpus-per-task')!='1' or directives.get('ntasks')!='1' or directives.get('nodes')!='1': p.error('budget assumes one CPU/task/node')
memory=re.fullmatch(r'(\d+)([MG])?',directives.get('mem',''))
if memory is None: p.error('fixed memory directive required')
memory_mib=int(memory[1])*(1024 if memory[2]=='G' else 1)
if memory_mib>7000: p.error('memory exceeds Jed standard per-CPU limit')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
code={str(path.relative_to(root)):sha(path) for path in (worker,script,launcher)}
identity=dict(original_plan=str(a.plan),original_plan_sha256=sha(a.plan),raw=str(a.raw),
    dependencies=str(a.dependencies),parent=str(a.parent),out=str(a.out),code_sha256=code,
    missing_repetitions=missing,maximum_cpu_hours=cpu_hours,maximum_cost_chf=cost,
    cpu_hour_rate_chf=rate,approved_cpu_hour_cap=a.max_cpu_hours,approved_cost_cap_chf=a.max_cost_chf,
    cpus_per_task=cpus,memory_mib=memory_mib,concurrency=concurrency)
print(json.dumps(dict(stage='planned',missing_repetitions=missing,maximum_simultaneous_workers=concurrency,
    maximum_cpu_hours=cpu_hours,maximum_cost_chf=cost,cost_is_rate_based_projection=True,
    automatic_retries=False,population_submitted=False,population_requires_passed_calibration_overlay=True)),flush=True)
def command(stage, minutes, dependency=None):
    name={'prepare':'prepare','recover':'array','merge':'merge'}[stage]
    cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name=pmean16-recover-'+name,
         f'--time={minutes//60:02d}:{minutes%60:02d}:00','--output='+str(a.out/'logs'/(name+'-%A_%a.out'))]
    if dependency:
        cmd += ['--dependency='+('afterok:' if stage=='recover' else 'afterany:')+dependency,'--kill-on-invalid-dep=yes']
    if stage=='recover': cmd += [f'--array=0-{missing-1}%{concurrency}']
    return cmd+[str(script),str(a.plan),str(a.raw),str(a.dependencies),str(a.parent),str(a.out),stage]
def now(): return datetime.now(timezone.utc).isoformat()
receipt_path=a.out/'submission.json'
receipt=dict(schema='pooled-mean-calibration-recovery-submission-v1',identity=identity,jobs=[],
    status='preparing',submitting=None,created_utc=now(),automatic_retries=False,population_submitted=False)
def save():
    receipt['updated_utc']=now(); temporary=receipt_path.with_name('.submission.json.tmp.'+str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,receipt_path)
def dispatch(stage,minutes,dependency=None):
    cmd=command(stage,minutes,dependency); print(shlex.join(cmd),flush=True)
    if not a.submit: return 'DRY_'+stage
    receipt['submitting']=dict(stage=stage,command=cmd); receipt['status']='submitting'; save()
    result=subprocess.run(cmd,check=True,cwd=root,text=True,stdout=subprocess.PIPE)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Scheduler response uncertain; inspect receipt and scheduler')
    receipt['jobs'].append(dict(stage=stage,job_id=job,allocation_minutes=minutes,submitted_utc=now()))
    receipt['submitting']=None; receipt['status']='preparing'; save(); return job
def launch():
    prepare=dispatch('prepare',prepare_minutes)
    recover=dispatch('recover',recovery_minutes,prepare)
    dispatch('merge',merge_minutes,recover)
    if a.submit:
        receipt['status']='submitted'; save()
        print(json.dumps(dict(stage='submitted',receipt=str(receipt_path),job_ids=[j['job_id'] for j in receipt['jobs']],
            population_submitted=False)),flush=True)
        status=subprocess.run(['squeue','--noheader','--jobs',','.join(j['job_id'] for j in receipt['jobs']),
            '--format=%i|%j|%T|%M'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        print(json.dumps(dict(stage='scheduler_snapshot',returncode=status.returncode,output=status.stdout,
            error=status.stderr,sampling_start_verified=False)),flush=True)
if not a.submit: launch()
else:
    required=(a.raw/'heldout/manifest.json',a.dependencies/'tables/anchors_prod/anchors.json',
              a.parent/'prepared.json',a.parent/'calibration/summary.json')
    if not all(path.is_file() for path in required): p.error('parent/source/dependency marker is missing')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.submission.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: p.error('another submission is in progress')
        if receipt_path.exists():
            old=json.loads(receipt_path.read_text())
            if old['identity']!=identity or old.get('status')!='submitted' or old.get('submitting') is not None:
                p.error('existing submission differs or is uncertain; inspect before any new action')
            print(json.dumps(dict(action='already_submitted',jobs=old['jobs'],population_submitted=False)),flush=True)
        else:
            if any(path.name!='.submission.lock' for path in a.out.iterdir()): p.error('nonempty output lacks launch receipt')
            (a.out/'logs').mkdir(); save(); launch()
PY

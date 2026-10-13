"""Recover only absent t16 repetitions; publish only the complete frozen cohort."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/scripts'), str(ROOT/'code/transformer')]
import pooled_mean_population as b
from recover_pooled_mean_calibration import bound, verify_bound

SCHEMA = 'pooled-population-recovery-v1'
CODE = ('code/scripts/recover_pooled_mean_population.py',
        'code/transformer/scitas/pooled_mean_population_recovery.sbatch')
MAX_MISSING = 1600
MINUTES = 490
CPUS = 2
RATE = .0055


def missing_repetitions(specs, reports, repetitions=32):
    missing = []
    for spec in specs:
        report = reports[spec['task_index']]
        b.require(report['status'] in ('complete', 'incomplete'), 'Parent task still active')
        for j in range(len(report['records']), repetitions):
            missing.append(dict(index=len(missing), task=spec, repetition=j))
    b.require(len(missing) <= MAX_MISSING, 'Recovery exceeds reviewed ceiling')
    return missing


def complete_report(original, additions, repetitions=32):
    result = deepcopy(original)
    first = len(result['records'])
    b.require(set(additions) == set(range(first, repetitions)), 'Incomplete or overlapping recovery')
    result['records'].extend(additions[j] for j in range(first, repetitions))
    b.require(result['records'][:first] == original['records'], 'Saved prefix changed')
    result.update(complete=True, status='complete', records_sha256=b.digest(result['records']),
                  scientific_reference_validated=False)
    result.pop('error', None)
    return result


def inputs(path):
    launch = json.loads(Path(path).read_text())
    b.require(launch['schema'] == SCHEMA and Path(launch['out']).is_absolute(), 'Invalid launch')
    for name, expected in launch['code_sha256'].items():
        b.require(b.sha(ROOT/name) == expected, 'Frozen recovery code changed: '+name)
    b.require(set(launch['code_sha256']) == set(CODE), 'Incomplete code binding')
    b.require(Path(launch['parent']).resolve() != Path(launch['out']).resolve()
              and Path(launch['parent']).resolve() not in Path(launch['out']).resolve().parents,
              'Recovery overlaps parent')
    b.require(b.sha(launch['plan']) == launch['plan_sha256'], 'Plan changed')
    return launch


def load(path):
    launch = inputs(path)
    out = Path(launch['out'])
    plan = json.loads((out/'recovery_plan.json').read_text())
    b.require(plan['id'] == b.digest({k:v for k,v in plan.items() if k != 'id'})
              and plan['launch_sha256'] == b.sha(path), 'Recovery identity changed')
    b.require(plan['term_cache_size'] in (4096, 16384), 'Unreviewed cache size')
    verify_bound(plan['prepared'])
    p, h, _ = b.read_plan(launch['plan'])
    ready, ph = b.read_prepared(p, h, launch['parent'], launch['dependencies'])
    b.require(ph == plan['prepared']['sha256'], 'Prepared cohort changed')
    return launch, plan, p, h, ready, ph


def prepare(path):
    launch = inputs(path); out = Path(launch['out']); parent = Path(launch['parent'])
    b.require(not (out/'recovery_plan.json').exists(), 'Recovery already prepared')
    accounting = subprocess.run(['sacct','-X','-n','-P','-j',launch['parent_jobs'],
        '--format=JobID%40,State'], check=True, capture_output=True, text=True).stdout
    rows = [row.split('|') for row in accounting.splitlines() if '|' in row]
    expected_jobs=set(launch['parent_jobs'].split(','))
    b.require({row[0].strip().split('_')[0] for row in rows} == expected_jobs,
              'Accounting does not cover both original jobs')
    queue=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i'],
        check=True,capture_output=True,text=True).stdout
    live_jobs={row.strip().split('_')[0] for row in queue.splitlines() if row.strip()}
    b.require(not (live_jobs & expected_jobs), 'Original jobs still queued')
    states = [row[1] for row in rows]
    b.require(states and all(s.split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT',
              'OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE') for s in states),
              'Original jobs are not all terminal')
    p,h,_ = b.read_plan(launch['plan'])
    ready,ph = b.read_prepared(p,h,parent,launch['dependencies'],full=True)
    gate = b.calibration_gate(parent,h,ph)
    replay = b.merge(launch['plan'],'population',launch['raw'],launch['dependencies'],parent,write=False)
    summary = parent/'population/summary.json'
    if summary.exists():
        b.require(json.loads(summary.read_text()) == replay, 'Parent merge does not replay')
    specs = b.tasks(ready,'population'); b.require(len(specs)==768, 'Cohort changed')
    reports = {}; bindings = {}
    for spec in specs:
        original = b.task_path(parent,'population',spec['task_index'])
        report = json.loads(original.read_text()); b.check_task(report,p,h,ph,spec)
        b.require(report['calibration_summary_sha256']==gate, 'Task gate changed')
        reports[spec['task_index']] = report; bindings[str(spec['task_index'])] = bound(original)
    missing = missing_repetitions(specs,reports,p['repetitions'])
    probes = []
    for i,task in enumerate((139,267)):
        probe = Path(launch['cache_probe'])/f'task-{i}.json'
        if not probe.exists(): break
        d = json.loads(probe.read_text())
        if not (d['passed'] is True and d['checks'] and all(d['checks'].values())
                and d['task']==task and d['original_plan_sha256']==h
                and d['parent_sha256']==bindings[str(task)]['sha256']
                and d['driver_sha256']==launch['cache_probe_driver_sha256']): break
        probes.append(bound(probe))
    cache = 16384 if len(probes)==2 else 4096
    common = ['plan.json','prepared.json','calibration/summary.json'] + [
        f'calibration/tasks/task-{i:04d}.json' for i in range(52)]
    overlay = out/'overlay'; common_bindings = {}
    for name in common:
        src = parent/name; dst = overlay/name; b.require(not dst.exists(), 'Overlay exists')
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
        common_bindings[name] = bound(dst)
    plan = dict(schema=SCHEMA,launch_sha256=b.sha(path),prepared=bound(parent/'prepared.json'),
        parent_tasks=bindings,common=common_bindings,missing=missing,term_cache_size=cache,
        cache_parity=probes if cache==16384 else [], parent_accounting=accounting,
        saved_repetitions=sum(len(r['records']) for r in reports.values()),
        maximum_cpu_hours=CPUS*(len(missing)*MINUTES+120)/60,
        maximum_cost_chf=CPUS*(len(missing)*MINUTES+120)/60*RATE,
        automatic_retries=False,later_contexts_submitted=False)
    plan['id']=b.digest(plan); b.save(out/'recovery_plan.json',plan)
    print(json.dumps(dict(event='recovery_prepared',missing=len(missing),cache=cache,
        saved=plan['saved_repetitions'],maximum_cost_chf=plan['maximum_cost_chf'])),flush=True)
    dispatch(path,plan)


def dispatch(path,plan):
    """One attempt, exact array length; a partial dispatch must be reviewed manually."""
    launch=inputs(path); out=Path(launch['out']); receipt=out/'dispatch.json'
    b.require(not receipt.exists(), 'Submission already attempted')
    jobs=[]; commands=[]
    b.save(receipt,dict(plan_id=plan['id'],jobs=jobs,commands=commands,complete=False))
    wrapper=ROOT/CODE[1]
    def submit(stage,extra,offset=0):
        cmd=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name=pmean16-finish-'+stage,
             '--account=lthc', '--time='+('08:10:00' if stage=='recover' else '01:00:00'),
             '--output='+str(out/'logs'/(stage+'-%A_%a.out')),*extra,str(wrapper),
             str(Path(path).resolve()),b.sha(path),stage,str(offset)]
        commands.append(cmd); b.save(receipt,dict(plan_id=plan['id'],jobs=jobs,commands=commands,complete=False))
        environment={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_'))}
        value=subprocess.run(cmd,cwd=ROOT,env=environment,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
        b.require(value.isdigit(),'Unrecognized job receipt'); jobs.append(value)
        b.save(receipt,dict(plan_id=plan['id'],jobs=jobs,commands=commands,complete=False))
    waves=[]
    for first in range(0,len(plan['missing']),768):
        last=min(first+768,len(plan['missing']))-1
        extra=[f'--array=0-{last-first}%64']
        submit('recover',extra,offset=first); waves.append(jobs[-1])
    submit('merge',['--dependency=afterany:'+':'.join(waves)] if waves else [])
    b.save(receipt,dict(plan_id=plan['id'],jobs=jobs,commands=commands,complete=True))
    print(json.dumps(dict(event='recovery_submitted',jobs=jobs)),flush=True)


def recover(path,index):
    began=time.monotonic(); launch,plan,p,h,ready,ph=load(path); out=Path(launch['out'])
    b.require(0<=index<len(plan['missing']), 'Index outside frozen recovery')
    item=plan['missing'][index]; spec=item['task']; j=item['repetition']
    b.require(b.tasks(ready,'population')[spec['task_index']]==spec,'Task changed')
    original=verify_bound(plan['parent_tasks'][str(spec['task_index'])])
    previous=json.loads(original.read_text()); b.check_task(previous,p,h,ph,spec)
    b.require(len(previous['records'])<=j<p['repetitions'],'Attempt overlaps saved prefix')
    result=out/'repetitions'/f'{index:04d}.json'; result.parent.mkdir(parents=True,exist_ok=True)
    with b.cal.writer_claim(result):
        b.require(not result.exists(),'Repetition already attempted')
        report=dict(schema=SCHEMA,plan_id=plan['id'],item=item,complete=False,record=None)
        b.save(result,report); settings=b.ARMS[spec['arm']]; dep=Path(launch['dependencies'])
        nodes,w=b.leggauss(settings['alpha_order'])
        kernel=b.RankedLSARowEvidence(64,1+nodes/2,depths=range(9),store=dep/'tables/anchors_prod',
            src=dep/'src',u_lo=settings['row_bounds'][0],u_hi=settings['row_bounds'][1],
            step=settings['row_step'],term_cache_size=plan['term_cache_size'])
        sampler=b.SharedBaseMCMC(64,kernel,w/2)
        def interrupted(sig,frame): raise InterruptedError(signal.Signals(sig).name)
        handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
        try:
            seconds=MINUTES*60-300-(time.monotonic()-began)
            b.require(seconds>0,'Allocation exhausted before sampling')
            print(json.dumps(dict(event='recovery_started',index=index,task=spec['task_index'],repetition=j)),flush=True)
            record=b.estimate(sampler,spec['tokens'],**{k:settings[k] for k in ('particles','temperatures','moves')},
                seed=b.seed_for(h,spec,j),resample_fraction=.75,seconds_limit=seconds)
            b.check_record(record,p,h,spec,j)
            report.update(complete=True,record=record,record_sha256=b.digest(record),
                          elapsed_seconds=time.monotonic()-began)
            b.save(result,report); print(json.dumps(dict(event='recovery_complete',index=index)),flush=True)
            return 0
        except (TimeoutError,InterruptedError) as error:
            report.update(error=str(error),elapsed_seconds=time.monotonic()-began); b.save(result,report)
            return 75
        finally:
            for s,handler in handlers.items(): signal.signal(s,handler)


def merge(path):
    launch,plan,p,h,ready,ph=load(path); out=Path(launch['out']); overlay=out/'overlay'
    b.require(not (out/'summary.json').exists(),'Merge already attempted')
    for binding in plan['common'].values(): verify_bound(binding)
    additions={}; missing=[]; result_hashes={}
    for item in plan['missing']:
        index=item['index']; f=out/'repetitions'/f'{index:04d}.json'
        if not f.exists(): missing.append(index); continue
        r=json.loads(f.read_text())
        b.require(r['schema']==SCHEMA and r['plan_id']==plan['id'] and r['item']==item,'Wrong repetition')
        if r['complete'] is not True: missing.append(index); continue
        b.require(r['record_sha256']==b.digest(r['record']),'Recovered record changed')
        b.check_record(r['record'],p,h,item['task'],item['repetition'])
        additions.setdefault(item['task']['task_index'],{})[item['repetition']]=r['record']
        result_hashes[str(index)]=b.sha(f)
    if missing:
        b.save(out/'summary.json',dict(complete=False,passed=False,missing=missing,
            plan_id=plan['id'],scientific_reference_validated=False)); return 3
    for spec in b.tasks(ready,'population'):
        index=spec['task_index']; source=verify_bound(plan['parent_tasks'][str(index)])
        original=json.loads(source.read_text()); b.check_task(original,p,h,ph,spec)
        report=complete_report(original,additions.get(index,{}),p['repetitions'])
        b.check_task(report,p,h,ph,spec); b.save(b.task_path(overlay,'population',index),report)
    result=b.merge(launch['plan'],'population',launch['raw'],launch['dependencies'],overlay)
    b.save(out/'summary.json',dict(plan_id=plan['id'],result_sha256=result_hashes,
        overlay=str(overlay),overlay_summary_sha256=b.sha(overlay/'population/summary.json'),
        complete=result['complete'],passed=result['passed'],scientific_reference_validated=result['scientific_reference_validated']))
    return 0 if result['scientific_reference_validated'] else 3


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('stage',choices=('prepare','recover','merge'))
    ap.add_argument('--launch',type=Path,required=True); ap.add_argument('--index',type=int)
    a=ap.parse_args()
    if a.stage=='prepare': prepare(a.launch); return 0
    if a.stage=='recover': return recover(a.launch,a.index)
    return merge(a.launch)

if __name__=='__main__': raise SystemExit(main())

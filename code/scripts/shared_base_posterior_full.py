"""Bounded persistent Slurm workers for the complete remaining posterior cohort.

Each allocation claims successive immutable tasks. Its afterany successor is
queued before sampling, so wall-time termination preserves automatic progress.
Only checkpoint timeouts are retried; unresolved numerical checks remain visible.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from prepare_shared_base_posterior_full import (ROOT,PLAN,WORKER,RESOURCES,read_plan,verify_inputs,
    verify_task_inputs,require,sha,fingerprint,atomic_json,immutable_json)
import shared_base_posterior_cluster as original

OUT=ROOT/'output/shared_base_posterior_full_2026-09-17'
SLOTS=64
GENERATIONS=32
MAX_ATTEMPTS=4
ALLOCATION_SECONDS=8*3600
TERMINAL={'ready','unresolved','failed','timeout_exhausted'}


def read(path):return json.loads(Path(path).read_text())


@contextmanager
def locked(path,nonblocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        try:yield stream
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def environment():
    return {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}


def command(launch_path,slot=None,generation=0,dependency=None):
    launch=read(launch_path);out=Path(launch['output'])
    name=f"sb-full-{launch['launch_id'][:8]}-s{slot if slot is not None else 'all'}-g{generation}"
    cmd=['sbatch','--parsable',f'--chdir={ROOT}',f'--job-name={name}',
         f'--output={out}/logs/full-%A_%a.out']
    if slot is None:cmd.append(f'--array=0-{SLOTS-1}%{SLOTS}')
    if dependency:cmd.extend([f'--dependency=afterany:{dependency}','--kill-on-invalid-dep=yes'])
    cmd.extend([str(ROOT/'code/transformer/scitas/shared_base_posterior_full.sbatch'),
                str(launch_path),str(slot) if slot is not None else 'array',str(generation),launch['venv']])
    return cmd,name


def find_job(name,created_date):
    """Reconcile a recorded submission intent; never submit on uncertain absence."""
    jobs=set()
    q=subprocess.run(['squeue','-h','-n',name,'-o','%i|%j'],check=True,capture_output=True,text=True,env=environment())
    a=subprocess.run(['sacct','-n','-X','-P','--starttime',created_date,'--name',name,
                      '--format=JobID%60,JobName%200'],check=True,capture_output=True,text=True,env=environment())
    for line in (q.stdout+'\n'+a.stdout).splitlines():
        parts=[p.strip() for p in line.split('|')]
        if len(parts)>=2 and parts[1]==name and re.fullmatch(r'\d+(?:_(?:\d+|\[[0-9,%:-]+\]))?',parts[0]):
            jobs.add(parts[0].split('_')[0])
    require(len(jobs)<=1,'Ambiguous successor jobs; inspect scheduler before continuing')
    return next(iter(jobs)) if jobs else None


def submit_once(intent_path,cmd,name,created_date):
    intent_path=Path(intent_path)
    with locked(intent_path.with_suffix('.lock')):
        if intent_path.exists():
            prior=read(intent_path)
            require(prior['command']==cmd and prior['name']==name,'Submission intent changed')
            if prior.get('job_id'):return prior['job_id']
            found=find_job(name,created_date)
            require(found is not None,'Previous submission is ambiguous: no automatic duplicate; inspect scheduler')
            prior.update(job_id=found,reconciled=True);atomic_json(intent_path,prior);return found
        intent=dict(command=cmd,name=name,status='intent',created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(intent_path,intent)
        try:
            result=subprocess.run(cmd,check=True,capture_output=True,text=True,cwd=ROOT,env=environment())
            response=result.stdout.strip()
            require(re.fullmatch(r'\d+(?:;[A-Za-z0-9_-]+)?',response) is not None,'Ambiguous sbatch output')
            job=response.split(';')[0]
            intent.update(status='submitted',job_id=job,scheduler_stdout=response,scheduler_stderr=result.stderr)
            atomic_json(intent_path,intent);print(json.dumps(dict(submitted=job,name=name)),flush=True);return job
        except Exception as error:
            intent.update(status='submission_uncertain',error=str(error));atomic_json(intent_path,intent);raise


def load_launch(path,check_dependencies=False):
    launch=read(path)
    require(launch['launch_id']==fingerprint({k:v for k,v in launch.items() if k!='launch_id'}),'Launch identity changed')
    require(launch['schema']=='shared-base-posterior-full-launch-v1' and Path(launch['repo_root']).resolve()==ROOT,'Wrong deployment')
    plan,digest=read_plan(Path(launch['plan']))
    require(digest==launch['plan_sha256'],'Frozen full plan changed')
    require((launch['slots'],launch['generations'],launch['max_timeout_attempts'])==(SLOTS,GENERATIONS,MAX_ATTEMPTS),
            'Allocation envelope changed')
    require(read(Path(launch['output'])/'launch.json')==launch,'Deployment receipt changed')
    if check_dependencies:original.verify_dependencies(launch['dependency_root'],plan)
    original.configure_dependency(launch['dependency_root'])
    return launch,plan


def queue_init(out,plan,launch):
    path=Path(out)/'queue.json'
    initial=dict(schema='shared-base-posterior-full-queue-v1',launch_id=launch['launch_id'],
                 tasks=[dict(id=t['id'],status='pending',attempts=0) for t in plan['tasks']])
    with locked(Path(out)/'queue.lock'):
        if path.exists():
            current=read(path)
            require(current['launch_id']==launch['launch_id'] and [r['id'] for r in current['tasks']]==[r['id'] for r in initial['tasks']],
                    'Queue identity changed')
        else:atomic_json(path,initial)


def claim(out,plan,slot):
    """Return task index and an open exclusive claim, or None if all work is held/done."""
    out=Path(out)
    with locked(out/'queue.lock'):
        state=read(out/'queue.json')
        # Finish untouched cases before resuming slow ones; never drop either.
        priority={'pending':0,'retry':1,'running':2}
        candidates=sorted((i for i,r in enumerate(state['tasks']) if r['status'] not in TERMINAL),
                          key=lambda i:(priority.get(state['tasks'][i]['status'],3),i))
        for index in candidates:
            row=state['tasks'][index];lockpath=out/'claims'/(row['id']+'.lock');lockpath.parent.mkdir(exist_ok=True)
            guard=lockpath.open('a+')
            try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:guard.close();continue
            receipt=out/'tasks'/(row['id']+'.json')
            previous=read(receipt) if receipt.exists() else None
            if previous and previous['status'] in TERMINAL:
                row['status']=previous['status'];guard.close();continue
            if row['attempts']>=MAX_ATTEMPTS:
                row['status']='timeout_exhausted'
                atomic_json(receipt,dict(schema='shared-base-posterior-full-task-v1',launch_id=state['launch_id'],
                            task=plan['tasks'][index],status='timeout_exhausted',attempt=row['attempts']))
                guard.close();continue
            row.update(status='running',attempts=row['attempts']+1,slot=slot)
            atomic_json(out/'queue.json',state)
            return index,guard,row['attempts']
        atomic_json(out/'queue.json',state)
    return None


def finish(out,index,status):
    require(status in TERMINAL|{'retry'},'Unknown task completion state')
    with locked(Path(out)/'queue.lock'):
        state=read(Path(out)/'queue.json');state['tasks'][index]['status']=status
        atomic_json(Path(out)/'queue.json',state)
        counts=dict(Counter(r['status'] for r in state['tasks']))
        atomic_json(Path(out)/'status.json',dict(expected_tasks=len(state['tasks']),counts=counts,
                    coverage_complete=sum(counts.get(k,0) for k in TERMINAL)==len(state['tasks']),
                    scientific_reference_validated=False,updated_utc=datetime.now(timezone.utc).isoformat()))


def task_paths(launch,task):return original.task_paths(launch,task)


def worker_command(launch,task,budget):
    cmd=[sys.executable,str(ROOT/'code/scripts/shared_base_posterior_full_worker.py'),'--input',
         str(Path(launch['inputs'])/task['setting']),'--out',str(Path(launch['output'])/'references'),
         '--index',str(task['index']),'--prefix',str(task['time']),'--alpha-order',str(task['order'])]
    for name in ('proposal','steps','burn','initial_draws','max_draws','seed'):
        value=budget if name in ('initial_draws','max_draws') else WORKER[name]
        cmd.extend(['--'+name.replace('_','-'),str(value)])
    return cmd


def file_stat(path):
    stat=Path(path).stat()
    return dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def audit_task(launch_path,index):
    launch,plan=load_launch(launch_path);task=plan['tasks'][index]
    verify_task_inputs(launch['inputs'],plan,task)
    original.runtime_setup(launch['dependency_root'])
    import shared_base_curve_audit as auditor
    import recover_shared_base_posterior_audit as recovery
    import revalidate_shared_base_depth as depth_check
    from validate_shared_base_mcmc import json_safe
    from source import seed_of
    paths=task_paths(launch,task);reference=read(paths['reference']);diagnostics=reference['estimate']['diagnostics']
    identity=reference['identity']
    require((reference['setting'],reference['index'],reference['time'],reference['order'])==
            (task['setting'],task['index'],task['time'],task['order']),'Reference target changed')
    require(identity['master_sampling_seed']==WORKER['seed'] and identity['burn']==WORKER['burn'] and
            identity['sampling_seed']==seed_of('shared-base-curve-reference-v1',WORKER['seed'],task['setting'],task['index'],task['time'],task['order']) and
            identity['proposal']['sampler_class']=='shared_base_surrogate.SharedBaseSurrogateMCMC' and
            identity['proposal']['proposal_spec']['surrogate_steps']==WORKER['steps'] and
            diagnostics['chains']==4 and diagnostics['thin']==1 and WORKER['initial_draws']<=diagnostics['draws']<=WORKER['max_draws'],
            'Reference sampling identity changed')
    hashes={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')};observations=[]
    with recovery.canonical_model_loader(auditor,observations):
        raw=auditor.audit_reference(Path(launch['inputs'])/task['setting'],Path(launch['output'])/'references',
                                    task['index'],task['time'],task['order'])
    require(raw['status'] in ('ready','needs_more_draws'),'Independent raw audit failed: '+str(raw.get('error')))
    depths=depth_check.diagnostic.summarize_depths(paths['checkpoint'],reference['estimate'])
    corrected=depth_check.adjust_depth_gate(reference,raw,depths)
    require(hashes=={str(paths[k]):sha(paths[k]) for k in ('reference','checkpoint','moments')},'Raw reference changed during audit')
    result=json_safe(dict(schema='shared-base-posterior-full-audit-v1',launch_id=launch['launch_id'],task=task,
                         raw_sha256=hashes,raw_stat={path:file_stat(path) for path in hashes},
                         raw_audit=raw,depth_diagnostics=depths,corrected=corrected,
                         canonical_model_readback=observations))
    atomic_json(paths['audit'],result)


def execute_task(launch_path,launch,plan,index,attempt,remaining,runtime):
    task=plan['tasks'][index];out=Path(launch['output']);paths=task_paths(launch,task)
    verify_task_inputs(launch['inputs'],plan,task)
    previous=read(paths['receipt']) if paths['receipt'].exists() else None
    if previous:
        require(previous['launch_id']==launch['launch_id'] and previous['task']==task and previous['runtime']==runtime,
                'Cannot resume a changed reference or numerical runtime')
        require(not previous.get('checkpoint_existed') or paths['checkpoint'].is_file(),'Saved chains disappeared')
    require(not paths['reference'].exists() or paths['checkpoint'].is_file(),'Report exists without saved chains')
    run_id=f"{os.environ.get('SLURM_JOB_ID','local')}-{time.time_ns()}"
    log=out/'worker_logs'/task['id']/(run_id+'.log');attempt_path=out/'attempts'/task['id']/(run_id+'.json')
    receipt=dict(schema='shared-base-posterior-full-task-v1',launch_id=launch['launch_id'],task=task,
                 status='running',runtime=runtime,attempt=attempt,worker_log=str(log),checkpoint_existed=paths['checkpoint'].exists())
    budget=(previous or {}).get('next_draw_budget',WORKER['initial_draws'])
    require(isinstance(budget,int) and WORKER['initial_draws']<=budget<=WORKER['max_draws'] and
            budget&(budget-1)==0,'Invalid saved adaptive budget')
    started=time.monotonic();last=[-1e9]
    def save(pid=None,elapsed=0):
        if pid is not None and elapsed-last[0]<30:return
        last[0]=elapsed;receipt.update(elapsed_seconds=time.monotonic()-started,progress=original.progress_record(Path(receipt['worker_log']),paths),
                                      checkpoint_existed=paths['checkpoint'].exists())
        atomic_json(paths['receipt'],receipt);atomic_json(attempt_path,receipt)
        print(json.dumps(dict(task=task['id'],status=receipt['status'],attempt=attempt,progress=receipt['progress'])),flush=True)
    save()
    try:
        while True:
            receipt['next_draw_budget']=budget;save()
            available=remaining-(time.monotonic()-started)
            if available<RESOURCES['audit_seconds']+600:
                receipt['status']='retry' if attempt<MAX_ATTEMPTS else 'timeout_exhausted';break
            stage_log=log.with_name(log.stem+f'-draws{budget}.log')
            receipt['worker_log']=str(stage_log)
            rc=original.run_child(worker_command(launch,task,budget),stage_log,
                                  max(60,int(available-RESOURCES['audit_seconds']-300)),save)
            receipt['worker_exit_code']=rc
            if rc in (0,2):
                auditlog=stage_log.with_name(stage_log.stem+'-audit.log')
                arc=original.run_child([sys.executable,str(Path(__file__)),'audit','--launch',str(launch_path),'--index',str(index)],
                                       auditlog,RESOURCES['audit_seconds'])
                receipt['audit_exit_code']=arc
                if arc==0:
                    audited=read(paths['audit']);receipt['audit_sha256']=sha(paths['audit'])
                    if audited['corrected']['audit']['ready']:
                        receipt['status']='ready';break
                    if budget>=WORKER['max_draws']:
                        receipt['status']='unresolved';break
                    budget=min(2*budget,WORKER['max_draws']);continue
                receipt['status']='retry' if arc==75 and attempt<MAX_ATTEMPTS else 'failed';break
            receipt['status']=('retry' if attempt<MAX_ATTEMPTS else 'timeout_exhausted') if rc==75 else 'failed';break
    except Exception as error:receipt.update(status='failed',error=str(error))
    save();finish(out,index,receipt['status'])


def summarize(launch_path):
    launch,plan=load_launch(launch_path);out=Path(launch['output'])
    from shared_base_curve_audit import aggregate_time
    from validate_shared_base_mcmc import json_safe
    from shared_base_long_context_results import select_references
    for name in ('pilot/base.json','pilot/refinement.json'):
        require(sha(Path(launch['inputs'])/name)==plan['inputs']['files'][name]['sha256'],'Completed pilot inputs changed')
    old=select_references(read(Path(launch['inputs'])/'pilot/base.json'),read(Path(launch['inputs'])/'pilot/refinement.json'),
                         read(ROOT/'code/transformer/configs/shared_base_posterior_plan_2026-09-16.json'),
                         read(ROOT/'code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json'))
    rows=[];audits={key:value for key,value in old.items()}
    for task in plan['tasks']:
        paths=task_paths(launch,task);receipt=read(paths['receipt']) if paths['receipt'].exists() else None
        row=dict(task=task,status='missing' if receipt is None else receipt['status'])
        if receipt is not None:
            require(receipt['launch_id']==launch['launch_id'] and receipt['task']==task,'Task receipt changed')
            if receipt['status'] in ('ready','unresolved'):
                require(sha(paths['audit'])==receipt['audit_sha256'],'Task audit changed')
                audit=read(paths['audit'])
                require(audit['launch_id']==launch['launch_id'] and audit['task']==task,'Audit identity changed')
                # Raw SHA256 was verified twice by the independent per-case audit.
                # Avoid rereading several TB of chains at aggregation time.
                require(set(audit['raw_stat'])==set(audit['raw_sha256']) and
                        all(file_stat(path)==stat for path,stat in audit['raw_stat'].items()),'Raw reference changed since audit')
                value=audit['corrected']['audit'];audits[task['depth'],task['index'],task['time'],task['order']]=value
                row['audit']=value
        rows.append(row)
    groups=[]
    for depth in (0,1,4,8):
        for t in (256,1024):
            orders=[32,64] if (depth,t)==(0,1024) else [16,32]
            selected=[a for (d,i,time_,o),a in audits.items() if (d,time_)==(depth,t) and o in orders]
            groups.append(dict(depth=depth,time=t,orders=orders,numerical_audit=aggregate_time(selected,list(range(256)),orders)))
    counts=dict(Counter(r['status'] for r in rows));coverage=all(r['status'] in TERMINAL for r in rows)
    result=json_safe(dict(schema='shared-base-posterior-full-summary-v1',launch_id=launch['launch_id'],
                         expected_new_tasks=4064,reused_pilot_sequences=8,rows=rows,groups=groups,counts=counts,
                         coverage_complete=coverage,all_numerical_checks_passed=all(g['numerical_audit']['passed'] for g in groups),
                         scope='All256 source sequences per depth. Missing, failed and unresolved cases are retained; no available-case scientific mean.'))
    result['status']='validated' if coverage and result['all_numerical_checks_passed'] else ('failed_numerical_checks' if coverage else 'incomplete')
    atomic_json(out/'summary.json',result)
    atomic_json(out/'status.json',dict(expected_tasks=len(plan['tasks']),counts=counts,coverage_complete=coverage,
                scientific_reference_validated=result['status']=='validated',status=result['status'],
                updated_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status=result['status'],coverage_complete=coverage,
                          all_numerical_checks_passed=result['all_numerical_checks_passed'])),flush=True)
    return result


def summary_exit(result):
    return 0 if result['coverage_complete'] and result['all_numerical_checks_passed'] else (2 if result['coverage_complete'] else 3)


def run_worker(launch_path,slot,generation):
    started=time.monotonic()
    launch,plan=load_launch(launch_path,True);out=Path(launch['output'])
    require(0<=slot<SLOTS and 0<=generation<GENERATIONS,'Outside fixed allocation envelope')
    with locked(out/'slots'/f'slot{slot:02d}.lock',nonblocking=True):
        queue_init(out,plan,launch)
        # An idle slot must not burn through generations while all work is held.
        # Claiming is bookkeeping only; its successor is queued before sampling.
        claimed=claim(out,plan,slot)
        if claimed is None:
            state=read(out/'queue.json')
            if all(r['status'] in TERMINAL for r in state['tasks']):
                with locked(out/'summary.lock'):
                    result=read(out/'summary.json') if (out/'summary.json').exists() else summarize(launch_path)
                    return summary_exit(result)
            return 0
        try:
            if generation+1<GENERATIONS:
                job=os.environ.get('SLURM_JOB_ID')
                require(job is not None,'Persistent production workers require a Slurm allocation')
                if os.environ.get('SLURM_ARRAY_JOB_ID'):
                    job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
                cmd,name=command(launch_path,slot,generation+1,job)
                submit_once(out/'submissions'/f'slot{slot:02d}-generation{generation+1:02d}.json',cmd,name,launch['created_date'])
            runtime=original.runtime_setup(launch['dependency_root'])
            while claimed is not None:
                index,guard,attempt=claimed
                try:
                    execute_task(launch_path,launch,plan,index,attempt,ALLOCATION_SECONDS-(time.monotonic()-started),runtime)
                except Exception as error:
                    task=plan['tasks'][index]
                    receipt=dict(schema='shared-base-posterior-full-task-v1',launch_id=launch['launch_id'],
                                 task=task,status='failed',attempt=attempt,error=str(error))
                    atomic_json(task_paths(launch,task)['receipt'],receipt)
                    finish(out,index,'failed')
                finally:guard.close();claimed=None
                # Do not spend a timeout attempt on the short tail of an allocation.
                if time.monotonic()-started>=ALLOCATION_SECONDS-2*3600:break
                claimed=claim(out,plan,slot)
        finally:
            if claimed is not None:claimed[1].close()
        state=read(out/'queue.json')
        if all(r['status'] in TERMINAL for r in state['tasks']):
            with locked(out/'summary.lock'):
                result=read(out/'summary.json') if (out/'summary.json').exists() else summarize(launch_path)
                return summary_exit(result)
        if generation+1==GENERATIONS:
            atomic_json(out/'slots'/f'slot{slot:02d}-envelope-ended.json',dict(generation=generation,pending=sum(r['status'] not in TERMINAL for r in state['tasks'])))


def launch(args):
    plan,digest=read_plan(args.plan)
    path=args.out/'launch.json'
    created_date=read(path)['created_date'] if path.exists() else datetime.now(timezone.utc).strftime('%Y-%m-%d')
    spec=dict(schema='shared-base-posterior-full-launch-v1',repo_root=str(ROOT),plan=str(args.plan.resolve()),plan_sha256=digest,
              output=str(args.out.resolve()),inputs=str(args.inputs.resolve()),dependency_root=str(args.dependency_root.resolve()),
              venv=str(args.venv.resolve()),slots=SLOTS,generations=GENERATIONS,max_timeout_attempts=MAX_ATTEMPTS,
              created_date=created_date)
    spec['launch_id']=fingerprint(spec)
    require(len(plan['tasks'])==4064,'Full rollout requires exactly the4064 missing order calculations')
    for protected in (ROOT/'code',ROOT/'results',ROOT/'notes',args.inputs.resolve(),args.dependency_root.resolve()):
        require(args.out.resolve()!=protected and args.out.resolve() not in protected.parents and protected not in args.out.resolve().parents,
                'Output overlaps immutable inputs or source files')
    path=args.out/'launch.json'
    print(json.dumps(dict(full_tasks=4064,new_sources=1016,context_lengths=[256,1024],slots=SLOTS,
                         allocation_hours=8,max_slot_generations=GENERATIONS,max_timeout_attempts=MAX_ATTEMPTS,
                         queue_strategy='At most64 running workers and64 afterany successors; no whole-cohort reservation.',
                         execute=args.submit)),flush=True)
    if not args.submit:
        print(f'Dry run: sbatch array0-63, each persistent worker reads {path}; no files written.',flush=True);return
    verify_inputs(args.inputs,plan);original.verify_dependencies(args.dependency_root,plan)
    require((args.venv/'bin/python').is_file(),'Prepared virtual environment missing')
    immutable_json(path,spec);(args.out/'logs').mkdir(exist_ok=True);queue_init(args.out,plan,spec)
    cmd,name=command(path);print(shlex.join(cmd),flush=True)
    submit_once(args.out/'submissions/initial.json',cmd,name,spec['created_date'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('launch');p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--out',type=Path,required=True,help='Large shared scratch output directory; raw chains must not fill home quota')
    p.add_argument('--inputs',type=Path,default=Path.home()/'Projects/shared_base_posterior_full_inputs_2026-09-17')
    p.add_argument('--dependency-root',type=Path,default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
    p.add_argument('--venv',type=Path,default=Path.home()/'venv-tf');p.add_argument('--submit',action='store_true');p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--launch',type=Path,required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--generation',type=int,required=True)
    p=sub.add_parser('audit');p.add_argument('--launch',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p=sub.add_parser('summary');p.add_argument('--launch',type=Path,required=True)
    a=ap.parse_args()
    if a.mode=='launch':
        require(not(a.submit and a.dry_run),'Choose submit or dry-run');launch(a)
    elif a.mode=='worker':return run_worker(a.launch,a.slot,a.generation)
    elif a.mode=='audit':audit_task(a.launch,a.index)
    else:
        require('SLURM_JOB_ID' in os.environ,'Full summary reads raw references and must run on a compute node')
        return summary_exit(summarize(a.launch))


if __name__=='__main__':sys.exit(main())

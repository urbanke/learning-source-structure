"""Four-worker, ten-hour copying reference campaign; no retries or payoff approval.

Each wave retains the original dense scientific plan and immutable numerical
implementation. This separate manifest replaces ONLY its serial dispatch policy.
It does not change seeds, samples, caches, reference selection, or acceptance.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import copying_payoff_dense as dense

ROOT=dense.ROOT
SCHEMA='copying-payoff-parallel-v1'
DEFAULT_OUT=ROOT/'output/copying_payoff_parallel_2026-09-20'
RESOURCES=dict(max_workers=4, reserved_memory_bytes_per_worker=5*2**29,
    total_memory_bytes=10*2**30, wall_seconds=36000, task_wall_seconds=900,
    startup_tasks=2, maximum_attempts_per_task=1, minimum_free_disk_bytes=5*2**30)
require,read,write,sha=dense.require,dense.read,dense.write,dense.sha


def code_hashes():
    return dict(dense.code_hashes(), **{f'code/scripts/{name}':sha(ROOT/'code/scripts'/name) for name in
        ('copying_payoff_parallel.py','merge_copying_payoff_parallel.py','merge_copying_payoff_dense.py')})


def prepare(out,times=(6,7,8),accuracy=dense.accuracy.DEFAULT_OUT):
    out=dense.pilot.ensure_output(out)
    require(list(times) in ([6],[7],[8],[6,7],[7,8],[6,7,8]),'Explicit ordered times6--8 required')
    require(not out.exists() or not any(out.iterdir()),'Campaign output must be new and empty')
    waves={}
    for observed in times:
        folder=out/f't{observed}';dense.prepare(folder,observed,accuracy)
        waves[str(observed)]=dense.bound(folder/'plan.json')
    campaign=dict(schema=SCHEMA,output=str(out),times=list(times),waves=waves,
        resources=RESOURCES,code_sha256=code_hashes(),
        merger=dense.bound(ROOT/'code/scripts/merge_copying_payoff_parallel.py'),
        dispatch_override='Replace only max_workers=1 and global worker lock with this separately bounded four-worker policy; retain every scientific task and one900-second attempt.',
        memory_reservation_basis='2.5GiB per worker; largest reachable phi cache at t6--8 is1.251GiB plus1.249GiB overhead; t5 measured peak1.412GiB.',
        population_reference_approved=False,payoff_forecast_approved=False)
    campaign['campaign_id']=dense.fingerprint(campaign)
    write(out/'campaign.json',campaign,immutable=True)
    return campaign


def load_campaign(out):
    out=dense.pilot.ensure_output(out);c=read(out/'campaign.json')
    require(c['schema']==SCHEMA and c['output']==str(out) and c['campaign_id']==dense.fingerprint(
        {k:v for k,v in c.items() if k!='campaign_id'}),'Campaign identity changed')
    require(c['resources']==RESOURCES and c['code_sha256']==code_hashes(),'Campaign code or resource policy changed')
    dense.verify_bound(c['merger'])
    require(Path(c['merger']['path'])==ROOT/'code/scripts/merge_copying_payoff_parallel.py','Merger path changed')
    require(c['times'] in ([6],[7],[8],[6,7],[7,8],[6,7,8])
        and set(c['waves'])==set(map(str,c['times'])),'Wave selection changed')
    for observed in c['times']:
        item=c['waves'][str(observed)];dense.verify_bound(item)
        require(Path(item['path'])==out/f't{observed}'/'plan.json','Wave path changed')
        p=dense.load_plan(Path(item['path']).parent)
        require(p['observed']==observed,'Scientific wave identity differs')
    return c


def validate_wave(campaign_root,wave_folder):
    c=load_campaign(campaign_root)
    require(Path(wave_folder).resolve() in {Path(i['path']).parent for i in c['waves'].values()},
        'Wave not in the frozen campaign')
    return c


def worker(out,observed,index):
    c=load_campaign(out);require(observed in c['times'],'Worker wave is not declared')
    folder=Path(out)/f't{observed}';plan=dense.load_plan(folder)
    require(type(index) is int and 0<=index<len(plan['tasks']),'Invalid worker index')
    os.environ.update(dense.pilot.environment(plan))
    require(dense.pilot.runtime()==plan['runtime'],'Numerical runtime differs')
    task=plan['tasks'][index];p=dense.paths(folder,task);attempt=read(p['attempt'])
    require(attempt['campaign_id']==c['campaign_id'] and attempt['plan_id']==plan['plan_id']
        and attempt['task']==task and attempt['status']=='running'
        and attempt['dispatcher_pid']==os.getppid(),'Worker requires its live bounded dispatcher')
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from eval_shared_base_models import atomic_npz
    from validate_shared_base_mcmc import json_safe
    from unseen_context_refinement import checkpoint_metadata
    began=time.monotonic()
    with dense.lock(p['attempt'].with_suffix('.worker.lock')):
        require(not p['checkpoint'].exists() and not p['report'].exists(),'No retry or silent overwrite')
        nodes,weights=np.polynomial.legendre.leggauss(task['order'])
        sampler=SharedBaseSurrogateMCMC(64,RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384),weights/2,surrogate_steps=64)
        tokens=np.asarray(task['tokens'],dtype=int);parent=plan['parents'].get(task['id'])
        if parent:
            for item in parent['files'].values():dense.verify_bound(item)
            shutil.copyfile(parent['files']['checkpoint']['path'],p['checkpoint'])
            require(sha(p['checkpoint'])==parent['files']['checkpoint']['sha256'],'Copied checkpoint differs')
            run=CurveChainRun.load(sampler,tokens,p['checkpoint'])
            dense.accuracy.verify_loaded_parent(run,checkpoint_metadata(p['checkpoint']))
            require(run.draw_counts.tolist()==[16384]*4 and run.seed==task['seed'],'Reused draws or seed changed')
            estimate=read(parent['files']['report']['path'])['estimate']
        else:run=CurveChainRun(sampler,tokens,chains=4,burn=16384,thin=1,seed=task['seed'])
        identity=dict(plan_id=plan['plan_id'],task=task,target=run.fingerprint,
            proposal=run.proposal_descriptor,runtime=dense.pilot.runtime())
        if parent:
            require(identity['target']==parent['identity']['target'] and identity['proposal']==parent['identity']['proposal'],
                'Reused target or proposal changed')
        write(p['identity'],identity,immutable=True)
        if not parent:
            saved=time.monotonic()
            def progress(event):
                nonlocal saved
                now=time.monotonic()
                if event['phase']=='burn' or event['draws']==event['requested_draws'] or now-saved>=120:
                    run.save(p['checkpoint']);saved=now
                print(json.dumps(dict(task=task['id'],**event)),flush=True)
            estimate=run.extend(16384,block_draws=2048,progress=progress);run.save(p['checkpoint'])
        del run
        moments,audit,fresh=dense.accuracy._audit_raw(task,identity,estimate,p['checkpoint'])
        atomic_npz(p['moments'],**moments)
        grids=selected_grid_checks(sampler,tokens,estimate);audit['final_state_grid_checks']=grids
        passed=audit['passed'] and all(g['passed'] for g in grids)
        write(p['audit'],json_safe(audit),immutable=True)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        result=json_safe(dict(schema=dense.SCHEMA,plan_id=plan['plan_id'],task=task,complete=True,
            audit_passed=bool(passed),estimate=fresh,elapsed_seconds=time.monotonic()-began,
            peak_rss_bytes=rss,reused=parent is not None,sampling_performed=parent is None,
            campaign_id=c['campaign_id'],population_reference_approved=False,payoff_forecast_approved=False,
            **{k+'_sha256':sha(p[k]) for k in ('audit','identity','checkpoint','moments')}))
        if parent:
            for item in parent['files'].values():dense.verify_bound(item)
            require(sha(p['checkpoint'])==parent['files']['checkpoint']['sha256'],'Reuse changed samples')
        write(p['report'],result,immutable=True)
        return result


def available_slots(active,completed,startup_done=False):
    """Conservative reserved-memory admission, independent of observed low RSS."""
    limit=RESOURCES['max_workers'] if startup_done or completed>=RESOURCES['startup_tasks'] else RESOURCES['startup_tasks']
    by_memory=RESOURCES['total_memory_bytes']//RESOURCES['reserved_memory_bytes_per_worker']
    return max(0,min(limit,by_memory)-active)


def terminate(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)


def resident_bytes(processes):
    """Read current child RSS; memory reservation remains the admission rule."""
    pids=[str(p.pid) for p in processes if p.poll() is None]
    if not pids:return 0
    result=subprocess.run(['ps','-o','rss=','-p',','.join(pids)],capture_output=True,text=True,timeout=5)
    require(result.returncode in (0,1),'Cannot check worker memory')
    return sum(int(line.strip())*1024 for line in result.stdout.splitlines() if line.strip())


def merge_wave(out,observed,plan,deadline):
    """Diagnostic merge gets its own process and at most900seconds; no inference."""
    target=Path(out)/'diagnostics'/f't{observed}'
    require(not target.exists(),'Do not overwrite an existing diagnostic merge')
    remaining=min(900,deadline-time.monotonic());require(remaining>0,'Campaign deadline before diagnostic merge')
    with (Path(out)/f'merge_t{observed}.log').open('x') as stream:
        process=subprocess.Popen([plan['python'],str(ROOT/'code/scripts/merge_copying_payoff_parallel.py'),
            '--input',str(Path(out)/f't{observed}'),'--campaign',str(out),'--out',str(target)],
            cwd=ROOT,env=dense.pilot.environment(plan),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=remaining)
            require(code in (0,2),f'Diagnostic merge failed with exit{code}')
            result=read(target/'summary.json')
            return {k:result[k] for k in ('complete','individual_audits_passed','paired_orders_passed','diagnostic_arrays_validated')}
        finally:terminate(process)


def campaign_summary(out,verify_arrays=True):
    c=load_campaign(out);waves={str(t):dense.summary(Path(out)/f't{t}',verify_arrays=verify_arrays) for t in c['times']}
    result=dict(schema=SCHEMA,campaign_id=c['campaign_id'],waves=waves,
        complete=all(w['complete'] for w in waves.values()),
        all_individual_audits_passed=all(w['all_individual_audits_passed'] for w in waves.values()),
        population_reference_approved=False,payoff_forecast_approved=False,
        next_action='Full-cohort shared-reference covariance and payoff-timing assessment; individual checks alone do not approve population claims.')
    write(Path(out)/'summary.json',result);return result


def dispatch(out):
    out=Path(out);c=load_campaign(out);active={};completed=0;startup_passed=0
    receipt=out/'allocation.json';require(not receipt.exists(),'Allocation already used; no automatic retry or second launch')
    for t in c['times']:
        p=dense.load_plan(out/f't{t}')
        require(not any(dense.paths(out/f't{t}',task)['attempt'].exists() for task in p['tasks']),
            'Campaign has prior attempts; inspect rather than restarting')
    previous_handlers={}
    def interrupted(signum,frame):raise InterruptedError(f'Campaign received signal{signum}')
    started=time.monotonic();deadline=started+RESOURCES['wall_seconds']
    state=dict(campaign_id=c['campaign_id'],dispatcher_pid=os.getpid(),started_unix=time.time(),
        status='running',worker_limit=4,wall_seconds=36000,completed_tasks=0,
        planned_tasks=sum(len(dense.load_plan(out/f't{t}')['tasks']) for t in c['times']),
        unresolved_task_ids=[],diagnostics={})
    def status():
        state.update(completed_tasks=completed,elapsed_seconds=time.monotonic()-started,
            active_workers=[dict(pid=v[0].pid,task=v[1]['task']['id'],time=v[1]['task']['time']) for v in active.values()],
            updated_unix=time.time())
        write(out/'status.json',state)
    locks=ExitStack();owns_allocation=False
    try:
        # Exclude original serial waves and competing new dispatchers throughout.
        for name in ('dispatcher','worker'):
            locks.enter_context(dense.lock(ROOT/f'output/.copying-payoff-dense-{name}.lock'))
        write(receipt,state,immutable=True);owns_allocation=True
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[signum]=signal.signal(signum,interrupted)
        for observed in c['times']:
            folder=out/f't{observed}';plan=dense.load_plan(folder);pending=list(range(len(plan['tasks'])))
            while pending or active:
                now=time.monotonic()
                require(now<deadline,'Campaign ten-hour wall limit reached')
                for index,item in list(active.items()):
                    process,a,p,began,stream=item;code=process.poll()
                    if code is None:
                        require(now-began<RESOURCES['task_wall_seconds'],'Worker900-second limit reached')
                        continue
                    stream.close();a.update(exit_code=code,elapsed_seconds=now-began,finished_unix=time.time())
                    if code:
                        a['status']='failed';write(p['attempt'],a);del active[index]
                        raise RuntimeError(f'Technical worker failure: t{observed} task{index}, exit{code}')
                    r=read(p['report']);require(r['campaign_id']==c['campaign_id'] and r['task']==a['task']
                        and r['plan_id']==plan['plan_id'] and r['complete'],'Worker report identity differs')
                    a['status']='ready' if r['audit_passed'] else 'unresolved';write(p['attempt'],a)
                    del active[index];completed+=1;startup_passed+=int(r['audit_passed'])
                    if not r['audit_passed']:state['unresolved_task_ids'].append(a['task']['id'])
                    print(json.dumps(dict(event='reference_complete',time=observed,task=index,
                        status=a['status'],completed=completed,peak_rss_bytes=r['peak_rss_bytes'])),flush=True)
                require(resident_bytes([v[0] for v in active.values()])<=RESOURCES['total_memory_bytes'],
                    'Active worker RSS exceeds10GiB campaign reserve')
                # Require two passing references before expanding beyond two workers.
                # Numerical failures remain recorded; independent references continue.
                slots=available_slots(len(active),startup_passed)
                # At startup do not replace an early finisher before its peer finishes.
                if completed<2:slots=min(slots,max(0,2-completed-len(active)))
                while pending and slots:
                    if deadline-time.monotonic()<RESOURCES['task_wall_seconds']:break
                    require(shutil.disk_usage(out).free>=RESOURCES['minimum_free_disk_bytes'],'Below5GiB free disk reserve')
                    index=pending.pop(0);task=plan['tasks'][index];p=dense.paths(folder,task)
                    a=dict(campaign_id=c['campaign_id'],plan_id=plan['plan_id'],task=task,
                        reserved_seconds=900,reserved_memory_bytes=RESOURCES['reserved_memory_bytes_per_worker'],
                        status='running',started_unix=time.time(),dispatcher_pid=os.getpid())
                    write(p['attempt'],a,immutable=True);stream=p['log'].open('x')
                    try:
                        process=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'worker','--out',str(out),
                            '--time',str(observed),'--index',str(index)],cwd=ROOT,env=dense.pilot.environment(plan),
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:
                        stream.close();a.update(status='launch_failed',finished_unix=time.time());write(p['attempt'],a);raise
                    active[index]=(process,a,p,time.monotonic(),stream);slots-=1
                    print(json.dumps(dict(event='reference_started',time=observed,task=index,pid=process.pid,
                        order=task['order'],active_workers=len(active))),flush=True)
                status()
                if pending and not active:
                    raise RuntimeError('No full900-second worker allocation remains; pending tasks preserved')
                if active:time.sleep(1)
            wave=dense.summary(folder)
            print(json.dumps(dict(event='wave_complete',time=observed,complete=wave['complete'],
                all_individual_audits_passed=wave['all_individual_audits_passed'])),flush=True)
            state['diagnostics'][str(observed)]=merge_wave(out,observed,plan,deadline);status()
        state['status']='complete';return campaign_summary(out)
    except BaseException as error:
        state.update(status='stopped',reason=str(error));raise
    finally:
        for process,a,p,began,stream in active.values():
            terminate(process);stream.close();a.update(status='interrupted',finished_unix=time.time(),
                elapsed_seconds=time.monotonic()-began,exit_code=process.returncode);write(p['attempt'],a)
        active.clear()
        if owns_allocation:
            state.update(completed_tasks=completed,finished_unix=time.time(),elapsed_seconds=time.monotonic()-started)
            write(receipt,state)
            status()
        locks.close()
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','summary'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--times',type=int,nargs='+',default=[6,7,8])
    parser.add_argument('--accuracy',type=Path,default=dense.accuracy.DEFAULT_OUT)
    parser.add_argument('--time',type=int);parser.add_argument('--index',type=int)
    args=parser.parse_args();print(f'Copying parallel: {args.action}; four workers, ten hours, no retries.',flush=True)
    if args.action=='prepare':result=prepare(args.out,args.times,args.accuracy)
    elif args.action=='run':result=dispatch(args.out)
    elif args.action=='worker':result=worker(args.out,args.time,args.index)
    else:result=campaign_summary(args.out)
    print(json.dumps({k:v for k,v in result.items() if k in
        ('campaign_id','times','complete','all_individual_audits_passed','population_reference_approved','payoff_forecast_approved')}),flush=True)


if __name__=='__main__':main()

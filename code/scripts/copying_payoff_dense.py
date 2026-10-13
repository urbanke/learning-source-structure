"""Bounded full-cohort informed-Bayes references, one explicitly selected time.

Default t5 retains all 768 original histories and both integration orders.
No automatic next wave, retry, inference, prior change or payoff approval.
Canonical references are shared; their aggregate errors are NOT independent.
"""
from __future__ import annotations
import argparse
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
import copying_payoff_accuracy as accuracy
from prepare_copying_payoff_references import canonicalize, reference_key

pilot=accuracy.pilot
ROOT=pilot.ROOT
SCHEMA='copying-payoff-dense-v1'
DEFAULT_OUT=ROOT/'output/copying_payoff_dense_2026-09-20'
COUNTS={5:38,6:81,7:159,8:248}
require,read,write,sha,fingerprint,lock=pilot.require,pilot.read,pilot.write,pilot.sha,pilot.fingerprint,pilot.lock
SAMPLE=dict(chains=4,burn=16384,thin=1,draws_per_chain=16384,surrogate_steps=64)


def code_hashes():
    return dict(accuracy.code_hashes(),**{'code/scripts/copying_payoff_dense.py':sha(__file__)})


def wave_tasks(inventory,observed,parents):
    require(type(observed) is int and observed in COUNTS,'Choose one time from5,6,7,8')
    require(inventory['histories_total']==768 and inventory['depths']==[0,2,3]
            and inventory['histories_per_condition']==256,'Original complete cohort required')
    refs=[r for r in inventory['references'] if r['time']==observed]
    uses=[u for u in inventory['uses'] if u['time']==observed]
    require(len(refs)==COUNTS[observed] and len({r['reference_key'] for r in refs})==len(refs),
            'Canonical reference coverage differs')
    expected={(g,i) for g in ('a1_L0','a1_L2','a1_L3') for i in range(256)}
    require(len(uses)==768 and {(u['condition'],u['index']) for u in uses}==expected,
            'Missing, duplicate or selected histories')
    bykey={r['reference_key']:r for r in refs}
    require(set(u['reference_key'] for u in uses)==set(bykey),'Unused or missing canonical prefix')
    for u in uses:
        r=bykey[u['reference_key']]
        require(u['queried_canonical_label']==r['tokens'][-1]
                and len(u['canonical_to_original_seen'])==r['n_seen']
                and u['canonical_to_original_seen'][r['tokens'][-1]]==u['queried_original_label'],
                'Context or alphabet mapping differs')
    tasks=[]
    for r in sorted(refs,key=lambda x:tuple(x['tokens'])):
        pattern,labels=canonicalize(r['tokens'])
        require(list(pattern)==r['tokens'] and len(pattern)==observed and len(labels)==r['n_seen']
                and reference_key(pattern)==r['reference_key'],'Invalid ordered canonical identity')
        for order in (16,32):
            name=r['reference_key']+f'_a{order}';parent=parents.get(name)
            task=dict(task_index=len(tasks),id=name,reference_key=r['reference_key'],tokens=r['tokens'],
                      time=observed,order=order,target_draws=16384,kind='reuse' if parent else 'fresh')
            if parent:
                p=parent['task']
                require(all(p[k]==task[k] for k in ('id','reference_key','tokens','time','order','target_draws')),
                        'Reuse changes the target or integration order')
                task['seed']=p['seed']
            else:task['seed']=int(fingerprint(['copying-dense',20260920,task])[:8],16)
            tasks.append(task)
    return tasks,uses


def bound(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'Bound file missing or linked')
    return dict(path=str(p.resolve()),sha256=sha(p),size_bytes=p.stat().st_size)


def verify_bound(item):
    p=Path(item['path']);require(p.is_file() and not p.is_symlink() and p.stat().st_size==item['size_bytes']
            and sha(p)==item['sha256'],'Frozen input changed: '+str(p))


def accuracy_gate(folder,observed):
    original=accuracy.load_plan(folder);reports=[];parents={}
    for task in original['tasks']:
        p=accuracy.paths(folder,task);r=read(p['report']);ident=read(p['identity']);audit=read(p['audit'])
        require(r['complete'] and r['audit_passed'] and r['task']==task and r['plan_id']==original['plan_id']
                and ident['task']==task and ident['plan_id']==original['plan_id']
                and all(r[k+'_sha256']==sha(p[k]) for k in ('checkpoint','moments','identity','audit')),
                'Passed accuracy record missing or changed')
        require(audit['passed'] and all(g['passed'] for g in audit['final_state_grid_checks']),
                'Accuracy numerical checks failed')
        if task['kind']=='exact_t4':
            actual=accuracy.exact_vector_check(r['estimate']['predictive_probs'],r['estimate']['mcse_probs'],task['expected_probs'])
            require(actual==audit['exact_vector_check'] and actual['passed'],'Exact short check changed')
        reports.append(r)
        if task['time']==observed:
            parents[task['id']]=dict(task=task,identity=ident,files={k:bound(p[k]) for k in
                ('checkpoint','moments','identity','audit','report')})
    require(accuracy.coverage_summary(original['tasks'],reports)['accuracy_diagnostic_passed'],
            'All42 accuracy tasks and21 paired orders must pass')
    return original,parents


def prepare(out,observed,accuracy_out=accuracy.DEFAULT_OUT):
    out=pilot.ensure_output(out)
    require(not out.exists() or not any(out.iterdir()),'Each wave needs a new empty output folder')
    original,parents=accuracy_gate(accuracy_out,observed)
    inventory=read(original['inventory']);tasks,uses=wave_tasks(inventory,observed,parents)
    for item in inventory['source_bindings'].values():
        require(sha(item['path'])==item['sha256'],'Original source file changed')
    protected=[Path(accuracy_out).resolve(),Path(original['pilot_path']).parent.resolve()]
    protected += [Path(i['path']).parent.resolve() for i in inventory['source_bindings'].values()]
    require(all(out!=p and p not in out.parents and out not in p.parents for p in protected),
            'Output overlaps protected originals')
    plan=dict(schema=SCHEMA,output=str(out),observed=observed,accuracy_plan=bound(Path(accuracy_out)/'plan.json'),
        inventory=bound(original['inventory']),source_bindings=inventory['source_bindings'],tasks=tasks,uses=uses,
        parents=parents,code_sha256=code_hashes(),runtime=original['runtime'],python=original['python'],
        environment=original['environment'],dependency_sha256=original['dependency_sha256'],prior=original['prior'],
        sampler=SAMPLE,resources=dict(max_workers=1,task_wall_seconds=900,total_worker_seconds=900*len(tasks),
        maximum_attempts_per_task=1,checkpoint_seconds=120,minimum_free_disk_bytes=5*2**30),
        population_reference_approved=False,payoff_forecast_approved=False,
        precision_scope='Per-reference diagnostics only. No aggregate precision approval or available-case population means.',
        required_merge='Map every prediction to all768 histories; sum oracle-weighted loss gradients per shared canonical reference before covariance/MCSE. Include paired integration-order checks and logarithm bias; keep sequence sampling uncertainty separate.')
    plan['plan_id']=fingerprint(plan);write(out/'plan.json',plan,immutable=True)
    return plan


def load_plan(out):
    out=pilot.ensure_output(out);p=read(out/'plan.json')
    require(p['schema']==SCHEMA and p['output']==str(out)
            and p['plan_id']==fingerprint({k:v for k,v in p.items() if k!='plan_id'}),'Wave identity changed')
    require(p['code_sha256']==code_hashes() and p['sampler']==SAMPLE,'Code or fixed draws changed')
    verify_bound(p['accuracy_plan']);verify_bound(p['inventory'])
    original=accuracy.load_plan(Path(p['accuracy_plan']['path']).parent)
    for k in ('runtime','python','environment','dependency_sha256','prior'):
        require(p[k]==original[k],'Prior, runtime or dependency identity changed')
    inventory=read(p['inventory']['path'])
    require(p['source_bindings']==inventory['source_bindings'],'Source inventory changed')
    for item in p['source_bindings'].values():
        require(sha(item['path'])==item['sha256'],'Original source file changed')
    tasks,uses=wave_tasks(inventory,p['observed'],p['parents'])
    require(tasks==p['tasks'] and uses==p['uses'],'Frozen wave selection changed')
    require(p['resources']==dict(max_workers=1,task_wall_seconds=900,total_worker_seconds=900*len(tasks),
        maximum_attempts_per_task=1,checkpoint_seconds=120,minimum_free_disk_bytes=5*2**30),'Resource limits changed')
    return p


def paths(out,task):
    return accuracy.paths(out,task)


def worker(out,index):
    plan=load_plan(out);require(type(index) is int and 0<=index<len(plan['tasks']),'Invalid task index')
    os.environ.update(pilot.environment(plan));require(pilot.runtime()==plan['runtime'],'Numerical runtime differs')
    task=plan['tasks'][index];p=paths(out,task);attempt=read(p['attempt'])
    require(attempt['plan_id']==plan['plan_id'] and attempt['task']==task and attempt['status']=='running'
            and attempt['dispatcher_pid']==os.getppid(),'Worker requires live bounded dispatcher')
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from eval_shared_base_models import atomic_npz
    from validate_shared_base_mcmc import json_safe
    from unseen_context_refinement import checkpoint_metadata
    began=time.monotonic()
    with lock(ROOT/'output/.copying-payoff-dense-worker.lock'):
        require(not p['checkpoint'].exists() and not p['report'].exists(),'No retry or silent overwrite')
        nodes,weights=np.polynomial.legendre.leggauss(task['order'])
        sampler=SharedBaseSurrogateMCMC(64,RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384),weights/2,surrogate_steps=64)
        tokens=np.asarray(task['tokens'],dtype=int)
        parent=plan['parents'].get(task['id'])
        if parent:
            for item in parent['files'].values():verify_bound(item)
            shutil.copyfile(parent['files']['checkpoint']['path'],p['checkpoint'])
            require(sha(p['checkpoint'])==parent['files']['checkpoint']['sha256'],'Copied checkpoint differs')
            run=CurveChainRun.load(sampler,tokens,p['checkpoint'])
            accuracy.verify_loaded_parent(run,checkpoint_metadata(p['checkpoint']))
            require(run.draw_counts.tolist()==[16384]*4 and run.seed==task['seed'],'Reused draws or seed changed')
            estimate=read(parent['files']['report']['path'])['estimate']
        else:run=CurveChainRun(sampler,tokens,chains=4,burn=16384,thin=1,seed=task['seed'])
        identity=dict(plan_id=plan['plan_id'],task=task,target=run.fingerprint,proposal=run.proposal_descriptor,runtime=pilot.runtime())
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
        moments,audit,fresh=accuracy._audit_raw(task,identity,estimate,p['checkpoint'])
        atomic_npz(p['moments'],**moments)
        grids=selected_grid_checks(sampler,tokens,estimate);audit['final_state_grid_checks']=grids
        passed=audit['passed'] and all(g['passed'] for g in grids)
        write(p['audit'],json_safe(audit),immutable=True)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        result=json_safe(dict(schema=SCHEMA,plan_id=plan['plan_id'],task=task,complete=True,audit_passed=bool(passed),
            estimate=fresh,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=rss,
            reused=parent is not None,sampling_performed=parent is None,
            population_reference_approved=False,payoff_forecast_approved=False,
            **{k+'_sha256':sha(p[k]) for k in ('audit','identity','checkpoint','moments')}))
        if parent:
            for item in parent['files'].values():verify_bound(item)
            require(sha(p['checkpoint'])==parent['files']['checkpoint']['sha256'],'Reuse changed samples')
        write(p['report'],result,immutable=True);return result


def coverage(plan,reports):
    expected={t['id']:t for t in plan['tasks']};found={r['task']['id']:r for r in reports}
    require(len(found)==len(reports) and set(found)<=set(expected)
            and all(r['task']==expected[k] and r['plan_id']==plan['plan_id'] and r['complete'] for k,r in found.items()),
            'Duplicate, selected or changed task report')
    return dict(schema=SCHEMA,plan_id=plan['plan_id'],observed=plan['observed'],expected_tasks=len(expected),
        completed_tasks=len(found),complete=len(found)==len(expected),missing_task_ids=sorted(set(expected)-set(found)),
        failed_audit_ids=[k for k,r in found.items() if not r['audit_passed']],
        all_individual_audits_passed=len(found)==len(expected) and all(r['audit_passed'] for r in reports),
        original_histories=768,aggregate_precision_checked=False,aggregate_precision_passed=False,
        population_reference_approved=False,payoff_forecast_approved=False,
        next_action='Independent full-cohort merge with shared-reference covariance and precision gates; no forecast yet.')


def summary(out,verify_arrays=True):
    plan=load_plan(out);reports=[];attempts=[]
    for task in plan['tasks']:
        p=paths(out,task)
        if p['attempt'].exists():attempts.append(read(p['attempt']))
        if not p['report'].exists():continue
        r=read(p['report'])
        for k in ('identity','audit')+(('checkpoint','moments') if verify_arrays else ()):
            require(r[k+'_sha256']==sha(p[k]),'Report input checksum changed')
        audit=read(p['audit'])
        require(r['audit_passed']==(audit['passed'] and all(g['passed'] for g in audit['final_state_grid_checks'])),
                'Audit acceptance flag changed')
        reports.append(r)
    result=coverage(plan,reports);result.update(attempts=attempts,all_saved_arrays_rehashed=verify_arrays)
    write(Path(out)/'summary.json',result);return result


def dispatch(out,only_tasks=None):
    plan=load_plan(out);selected=sorted(set(range(len(plan['tasks'])) if only_tasks is None else only_tasks))
    require(selected and all(type(i) is int and 0<=i<len(plan['tasks']) for i in selected),'Invalid declared task indices')
    with lock(ROOT/'output/.copying-payoff-dense-dispatcher.lock'):
        for index in selected:
            task=plan['tasks'][index];p=paths(out,task)
            if p['attempt'].exists():continue
            spent=sum(read(paths(out,t)['attempt'])['reserved_seconds'] for t in plan['tasks'] if paths(out,t)['attempt'].exists())
            require(spent+900<=plan['resources']['total_worker_seconds'],'Fixed allocation budget exhausted')
            require(shutil.disk_usage(out).free>=plan['resources']['minimum_free_disk_bytes'],'Below5GiB disk reserve')
            a=dict(plan_id=plan['plan_id'],task=task,reserved_seconds=900,status='running',started_unix=time.time(),dispatcher_pid=os.getpid())
            write(p['attempt'],a,immutable=True);began=time.monotonic();process=None
            print(json.dumps(dict(task=task['id'],status='starting',kind=task['kind'],seconds=900)),flush=True)
            try:
                with p['log'].open('x') as stream:
                    process=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'worker','--out',str(out),'--index',str(index)],
                        cwd=ROOT,env=pilot.environment(plan),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code=process.wait(timeout=900)
                        a.update(status='failed' if code else 'ready' if read(p['report'])['audit_passed'] else 'unresolved',exit_code=code)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        a.update(status='time_limit',exit_code=75)
            except BaseException:
                if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                a['status']='interrupted';raise
            finally:
                a.update(elapsed_seconds=time.monotonic()-began,finished_unix=time.time());write(p['attempt'],a)
            summary(out,verify_arrays=False)
    return summary(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','worker','summary'))
    p.add_argument('--time',type=int,choices=tuple(COUNTS),default=5)
    p.add_argument('--out',type=Path);p.add_argument('--accuracy',type=Path,default=accuracy.DEFAULT_OUT)
    p.add_argument('--only-task',type=int,action='append');p.add_argument('--index',type=int)
    a=p.parse_args();out=a.out or DEFAULT_OUT/f't{a.time}'
    print(f'Copying references: {a.action}, output={out}; no automatic next wave.',flush=True)
    if a.action=='prepare':result=prepare(out,a.time,a.accuracy)
    elif a.action=='run':result=dispatch(out,a.only_task)
    elif a.action=='worker':result=worker(out,a.index)
    else:result=summary(out)
    print(json.dumps({k:v for k,v in result.items() if k in
        ('plan_id','observed','complete','expected_tasks','completed_tasks','audit_passed','all_individual_audits_passed','population_reference_approved')}),flush=True)


if __name__=='__main__':main()

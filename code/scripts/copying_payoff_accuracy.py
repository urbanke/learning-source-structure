"""Bounded exact-short reproduction and continuation of the frozen Bayes pilot.

Thirty fresh t4 references reproduce the complete exact catalog; all twelve
pilot chains receive the same fixed fourfold total draw budget. Every source,
runtime, implementation and original checkpoint is bound before dispatch.
This accuracy diagnostic cannot approve a payoff forecast or population mean.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
from statistics import NormalDist
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
import copying_payoff_bayes_pilot as pilot
from prepare_copying_payoff_references import reference_key
from shared_base_short_evidence import SharedBaseShortEvidence,_patterns

SCHEMA='copying-payoff-accuracy-v1'
DEFAULT_OUT=ROOT/'output/copying_payoff_accuracy_2026-09-19'
TARGET_DRAWS=16384
RESOURCES=dict(max_workers=1,task_wall_seconds=900,total_worker_seconds=37800,
               maximum_attempts_per_task=1,checkpoint_seconds=120)
FAMILY_COMPARISONS=64*(30+21)
FAMILY_ALPHA=.05
DETERMINISTIC_ALLOWANCE=1e-7
Z_MULTIPLIER=NormalDist().inv_cdf(1-FAMILY_ALPHA/(2*FAMILY_COMPARISONS))
require,sha,fingerprint,read,write,lock=pilot.require,pilot.sha,pilot.fingerprint,pilot.read,pilot.write,pilot.lock


def code_hashes():
    result=pilot.code_hashes()
    extra=['code/scripts/copying_payoff_accuracy.py','code/scripts/unseen_context_refinement.py',
           'code/scripts/shared_base_surrogate_summary.py','code/scripts/shared_base_full_context_summary.py',
           'code/scripts/depth_rhat_statistics.py','code/transformer/shared_base_short_evidence.py',
           'code/transformer/eval_shared_base_surrogate.py']
    result.update({name:sha(ROOT/name) for name in extra})
    return result


def build_tasks(pilot_plan,catalog):
    tasks=[]
    for pattern in _patterns(4):
        for order in (16,32):
            task=dict(task_index=len(tasks),id=reference_key(pattern)+f'_a{order}',kind='exact_t4',
                tokens=list(pattern),time=4,order=order,reference_key=reference_key(pattern),
                baseline_draws=0,target_draws=TARGET_DRAWS,
                expected_probs=np.asarray(catalog.next_probs(pattern),dtype=float).tolist())
            task['seed']=int(fingerprint(['copying-accuracy-exact-t4',20260919,task])[:8],16)
            tasks.append(task)
    require(len(pilot_plan['tasks'])==12,'All twelve original pilot tasks required')
    for index,parent in enumerate(pilot_plan['tasks']):
        require(parent['task_index']==index and parent['time'] in (5,8,16)
                and parent['order'] in (16,32),'Pilot task coordinates changed')
        tasks.append(dict(task_index=len(tasks),id=parent['id'],kind='pilot_continuation',
            tokens=parent['tokens'],time=parent['time'],order=parent['order'],seed=parent['seed'],
            reference_key=parent['reference_key'],parent_index=index,baseline_draws=4096,
            target_draws=TARGET_DRAWS))
    require(len(tasks)==42 and len({t['id'] for t in tasks})==42,'Duplicate or missing accuracy task')
    return tasks


def _vector_check(left,se,right,allowance):
    left,se,right=map(lambda x:np.asarray(x,dtype=float),(left,se,right))
    require(left.shape==se.shape==right.shape==(64,) and np.isfinite(left).all()
            and np.isfinite(right).all() and np.isfinite(se).all() and np.min(se)>=0
            and np.min(left)>0 and np.min(right)>0 and abs(left.sum()-1)<=1e-7
            and abs(right.sum()-1)<=1e-7,'Invalid predictive vector or MCSE')
    error=abs(left-right);limit=Z_MULTIPLIER*se+allowance
    return dict(passed=bool(np.all(error<=limit)),failing_coordinates=np.flatnonzero(error>limit).tolist(),
        maximum_absolute_probability_error=float(error.max()),maximum_mcse=float(se.max()),
        maximum_excess_over_allowance=float(np.max(error-limit)),z_multiplier=Z_MULTIPLIER,
        family_comparisons=FAMILY_COMPARISONS,family_alpha=FAMILY_ALPHA,
        deterministic_probability_allowance=allowance,
        absolute_probability_errors=error.tolist(),coordinate_mcse=se.tolist(),
        coordinate_allowances=limit.tolist(),
        interpretation='Simultaneous two-sided normal-MCSE diagnostic with Bonferroni allowance across all30 exact-vector and21 paired-order comparisons; estimated MCMC errors do not yield a rigorous coverage guarantee.')


def exact_vector_check(prediction,mcse,expected):
    return _vector_check(prediction,mcse,expected,DETERMINISTIC_ALLOWANCE)


def paired_order_check(left_pred,left_mcse,right_pred,right_mcse):
    left,right=np.asarray(left_mcse,dtype=float),np.asarray(right_mcse,dtype=float)
    require(left.shape==right.shape==(64,) and np.isfinite(left).all() and np.isfinite(right).all()
            and np.min(left)>=0 and np.min(right)>=0,'Invalid paired MCSE')
    return _vector_check(left_pred,np.sqrt(left**2+right**2),right_pred,2*DETERMINISTIC_ALLOWANCE)


def coverage_summary(tasks,reports):
    expected={t['id']:t for t in tasks};found={r['task']['id']:r for r in reports}
    require(len(tasks)==len(expected)==42 and len(found)==len(reports) and set(found)<=set(expected),
            'Duplicate, missing declared scope or unknown accuracy report')
    require(all(r['task']==expected[key] for key,r in found.items()),'Accuracy report task changed')
    complete=len(found)==42;paired=[]
    for reference in sorted({t['reference_key'] for t in tasks}):
        group={r['task']['order']:r for r in reports if r['task']['reference_key']==reference}
        if set(group)!={16,32}:continue
        estimates=[group[o]['estimate'] for o in (16,32)]
        check=paired_order_check(estimates[0]['predictive_probs'],estimates[0]['mcse_probs'],
                                 estimates[1]['predictive_probs'],estimates[1]['mcse_probs'])
        paired.append(dict(reference_key=reference,**check))
    all_tasks=complete and all(r['audit_passed'] for r in reports)
    passed=all_tasks and len(paired)==21 and all(p['passed'] for p in paired)
    return dict(schema=SCHEMA,complete=complete,expected_tasks=42,completed_tasks=len(reports),
        missing_task_ids=sorted(set(expected)-set(found)),all_tasks_passed=all_tasks,
        accuracy_diagnostic_passed=passed,paired_order_checks=paired,
        failed_task_ids=[r['task']['id'] for r in reports if not r['audit_passed']],
        payoff_forecast_approved=False,population_reference_approved=False,
        interpretation='All42 fixed tasks and both integration orders are required. A pass concerns these exact-short and pilot accuracy diagnostics only; no selected-cohort payoff or population mean is approved.')


def paths(out,task):
    result=pilot.task_paths(out,task)
    result.update(parent_checkpoint=result['checkpoint'].with_name('parent_chains.npz'),
                  audit=result['checkpoint'].with_name('raw_audit.json'),
                  prepared=result['checkpoint'].with_name('prepared.json'))
    return result


def _bound(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Frozen parent missing or linked: '+str(path))
    return dict(path=str(path.resolve()),sha256=sha(path),size_bytes=path.stat().st_size)


def verify_parent(parent):
    for item in parent['files'].values():
        path=Path(item['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['size_bytes']
                and sha(path)==item['sha256'],'Frozen pilot parent changed: '+str(path))
    report=read(parent['files']['report']['path']);identity=read(parent['files']['identity']['path'])
    require(report['identity_sha256']==parent['files']['identity']['sha256']
            and report['checkpoint_sha256']==parent['files']['checkpoint']['sha256']
            and report['moments_sha256']==parent['files']['moments']['sha256']
            and identity==parent['identity'],'Pilot parent bindings differ')
    require(report['complete'] is True and report['estimate']['diagnostics']['draws']==4096,
            'Pilot parent is incomplete or has a different draw budget')
    return report


def prepare(out,pilot_out=pilot.DEFAULT_OUT):
    out=pilot.ensure_output(out);pilot_out=Path(pilot_out).resolve()
    require(not out.exists() or not any(out.iterdir()),'Accuracy output must be a new empty directory')
    require(out.resolve()!=pilot_out and out.resolve() not in pilot_out.parents and pilot_out not in out.resolve().parents,
            'Accuracy output overlaps frozen pilot')
    original=pilot.load_plan(pilot_out);catalog=SharedBaseShortEvidence();parents={}
    for index,task in enumerate(original['tasks']):
        p=pilot.task_paths(pilot_out,task);report=read(p['report']);identity=read(p['identity'])
        require(report['task']==task and report['plan_id']==original['plan_id'] and report['complete']
                and identity['task']==task and identity['plan_id']==original['plan_id'],
                'Need every original pilot task, including failed numerical diagnostics')
        item=dict(task=task,identity=identity,files={name:_bound(p[name]) for name in
                      ('checkpoint','report','moments','identity','attempt')})
        verify_parent(item);parents[str(index)]=item
    plan=dict(schema=SCHEMA,output=str(out.resolve()),pilot_path=str(pilot_out/'plan.json'),
        pilot_plan_sha256=sha(pilot_out/'plan.json'),pilot_plan_id=original['plan_id'],parents=parents,
        inventory=original['inventory'],inventory_sha256=original['inventory_sha256'],
        tasks=build_tasks(original,catalog),runtime=original['runtime'],python=original['python'],
        environment=original['environment'],dependency_sha256=original['dependency_sha256'],
        prior=original['prior'],sampler=dict(chains=4,burn=16384,thin=1,draws_per_chain=TARGET_DRAWS,surrogate_steps=64),
        resources=RESOURCES,code_sha256=code_hashes(),catalog_validation=catalog.validation,
        comparison=dict(family_alpha=FAMILY_ALPHA,family_comparisons=FAMILY_COMPARISONS,
                        z_multiplier=Z_MULTIPLIER,deterministic_probability_allowance=DETERMINISTIC_ALLOWANCE),
        payoff_forecast_approved=False,population_reference_approved=False)
    for protected in read(original['inventory'])['candidate_scan']['roots']:
        path=Path(protected).resolve()
        require(out.resolve()!=path and path not in out.resolve().parents and out.resolve() not in path.parents,
                'Accuracy output overlaps an original reference tree')
    plan['plan_id']=fingerprint(plan);write(out/'plan.json',plan,immutable=True)
    return plan


def load_plan(out):
    out=pilot.ensure_output(out);plan=read(out/'plan.json')
    require(plan['schema']==SCHEMA and plan['plan_id']==fingerprint({k:v for k,v in plan.items() if k!='plan_id'})
            and plan['output']==str(out.resolve()),'Accuracy plan identity differs')
    require(plan['resources']==RESOURCES and plan['code_sha256']==code_hashes(), 'Accuracy code or limits changed')
    require(sha(plan['pilot_path'])==plan['pilot_plan_sha256'] and sha(plan['inventory'])==plan['inventory_sha256'],
            'Pilot plan or source inventory changed')
    original=pilot.load_plan(Path(plan['pilot_path']).parent);catalog=SharedBaseShortEvidence()
    require(plan['tasks']==build_tasks(original,catalog) and plan['catalog_validation']==catalog.validation,
            'Exact catalog or fixed42-task coverage changed')
    require(plan['runtime']==original['runtime'] and plan['prior']==original['prior']
            and plan['environment']==original['environment'] and plan['dependency_sha256']==original['dependency_sha256'],
            'Numerical prior, runtime or deployment changed')
    require(plan['sampler']==dict(chains=4,burn=16384,thin=1,draws_per_chain=16384,surrogate_steps=64),
            'Fixed sampling settings changed')
    require(set(plan['parents'])==set(map(str,range(12))),'Missing original pilot parents')
    return plan


def verify_loaded_parent(run,metadata):
    """Loading may reconstruct caches, but must restore every saved chain state and RNG."""
    require(run.fingerprint==metadata['fingerprint'] and run.proposal_descriptor==metadata['proposal_descriptor']
            and run.initializations==metadata['initializations'],'Loaded target or initialization changed')
    require({k:getattr(run,k) for k in ('chains','burn','thin','seed')}==metadata['config'],
            'Loaded chain configuration differs')
    require([r.bit_generator.state for r in run._rngs]==metadata['rng_states'],
            'Loaded chain RNG states changed')
    actual=[dict(alpha_index=s.alpha_index,ranks=s.ranks.tolist(),log_likelihood=s.log_likelihood,
                 log_target=s.log_target) for s in run._states]
    require(actual==metadata['states'],'Loaded chain endpoint states changed')


def _audit_raw(task,identity,estimate,path):
    from shared_base_surrogate_summary import raw_checkpoint
    from shared_base_full_context_summary import Audit
    from full_context_statistics import diagnostic_gates
    from depth_rhat_statistics import coordinate_summary
    audit=Audit()
    adapted=dict(identity=dict(numerical_problem=identity['target'],proposal_descriptor=identity['proposal']),
                 settings=dict(chains=4,burn=16384,thin=1,seed=task['seed']),estimate=estimate)
    moments,fresh=raw_checkpoint(audit,'canonical_accuracy',path,adapted)
    corrected=deepcopy(fresh);details=[]
    with np.load(path,allow_pickle=False) as stored:
        depths=np.stack([stored[f'depths_{i}'] for i in range(4)])
    for depth in range(9):
        row=coordinate_summary(depths[:,:,depth]);details.append(row)
        value=row['rescaled']['rhat']
        if row['rescaled']['status']=='constant_uninformative' and row['min']==row['max']:value=1.
        elif row['rescaled']['status']!='finite':value=None
        corrected['diagnostics']['rhat_depth'][depth]=value
    gates=diagnostic_gates(corrected)
    return moments,dict(raw_checks=audit.checks,raw_inputs=audit.inputs,
        classic_gates=diagnostic_gates(fresh),corrected_gates=gates,depth_diagnostics=details,
        passed=all(gates.values())),fresh


def worker(out,index):
    plan=load_plan(out);require(0<=index<42,'Accuracy task index outside0..41')
    os.environ.update(pilot.environment(plan));require(pilot.runtime()==plan['runtime'],'Accuracy runtime differs')
    task=plan['tasks'][index];p=paths(out,task);attempt=read(p['attempt'])
    require(attempt['plan_id']==plan['plan_id'] and attempt['task']==task and attempt['status']=='running'
            and attempt['dispatcher_pid']==os.getppid(),'Worker requires its bounded dispatcher allocation')
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from eval_shared_base_models import atomic_npz
    from validate_shared_base_mcmc import json_safe
    from unseen_context_refinement import checkpoint_metadata,verify_preserved_samples
    with lock(Path(out)/'worker.lock'):
        require(not p['checkpoint'].exists() and not p['report'].exists(),'No automatic retry or restart of an accuracy task')
        nodes,weights=np.polynomial.legendre.leggauss(task['order'])
        sampler=SharedBaseSurrogateMCMC(64,RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384),weights/2,surrogate_steps=64)
        tokens=np.asarray(task['tokens'],dtype=int);parent=None
        if task['kind']=='pilot_continuation':
            parent=plan['parents'][str(task['parent_index'])];verify_parent(parent)
            for name in ('parent_checkpoint','checkpoint'):
                p[name].parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(parent['files']['checkpoint']['path'],p[name])
                require(sha(p[name])==parent['files']['checkpoint']['sha256'],'Isolated checkpoint copy differs')
            metadata=checkpoint_metadata(p['parent_checkpoint'])
            run=CurveChainRun.load(sampler,tokens,p['checkpoint']);verify_loaded_parent(run,metadata)
            require(run.draw_counts.tolist()==[4096]*4,'Saved pilot draw count changed')
            write(p['prepared'],dict(plan_id=plan['plan_id'],parent_files=parent['files'],
                exact_rng_restored_before_extension=True),immutable=True)
        else:run=CurveChainRun(sampler,tokens,chains=4,burn=16384,thin=1,seed=task['seed'])
        identity=dict(plan_id=plan['plan_id'],task=task,target=run.fingerprint,proposal=run.proposal_descriptor,runtime=pilot.runtime())
        if parent is not None:
            require(identity['target']==parent['identity']['target'] and identity['proposal']==parent['identity']['proposal'],
                    'Continuation target or proposal differs from the original pilot')
        write(p['identity'],identity,immutable=True);saved=time.monotonic();began=saved
        def progress(event):
            nonlocal saved
            now=time.monotonic()
            if event['phase']=='burn' or event['draws']==event['requested_draws'] or now-saved>=120:
                run.save(p['checkpoint']);saved=now
            print(json.dumps(dict(task=task['id'],**event)),flush=True)
        estimate=run.extend(TARGET_DRAWS,block_draws=2048,progress=progress);run.save(p['checkpoint'])
        del run
        if parent is not None:
            verify_preserved_samples(p['parent_checkpoint'],p['checkpoint'],task,complete=True);verify_parent(parent)
        moments,audit,fresh=_audit_raw(task,identity,estimate,p['checkpoint'])
        atomic_npz(p['moments'],**moments)
        grids=selected_grid_checks(sampler,tokens,estimate)
        exact=exact_vector_check(fresh['predictive_probs'],fresh['mcse_probs'],task['expected_probs']) if task['kind']=='exact_t4' else None
        audit.update(exact_vector_check=exact,final_state_grid_checks=grids)
        passed=audit['passed'] and all(g['passed'] for g in grids) and (exact is None or exact['passed'])
        write(p['audit'],json_safe(audit),immutable=True)
        report=json_safe(dict(schema=SCHEMA,plan_id=plan['plan_id'],task=task,complete=True,audit_passed=bool(passed),
            estimate=fresh,audit_sha256=sha(p['audit']),identity_sha256=sha(p['identity']),
            checkpoint_sha256=sha(p['checkpoint']),moments_sha256=sha(p['moments']),elapsed_seconds=time.monotonic()-began,
            parent_samples_preserved=parent is not None,payoff_forecast_approved=False,population_reference_approved=False))
        write(p['report'],report,immutable=True)
        return report


def summary(out):
    plan=load_plan(out);reports=[];attempts=[]
    for task in plan['tasks']:
        p=paths(out,task)
        if p['attempt'].exists():attempts.append(read(p['attempt']))
        if not p['report'].exists():continue
        r=read(p['report'])
        require(r['plan_id']==plan['plan_id'] and r['task']==task and r['complete']
                and all(r[name+'_sha256']==sha(p[name]) for name in ('audit','identity','checkpoint','moments')),
                'Accuracy report binding changed')
        audit=read(p['audit']);expected=audit['passed'] and all(g['passed'] for g in audit['final_state_grid_checks'])
        if task['kind']=='exact_t4':
            check=exact_vector_check(r['estimate']['predictive_probs'],r['estimate']['mcse_probs'],task['expected_probs'])
            require(check==audit['exact_vector_check'],'Exact vector comparison changed');expected=expected and check['passed']
        require(r['audit_passed']==expected,'Task acceptance flag changed')
        reports.append(r)
    result=coverage_summary(plan['tasks'],reports);result.update(plan_id=plan['plan_id'],attempts=attempts)
    write(Path(out)/'summary.json',result);return result


def dispatch(out,only_tasks=None):
    plan=load_plan(out);selected=sorted(set(range(42) if only_tasks is None else only_tasks))
    require(selected and all(type(i)is int and 0<=i<42 for i in selected),'Need declared accuracy indices0..41')
    with lock(Path(out)/'dispatcher.lock'):
        for index in selected:
            task=plan['tasks'][index];p=paths(out,task)
            if p['attempt'].exists():continue
            spent=sum(read(paths(out,t)['attempt'])['reserved_seconds'] for t in plan['tasks'] if paths(out,t)['attempt'].exists())
            require(spent+900<=37800,'Fixed accuracy allocation budget exhausted')
            attempt=dict(plan_id=plan['plan_id'],task=task,reserved_seconds=900,status='running',
                         started_unix=time.time(),dispatcher_pid=os.getpid())
            write(p['attempt'],attempt,immutable=True);began=time.monotonic();process=None
            try:
                with p['log'].open('x') as stream:
                    process=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'worker','--out',str(out),'--index',str(index)],
                        cwd=ROOT,env=pilot.environment(plan),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code=process.wait(timeout=900)
                        status='failed' if code!=0 else 'ready' if read(p['report'])['audit_passed'] else 'unresolved'
                        attempt.update(status=status,exit_code=code)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        attempt.update(status='time_limit',exit_code=75)
            except BaseException:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                attempt['status']='interrupted';raise
            finally:
                attempt.update(elapsed_seconds=time.monotonic()-began,finished_unix=time.time());write(p['attempt'],attempt)
            summary(out)
    return summary(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','summary'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--pilot',type=Path,default=pilot.DEFAULT_OUT)
    parser.add_argument('--only-task',type=int,action='append');parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.out,args.pilot)
    elif args.action=='run':result=dispatch(args.out,args.only_task)
    elif args.action=='worker':
        require(args.index is not None,'Worker needs a declared task index');result=worker(args.out,args.index)
    else:result=summary(args.out)
    print(json.dumps({k:v for k,v in result.items() if k in ('plan_id','complete','completed_tasks','accuracy_diagnostic_passed','audit_passed')}),flush=True)


if __name__=='__main__':main()

"""Bounded all-context preparation for the fixed 192-history pooled comparison.

Timing is permitted without accuracy approval. Scientific sampling requires the
actual short-calibration decision and long-context refinement. Every result
retains the full planned coverage; failed histories are never dropped.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
sys.path.insert(0, str(ROOT/'code/scripts'))
import pooled_predictive_calibration as calibration
import pooled_predictive_pilot as pilot
from shared_base_predictive_smc import (estimate, pool_repetitions, checked_probs,
    component_prediction, combine_components, mixture_regret_mcse)
from shared_base_short_evidence import SharedBaseShortEvidence
from mixture_family_evidence import FamilyEvidence
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC

PLAN = ROOT/'code/transformer/configs/pooled_predictive_population_plan_2026-09-19.json'
FAMILIES = pilot.FAMILIES
TIMES = (16,64,256,1024)
ALL_TIMES = (1,4,*TIMES)
INDICES = tuple(range(5,69))
VARIANTS = {
    'A': dict(particles=8192,temperatures=64,moves=4,alpha_order=16,row_bounds=[-32.,32.],row_step=1/32),
    'B': dict(particles=16384,temperatures=128,moves=4,alpha_order=16,row_bounds=[-32.,32.],row_step=1/32),
    'C': dict(particles=16384,temperatures=128,moves=4,alpha_order=32,row_bounds=[-32.,32.],row_step=1/32),
    'D': dict(particles=16384,temperatures=128,moves=4,alpha_order=16,row_bounds=[-40.,40.],row_step=1/64),
}
STAGES = ('timing','refinement_timing','refine','population')
CODE = tuple(sorted(set(calibration.CODE + pilot.CODE)))
require, sha, digest, atomic_json = calibration.require, calibration.sha, calibration.digest, calibration.atomic_json


def read_plan(path=PLAN):
    plan=json.loads(Path(path).read_text())
    require(plan['schema']=='pooled-predictive-population-plan-v1' and plan['times']==list(TIMES)
        and plan['variants']==VARIANTS and plan['indices']==list(INDICES), 'Population plan changed')
    require(plan['family_order']==list(FAMILIES) and plan['family_prior']==[1/3]*3
        and plan['development_indices']==[1,2,3,4] and plan['n_per_family']==64 and plan['publication_arm']=='B' and plan['cheap_numerics']==dict(
            default_bounds=[-30.,30.],default_step=1/64,refined_bounds=[-40.,40.],refined_step=1/128), 'Selection changed')
    require(plan['stages']==dict(timing=dict(repetitions=1,seconds_per_replica=1200,allocation_minutes=30,concurrency=12),
        refinement_timing=dict(repetitions=1,seconds_per_replica=2400,allocation_minutes=45,concurrency=12),
        refine=dict(repetitions={'A':64,'B':32,'C':32,'D':32},seconds_per_replica=1800,allocation_minutes=60,concurrency=64),
        population=dict(repetitions=64,seconds_per_replica=1800,allocation_minutes=45,concurrency=64)), 'Work limits changed')
    require(plan['resource_limits']==dict(maximum_total_projected_cpu_hours=4096,maximum_attempts_per_task=2,
        timing_safety_factor=2.,maximum_array_size=768), 'Resource limits changed')
    require(plan['criteria']==dict(tolerance_bits=.01,mcse_multiplier=3.,minimum_replicates=32,
        minimum_effective_replica_count=8.,maximum_likelihood_share=.2,normalization_tolerance=1e-7,
        maximum_relative_mcse=.1,cohort_mcse_multiplier=2.,cohort_bias_multiplier=2.,
        cohort_absolute_budget_bits=.01/3,cohort_se_fraction=.25), 'Numerical criteria changed')
    require(set(plan['code_sha256'])==set(CODE), 'Incomplete frozen code binding')
    for name,value in plan['code_sha256'].items(): require(sha(ROOT/name)==value,'Frozen implementation changed: '+name)
    selection=plan['selection']
    require(digest(selection)==plan['selection_digest'] and selection['indices']==list(INDICES)
        and selection['family_order']==list(FAMILIES) and selection['n_total']==192
        and selection['observation_times']==[16,64], 'Original fixed selection changed')
    require(sha(calibration.PLAN)==plan['calibration_plan_sha256'],'Exact calibration plan changed')
    return plan,sha(path)


def task_specs(plan,stage):
    require(stage in STAGES,'Unknown stage')
    tasks=[]
    for observed in TIMES:
        for family in FAMILIES:
            indices=[1] if stage in ('timing','refinement_timing') else plan['development_indices'] if stage=='refine' else list(INDICES)
            for index in indices:
                arms=['P256','P1024','P8192'] if stage=='timing' else ['B','C','D'] if stage in ('refinement_timing','population') else list(VARIANTS)
                for arm in arms:
                    settings=dict(VARIANTS['A' if arm.startswith('P') else arm])
                    if arm.startswith('P'):settings['particles']=int(arm[1:])
                    count=plan['stages'][stage]['repetitions']
                    if isinstance(count,dict):count=count[arm]
                    tasks.append(dict(task_index=len(tasks),stage=stage,family=family,index=index,observed=observed,
                        arm=arm,settings=settings,repetitions=count,development=stage!='population'))
    return tasks


def seed_for(plan,spec,replica):
    return int(digest(['pooled-population-v1',plan['seed'],spec,replica])[:16],16)


def calibration_decision(folder,*,require_pass=False):
    """Read the real gate; if required, independently replay all bound shards."""
    folder=Path(folder);path=folder/'summary.json'
    if not path.exists():
        require(not require_pass,'Exact-short calibration report is missing')
        return dict(available=False,complete=False,exact_short_passed=False)
    saved=json.loads(path.read_text())
    require(saved['schema']=='pooled-predictive-exact-short-summary-v1','Wrong calibration report')
    info=dict(available=True,summary_sha256=sha(path),complete=saved['complete'],
        exact_short_passed=saved['exact_short_passed'],attempt=saved['identity']['attempt'])
    if not require_pass:return info
    require(saved['complete'] is True and saved['exact_short_passed'] is True
        and not saved['missing_tasks'] and not saved['unfinished_tasks'], 'Exact-short accuracy has not passed')
    p,h=calibration.read_plan(calibration.PLAN);attempt=saved['identity']['attempt']
    require(attempt=='stronger' and saved['identity']['plan_sha256']==h and saved['criteria']==p['criteria'],
        'Need the unchanged stronger calibration')
    require(saved['identity']['settings']==p['attempts'][attempt] and saved['identity']['numerics']==p['numerics']
        and saved['identity']['code_sha256']==p['code_sha256'] and saved['identity']['reference_sha256']==p['reference_sha256'],
        'Calibration settings or code differ')
    catalog=SharedBaseShortEvidence();grouped={c['id']:[] for c in p['cases']}
    count=len(p['cases'])*p['attempts'][attempt]['replicates']//p['replicas_per_task']
    expected={f'shards/task-{i:04d}.json' for i in range(count)}
    require(set(saved['shard_sha256'])==expected,'Calibration coverage changed')
    for i in range(count):
        key=f'shards/task-{i:04d}.json';require(sha(folder/key)==saved['shard_sha256'][key],'Calibration shard changed')
        item=json.loads((folder/key).read_text());spec=calibration.task_spec(p,attempt,i)
        require(item['exact_reference']==calibration.exact_reference(catalog,spec['case']['tokens']), 'Exact catalog changed')
        rows=calibration.validate_report(item,saved['identity'],spec,p,attempt)
        require(item['complete'],'Calibration shard incomplete')
        grouped[spec['case']['id']].extend(r['estimate'] for r in rows)
    rebuilt=[dict(case=c,**calibration.summarize(grouped[c['id']],calibration.exact_reference(catalog,c['tokens']),
              p['attempts'][attempt]['replicates'])) for c in p['cases']]
    require(rebuilt==saved['cases'] and all(r['passed'] for r in rebuilt),'Calibration decision does not replay')
    info['independent_gate_replay']=True
    return info


def load_case(raw,plan,spec):
    """Family identity/oracle only scores predictions; observed prefix is x[:t]."""
    raw=Path(raw);family,index,t=spec['family'],spec['index'],spec['observed'];selection=plan['selection']
    cases=selection['development_cases'] if spec['development'] else selection['cases']
    wanted=[c for c in cases if c['family']==family and c['index']==index]
    require(len(wanted)==1 and t in ALL_TIMES,'Case outside fixed selection')
    cohort=family+'_prior';names=['heldout/manifest.json',f'heldout/{cohort}.json',f'heldout/{cohort}.npz']
    names += [f'eval/{model}/{cohort}.{ext}' for model in ('pooled_seed0','pooled_seed1') for ext in ('json','npz')]
    hashes={name:sha(raw/name) for name in names}
    require(all(hashes[k]==selection['input_sha256'][k] for k in names),'Frozen pooled input changed')
    source=json.loads((raw/f'heldout/{cohort}.json').read_text())
    require(source['plan_id']==selection['source_plan_id'] and source['data_sha256']==hashes[f'heldout/{cohort}.npz'],'Source receipt changed')
    col=ALL_TIMES.index(t)
    with np.load(raw/f'heldout/{cohort}.npz',allow_pickle=False) as a:
        require(np.array_equal(a['prefixes'],ALL_TIMES) and np.array_equal(a['sequence_seed'][index],wanted[0]['sequence_seed']), 'Source time/seed mismatch')
        tokens=a['tokens'][index,:t].copy();oracle=a['oracle_rows'][index,col].copy()
        require(len(tokens)==t and np.array_equal(oracle,a['transition_probs'][index,tokens[-1]]),'Prefix/oracle off by one')
        checked_probs(oracle,64)
    scores={}
    for model in ('pooled_seed0','pooled_seed1'):
        r=json.loads((raw/f'eval/{model}/{cohort}.json').read_text())
        require(r['data_sha256']==hashes[f'eval/{model}/{cohort}.npz'] and r['source_data_sha256']==source['data_sha256']
            and r['identity']['name']==model and r['identity']['source_plan_id']==selection['source_plan_id'],'Model receipt differs')
        with np.load(raw/f'eval/{model}/{cohort}.npz',allow_pickle=False) as a:
            require(np.array_equal(a['prefixes'],ALL_TIMES) and np.array_equal(a['sequence_seed'][index],wanted[0]['sequence_seed']),'Model pairing differs')
            scores[model]=float(a['next_kl_bits'][index,col]);require(np.isfinite(scores[model]),'Nonfinite transformer score')
    return tokens,oracle,scores,hashes


def checked_component(value,d=64):
    p=checked_probs(value['predictive_probs'],d);z=float(value['log_evidence'])
    require(np.isfinite(z) and z<=1e-12,'Invalid absolute component evidence')
    joint=np.asarray(value['log_joint_next'])
    require(joint.shape==(d,) and np.isfinite(joint).all() and
        np.max(np.abs(joint-z-np.log(p)))<=1e-9 and abs(logsumexp(joint)-z)<=1e-7,
        'Component joint/evidence/normalization mismatch')
    return value


def regret(oracle,p):
    q=checked_probs(oracle);p=checked_probs(p,len(q))
    return float(q@(np.log(q)-np.log(p))/np.log(2))


def component_uncertainty(records):
    pooled=pool_repetitions(records);n=len(records)
    require(n>=2,'At least two repetitions required')
    logs=np.asarray([r['log_evidence'] for r in records]);z=np.exp(logs-logs.max())
    joint=z[:,None]*np.array([r['predictive_probs'] for r in records])
    evidence=float(z.std(ddof=1)/math.sqrt(n)/z.mean()/np.log(2))
    influence=(joint/joint.mean(0)-z[:,None]/z.mean())/np.log(2)
    return dict(pooled=pooled,evidence_mcse_bits=evidence,
        predictive_log_mcse_bits=(influence.std(0,ddof=1)/math.sqrt(n)).tolist())


def refinement_pair(left_records,right_records):
    a,b=component_uncertainty(left_records),component_uncertainty(right_records)
    evidence_difference=abs(a['pooled']['log_evidence']-b['pooled']['log_evidence'])/np.log(2)
    prob_difference=np.abs(np.log2(a['pooled']['predictive_probs'])-np.log2(b['pooled']['predictive_probs']))
    evidence_allowance=float(evidence_difference+3*math.hypot(a['evidence_mcse_bits'],b['evidence_mcse_bits']))
    predictive_allowance=prob_difference+3*np.hypot(a['predictive_log_mcse_bits'],b['predictive_log_mcse_bits'])
    return dict(passed=evidence_allowance<=.01 and bool(np.max(predictive_allowance)<=.01),
        evidence_difference_bits=float(evidence_difference),evidence_allowance_bits=evidence_allowance,
        predictive_allowance_bits=predictive_allowance.tolist())


def mixture_uncertainty(records,fixed_components,oracle):
    """Covariance-aware ratio uncertainty plus conservative curvature/jackknife."""
    n=len(records);require(n>=2,'At least two repetitions required')
    q=checked_probs(oracle);logs=np.array([r['log_evidence'] for r in records])
    fixed_logs=np.array([r['log_evidence'] for r in fixed_components])
    shift=float(max(logs.max(),fixed_logs.max()));z=np.exp(logs-shift)/3
    joint=z[:,None]*np.array([checked_probs(r['predictive_probs'],len(q)) for r in records])
    fixed_z=np.exp(fixed_logs-shift)/3
    fixed_p=np.array([checked_probs(r['predictive_probs'],len(q)) for r in fixed_components])
    a=fixed_z@fixed_p+joint.mean(0);d=fixed_z.sum()+z.mean()
    va=joint.var(0,ddof=1)/n;vd=float(z.var(ddof=1)/n)
    curvature=float((q@(va/a**2)+vd/d**2)/(2*np.log(2)))
    p=checked_probs(a/d,len(q));full=regret(q,p)
    leave=[]
    for i in range(n):
        aa=fixed_z@fixed_p+(joint.sum(0)-joint[i])/(n-1)
        dd=fixed_z.sum()+(z.sum()-z[i])/(n-1)
        leave.append(regret(q,checked_probs(aa/dd,len(q))))
    jack=abs((n-1)*(float(np.mean(leave))-full))
    relative=float(max(np.max(np.sqrt(va)/a),math.sqrt(vd)/d))
    bias=max(curvature,jack)
    return dict(mcse_bits=mixture_regret_mcse(records,fixed_components,q),curvature_bias_bits=curvature,
        jackknife_bias_bits=jack,bias_allowance_bits=bias,relative_mcse_max=relative,
        passed=relative<=.1 and jack<=2*curvature+1e-12)


def balanced_summary(rows):
    """Exactly 64 paired histories per family; source and numerical SE separate."""
    require(len(rows)==192 and {(r['family'],r['index']) for r in rows}=={(f,i) for f in FAMILIES for i in INDICES},
        'Every one of the 192 fixed histories is required')
    require(all(np.isfinite([r['bayes_regret_bits'],r['mcse_bits'],r['bias_allowance_bits'],
        *r['transformer_regret_bits'].values()]).all() and r['mcse_bits']>=0 and r['bias_allowance_bits']>=0
        and set(r['transformer_regret_bits'])=={'pooled_seed0','pooled_seed1'} for r in rows),'Invalid paired scores')
    ordered=sorted(rows,key=lambda r:(FAMILIES.index(r['family']),r['index']))
    def stats(x):
        x=np.asarray(x).reshape(3,64);return dict(mean=float(x.mean()),se=float(np.sqrt(np.sum(x.var(1,ddof=1)/64))/3),
            family_means=x.mean(1).tolist(),family_se=(x.std(1,ddof=1)/8).tolist())
    bayes=np.array([r['bayes_regret_bits'] for r in ordered]);b=stats(bayes)
    numerical=float(np.sqrt(sum(r['mcse_bits']**2 for r in ordered))/192)
    bias=float(np.mean([r['bias_allowance_bits'] for r in ordered]))
    # Remove independent numerical variance only for setting the source-SE budget.
    strata=bayes.reshape(3,64)
    num_var=np.array([r['mcse_bits']**2 for r in ordered]).reshape(3,64)
    denoised=float(np.sqrt(np.maximum(0.,strata.var(1,ddof=1)-num_var.mean(1)).sum()/64)/3)
    budget=min(.01/3,.25*denoised);allowance=2*numerical+2*bias
    return dict(n_total=192,n_per_family=64,bayes=b,
        transformers={name:stats([r['transformer_regret_bits'][name] for r in ordered]) for name in ('pooled_seed0','pooled_seed1')},
        paired={name:stats([r['transformer_regret_bits'][name]-r['bayes_regret_bits'] for r in ordered]) for name in ('pooled_seed0','pooled_seed1')},
        numerical_mcse_bits=numerical,bias_allowance_bits=bias,numerical_allowance_bits=allowance,
        numerical_budget_bits=budget,denoised_source_se_bits=denoised,precision_passed=allowance<=budget)


def validate_estimate(value,plan,spec,replica):
    require(value['schema']=='shared-base-predictive-smc-repetition-v1' and value['n_observed']==spec['observed']
        and value['seed']==seed_for(plan,spec,replica) and value['resample_fraction']==.75
        and all(value[k]==spec['settings'][k] for k in ('particles','temperatures','moves')),
        'Repetition settings or source length changed')
    checked_component(value)
    require(np.isfinite(value['maximum_conditional_normalization_error']) and 0<=value['maximum_conditional_normalization_error']<=1e-7,
        'Underlying predictive kernel failed normalization')
    require(len(value['effective_sample_sizes'])==spec['settings']['temperatures']
        and np.isfinite(value['effective_sample_sizes']).all() and min(value['effective_sample_sizes'])>0,
        'Invalid particle diagnostics')
    for k in ('elapsed_seconds','sampler_seconds','prediction_seconds'):
        require(np.isfinite(value[k]) and value[k]>=0,'Invalid repetition timing')


def record_summary(records,cheap,oracle,scores):
    shared=pool_repetitions(records)
    mixture=combine_components([cheap['product']['log_evidence'],shared['log_evidence'],cheap['uniform_base']['log_evidence']],
        [cheap['product']['predictive_probs'],shared['predictive_probs'],cheap['uniform_base']['predictive_probs']])
    checked_component(mixture)
    result=dict(shared_base=shared,mixture=mixture,bayes_regret_bits=regret(oracle,mixture['predictive_probs']),
        transformer_regret_bits=scores)
    if len(records)>=2:
        u=component_uncertainty(records);m=mixture_uncertainty(records,[cheap['product'],cheap['uniform_base']],oracle)
        result.update(component_uncertainty=u,mixture_uncertainty=m)
        result['basic_passed']=(len(records)>=32 and shared['effective_replica_count']>=8 and
            shared['maximum_single_replica_likelihood_share']<=.2 and m['passed'] and
            3*u['evidence_mcse_bits']<=.01 and 3*max(u['predictive_log_mcse_bits'])<=.01)
    else:result['basic_passed']=False
    return result


def stage_path(out,stage,index):return Path(out)/stage/'tasks'/f'task-{index:04d}.json'


def load_stage(plan,plan_hash,out,stage):
    reports={};missing=[]
    for spec in task_specs(plan,stage):
        path=stage_path(out,stage,spec['task_index'])
        if not path.exists():missing.append(spec['task_index']);continue
        r=json.loads(path.read_text())
        require(r['schema']=='pooled-predictive-population-task-v1' and r['task']==spec
            and r['identity']['plan_sha256']==plan_hash and r['identity']['driver_sha256']==sha(__file__), 'Task identity changed')
        require(r['records_sha256']==digest(r['records']),'Saved repetition payload changed')
        require(len(r['records'])<=spec['repetitions'] and r['complete']==(len(r['records'])==spec['repetitions']), 'Task coverage changed')
        for i,record in enumerate(r['records']):validate_estimate(record,plan,spec,i)
        for value in r['cheap_components'].values():checked_component(value)
        require(r['cheap_components_sha256']==digest(r['cheap_components']), 'Fixed family components changed')
        if r['records']:
            rebuilt=record_summary(r['records'],r['cheap_components'],r['oracle'],r['transformer_regret_bits'])
            require(rebuilt==r['summary'],'Task statistics do not replay')
        reports[spec['task_index']]=r
        if not r['complete']:missing.append(spec['task_index'])
    return reports,missing


def read_stage_gate(out,stage,plan_hash,observed):
    path=Path(out)/stage/'summary.json';require(path.is_file(),'Missing prerequisite stage: '+stage)
    value=json.loads(path.read_text())
    require(value['plan_sha256']==plan_hash and value['driver_sha256']==sha(__file__),'Prerequisite implementation changed')
    context=value['by_context'][str(observed)]
    for name,h in context['task_sha256'].items():require(sha(Path(out)/name)==h,'Prerequisite task changed')
    require(context['complete'] is True and context['passed'] is True,
        'Prerequisite context has not passed: '+stage+' t'+str(observed))
    return context


def scientific_gate(plan,plan_hash,out,stage,calibration_root,observed):
    if stage in ('timing','refinement_timing'):
        return dict(calibration=calibration_decision(calibration_root),scientific_reference_validated=False)
    cal=calibration_decision(calibration_root,require_pass=True)
    timing=read_stage_gate(out,'refinement_timing',plan_hash,observed)
    require(timing['resource_projection']['within_budget'],'Measured total campaign exceeds its declared cost cap')
    gate=dict(calibration=cal,refinement_timing_context_sha256=digest(timing))
    if stage=='population':
        refine=read_stage_gate(out,'refine',plan_hash,observed)
        require(refine['all_component_and_mixture_refinements_passed'],'Long-context numerical refinement did not pass')
        gate['refinement_context_sha256']=digest(refine)
    return gate


@contextmanager
def heartbeat(path,payload):
    done=threading.Event();start=time.monotonic()
    def emit():
        while not done.wait(30):
            item=dict(**payload,status='running',elapsed_seconds=time.monotonic()-start,
                within_replica_checkpoint=False,scientific_reference_validated=False)
            atomic_json(path,item);print(json.dumps(item),flush=True)
    thread=threading.Thread(target=emit,daemon=True);thread.start()
    try:yield
    finally:done.set();thread.join()


def run(plan_path,stage,index,raw,dependency_root,out,calibration_root):
    plan,plan_hash=read_plan(plan_path);specs=task_specs(plan,stage)
    require(0<=index<len(specs),'Task outside plan');spec=specs[index]
    out=Path(out);raw=Path(raw);dep=Path(dependency_root)
    path=stage_path(out,stage,index);path.parent.mkdir(parents=True,exist_ok=True)
    gate=scientific_gate(plan,plan_hash,out,stage,calibration_root,spec['observed'])
    cp,_=calibration.read_plan(calibration.PLAN)
    store,src=dep/'tables/anchors_prod',dep/'src'
    dependencies=calibration.dependencies(cp,store,src)
    environment=calibration.numerical_environment(store,src)
    tokens,oracle,scores,hashes=load_case(raw,plan,spec)
    ident=dict(plan_sha256=plan_hash,driver_sha256=sha(__file__),code_sha256=plan['code_sha256'],
        source_sha256=hashes,dependency_sha256=dependencies,numerical_environment=environment,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=calibration.scipy.__version__))
    def stop(sig,_):raise InterruptedError('Signal '+signal.Signals(sig).name)
    with calibration.writer_claim(path):
        if path.exists():
            report=json.loads(path.read_text());require(report['identity']==ident and report['task']==spec,'Resume identity changed')
            require(report['records_sha256']==digest(report['records']),'Saved repetitions changed')
            for j,r in enumerate(report['records']):validate_estimate(r,plan,spec,j)
            require(report['cheap_components_sha256']==digest(report['cheap_components']),'Saved components changed')
            require(report['tokens']==tokens.tolist() and report['oracle']==oracle.tolist()
                and report['transformer_regret_bits']==scores,'Resume source/score identity changed')
            require(report['gate']==gate or stage in ('timing','refinement_timing'),'Scientific gate changed on resume')
            if report['complete']:return 0
            require(report['attempts']<plan['resource_limits']['maximum_attempts_per_task'],'Task retry cap reached; inspect saved outputs')
            report['attempts']+=1
        else:
            bounds=(-40.,40.) if spec['arm']=='D' else (-30.,30.)
            step=1/128 if spec['arm']=='D' else 1/64
            cheap={f:component_prediction(FamilyEvidence(f,bounds=bounds,step=step),tokens) for f in ('product','uniform_base')}
            for c in cheap.values():checked_component(c)
            report=dict(schema='pooled-predictive-population-task-v1',identity=ident,task=spec,gate=gate,tokens=tokens.tolist(),
                oracle=oracle.tolist(),transformer_regret_bits=scores,cheap_components=cheap,cheap_components_sha256=digest(cheap),
                attempts=1,records=[],records_sha256=digest([]),complete=False,status='starting',
                scientific_reference_validated=False,within_replica_checkpoint=False,summary=None)
        atomic_json(path,report)
        settings=spec['settings'];nodes,weights=leggauss(settings['alpha_order'])
        kernel=RankedLSARowEvidence(64,1+nodes/2,depths=range(9),store=store,src=src,
            u_lo=settings['row_bounds'][0],u_hi=settings['row_bounds'][1],step=settings['row_step'])
        sampler=SharedBaseMCMC(64,kernel,weights/2)
        handlers={s:signal.signal(s,stop) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
        started=time.monotonic()
        try:
            for j in range(len(report['records']),spec['repetitions']):
                print(json.dumps(dict(event='repetition_started',task=index,stage=stage,family=spec['family'],
                    index=spec['index'],observed=spec['observed'],arm=spec['arm'],replica=j,settings=settings)),flush=True)
                with heartbeat(path.with_suffix('.progress.json'),dict(task=index,stage=stage,completed_replicates=j)):
                    value=estimate(sampler,tokens,**{k:settings[k] for k in ('particles','temperatures','moves')},
                        seed=seed_for(plan,spec,j),resample_fraction=.75,seconds_limit=plan['stages'][stage]['seconds_per_replica'])
                validate_estimate(value,plan,spec,j);report['records'].append(value)
                report['records_sha256']=digest(report['records']);report['complete']=j+1==spec['repetitions']
                report['status']='complete' if report['complete'] else 'running'
                report['summary']=record_summary(report['records'],report['cheap_components'],oracle,scores)
                report['invocation_elapsed_seconds']=time.monotonic()-started
                atomic_json(path,report)
        except (TimeoutError,InterruptedError) as error:
            report.update(status='time_limit',error=str(error),invocation_elapsed_seconds=time.monotonic()-started)
            atomic_json(path,report);return 75
        finally:
            for s,h in handlers.items():signal.signal(s,h)
    return 0


def resource_projection(plan,reports,observed):
    """Independent 1024-CPU-hour budget per context, using all three families."""
    require(observed in TIMES,'Unknown context')
    selected=[r for r in reports.values() if r['task']['observed']==observed]
    keys={(r['task']['family'],r['task']['arm']) for r in selected if r['complete']}
    if keys!={(f,a) for f in FAMILIES for a in ('B','C','D')}:
        return dict(within_budget=False,reason='All three families and B/C/D kernel timings are required')
    timing={a:max(r['records'][0]['elapsed_seconds'] for r in selected if r['task']['arm']==a) for a in ('B','C','D')}
    population=sum(timing[a]*192*64 for a in ('B','C','D'))
    refinement=timing['B']*12*(64+32)+timing['C']*12*32+timing['D']*12*32
    reserve=(36*1800+36*2700)/4
    total=(2*(population+refinement)+reserve)*plan['resource_limits']['timing_safety_factor']/3600
    return dict(projected_total_cpu_hours=total,maximum_cpu_hours=1024,within_budget=total<=1024,
        measured_max_replica_seconds=timing,
        includes='All 192 histories for this context, B/C/D rules and 64 repetitions; all 12 development histories; two attempts and a factor-two runtime margin. Four fixed context budgets sum to 4096 CPU hours. Projection is not a guarantee.')


def exact_short_rows(plan,raw,dep):
    cp,_=calibration.read_plan();calibration.dependencies(cp,Path(dep)/'tables/anchors_prod',Path(dep)/'src')
    calibration.numerical_environment(Path(dep)/'tables/anchors_prod',Path(dep)/'src')
    shared=SharedBaseShortEvidence();fixed={f:FamilyEvidence(f) for f in ('product','uniform_base')}
    groups={}
    for t in (1,4):
        rows=[]
        for f in FAMILIES:
            for i in INDICES:
                spec=dict(family=f,index=i,observed=t,development=False)
                x,q,scores,hashes=load_case(raw,plan,spec)
                components=[component_prediction(fixed['product'],x),calibration.exact_reference(shared,x.tolist()),
                    component_prediction(fixed['uniform_base'],x)]
                for c in components:checked_component(c)
                mixture=combine_components([c['log_evidence'] for c in components],[c['predictive_probs'] for c in components])
                rows.append(dict(family=f,index=i,bayes_regret_bits=regret(q,mixture['predictive_probs']),
                    transformer_regret_bits=scores,mcse_bits=0.,bias_allowance_bits=0.))
        groups[t]=dict(rows=rows,summary=balanced_summary(rows),reference='Existing validated exact short catalog and row-evidence helpers')
    return groups


def cohort_refinement(summaries):
    """Reject a coherent integration shift even when every case passes alone."""
    require(set(summaries)=={'B','C','D'},'All three complete cohort rules are required')
    pairs={}
    for arm in ('C','D'):
        a,b=summaries['B'],summaries[arm]
        difference=abs(a['bayes']['mean']-b['bayes']['mean'])
        stochastic=3*math.hypot(a['numerical_mcse_bits'],b['numerical_mcse_bits'])+a['bias_allowance_bits']+b['bias_allowance_bits']
        pairs['B_'+arm]=dict(absolute_difference_bits=difference,stochastic_agreement_limit_bits=stochastic,
            resolution_limit_bits=.01/3,passed=difference<=.01/3 and difference<=stochastic)
    return dict(passed=all(s['precision_passed'] for s in summaries.values()) and all(c['passed'] for c in pairs.values()),pairs=pairs)


def merge(plan_path,stage,raw,dep,out,calibration_root):
    plan,h=read_plan(plan_path);out=Path(out)
    reports,missing=load_stage(plan,h,out,stage)
    cp,_=calibration.read_plan(calibration.PLAN)
    dependency_hashes=calibration.dependencies(cp,Path(dep)/'tables/anchors_prod',Path(dep)/'src')
    environment=calibration.numerical_environment(Path(dep)/'tables/anchors_prod',Path(dep)/'src')
    for report in reports.values():
        x,q,scores,hashes=load_case(raw,plan,report['task'])
        require(report['identity']['source_sha256']==hashes and report['tokens']==x.tolist()
            and report['oracle']==q.tolist() and report['transformer_regret_bits']==scores
            and report['identity']['dependency_sha256']==dependency_hashes
            and report['identity']['numerical_environment']==environment
            and report['identity']['code_sha256']==plan['code_sha256'], 'Merge source or numerical provenance changed')
    complete=not missing;result=dict(schema='pooled-predictive-population-stage-v1',stage=stage,plan_sha256=h,
        driver_sha256=sha(__file__),complete=complete,missing_or_unfinished_tasks=missing,
        by_context={str(t):dict(complete=False,passed=False,withheld_reason='Incomplete tasks',task_sha256={}) for t in TIMES},
        task_sha256={str(stage_path(out,stage,i).relative_to(out)):sha(stage_path(out,stage,i)) for i in reports},
        scientific_reference_validated=False,population_reference_validated=False,passed=False)
    for t in TIMES:
        expected=[q['task_index'] for q in task_specs(plan,stage) if q['observed']==t]
        context=result['by_context'][str(t)]
        context['complete']=all(i in reports and reports[i]['complete'] for i in expected)
        context['task_sha256']={str(stage_path(out,stage,i).relative_to(out)):sha(stage_path(out,stage,i)) for i in expected if i in reports}
    if stage=='timing':
        result.update(passed=complete,timings=[dict(task=r['task'],complete=r['complete'],status=r['status'],
            completed_seconds=[x['elapsed_seconds'] for x in r['records']],elapsed_seconds=r.get('invocation_elapsed_seconds')) for r in reports.values()],
            scope='Performance only. Timing has no numerical-accuracy approval and cannot unlock population sampling.')
        for c in result['by_context'].values():
            c['passed']=c['complete'];c['withheld_reason']=None if c['passed'] else 'Incomplete timing wave'
    elif stage=='refinement_timing':
        for t in TIMES:
            c=result['by_context'][str(t)];projection=resource_projection(plan,reports,t)
            c.update(resource_projection=projection,passed=c['complete'] and projection['within_budget'])
            c['withheld_reason']=None if c['passed'] else 'Incomplete timing or projected cost exceeds the fixed context budget'
        result['passed']=complete and all(c['passed'] for c in result['by_context'].values())
    else:
        result['calibration']=calibration_decision(calibration_root,require_pass=True)
        for t in TIMES:
            if any(r['task']['observed']==t for r in reports.values()):
                scientific_gate(plan,h,out,stage,calibration_root,t)
        grouped={}
        for r in reports.values():
            s=r['task'];grouped.setdefault((s['observed'],s['family'],s['index']),{})[s['arm']]=r
        checks=[];published={}
        for key,arms in grouped.items():
            wanted=set(VARIANTS) if stage=='refine' else {'B','C','D'}
            if set(arms)!=wanted or not all(r['complete'] for r in arms.values()):continue
            pairs=[('A','B'),('B','C'),('B','D')] if stage=='refine' else [('B','C'),('B','D')]
            pairchecks={a+'_'+b:refinement_pair(arms[a]['records'],arms[b]['records']) for a,b in pairs}
            cheapchecks={}
            for a,b in pairs:
                for f in ('product','uniform_base'):
                    ca,cb=arms[a]['cheap_components'][f],arms[b]['cheap_components'][f]
                    error=max(abs(ca['log_evidence']-cb['log_evidence'])/np.log(2),
                        float(np.max(np.abs(np.log2(ca['predictive_probs'])-np.log2(cb['predictive_probs'])))))
                    cheapchecks[a+'_'+b+'_'+f]=dict(maximum_log_error_bits=error,passed=error<=.01)
            riskchecks={}
            for a,b in pairs:
                sa,sb=arms[a]['summary'],arms[b]['summary'];ua,ub=sa['mixture_uncertainty'],sb['mixture_uncertainty']
                delta=abs(sa['bayes_regret_bits']-sb['bayes_regret_bits'])
                allowance=delta+3*math.hypot(ua['mcse_bits'],ub['mcse_bits'])+ua['bias_allowance_bits']+ub['bias_allowance_bits']
                riskchecks[a+'_'+b]=dict(allowance_bits=allowance,passed=allowance<=.01)
            ok=all(r['summary']['basic_passed'] for r in arms.values()) and all(c['passed'] for c in pairchecks.values()) and all(c['passed'] for c in riskchecks.values()) and all(c['passed'] for c in cheapchecks.values())
            checks.append(dict(observed=key[0],family=key[1],index=key[2],passed=ok,component_refinement=pairchecks,mixture_refinement=riskchecks,cheap_component_refinement=cheapchecks))
        if stage=='refine':
            allpass=complete and len(checks)==48 and all(c['passed'] for c in checks)
            result.update(all_component_and_mixture_refinements_passed=allpass,passed=allpass,checks=checks)
            for t in TIMES:
                c=result['by_context'][str(t)];subset=[r for r in checks if r['observed']==t]
                ok=c['complete'] and len(subset)==12 and all(r['passed'] for r in subset)
                c.update(passed=ok,all_component_and_mixture_refinements_passed=ok,
                    withheld_reason=None if ok else 'All 12 histories and independent component/mixture refinements must pass')
        else:
            for t in TIMES:
                selected=[c for c in checks if c['observed']==t]
                if len(selected)!=192 or not all(c['passed'] for c in selected):continue
                rules={}
                for arm in ('B','C','D'):
                    rows=[]
                    for f in FAMILIES:
                        for i in INDICES:
                            r=grouped[t,f,i][arm];s=r['summary'];u=s['mixture_uncertainty']
                            rows.append(dict(family=f,index=i,bayes_regret_bits=s['bayes_regret_bits'],transformer_regret_bits=s['transformer_regret_bits'],
                                mcse_bits=u['mcse_bits'],bias_allowance_bits=u['bias_allowance_bits']))
                    rules[arm]=dict(rows=rows,summary=balanced_summary(rows))
                aggregate=cohort_refinement({arm:rule['summary'] for arm,rule in rules.items()})
                refinement=aggregate['pairs'];approved=aggregate['passed']
                if approved:
                    published[t]=dict(**rules['B'],refinement_rules={a:v['summary'] for a,v in rules.items()},
                        cohort_refinement=refinement,publication_arm='B',numerically_validated=True)
                result['by_context'][str(t)].update(passed=approved,withheld_reason=None if approved else 'Cohort numerical budget or aggregate refinement failed')
            result.update(passed=complete and len(published)==4,groups=published,
                withheld_times=[t for t in TIMES if t not in published],checks=checks)
            if published:
                result['exact_short']=exact_short_rows(plan,raw,dep)
                result['population_reference_validated']=complete and len(published)==4
                result['validated_contexts']=sorted(published)
                result['scope']='Only complete numerically accepted contexts; same 192 histories for Bayes and both transformers. Numerical diagnostics do not prove posterior convergence.'
    atomic_json(out/stage/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('stage','complete','passed','scientific_reference_validated','population_reference_validated')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','merge','gate'))
    p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--task-index',type=int);p.add_argument('--observed',type=int,choices=TIMES,action='append');p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);a=p.parse_args()
    if a.command=='gate':
        plan,h=read_plan(a.plan)
        approval={str(t):scientific_gate(plan,h,a.out,a.stage,a.calibration,t) for t in (a.observed or TIMES)}
        atomic_json(a.out/a.stage/'gate.json',dict(plan_sha256=h,driver_sha256=sha(__file__),stage=a.stage,
            approved_to_execute=True,gate=approval,scientific_reference_validated=False))
        print(json.dumps(dict(stage=a.stage,approved_to_execute=True,population_auto_submitted=False)),flush=True)
        return 0
    if a.command=='run':
        if a.task_index is None:p.error('run needs --task-index')
        return run(a.plan,a.stage,a.task_index,a.raw,a.dependencies,a.out,a.calibration)
    if a.task_index is not None:p.error('merge uses all planned tasks')
    r=merge(a.plan,a.stage,a.raw,a.dependencies,a.out,a.calibration)
    return 0 if all(r['by_context'][str(t)]['passed'] for t in (a.observed or TIMES)) else 3


if __name__=='__main__':raise SystemExit(main())

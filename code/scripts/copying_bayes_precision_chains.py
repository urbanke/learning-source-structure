"""Fixed copying precision continuation, preserving old outputs and gates.

Later chains retain every draw and RNG state. Earlier chains are fresh Linux
runs of the same prior. Draw targets and resource bounds are frozen before
sampling; no adaptive retries or incomplete-cohort publication is permitted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import copying_bayes_dense16_recovery as r

b=r.base
SCHEMA='copying-dense16-precision-v1'


def codes():return {str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__),**b.codes(),**r.codes()}


def prepare(out,allocation,parent,final,banks):
    b.require(not (out/'plan.json').exists(),'No repeated preparation')
    a=b.read(allocation);old=b.load(parent);latest=b.read(final/'plan.json')
    b.require(a['schema']=='copying-precision-fixed-draw-allocation-v1' and a['maximum_cost_chf']<84,'Allocation exceeds bound')
    tasks=[];retained={}
    for i,row in enumerate(a['tasks']):
        task=row['task'];files={};counts=[0]*4
        if not row.get('early_fresh'):
            report_path=Path(row['report_path']);b.require(b.sha(report_path)==row['report_sha256'],'Report changed')
            report=b.read(report_path);folder=report_path.parents[2];paths=b.accuracy.paths(folder,task)
            b.require(report['complete'] and report['audit_passed'] and report['moments_sha256']==row['moments_sha256'],'Parent reference changed')
            files={'report':dict(path=str(report_path),sha256=row['report_sha256'])}
            for k in ('identity','audit','moments','checkpoint'):files[k]=dict(path=str(paths[k]),sha256=report[k+'_sha256'])
            counts=[row['draws']]*4
        tasks.append(dict(index=i,task=task,target_draws=row['target_draws'],parent_files=files,parent_draw_counts=counts,
          early_fresh=bool(row.get('early_fresh')),reserved_seconds=row['reserved_seconds'],cpus=row['cpus']))
        if i%1000==0:print(json.dumps(dict(event='freeze_reference',completed=i,total=len(a['tasks']))),flush=True)
    for key in a['retained']:
        matches=[x for x in latest['tasks'] if x['task']['id']==key]
        if matches:
            path=b.accuracy.paths(final,matches[0]['task'])['report'];retained[key]=dict(path=str(path),sha256=b.sha(path))
        else:retained[key]=latest['retained_report_bindings'][key]
    b.require(len({t['task']['id'] for t in tasks})==len(tasks) and len({t['task']['seed'] for t in tasks})==len(tasks),'Duplicate task or random seed')
    plan=dict(schema=SCHEMA,parent=str(parent),final=str(final),banks=str(banks),runtime=old['ready']['runtime'],
      parent_plan=r.bind(parent/'plan.json'),final_plan=r.bind(final/'plan.json'),allocation=r.bind(allocation),
      code_sha256=codes(),tasks=tasks,retained=retained,shards=a['shards'],forecast=a['forecast'],
      resources=dict(maximum_cost_chf=a['maximum_cost_chf'],automatic_retries=False),
      early_runtime='Fresh independent Linux chains; no cross-platform checkpoint conversion',
      original_histories=768,times=list(range(1,17)),population_reference_approved=False)
    plan['plan_id']=b.fp(plan);b.write(out/'plan.json',plan,immutable=True)
    print(json.dumps(dict(event='prepared',tasks=len(tasks),shards=len(a['shards']),ceiling_chf=a['maximum_cost_chf'])),flush=True)


def load(out):
    p=b.read(out/'plan.json')
    b.require(p['schema']==SCHEMA and p['plan_id']==b.fp({k:v for k,v in p.items() if k!='plan_id'}),'Plan changed')
    b.require(p['code_sha256']==codes(),'Code changed')
    for k in ('parent_plan','final_plan','allocation'):r.verify(p[k])
    old=b.load(Path(p['parent']));b.require(old['ready']['runtime']==p['runtime'],'Runtime changed')
    return p,old


def worker(out,index):
    p,old=load(out);b.require(0<=index<len(p['tasks']),'Invalid task')
    # Reuse the frozen continuation, raw audit, grid checks and preservation audit.
    # Only this process's plan reader changes, never the sampler or acceptance rules.
    previous=r.load;r.load=lambda folder:(p,old)
    try:return r.worker(out,index)
    finally:r.load=previous


def shard(out,index):
    p,_=load(out);s=p['shards'][index];folder=out/'shards'/str(index);folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();rows=[]
    for i in s['tasks']:
        spec=p['tasks'][i];remaining=s['wall_seconds']-90-(time.monotonic()-start)
        if remaining<=60:break
        print(json.dumps(dict(event='reference_started',index=i,id=spec['task']['id'],target=spec['target_draws'])),flush=True)
        log=folder/f'task-{i}.log'
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(i)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=min(remaining,spec['reserved_seconds']))
            except subprocess.TimeoutExpired:b.frequency.stop(process);code=75
        rows.append(dict(index=i,returncode=code))
        b.write(folder/'status.json',dict(plan_id=p['plan_id'],assigned=s['tasks'],attempts=rows))
        print(json.dumps(dict(event='reference_complete',index=i,returncode=code)),flush=True)
    return 0 if len(rows)==len(s['tasks']) and all(x['returncode']==0 for x in rows) else 75


def checked_moments(m,estimate,seen,draws):
    p=m['mean'];cm=m['chain_means']
    b.require(draws in (16384,32768,65536,131072,262144,524288,1048576)
      and p.shape==(64,) and cm.shape==(4,64) and int(m['chains'])==4
      and int(m['draws'])==draws==estimate['diagnostics']['draws'],'Moment shape or budget differs')
    b.require(all(np.isfinite(v).all() for v in m.values()) and p.min()>0 and cm.min()>0
      and abs(p.sum()-1)<1e-7 and np.max(abs(cm.sum(1)-1))<1e-7,'Invalid probabilities')
    b.require(np.allclose(p,estimate['predictive_probs'],rtol=0,atol=2e-14)
      and np.allclose(p,cm.mean(0),rtol=0,atol=2e-14),'Readback differs')
    b.require(np.max(abs(p[seen:]-p[seen]))<2e-14 and np.max(abs(cm[:,seen:]-cm[:,seen,None]))<2e-14,'Unequal unseen probability')
    for k in ('cov_batch','cov_chain'):
        c=m[k];b.require(c.shape==(64,64) and np.max(abs(c-c.T))<1e-14 and np.linalg.eigvalsh(c).min()>-1e-12,'Invalid covariance')
        b.require(np.max(abs(c[:,seen:]-c[:,seen,None]))<2e-14,'Unequal unseen covariance')
    expected=np.sqrt(np.maximum(0,np.maximum(np.diag(m['cov_batch']),np.diag(m['cov_chain']))))
    b.require(np.allclose(expected,m['mcse_probs'],rtol=1e-10,atol=2e-14),'MCSE differs')


def merge(out):
    p,_=load(out);refs={16:{},32:{}};missing=[];failed=[];bindings={}
    entries=[(spec,b.accuracy.paths(out,spec['task'])['report']) for spec in p['tasks']]
    for key,bind in p['retained'].items():r.verify(bind);entries.append((None,Path(bind['path'])))
    for spec,path in entries:
        if not path.exists():missing.append(spec['task']['id']);continue
        report=b.read(path);task=report['task'];task=task.get('task',task)
        folder=path.parents[2];paths=b.accuracy.paths(folder,task);ident=b.read(paths['identity']);audit=b.read(paths['audit'])
        b.require(report['complete'] and ident['runtime']==p['runtime'],'Incomplete or different runtime')
        if spec:b.require(report['task']==ident['task']==spec and report['plan_id']==ident['plan_id']==p['plan_id'],'Reference identity differs')
        for k in ('identity','audit','moments','checkpoint'):
            b.require(b.sha(paths[k])==report[k+'_sha256'],'Artifact changed');bindings[str(paths[k])]=report[k+'_sha256']
        bindings[str(path)]=b.sha(path)
        passed=bool(audit['passed'] and all(g['passed'] for g in audit['final_state_grid_checks']))
        b.require(report['audit_passed']==passed,'Audit flag differs')
        if not passed:failed.append(task['id'])
        m=b.archive(paths['moments']);checked_moments(m,report['estimate'],len(set(task['tokens'])),int(m['draws']))
        if spec and spec['parent_files']:r.preserve(spec['parent_files']['checkpoint']['path'],paths['checkpoint'],spec['target_draws'])
        refs[task['order']][task['reference_key']]=m
    result=dict(schema=SCHEMA,plan_id=p['plan_id'],complete=not missing,missing=missing,failed_audits=failed,
      aggregate_precision_passed=False,population_reference_approved=False,publication_status='diagnostic_nonpublication')
    if missing or failed:b.write(out/'summary.json',result);return 3
    early={o:{k:v for k,v in refs[o].items() if int(k[1:5])<=8} for o in refs}
    late={o:{k:v for k,v in refs[o].items() if int(k[1:5])>=9} for o in refs}
    b.require(len(early[16])==len(early[32])==526 and len(late[16])==len(late[32])==5086,'Incomplete canonical cohort')
    pair_failed=r.paired_failures(late)
    for k,x in early[16].items():
        y=early[32][k]
        if not b.accuracy.paired_order_check(x['mean'],x['mcse_probs'],y['mean'],y['mcse_probs'])['passed']:pair_failed.append(k)
    result['failed_paired_orders']=pair_failed
    if pair_failed:b.write(out/'summary.json',result);return 3
    bankfolder=Path(p['banks']);br=b.read(bankfolder/'frequency_summary.json')
    b.require(br['complete'] and br['banks']==256 and br['retained_bank_parity_passed']
      and b.sha(bankfolder/'frequency_arrays.npz')==br['arrays_sha256'],'Frequency refinement incomplete')
    arrays=b.archive(bankfolder/'frequency_arrays.npz');data=b.archive(Path(p['parent'])/'input.npz');groups=[]
    for g in b.CONDITIONS:
        losses={};mcse={};bias={}
        for order in (16,32):
            loss=np.empty((256,16));loss[:,:4]=data[g+('_bayes_loss' if order==32 else '_bayes_loss_a16')][:,:4]
            mcse[order]=np.zeros(16);bias[order]=np.zeros(16)
            for n in range(5,17):
                oracles=[];keys=[]
                for i,row in enumerate(data[g+'_tokens']):
                    pattern,labels=b.canonicalize(row[:n]);key=b.reference_key(pattern)
                    oracle=data[g+'_rows'][i,row[n-1],b.alphabet_map(labels)];m=refs[order][key]
                    loss[i,n-1]=-oracle@np.log2(m['mean']);oracles.append(oracle);keys.append(key)
                err=b.shared_error(np.asarray(oracles),keys,refs[order]);mcse[order][n-1]=err['numerical_mcse_bits'];bias[order][n-1]=err['log_bias_proxy_bits']
            losses[order]=loss;arrays[g+f'_bayes_loss_a{order}']=loss
        f,c,fl,cl=[arrays[g+'_'+k] for k in ('stationary_loss','stationary_coarse_loss','stationary_leave_one_bank_out_loss','stationary_coarse_leave_one_bank_out_loss')]
        checks={scope:b.allowance(f,c,fl,cl,losses[32],losses[16],mcse[32],mcse[16],bias[32],bias[16],cum)
          for scope,cum in (('instantaneous',False),('cumulative',True))}
        groups.append(dict(condition=g,**checks))
        for role in ('ordered','shuffled'):
            for seed in (0,1):arrays[g+f'_{role}_seed{seed}']=data[g+f'_{role}_seed{seed}']
    approved=all(g[k]['passed'] for g in groups for k in ('instantaneous','cumulative'))
    b.save(out/'arrays.npz',**arrays);b.write(out/'input_bindings.json',bindings,immutable=True)
    result.update(groups=groups,aggregate_precision_passed=approved,population_reference_approved=approved,
      publication_status='approved' if approved else 'diagnostic_nonpublication',arrays_sha256=b.sha(out/'arrays.npz'),
      input_bindings_sha256=b.sha(out/'input_bindings.json'),original_histories=768,times=list(range(1,17)),
      comparator='Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.',payoff_forecast_approved=False)
    b.write(out/'summary.json',result)
    return 0 if approved else 3


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','worker','shard','merge'))
    a.add_argument('--out',type=Path,required=True);a.add_argument('--index',type=int)
    for k in ('allocation','parent','final','banks'):a.add_argument('--'+k,type=Path)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.allocation,v.parent,v.final,v.banks);return 0
    return globals()[v.stage](v.out,v.index) if v.stage in ('worker','shard') else merge(v.out)
if __name__=='__main__':raise SystemExit(main())

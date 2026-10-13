"""Portable bounded extension of the approved copying comparison through t=16.

The original 768 histories, posterior, 16/32 rules and numerical error budget
are unchanged. Each task belongs to exactly one shard, with one attempt.
The stationary comparator remains a source-induced stationary-law surrogate,
not exact finite-shuffle Bayes. No incomplete-cohort mean is published.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import NormalDist
import numpy as np
import copying_payoff_accuracy as accuracy
import copying_frequency_paired_refinement as frequency
from prepare_copying_payoff_references import inventory_tokens, canonicalize, reference_key, alphabet_map
from merge_copying_payoff_dense import moments_checked, shared_error

ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=('a1_L0','a1_L2','a1_L3')
COUNTS=(374,490,588,654,714,743,758,765)
SCHEMA='copying-bayes-dense16-v1'
SAMPLE=dict(chains=4,burn=16384,thin=1,draws_per_chain=16384,surrogate_steps=64)
DEP='code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json'
require,read,write,sha,fp=accuracy.require,accuracy.read,accuracy.write,accuracy.sha,accuracy.fingerprint


def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z}


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)


def codes():
    names=('code/scripts/copying_bayes_dense16.py','code/scripts/copying_frequency_paired_refinement.py',
      'code/scripts/merge_copying_payoff_dense.py','code/scripts/copying_payoff_dense.py',
      'code/transformer/stationary_frequency_moments.py','code/transformer/scitas/copying_bayes_dense16.sbatch')
    return dict(accuracy.code_hashes(),**{name:sha(ROOT/name) for name in names})


def tasks_for(data):
    refs,uses=inventory_tokens({g:data[g+'_tokens'] for g in CONDITIONS},range(9,17))
    require(tuple(sum(r['time']==n for r in refs.values()) for n in range(9,17))==COUNTS,'Canonical coverage differs')
    tasks=[]
    for key,r in sorted(refs.items()):
        for order in (16,32):
            task=dict(task_index=len(tasks),id=key+f'_a{order}',reference_key=key,tokens=r['tokens'],
                      time=r['time'],order=order,target_draws=16384)
            task['seed']=int(fp(['copying-dense16-linux',20260922,task])[:32],16)
            tasks.append(task)
    require(len(tasks)==10172 and len({t['seed'] for t in tasks})==len(tasks),'Task count or seed collision')
    require(len(uses)==768*8,'Incomplete original cohort')
    return tasks


def bundle(out):
    """Transport an approved release, not old runtime-dependent sampler state."""
    from assess_copying_payoff import check_output
    old=ROOT/'output/copying_payoff_alpha_refined_assessment_2026-09-21'
    require(check_output(old)==0,'Old release is not approved')
    validation=read(old/'validation.json');data=archive(old/'arrays.npz')
    bindings={str((old/name).relative_to(ROOT)):sha(old/name) for name in ('validation.json','summary.json','arrays.npz')}
    for path,digest in validation['direct_input_sha256'].items():
        p=Path(path);require(p.is_file() and sha(p)==digest,'Old direct input changed: '+str(p))
    old_seeds=set()
    for n in range(5,9):
        path=ROOT/f'output/copying_payoff_alpha_refined_diagnostics_2026-09-21/t{n}/summary.json'
        require(validation['direct_input_sha256'].get(str(path))==sha(path),'Unbound early numerical errors')
        report=read(path);bindings[str(path.relative_to(ROOT))]=sha(path)
        for g in report['diagnostic_groups']:
            for order in (16,32):
                for metric in ('numerical_mcse_bits','log_bias_proxy_bits'):
                    key=g['condition']+f'_old_{order}_'+metric
                    if key not in data:data[key]=np.zeros(8)
                    data[key][n-1]=g['rules'][str(order)][metric]
        for p,digest in report['input_sha256'].items():
            if p.endswith('/identity.json'):
                require(sha(p)==digest,'Old identity changed')
                ident=read(p)
                if 'task' in ident:old_seeds.add(ident['task']['seed'])
    base=ROOT/'output/shared_base_shuffle_download_2026-09-18'
    for g in CONDITIONS:
        path=base/'cohorts'/f'{g}.npz';z=archive(path)
        require(z['tokens'].shape==(256,1025) and z['transition'].shape==(256,64,64),'Incomplete original source')
        data[g+'_tokens']=z['tokens'][:,:17];data[g+'_rows']=z['transition']
        bindings[str(path.relative_to(ROOT))]=sha(path)
        for role in ('ordered','shuffled'):
            for seed in (0,1):
                path=base/'eval'/f'{role}_seed{seed}'/f'{g}.npz';x=archive(path)['expected_loss_bits'][:,:16]
                require(np.array_equal(x[:,:8],data[g+f'_{role}_seed{seed}']),'Old neural score identity differs')
                data[g+f'_{role}_seed{seed}']=x;bindings[str(path.relative_to(ROOT))]=sha(path)
    tasks=tasks_for(data);require(not old_seeds.intersection(t['seed'] for t in tasks),'Cross-time seed collision')
    require(not out.exists(),'Bundle already exists');out.mkdir(parents=True)
    save(out/'input.npz',**data)
    provenance=dict(approved_release=bindings,old_validation=validation,old_seeds=sorted(old_seeds),
                    deployment='Fresh Linux draws; old approved t1..8 arrays retained. No claimed runtime parity.')
    write(out/'input_receipt.json',provenance,immutable=True)
    plan=dict(schema=SCHEMA,times=list(range(9,17)),sampler=SAMPLE,original_histories=768,
              shards=256,task_seconds=900,shard_seconds=28680,orders=[16,32],automatic_retries=False,
              input_sha256=sha(out/'input.npz'),receipt_sha256=sha(out/'input_receipt.json'),
              dependency_manifest_sha256=sha(ROOT/DEP),code_sha256=codes(),
              resources=dict(maximum_cpu_hours=2060,rate_chf=.0055,maximum_cost_chf=11.33),
              prior=dict(alphabet=64,alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation',initial_token='q'))
    plan['plan_id']=fp(plan);write(out/'plan.json',plan,immutable=True)
    print(json.dumps(dict(bundle=str(out),plan_id=plan['plan_id'],tasks=len(tasks))))


def load(out,prepared=True):
    p=read(out/'plan.json');require(p['plan_id']==fp({k:v for k,v in p.items() if k!='plan_id'}),'Plan changed')
    require(p['schema']==SCHEMA and p['sampler']==SAMPLE and p['code_sha256']==codes(),'Code or sampler changed')
    require(p['input_sha256']==sha(out/'input.npz') and p['receipt_sha256']==sha(out/'input_receipt.json'),'Input changed')
    require(p['dependency_manifest_sha256']==sha(ROOT/DEP),'Dependency manifest changed')
    if prepared:
        ready=read(out/'prepared.json');require(ready['plan_id']==p['plan_id'],'Preparation differs')
        os.environ.update(ready['environment']);require(accuracy.pilot.runtime()==ready['runtime'],'Runtime changed')
        for name,stamp in ready['dependency_stat'].items():
            s=(Path(ready['dependencies'])/name).stat()
            require([s.st_size,s.st_mtime_ns]==stamp,'Sealed dependency changed: '+name)
        require(ready['tasks_sha256']==sha(out/'tasks.json') and ready['kinds_sha256']==sha(out/'kinds.json'),'Task assignment changed')
        p['ready']=ready
    return p


def prepare(out,dependencies):
    p=load(out,False);require(not (out/'prepared.json').exists(),'No repeated preparation')
    deps=read(ROOT/DEP);stamp={}
    for name,record in deps['files'].items():
        path=dependencies/name;require(path.is_file() and not path.is_symlink() and sha(path)==record['sha256'],'Dependency hash differs: '+name)
        s=path.stat();stamp[name]=[s.st_size,s.st_mtime_ns]
    environment={'LSA_STORE':str(dependencies/'tables/anchors_prod'),'PMM_SRC':str(dependencies/'src')}
    # Ban inherited numerical overrides; the legacy copying kernel used defaults.
    require(not any(k.startswith('PMM_') and k!='PMM_SRC' for k in os.environ),'Unexpected numerical environment override')
    os.environ.update(environment);data=archive(out/'input.npz');tasks=tasks_for(data)
    require(not set(read(out/'input_receipt.json')['old_seeds']).intersection(t['seed'] for t in tasks),'Old/new seeds collide')
    kinds=set()
    for g in CONDITIONS:
        for row in data[g+'_tokens']:
            for n in range(1,17):
                prefix=row[:n].tolist();kinds.add(frequency.frequency_type(prefix))
                for label in list(dict.fromkeys(prefix))+[next(x for x in range(64) if x not in prefix)]:
                    kinds.add(frequency.frequency_type(prefix+[label]))
    write(out/'tasks.json',tasks,immutable=True);write(out/'kinds.json',sorted(kinds),immutable=True)
    write(out/'prepared.json',dict(plan_id=p['plan_id'],runtime=accuracy.pilot.runtime(),environment=environment,
        dependencies=str(dependencies),dependency_stat=stamp,tasks_sha256=sha(out/'tasks.json'),
        kinds_sha256=sha(out/'kinds.json'),counts=COUNTS,task_count=len(tasks),frequency_types=len(kinds)),immutable=True)
    print(json.dumps(dict(event='prepared',tasks=len(tasks),frequency_types=len(kinds))),flush=True)


def worker(out,index):
    p=load(out);task=read(out/'tasks.json')[index];paths=accuracy.paths(out,task)
    attempt=read(paths['attempt']);require(attempt['dispatcher_pid']==os.getppid() and attempt['task']==task
        and attempt['plan_id']==p['plan_id'] and attempt['status']=='running','Invalid task dispatch')
    require(not paths['checkpoint'].exists() and not paths['report'].exists(),'No retry or overwrite')
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from validate_shared_base_mcmc import json_safe
    began=time.monotonic();nodes,weights=np.polynomial.legendre.leggauss(task['order'])
    sampler=SharedBaseSurrogateMCMC(64,RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384),weights/2,surrogate_steps=64)
    tokens=np.asarray(task['tokens'],int);run=CurveChainRun(sampler,tokens,chains=4,burn=16384,thin=1,seed=task['seed'])
    identity=dict(plan_id=p['plan_id'],task=task,target=run.fingerprint,proposal=run.proposal_descriptor,runtime=accuracy.pilot.runtime())
    write(paths['identity'],identity,immutable=True);saved=time.monotonic()
    def progress(event):
        nonlocal saved
        if event['phase']=='burn' or event['draws']==event['requested_draws'] or time.monotonic()-saved>=120:
            run.save(paths['checkpoint']);saved=time.monotonic()
        print(json.dumps(dict(task=task['id'],**event)),flush=True)
    estimate=run.extend(16384,block_draws=2048,progress=progress);run.save(paths['checkpoint']);del run
    moments,audit,fresh=accuracy._audit_raw(task,identity,estimate,paths['checkpoint']);save(paths['moments'],**moments)
    audit['final_state_grid_checks']=selected_grid_checks(sampler,tokens,estimate)
    passed=audit['passed'] and all(g['passed'] for g in audit['final_state_grid_checks'])
    write(paths['audit'],json_safe(audit),immutable=True)
    write(paths['report'],json_safe(dict(plan_id=p['plan_id'],task=task,complete=True,audit_passed=bool(passed),
        estimate=fresh,elapsed_seconds=time.monotonic()-began,
        **{k+'_sha256':sha(paths[k]) for k in ('audit','identity','checkpoint','moments')})),immutable=True)


def assigned(tasks,index,shards):
    require(type(index) is int and 0<=index<shards,'Invalid shard')
    return tasks[index::shards]


def shard(out,index):
    p=load(out);tasks=assigned(read(out/'tasks.json'),index,p['shards']);began=time.monotonic()
    directory=out/'shards'/str(index);directory.mkdir(parents=True,exist_ok=False)
    attempts=[]
    for task in tasks:
        remaining=p['shard_seconds']-(time.monotonic()-began)
        if remaining<60:break
        paths=accuracy.paths(out,task);paths['attempt'].parent.mkdir(parents=True,exist_ok=False)
        attempt=dict(plan_id=p['plan_id'],task=task,status='running',dispatcher_pid=os.getpid(),shard=index)
        write(paths['attempt'],attempt,immutable=True)
        with paths['log'].open('x') as stream:
            proc=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(task['task_index'])],
                                  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(p['task_seconds'],remaining))
            except subprocess.TimeoutExpired:frequency.stop(proc);code=75
        attempt.update(status='complete' if code==0 else 'failed',returncode=code)
        write(paths['attempt'],attempt);attempts.append(attempt)
        write(directory/'status.json',dict(plan_id=p['plan_id'],assigned=[t['id'] for t in tasks],attempts=attempts))
        print(json.dumps(dict(event='task_finished',task=task['id'],returncode=code,completed=len(attempts))),flush=True)
    complete=len(attempts)==len(tasks) and all(x['returncode']==0 for x in attempts)
    return 0 if complete else 75


def bank(out,index):
    p=load(out);require(0<=index<16,'Invalid bank');directory=out/'banks'/str(index);directory.mkdir(parents=True,exist_ok=False)
    kinds=[(x[0],tuple(x[1])) for x in read(out/'kinds.json')];began=time.monotonic()
    fine,coarse,nf,nc=frequency.generate_bank(index,kinds)
    save(directory/'moments.npz',fine=fine,coarse=coarse)
    write(directory/'report.json',dict(plan_id=p['plan_id'],index=index,complete=True,
        arrays_sha256=sha(directory/'moments.npz'),elapsed_seconds=time.monotonic()-began,latent_rows=[nf,nc]),immutable=True)
    print(json.dumps(dict(event='bank_complete',index=index,seconds=time.monotonic()-began)),flush=True)


def score_frequency(joints,data,kinds):
    index={kind:i for i,kind in enumerate(kinds)};total=joints.sum(0);full=total/16;leave=(total-joints)/15;result={}
    for g in CONDITIONS:
        loss=np.empty((256,16));loo=np.empty((16,256,16))
        for i,row in enumerate(data[g+'_tokens']):
            for n in range(1,17):
                prefix=row[:n].tolist();den=index[frequency.frequency_type(prefix)]
                cols=[index[frequency.frequency_type(prefix+[a])] for a in range(64)]
                prediction=full[cols]/full[den];predictions=leave[:,cols]/leave[:,den,None]
                require(abs(prediction.sum()-1)<1e-10 and np.max(abs(predictions.sum(1)-1))<1e-10,'Frequency normalization failed')
                oracle=data[g+'_rows'][i,row[n-1]];loss[i,n-1]=-oracle@np.log2(prediction);loo[:,i,n-1]=-np.log2(predictions)@oracle
        result[g]=(loss,loo)
    return result


def allowance(fine,coarse,fl,cl,b32,b16,m32,m16,bias32,bias16,cumulative=False):
    if cumulative:
        fine,coarse,fl,cl,b32,b16=[np.cumsum(a,axis=-1) for a in (fine,coarse,fl,cl,b32,b16)]
        m32,m16=[np.sqrt(np.cumsum(a*a)) for a in (m32,m16)];bias32,bias16=[np.cumsum(a) for a in (bias32,bias16)]
    bank=frequency._delete_metrics(fine,fl);comparison=frequency._delete_metrics(fine-coarse,fl-cl)
    gain=fine-b32;se=gain.std(0,ddof=1)/16
    total=2*np.hypot(m32,bank['mcse'])+2*(bias32+bank['bias'])+abs((b16-b32).mean(0))+3*np.hypot(m32,m16)+bias32+bias16+abs((fine-coarse).mean(0))+3*comparison['mcse']+comparison['bias']
    budget=np.minimum(.01/3,.25*se)
    return dict(passed=bool(np.all(total<=budget)),mean_bits=gain.mean(0).tolist(),source_se_bits=se.tolist(),
                numerical_allowance_bits=total.tolist(),numerical_budget_bits=budget.tolist())


def merge(out):
    p=load(out);tasks=read(out/'tasks.json');refs={16:{},32:{}};missing=[];failed=[];bindings={}
    for t in tasks:
        paths=accuracy.paths(out,t)
        if not paths['report'].exists():missing.append(t['id']);continue
        r=read(paths['report']);a=read(paths['audit']);ident=read(paths['identity'])
        require(r['complete'] and r['task']==t and r['plan_id']==p['plan_id'] and ident['task']==t
                and ident['plan_id']==p['plan_id'] and ident['runtime']==p['ready']['runtime'],'Reference identity differs')
        for key in ('audit','identity','checkpoint','moments'):
            require(sha(paths[key])==r[key+'_sha256'],'Reference artifact changed');bindings[str(paths[key])]=r[key+'_sha256']
        bindings[str(paths['report'])]=sha(paths['report'])
        require(r['audit_passed']==bool(a['passed'] and all(g['passed'] for g in a['final_state_grid_checks'])),'Audit flag differs')
        if not r['audit_passed']:failed.append(t['id'])
        m=archive(paths['moments']);moments_checked(m,r['estimate'],len(set(t['tokens'])));refs[t['order']][t['reference_key']]=m
    banks=[]
    for i in range(16):
        path=out/'banks'/str(i)
        if not (path/'report.json').exists():missing.append('bank-'+str(i));continue
        r=read(path/'report.json');require(r['complete'] and r['index']==i and r['plan_id']==p['plan_id']
            and sha(path/'moments.npz')==r['arrays_sha256'],'Bank binding differs');banks.append(archive(path/'moments.npz'))
        bindings[str(path/'report.json')]=sha(path/'report.json');bindings[str(path/'moments.npz')]=r['arrays_sha256']
    result=dict(schema=SCHEMA,plan_id=p['plan_id'],complete=not missing,missing=missing,failed_audits=failed,
                aggregate_precision_passed=False,population_reference_approved=False,publication_status='diagnostic_nonpublication')
    if missing or failed:write(out/'summary.json',result);return 3
    z=NormalDist().inv_cdf(1-.05/(2*64*(len(tasks)//2)));pairs=[]
    for key,left in refs[16].items():
        right=refs[32][key];error=abs(left['mean']-right['mean']);limit=z*np.hypot(left['mcse_probs'],right['mcse_probs'])+2e-7
        if np.any(error>limit):pairs.append(key)
    result['failed_paired_orders']=pairs;result['family_pairs']=len(tasks)//2
    if pairs:write(out/'summary.json',result);return 3
    data=archive(out/'input.npz');kinds=[(x[0],tuple(x[1])) for x in read(out/'kinds.json')]
    fine=score_frequency(np.stack([b['fine'] for b in banks]),data,kinds)
    coarse=score_frequency(np.stack([b['coarse'] for b in banks]),data,kinds);arrays={};groups=[]
    for g in CONDITIONS:
        f,fl=fine[g];c,cl=coarse[g]
        for new,old in ((f,'stationary_loss'),(fl,'stationary_leave_one_bank_out_loss'),
                        (c,'stationary_coarse_loss'),(cl,'stationary_coarse_leave_one_bank_out_loss')):
            require(np.allclose(new[...,:8],data[g+'_'+old],rtol=0,atol=5e-12),'Old stationary-bank replay failed')
            new[...,:8]=data[g+'_'+old]
        losses={};mcse={};bias={}
        for order in (16,32):
            loss=np.empty((256,16));loss[:,:8]=data[g+('_bayes_loss' if order==32 else '_bayes_loss_a16')]
            mcse[order]=np.zeros(16);bias[order]=np.zeros(16)
            mcse[order][:8]=data[g+f'_old_{order}_numerical_mcse_bits'];bias[order][:8]=data[g+f'_old_{order}_log_bias_proxy_bits']
            for n in range(9,17):
                oracles=[];keys=[]
                for i,row in enumerate(data[g+'_tokens']):
                    pattern,labels=canonicalize(row[:n]);key=reference_key(pattern)
                    oracle=data[g+'_rows'][i,row[n-1],alphabet_map(labels)];m=refs[order][key]
                    loss[i,n-1]=-oracle@np.log2(m['mean']);oracles.append(oracle);keys.append(key)
                err=shared_error(np.asarray(oracles),keys,refs[order]);mcse[order][n-1]=err['numerical_mcse_bits'];bias[order][n-1]=err['log_bias_proxy_bits']
            losses[order]=loss;arrays[g+f'_bayes_loss_a{order}']=loss
        checks={scope:allowance(f,c,fl,cl,losses[32],losses[16],mcse[32],mcse[16],bias[32],bias[16],cum)
                for scope,cum in (('instantaneous',False),('cumulative',True))}
        groups.append(dict(condition=g,**checks));arrays.update({g+'_stationary_loss':f,g+'_stationary_coarse_loss':c,
            g+'_stationary_leave_one_bank_out_loss':fl,g+'_stationary_coarse_leave_one_bank_out_loss':cl})
        for role in ('ordered','shuffled'):
            for seed in (0,1):arrays[g+f'_{role}_seed{seed}']=data[g+f'_{role}_seed{seed}']
    approved=all(x[k]['passed'] for x in groups for k in ('instantaneous','cumulative'))
    save(out/'arrays.npz',**arrays);write(out/'input_bindings.json',bindings,immutable=True)
    result.update(groups=groups,aggregate_precision_passed=approved,population_reference_approved=approved,
        publication_status='approved' if approved else 'diagnostic_nonpublication',arrays_sha256=sha(out/'arrays.npz'),
        input_bindings_sha256=sha(out/'input_bindings.json'),original_histories=768,times=list(range(1,17)),
        comparator='Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.',
        uncertainty='Shared canonical gradients; common-bank delete-one covariance; independent-time MCMC seeds.',
        payoff_forecast_approved=False,scope='Instantaneous and cumulative gains only; payoff crossing bootstrap not performed.')
    write(out/'summary.json',result);return 0 if approved else 3


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('bundle','prepare','shard','worker','bank','merge'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dependencies',type=Path);parser.add_argument('--index',type=int)
    a=parser.parse_args()
    if a.stage=='bundle':bundle(a.out)
    elif a.stage=='prepare':prepare(a.out,a.dependencies)
    elif a.stage=='shard':return shard(a.out,a.index)
    elif a.stage=='worker':worker(a.out,a.index)
    elif a.stage=='bank':bank(a.out,a.index)
    else:return merge(a.out)
    return 0
if __name__=='__main__':raise SystemExit(main())

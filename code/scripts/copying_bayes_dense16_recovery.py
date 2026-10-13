"""Complete only missing or numerically failed dense copying references.

The parent allocation is immutable. Continue its saved RNG/states and every
retained draw; keep all successful predictions and all sixteen common banks.
No selected-cohort mean, automatic retry or weaker numerical gate is allowed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from statistics import NormalDist
import numpy as np
import copying_bayes_dense16 as base
from unseen_context_refinement import checkpoint_metadata,checkpoint_counts,prefix_hash

ROOT=base.ROOT
require,read,write,sha,fp=base.require,base.read,base.write,base.sha,base.fp
SCHEMA='copying-dense16-recovery-v1'
WRAPPER='code/transformer/scitas/copying_bayes_dense16_recovery.sbatch'


def codes():
    return {str(Path(__file__).relative_to(ROOT)):sha(__file__),WRAPPER:sha(ROOT/WRAPPER)}


def bind(path):
    path=Path(path);return dict(path=str(path),sha256=sha(path))


def verify(item):
    require(sha(item['path'])==item['sha256'],'Frozen parent changed: '+item['path'])


def moments_checked(m,estimate,seen,draws):
    """The legacy moment audit, allowing only an explicitly bound larger draw count."""
    p=m['mean'];cm=m['chain_means']
    require(p.shape==(64,) and cm.shape==(4,64) and int(m['chains'])==4
            and int(m['draws'])==draws==estimate['diagnostics']['draws'],'Moment shape or retained budget differs')
    require(draws in (16384,65536,262144) and all(np.isfinite(v).all() for v in m.values())
            and p.min()>0 and cm.min()>0 and abs(p.sum()-1)<1e-7
            and np.max(abs(cm.sum(1)-1))<1e-7,'Invalid predictive moments')
    require(np.allclose(p,estimate['predictive_probs'],rtol=0,atol=2e-14)
            and np.allclose(p,cm.mean(0),rtol=0,atol=2e-14),'Predictive readbacks disagree')
    require(np.max(abs(p[seen:]-p[seen]))<2e-14 and np.max(abs(cm[:,seen:]-cm[:,seen,None]))<2e-14,'Unequal unseen probability')
    for name in ('cov_batch','cov_chain'):
        c=m[name];require(c.shape==(64,64) and np.max(abs(c-c.T))<1e-14 and np.linalg.eigvalsh(c).min()>-1e-12,'Invalid covariance')
        require(np.max(abs(c[:,seen:]-c[:,seen,None]))<2e-14,'Unequal unseen covariance')
    expected=np.sqrt(np.maximum(0,np.maximum(np.diag(m['cov_batch']),np.diag(m['cov_chain']))))
    require(np.allclose(expected,m['mcse_probs'],rtol=1e-10,atol=2e-14),'Moment MCSE differs')


def paired_failures(refs,total_pairs=5086):
    z=NormalDist().inv_cdf(1-.05/(2*64*total_pairs));failed=[]
    for key in sorted(set(refs[16]) & set(refs[32])):
        a,b=refs[16][key],refs[32][key]
        if np.any(abs(a['mean']-b['mean'])>z*np.hypot(a['mcse_probs'],b['mcse_probs'])+2e-7):failed.append(key)
    return failed


def parent_record(parent,plan,t):
    paths=base.accuracy.paths(parent,t);r=read(paths['report']);a=read(paths['audit']);i=read(paths['identity'])
    require(r['complete'] and r['task']==i['task']==t and r['plan_id']==i['plan_id']==plan['plan_id']
            and i['runtime']==plan['ready']['runtime'],'Parent record identity differs')
    for k in ('audit','identity','checkpoint','moments'):
        require(sha(paths[k])==r[k+'_sha256'],'Parent record hash differs: '+t['id'])
    require(r['audit_passed']==bool(a['passed'] and all(g['passed'] for g in a['final_state_grid_checks'])),'Parent gate flag differs')
    m=base.archive(paths['moments']);moments_checked(m,r['estimate'],len(set(t['tokens'])),16384)
    return r,m,{k:bind(paths[k]) for k in ('report','audit','identity','checkpoint','moments')}


def prepare(out,parent,terminal_receipt):
    p=base.load(parent);require(not (out/'plan.json').exists(),'Recovery already prepared')
    terminal=read(terminal_receipt)
    require(terminal['active_jobs']==[] and terminal['terminal_verified'] is True,'Original allocation not verified terminal')
    tasks=read(parent/'tasks.json');refs={16:{},32:{}};bindings={};reasons={};source={}
    for t in tasks:
        paths=base.accuracy.paths(parent,t)
        if paths['report'].exists():
            r,m,files=parent_record(parent,p,t);refs[t['order']][t['reference_key']]=m;source[t['id']]=files
            bindings[t['id']]=files['report']
            if not r['audit_passed']:reasons[t['id']]=['individual_audit']
        else:
            reasons[t['id']]=['missing']
            files={k:bind(paths[k]) for k in ('identity','checkpoint') if paths[k].exists()}
            require(not files or set(files)=={'identity','checkpoint'},'Partial checkpoint lacks identity')
            source[t['id']]=files
        if t['task_index']%500==0:print({'event':'inspect','done':t['task_index']},flush=True)
    pairs=paired_failures(refs)
    for key in pairs:
        for order in (16,32):reasons.setdefault(key+f'_a{order}',[]).append('paired_order')
    specs=[]
    for t in tasks:
        if t['id'] not in reasons:continue
        files=source[t['id']];checkpoint=files.get('checkpoint')
        counts=checkpoint_counts(checkpoint['path']) if checkpoint else [0]*4
        target=65536 if any(x!='missing' for x in reasons[t['id']]) else 16384
        require(max(counts)<=target,'Parent already exceeds continuation budget')
        specs.append(dict(index=len(specs),task=t,reasons=reasons[t['id']],target_draws=target,
                          parent_files=files,parent_draw_counts=counts))
    bank_files=[]
    for index in range(16):
        folder=parent/'banks'/str(index);r=read(folder/'report.json')
        require(r['complete'] and r['index']==index and r['plan_id']==p['plan_id'] and sha(folder/'moments.npz')==r['arrays_sha256'],'Parent bank differs')
        bank_files.append(dict(report=bind(folder/'report.json'),moments=bind(folder/'moments.npz')))
    max_cpu=2*len(specs)+4
    require(max_cpu*.0055<=30,'Recovery exceeds authorized CHF30 ceiling')
    plan=dict(schema=SCHEMA,parent=str(parent),parent_plan=bind(parent/'plan.json'),parent_prepared=bind(parent/'prepared.json'),
        terminal_receipt=bind(terminal_receipt),parent_code_sha256=p['code_sha256'],code_sha256=codes(),
        runtime=p['ready']['runtime'],tasks=specs,original_tasks=len(tasks),retained_report_bindings=bindings,
        bank_files=bank_files,paired_failures=pairs,resources=dict(task_seconds=6900,maximum_concurrent=128,
        allocated_cpus_per_task=1,memory_gib=6,maximum_cpu_hours=max_cpu,maximum_cost_chf=max_cpu*.0055),
        automatic_retries=False,acceptance='Unchanged individual,5086 paired-order,and complete-cohort gain error gates')
    plan['plan_id']=fp(plan);write(out/'plan.json',plan,immutable=True)
    print({'event':'prepared','tasks':len(specs),'paired_failures':len(pairs),'resources':plan['resources']},flush=True)


def load(out):
    p=read(out/'plan.json');require(p['schema']==SCHEMA and p['plan_id']==fp({k:v for k,v in p.items() if k!='plan_id'}),'Recovery plan changed')
    require(p['code_sha256']==codes(),'Recovery code changed');verify(p['parent_plan']);verify(p['parent_prepared']);verify(p['terminal_receipt'])
    old=base.load(Path(p['parent']));require(old['code_sha256']==p['parent_code_sha256'] and old['ready']['runtime']==p['runtime'],'Parent runtime or source changed')
    return p,old


def preserve(parent,child,target):
    a,b=checkpoint_metadata(parent),checkpoint_metadata(child)
    for k in ('format','fingerprint','config','initializations','proposal_descriptor'):require(a[k]==b[k],'Continuation changed '+k)
    counts=checkpoint_counts(parent);require(checkpoint_counts(child)==[target]*4,'Incomplete continued chains')
    for chain,n in enumerate(counts):
        for kind in ('predictions','depths','alpha_indices','scalars'):
            name=f'{kind}_{chain}';require(prefix_hash(child,name,n)==a['array_hashes'][name],'Retained samples changed')
    with np.load(parent,allow_pickle=False) as x,np.load(child,allow_pickle=False) as y:
        require(np.all(~x['burn_done'] | y['burn_done']),'Completed burn-in was lost')


def worker(out,index):
    p,old=load(out);spec=p['tasks'][index];t=spec['task'];paths=base.accuracy.paths(out,t)
    paths['checkpoint'].parent.mkdir(parents=True,exist_ok=False)
    for item in spec['parent_files'].values():verify(item)
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from validate_shared_base_mcmc import json_safe
    n,w=np.polynomial.legendre.leggauss(t['order']);sampler=SharedBaseSurrogateMCMC(64,
        RankedLSARowEvidence(64,1+n/2,term_cache_size=16384),w/2,surrogate_steps=64)
    tokens=np.asarray(t['tokens'],int);parent=spec['parent_files'].get('checkpoint');began=time.monotonic()
    if parent:
        shutil.copyfile(parent['path'],paths['checkpoint']);require(sha(paths['checkpoint'])==parent['sha256'],'Copied checkpoint differs')
        run=CurveChainRun.load(sampler,tokens,paths['checkpoint']);base.accuracy.verify_loaded_parent(run,checkpoint_metadata(parent['path']))
        original=read(spec['parent_files']['identity']['path']);require(run.fingerprint==original['target'] and run.proposal_descriptor==original['proposal'],'Posterior or proposal changed')
    else:run=CurveChainRun(sampler,tokens,chains=4,burn=16384,thin=1,seed=t['seed'])
    require(run.seed==t['seed'] and run.draw_counts.tolist()==spec['parent_draw_counts'],'Wrong seed or retained draws')
    identity=dict(plan_id=p['plan_id'],task=spec,target=run.fingerprint,proposal=run.proposal_descriptor,runtime=p['runtime'])
    write(paths['identity'],identity,immutable=True);saved=time.monotonic()
    def progress(event):
        nonlocal saved
        if event['phase']=='burn' or event['draws']==event['requested_draws'] or time.monotonic()-saved>=120:
            run.save(paths['checkpoint']);saved=time.monotonic()
        print(dict(task=t['id'],**event),flush=True)
    estimate=run.extend(spec['target_draws'],block_draws=2048,progress=progress);run.save(paths['checkpoint']);del run
    if parent:preserve(parent['path'],paths['checkpoint'],spec['target_draws'])
    m,a,fresh=base.accuracy._audit_raw(t,identity,estimate,paths['checkpoint']);base.save(paths['moments'],**m)
    a['final_state_grid_checks']=selected_grid_checks(sampler,tokens,estimate);passed=a['passed'] and all(g['passed'] for g in a['final_state_grid_checks'])
    write(paths['audit'],json_safe(a),immutable=True)
    write(paths['report'],json_safe(dict(plan_id=p['plan_id'],task=spec,complete=True,audit_passed=bool(passed),estimate=fresh,
        elapsed_seconds=time.monotonic()-began,parent_draws_preserved=bool(parent),
        **{k+'_sha256':sha(paths[k]) for k in ('audit','identity','checkpoint','moments')})),immutable=True)
    for item in spec['parent_files'].values():verify(item)
    return 0 if passed else 3


def dispatch(out,index):
    p,_=load(out);require(0<=index<len(p['tasks']),'Invalid recovery index')
    directory=out/'attempts';directory.mkdir(exist_ok=True)
    with (directory/f'{index}.lock').open('x') as stream:
        proc=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(index)],start_new_session=True)
        try:code=proc.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:base.frequency.stop(proc);code=75
    write(directory/f'{index}.json',dict(plan_id=p['plan_id'],index=index,returncode=code),immutable=True)
    return code


def merge(out):
    p,old=load(out);parent=Path(p['parent']);specs={x['task']['id']:x for x in p['tasks']};tasks=read(parent/'tasks.json')
    refs={16:{},32:{}};missing=[];failed=[];bindings={}
    for t in tasks:
        if t['id'] in specs:
            spec=specs[t['id']];paths=base.accuracy.paths(out,t)
            if not paths['report'].exists():missing.append(t['id']);continue
            r=read(paths['report']);a=read(paths['audit']);ident=read(paths['identity'])
            require(r['task']==ident['task']==spec and r['plan_id']==ident['plan_id']==p['plan_id'] and r['complete']
                and ident['runtime']==p['runtime'],'Recovery record identity differs')
            for k in ('checkpoint','moments','identity','audit'):
                require(sha(paths[k])==r[k+'_sha256'],'Recovery artifact changed');bindings[str(paths[k])]=r[k+'_sha256']
            require(r['audit_passed']==bool(a['passed'] and all(g['passed'] for g in a['final_state_grid_checks'])),'Recovery gate flag differs')
            m=base.archive(paths['moments']);moments_checked(m,r['estimate'],len(set(t['tokens'])),spec['target_draws'])
            if spec['parent_files'].get('checkpoint'):preserve(spec['parent_files']['checkpoint']['path'],paths['checkpoint'],spec['target_draws'])
            bindings[str(paths['report'])]=sha(paths['report'])
        else:
            verify(p['retained_report_bindings'][t['id']]);r,m,files=parent_record(parent,old,t)
            bindings.update({i['path']:i['sha256'] for i in files.values()})
        if not r['audit_passed']:failed.append(t['id'])
        refs[t['order']][t['reference_key']]=m
    result=dict(schema=SCHEMA,plan_id=p['plan_id'],complete=not missing,missing=missing,failed_audits=failed,
        aggregate_precision_passed=False,population_reference_approved=False,publication_status='diagnostic_nonpublication')
    if missing or failed:write(out/'summary.json',result);return 3
    pairs=paired_failures(refs);result['failed_paired_orders']=pairs
    if pairs:write(out/'summary.json',result);return 3
    banks=[]
    for b in p['bank_files']:
        for i in b.values():verify(i);bindings[i['path']]=i['sha256']
        banks.append(base.archive(b['moments']['path']))
    data=base.archive(parent/'input.npz');kinds=[(x[0],tuple(x[1])) for x in read(parent/'kinds.json')]
    fine=base.score_frequency(np.stack([b['fine'] for b in banks]),data,kinds)
    coarse=base.score_frequency(np.stack([b['coarse'] for b in banks]),data,kinds);arrays={};groups=[]
    for g in base.CONDITIONS:
        f,fl=fine[g];c,cl=coarse[g]
        for new,name in ((f,'stationary_loss'),(fl,'stationary_leave_one_bank_out_loss'),(c,'stationary_coarse_loss'),(cl,'stationary_coarse_leave_one_bank_out_loss')):
            require(np.allclose(new[...,:8],data[g+'_'+name],rtol=0,atol=5e-12),'Old common-bank replay differs');new[...,:8]=data[g+'_'+name]
        losses={};mcse={};bias={}
        for order in (16,32):
            loss=np.empty((256,16));loss[:,:8]=data[g+('_bayes_loss' if order==32 else '_bayes_loss_a16')]
            mcse[order]=np.zeros(16);bias[order]=np.zeros(16)
            mcse[order][:8]=data[g+f'_old_{order}_numerical_mcse_bits'];bias[order][:8]=data[g+f'_old_{order}_log_bias_proxy_bits']
            for n in range(9,17):
                oracles=[];keys=[]
                for i,row in enumerate(data[g+'_tokens']):
                    pattern,labels=base.canonicalize(row[:n]);key=base.reference_key(pattern)
                    oracle=data[g+'_rows'][i,row[n-1],base.alphabet_map(labels)];m=refs[order][key]
                    loss[i,n-1]=-oracle@np.log2(m['mean']);oracles.append(oracle);keys.append(key)
                err=base.shared_error(np.asarray(oracles),keys,refs[order]);mcse[order][n-1]=err['numerical_mcse_bits'];bias[order][n-1]=err['log_bias_proxy_bits']
            losses[order]=loss;arrays[g+f'_bayes_loss_a{order}']=loss
        checks={scope:base.allowance(f,c,fl,cl,losses[32],losses[16],mcse[32],mcse[16],bias[32],bias[16],cum)
                for scope,cum in (('instantaneous',False),('cumulative',True))}
        groups.append(dict(condition=g,**checks));arrays.update({g+'_stationary_loss':f,g+'_stationary_coarse_loss':c,
            g+'_stationary_leave_one_bank_out_loss':fl,g+'_stationary_coarse_leave_one_bank_out_loss':cl})
        for role in ('ordered','shuffled'):
            for seed in (0,1):arrays[g+f'_{role}_seed{seed}']=data[g+f'_{role}_seed{seed}']
    approved=all(g[k]['passed'] for g in groups for k in ('instantaneous','cumulative'))
    base.save(out/'arrays.npz',**arrays);write(out/'input_bindings.json',bindings,immutable=True)
    result.update(groups=groups,aggregate_precision_passed=approved,population_reference_approved=approved,
        publication_status='approved' if approved else 'diagnostic_nonpublication',arrays_sha256=sha(out/'arrays.npz'),
        input_bindings_sha256=sha(out/'input_bindings.json'),original_histories=768,times=list(range(1,17)),
        comparator='Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.',payoff_forecast_approved=False,
        scope='Instantaneous and cumulative gains only; payoff crossing bootstrap not performed.')
    write(out/'summary.json',result);return 0 if approved else 3


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=('prepare','dispatch','worker','merge'))
    q.add_argument('--out',type=Path,required=True);q.add_argument('--parent',type=Path);q.add_argument('--terminal-receipt',type=Path);q.add_argument('--index',type=int)
    a=q.parse_args()
    if a.stage=='prepare':prepare(a.out,a.parent,a.terminal_receipt);return 0
    if a.stage=='dispatch':return dispatch(a.out,a.index)
    if a.stage=='worker':return worker(a.out,a.index)
    return merge(a.out)
if __name__=='__main__':raise SystemExit(main())

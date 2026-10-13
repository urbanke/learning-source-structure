"""Read-only full-cohort t5 diagnostics; never approve a payoff-time forecast.

Canonical references are shared random estimates: sum every oracle-weighted
loss gradient for a prefix before covariance projection. Independent prefixes
add variances; sequence variation and source-bank error are reported separately.
No sampling, failed-case subset, probability clipping or new precision budget.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import copying_payoff_dense as dense
from prepare_copying_payoff_references import canonicalize, reference_key, alphabet_map

ROOT=dense.ROOT
require,read,sha=dense.require,dense.read,dense.sha
LN2=np.log(2.)
FREQ=ROOT/'output/copying_frequency_extension_2026-09-19'
VALID=ROOT/'results/transformer/copying_frequency_extension_2026-09-19/validation.json'
SCORES=ROOT/'output/shared_base_shuffle_download_2026-09-18'


def stats(x):
    x=np.asarray(x);require(x.shape==(256,) and np.isfinite(x).all(),'All256 paired losses required')
    return dict(candidate_mean_bits=float(x.mean()),observed_sequence_se_bits=float(x.std(ddof=1)/16))


def moments_checked(m,estimate,n_seen):
    p=m['mean'];cm=m['chain_means']
    require(p.shape==(64,) and cm.shape==(4,64) and int(m['chains'])==4 and int(m['draws'])==16384,
            'Predictive moment shape or draw count changed')
    require(all(np.isfinite(v).all() for v in m.values()) and np.min(p)>0 and np.min(cm)>0
            and abs(p.sum()-1)<1e-7 and np.max(abs(cm.sum(1)-1))<1e-7,'Invalid predictive moments')
    require(np.allclose(p,estimate['predictive_probs'],rtol=0,atol=2e-14)
            and np.allclose(p,cm.mean(0),rtol=0,atol=2e-14),'Predictive readbacks disagree')
    require(np.max(abs(p[n_seen:]-p[n_seen]))<2e-14
            and np.max(abs(cm[:,n_seen:]-cm[:,n_seen,None]))<2e-14,'Unseen labels must share equal mass')
    for key in ('cov_batch','cov_chain'):
        c=m[key];require(c.shape==(64,64) and np.max(abs(c-c.T))<1e-14
            and np.linalg.eigvalsh(c).min()>-1e-12,'Invalid predictive covariance')
        require(np.max(abs(c[:,n_seen:]-c[:,n_seen,None]))<2e-14,'Unequal unseen covariance')
    expected=np.sqrt(np.maximum(0,np.maximum(np.diag(m['cov_batch']),np.diag(m['cov_chain']))))
    require(np.allclose(expected,m['mcse_probs'],rtol=1e-10,atol=2e-14),'Moment MCSE mismatch')


def projected(gradient,m):
    g=gradient-gradient.mean()
    return max(0.,float(g@m['cov_batch']@g),float(g@m['cov_chain']@g))


def shared_error(oracles,keys,references):
    """Conditional numerical error of a mean; repeated uses do not add draws."""
    require(len(oracles)==len(keys) and len(keys)>0,'Empty or unpaired loss gradients')
    variance=curvature=jackknife=bias=0.;n=len(keys)
    for key in sorted(set(keys)):
        weight=oracles[np.asarray(keys)==key].sum(0)/n
        m=references[key];p=m['mean'];variance+=projected(-weight/(p*LN2),m)
        diagonal=np.maximum(0,np.maximum(np.diag(m['cov_batch']),np.diag(m['cov_chain'])))
        curve=float(np.sum(weight*diagonal/p**2)/(2*LN2))
        leave=(4*p-m['chain_means'])/3
        require(np.min(leave)>0,'Invalid leave-one-chain predictive mean')
        jack=3*abs(float((-np.log2(leave)@weight).mean()+np.log2(p)@weight))
        curvature+=curve;jackknife+=jack;bias+=max(curve,jack)
    return dict(numerical_mcse_bits=float(np.sqrt(variance)),
        curvature_proxy_bits=curvature,chain_jackknife_proxy_bits=jackknife,log_bias_proxy_bits=bias)


def bank_error(loss,leave):
    """One common bank estimate serves all histories; jackknife after averaging."""
    require(leave.ndim==2 and leave.shape[1]==len(loss) and leave.shape[0]>=2,'Prior bank coverage differs')
    means=leave.mean(1);b=len(means)
    return dict(prior_bank_mcse_bits=float(np.sqrt((b-1)/b*np.sum((means-means.mean())**2))),
                prior_bank_jackknife_bias_proxy_bits=float((b-1)*abs(means.mean()-loss.mean())))


def load_references(folder):
    plan=dense.load_plan(folder);require(plan['observed']==5 and len(plan['tasks'])==76,'This merger is frozen to t5')
    require(len({t['seed'] for t in plan['tasks']})==76,'Canonical runs must have distinct seeds')
    missing=[t['id'] for t in plan['tasks'] if not dense.paths(folder,t)['report'].exists()]
    result=dict(complete=not missing,expected_tasks=76,original_histories=768,missing_task_ids=missing,
        individual_audits_passed=False,paired_orders_passed=False,diagnostic_arrays_validated=False,
        aggregate_precision_passed=False,population_reference_approved=False,payoff_forecast_approved=False,
        timing_assessment_pending=True,finite_shuffle_exact=False,published_mean_gain_bits=None)
    if missing:return plan,result,{},{}
    refs={16:{},32:{}};bindings={};failed=[]
    for t in plan['tasks']:
        paths=dense.paths(folder,t);r=read(paths['report']);a=read(paths['audit']);ident=read(paths['identity'])
        require(r['plan_id']==plan['plan_id'] and r['task']==t and r['complete']
                and ident['plan_id']==plan['plan_id'] and ident['task']==t,'Reference identity changed')
        for key in ('audit','identity','checkpoint','moments'):
            require(sha(paths[key])==r[key+'_sha256'],'Saved raw/audit binding changed')
            bindings[str(paths[key])]=r[key+'_sha256']
        bindings[str(paths['report'])]=sha(paths['report'])
        require(r['audit_passed']==(a['passed'] and all(g['passed'] for g in a['final_state_grid_checks'])),
                'Individual audit flag changed')
        if not r['audit_passed']:failed.append(t['id'])
        with np.load(paths['moments'],allow_pickle=False) as z:m={k:z[k].copy() for k in z}
        moments_checked(m,r['estimate'],len(set(t['tokens'])))
        refs[t['order']][t['reference_key']]=m
    result['failed_audit_ids']=failed;result['individual_audits_passed']=not failed
    pairs=[]
    for key,left in refs[16].items():
        right=refs[32][key]
        check=dense.accuracy.paired_order_check(left['mean'],left['mcse_probs'],right['mean'],right['mcse_probs'])
        pairs.append(dict(reference_key=key,**check))
    result.update(paired_order_checks=pairs,paired_orders_passed=all(p['passed'] for p in pairs),
        paired_order_scope='Existing normal-MCSE diagnostic: inherited64x51 Bonferroni allowance covers these38x64 coordinates; not a rigorous convergence guarantee.')
    return plan,result,refs,bindings


def map_history(tokens,transition,use,task):
    pattern,labels=canonicalize(tokens[:5])
    digest=hashlib.sha256(np.asarray(tokens[:5],dtype='<i8').tobytes()).hexdigest()
    require(reference_key(pattern)==use['reference_key'] and list(pattern)==task['tokens']
            and list(labels)==use['canonical_to_original_seen'] and digest==use['original_prefix_sha256']
            and int(tokens[4])==use['queried_original_label'],'Original ordered history or alphabet map changed')
    mapping=alphabet_map(labels);oracle=transition[int(tokens[4]),mapping]
    require(oracle.shape==(64,) and np.isfinite(oracle).all() and np.min(oracle)>=0
            and abs(oracle.sum()-1)<1e-7,'Invalid true query-row distribution')
    return mapping,oracle


def frozen_inputs(plan):
    receipt=read(VALID);require(receipt['passed'] and receipt['script_sha256']==sha(ROOT/'code/scripts/publish_copying_frequency_extension.py'),
                              'Frequency comparator independent audit changed')
    require(receipt['summary_sha256']==sha(VALID.parent/'summary.json'),'Published frequency summary changed')
    bindings={str(VALID):sha(VALID),str(VALID.parent/'summary.json'):receipt['summary_sha256']}
    for name,digest in receipt['input_sha256'].items():
        require(sha(ROOT/name)==digest,'Audited frequency input changed');bindings[str(ROOT/name)]=digest
    plans={stage:read(FREQ/stage/'plan.json') for stage in ('stratified','refined')}
    for stage,p in plans.items():
        s=read(FREQ/stage/'summary.json');require(s['complete'] and s['times']==list(range(1,9))
            and s['plan_sha256']==sha(FREQ/stage/'plan.json') and not s['finite_shuffle_exact'],'Frequency scope changed')
        for name,digest in {**p['code_sha256'],**p['inputs_sha256'],**p['original_scoring_inputs_sha256']}.items():
            require(sha(ROOT/name)==digest,'Frozen frequency source or model changed');bindings[str(ROOT/name)]=digest
    return bindings,plans['refined']


def model_loss(name,seed,role,source,bindings):
    folder=SCORES/'eval'/f'{role}_seed{seed}';p=folder/(name+'.npz');protocol=read(folder/'protocol.json');status=read(folder/'status.json')
    require(status['complete'] and status['scientific_results'] and protocol['role']==role and protocol['seed']==seed
        and protocol['evaluation_code_sha256']==sha(ROOT/'code/transformer/shared_base_shuffle_eval.py')
        and protocol['model_code_sha256']==sha(ROOT/'code/transformer/model.py'),'Neural evaluation identity changed')
    with np.load(p,allow_pickle=False) as z:
        require(str(z['protocol_sha256'])==sha(folder/'protocol.json') and str(z['source_tokens_sha256'])==source['tokens_sha256']
            and np.array_equal(z['times'],np.arange(1,1025)) and z['expected_loss_bits'].shape==(256,1024),'Neural cohort/time differs')
        value=z['expected_loss_bits'][:,4].copy()
    require(np.isfinite(value).all(),'Nonfinite neural losses')
    for f in (p,folder/'protocol.json',folder/'status.json'):bindings[str(f)]=sha(f)
    return value


def merge(folder):
    plan,result,refs,bindings=load_references(folder)
    result.update(schema='copying-payoff-dense-diagnostics-v1',plan_id=plan['plan_id'],time=5,
        interpretation='Complete-cohort diagnostic at one context length. Timing definition and total uncertainty must still be assessed; no crossing inferred.',
        comparator='Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.',
        informed_bayes='Original full shared-base prior, same ordered histories; source parameters used only for scoring.',
        uncertainty='Shared-prefix MCMC covariance; local log-bias proxies and independent source-bank errors. Observed paired sequence SE is separate and still contains numerical noise. No hard loss-precision budget imposed.')
    if not result['complete'] or not result['individual_audits_passed'] or not result['paired_orders_passed']:return result,{}
    extra,freqplan=frozen_inputs(plan);bindings.update(extra);arrays={};groups=[]
    tasks={t['reference_key']:t for t in plan['tasks']}
    for name,source in plan['source_bindings'].items():
        with np.load(source['path'],allow_pickle=False) as z:tokens=z['tokens'];transition=z['transition']
        require(tokens.shape==(256,1025) and transition.shape==(256,64,64),'Incomplete source cohort')
        uses=sorted([u for u in plan['uses'] if u['condition']==name],key=lambda u:u['index'])
        require([u['index'] for u in uses]==list(range(256)),'Selected or duplicated histories')
        mapped=[map_history(tokens[i],transition[i],u,tasks[u['reference_key']]) for i,u in enumerate(uses)]
        oracles=np.stack([v[1] for v in mapped]);keys=[u['reference_key'] for u in uses]
        group=dict(condition=name,n_sequences=256,rules={})
        with np.load(FREQ/'refined'/(name+'.npz'),allow_pickle=False) as z:
            require(z['stationary_loss'].shape==(256,8) and z['stationary_leave_one_bank_out_loss'].shape==(16,256,8),'Stationary coverage changed')
            stationary=z['stationary_loss'][:,4];leave=z['stationary_leave_one_bank_out_loss'][:,:,4]
        with np.load(FREQ/'stratified'/(name+'.npz'),allow_pickle=False) as z:coarse=z['stationary_loss'][:,4]
        require(np.isfinite(stationary).all() and np.isfinite(leave).all() and np.isfinite(coarse).all(),'Invalid stationary losses')
        bank=bank_error(stationary,leave);bank['coarse_refined_mean_shift_bits']=float((stationary-coarse).mean())
        group['stationary_numerics']=bank;arrays[name+'_stationary_loss']=stationary
        for order in (16,32):
            prediction=np.stack([refs[order][k]['mean'] for k in keys]);loss=-np.sum(oracles*np.log2(prediction),axis=1)
            error=shared_error(oracles,keys,refs[order]);gain=stationary-loss
            row=dict(bayes_loss=stats(loss),stationary_minus_bayes=stats(gain),**error)
            row['stationary_minus_bayes'].update(combined_numerical_mcse_bits=float(np.hypot(error['numerical_mcse_bits'],bank['prior_bank_mcse_bits'])),
                combined_log_bias_proxy_bits=error['log_bias_proxy_bits']+bank['prior_bank_jackknife_bias_proxy_bits'])
            group['rules'][str(order)]=row;arrays[name+f'_bayes_loss_a{order}']=loss
            original=np.empty_like(prediction)
            for i,(mapping,_) in enumerate(mapped):original[i,mapping]=prediction[i]
            arrays[name+f'_original_label_probs_a{order}']=original
        difference=arrays[name+'_bayes_loss_a16']-arrays[name+'_bayes_loss_a32']
        group['order_loss_difference']=dict(**stats(difference),numerical_mcse_bits=float(np.hypot(*[group['rules'][str(o)]['numerical_mcse_bits'] for o in (16,32)])))
        group['neural']={}
        for seed in (0,1):
            ordered=model_loss(name,seed,'ordered',source,bindings);shuffled=model_loss(name,seed,'shuffled',source,bindings)
            for role,v in (('ordered',ordered),('shuffled',shuffled)):arrays[name+f'_{role}_seed{seed}']=v
            group['neural'][str(seed)]=dict(gain=stats(shuffled-ordered),ordered_minus_bayes=stats(ordered-arrays[name+'_bayes_loss_a32']),
                                           shuffled_minus_stationary=stats(shuffled-stationary))
        groups.append(group)
    result.update(diagnostic_arrays_validated=True,diagnostic_groups=groups,input_sha256=bindings,
        merger_sha256=sha(__file__),wave_plan_sha256=sha(Path(folder)/'plan.json'))
    return result,arrays


def exit_status(result):
    if not result['complete']:return 3
    if not result['individual_audits_passed'] or not result['paired_orders_passed']:return 2
    require(result['diagnostic_arrays_validated'],'Complete diagnostics were not validated')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=dense.DEFAULT_OUT/'t5')
    p.add_argument('--out',type=Path,default=ROOT/'output/copying_payoff_dense_diagnostics_2026-09-20/t5');a=p.parse_args()
    require(not a.out.exists(),'Use a new diagnostics folder; preserve previous results')
    result,arrays=merge(a.input);a.out.mkdir(parents=True)
    if arrays:
        np.savez_compressed(a.out/'arrays.npz',**arrays);result['arrays_sha256']=sha(a.out/'arrays.npz')
    dense.write(a.out/'summary.json',result)
    print({k:result[k] for k in ('complete','individual_audits_passed','paired_orders_passed','diagnostic_arrays_validated','payoff_forecast_approved')})
    raise SystemExit(exit_status(result))

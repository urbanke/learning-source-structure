"""Append independent copying frequency banks without altering existing outputs.

The numerical kernel, seeds 0..15, original 768 histories, both integration
orders, and numerical gates are retained. This creates a new diagnostic
layer only; a frequency refinement alone cannot approve the copying figure.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import copying_bayes_dense16 as base

SCHEMA = 'copying-dense16-frequency-precision-v1'
NEW_BANKS = list(range(16, 256))
BINDINGS = ('plan.json', 'prepared.json', 'input.npz', 'kinds.json')


def codes():
    return {str(Path(__file__).relative_to(base.ROOT)): base.sha(__file__),
            **base.codes()}


def prepare(out, parent, final):
    base.require(not (out/'plan.json').exists(), 'No repeated preparation')
    p = base.load(parent)
    last = base.read(final/'summary.json')
    base.require(last['complete'] and not last['failed_audits']
                 and not last['failed_paired_orders'], 'Parent inference is incomplete')
    banks = []
    for i in range(16):
        path = parent/'banks'/str(i)
        r = base.read(path/'report.json')
        base.require(r['complete'] and r['index'] == i and r['plan_id'] == p['plan_id']
                     and base.sha(path/'moments.npz') == r['arrays_sha256'], 'Parent bank differs')
        banks.append(dict(index=i, path=str(path), report_sha256=base.sha(path/'report.json'),
                          moments_sha256=r['arrays_sha256']))
    plan = dict(schema=SCHEMA, parent=str(parent), final=str(final),
                parent_bindings={name:base.sha(parent/name) for name in BINDINGS},
                final_summary_sha256=base.sha(final/'summary.json'),
                final_plan_sha256=base.sha(final/'plan.json'),
                code_sha256=codes(), existing_banks=banks, new_banks=NEW_BANKS,
                total_banks=256, original_histories=768, times=list(range(1,17)),
                runtime=p['ready']['runtime'],
                resources=dict(bank_wall_minutes=11, banks=240, allocated_cpus=1,
                  memory_gib=2, maximum_concurrent=128, preparation_minutes=5,
                  assessment_minutes=55, maximum_cpu_hours=45,
                  maximum_cost_chf=.2475, rate_chf_per_cpu_hour=.0055),
                automatic_retries=False, population_reference_approved=False)
    plan['plan_id'] = base.fp(plan)
    base.write(out/'plan.json', plan, immutable=True)
    print(json.dumps(dict(event='prepared', plan_id=plan['plan_id'], resources=plan['resources'])), flush=True)


def load(out):
    p = base.read(out/'plan.json')
    base.require(p['schema']==SCHEMA and p['plan_id']==base.fp({k:v for k,v in p.items() if k!='plan_id'}), 'Plan changed')
    base.require(p['code_sha256']==codes() and p['new_banks']==NEW_BANKS, 'Frozen implementation changed')
    parent=Path(p['parent'])
    for name,digest in p['parent_bindings'].items():
        base.require(base.sha(parent/name)==digest, 'Parent input changed: '+name)
    old=base.load(parent)
    base.require(old['ready']['runtime']==p['runtime'], 'Runtime changed')
    return p


def bank(out,index):
    p=load(out);base.require(index in NEW_BANKS, 'Invalid new bank')
    folder=out/'banks'/str(index);folder.mkdir(parents=True,exist_ok=False)
    kinds=[(x[0],tuple(x[1])) for x in base.read(Path(p['parent'])/'kinds.json')]
    print(json.dumps(dict(event='bank_started',index=index,frequency_types=len(kinds))),flush=True)
    start=time.monotonic()
    original=base.frequency.stationary_joint_type; completed=0
    def counted(*args,**kwargs):
        nonlocal completed
        value=original(*args,**kwargs);completed+=1
        if completed==1 or completed%100==0:
            print(json.dumps(dict(event='bank_moments',index=index,completed=completed,total=2*len(kinds))),flush=True)
        return value
    base.frequency.stationary_joint_type=counted
    try:fine,coarse,nf,nc=base.frequency.generate_bank(index,kinds)
    finally:base.frequency.stationary_joint_type=original
    base.save(folder/'moments.npz',fine=fine,coarse=coarse)
    base.write(folder/'report.json',dict(plan_id=p['plan_id'],index=index,complete=True,
      arrays_sha256=base.sha(folder/'moments.npz'),elapsed_seconds=time.monotonic()-start,
      latent_rows=[nf,nc]),immutable=True)
    print(json.dumps(dict(event='bank_complete',index=index,seconds=time.monotonic()-start)),flush=True)


def score_frequency(joints,data,kinds):
    """Identical pooled ratio and delete-bank estimator, generalized to B banks."""
    B=len(joints);base.require(B>=2,'At least two banks required')
    index={kind:i for i,kind in enumerate(kinds)};total=joints.sum(0)
    full=total/B;leave=(total-joints)/(B-1);result={}
    for g in base.CONDITIONS:
        loss=np.empty((256,16));loo=np.empty((B,256,16))
        for i,row in enumerate(data[g+'_tokens']):
            for n in range(1,17):
                prefix=row[:n].tolist();den=index[base.frequency.frequency_type(prefix)]
                cols=[index[base.frequency.frequency_type(prefix+[a])] for a in range(64)]
                prediction=full[cols]/full[den];predictions=leave[:,cols]/leave[:,den,None]
                base.require(abs(prediction.sum()-1)<1e-10 and np.max(abs(predictions.sum(1)-1))<1e-10,'Frequency normalization failed')
                oracle=data[g+'_rows'][i,row[n-1]]
                loss[i,n-1]=-oracle@np.log2(prediction);loo[:,i,n-1]=-np.log2(predictions)@oracle
        result[g]=(loss,loo)
    return result


def assess(out):
    p=load(out);parent=Path(p['parent']);banks=[];bindings={}
    for i in range(256):
        folder=(parent if i<16 else out)/'banks'/str(i)
        r=base.read(folder/'report.json')
        expected=base.read(parent/'plan.json')['plan_id'] if i<16 else p['plan_id']
        base.require(r['complete'] and r['index']==i and r['plan_id']==expected
                     and base.sha(folder/'moments.npz')==r['arrays_sha256'],'Bank binding differs')
        if i<16:
            bound=p['existing_banks'][i]
            base.require(base.sha(folder/'report.json')==bound['report_sha256'] and r['arrays_sha256']==bound['moments_sha256'],'Retained bank changed')
        banks.append(base.archive(folder/'moments.npz'))
        bindings[str(folder/'moments.npz')]=r['arrays_sha256']
    data=base.archive(parent/'input.npz');kinds=[(x[0],tuple(x[1])) for x in base.read(parent/'kinds.json')]
    scores={};arrays={}
    for key in ('fine','coarse'):
        joints=np.stack([b[key] for b in banks])
        before=base.score_frequency(joints[:16],data,kinds)
        replay=score_frequency(joints[:16],data,kinds)
        for g in base.CONDITIONS:
            base.require(all(np.array_equal(a,b) for a,b in zip(before[g],replay[g])), '16-bank parity failed')
        scores[key]=score_frequency(joints,data,kinds)
        print(json.dumps(dict(event='frequency_scored',rule=key)),flush=True)
    groups=[]
    for g in base.CONDITIONS:
        f,fl=scores['fine'][g];c,cl=scores['coarse'][g]
        arrays.update({g+'_stationary_loss':f,g+'_stationary_coarse_loss':c,
          g+'_stationary_leave_one_bank_out_loss':fl,g+'_stationary_coarse_leave_one_bank_out_loss':cl})
        rows={}
        for scope,cum in (('instantaneous',False),('cumulative',True)):
            ff,cc,ffl,ccl=[np.cumsum(x,axis=-1) if cum else x for x in (f,c,fl,cl)]
            bank=base.frequency._delete_metrics(ff,ffl)
            pair=base.frequency._delete_metrics(ff-cc,ffl-ccl)
            rows[scope]=dict(bank_mcse_bits=bank['mcse'].tolist(),bank_bias_bits=bank['bias'].tolist(),
              paired_mean_shift_bits=abs((ff-cc).mean(0)).tolist(),paired_mcse_bits=pair['mcse'].tolist(),paired_bias_bits=pair['bias'].tolist())
        groups.append(dict(condition=g,**rows))
    base.save(out/'frequency_arrays.npz',**arrays)
    base.write(out/'frequency_summary.json',dict(schema=SCHEMA,plan_id=p['plan_id'],complete=True,
      banks=256,groups=groups,retained_bank_parity_passed=True,population_reference_approved=False,
      arrays_sha256=base.sha(out/'frequency_arrays.npz'),input_bindings=bindings),immutable=True)
    print(json.dumps(dict(event='frequency_assessment_complete',banks=256)),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage',choices=('prepare','bank','assess'));a.add_argument('--out',type=Path,required=True)
    a.add_argument('--parent',type=Path);a.add_argument('--final',type=Path);a.add_argument('--index',type=int)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.final)
    elif v.stage=='bank':bank(v.out,v.index)
    else:assess(v.out)
if __name__=='__main__':main()

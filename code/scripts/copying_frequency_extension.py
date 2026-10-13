"""Bounded stationary-frequency comparison through eight observed tokens.

Reuse both frozen source banks and all original neural evaluation histories.
This validates a surrogate, not the exact finite-shuffle law or a payoff time.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import signal
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
from stationary_frequency_moments import frequency_type,stationary_joint_type
from copying_frequency_calibration import MANIFEST,BASE,sha,require

OUT=ROOT/'output/copying_frequency_extension_2026-09-19'
BANKS=ROOT/'output/copying_frequency_calibration_2026-09-19'


def write(path,value):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def key(kind):return json.dumps(kind,separators=(',',':'))


def bank_weights(settings):
    banks,order,draws=settings
    _,weights=np.polynomial.legendre.leggauss(order)
    row=[]
    for weight in weights:
        for depth in range(9):
            n=1 if depth==0 else draws
            row.extend([weight/18/n]*n)
    value=np.asarray(row)
    require(abs(value.sum()-1)<1e-13,'Incorrect latent-prior weights')
    return banks,value


def load_inputs():
    manifest=json.loads(MANIFEST.read_text());conditions=manifest['conditions']
    inputs={str(MANIFEST.relative_to(ROOT)):sha(MANIFEST)};groups=[]
    for condition in conditions:
        path=BASE/'cohorts'/(condition['id']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            tokens=z['tokens'].copy();rows=z['transition'].copy()
        require(tokens.shape==(256,1025) and rows.shape==(256,64,64),'Incomplete frozen cohort')
        # This calibrated reader checks all source/model identities independently
        # on the same inputs before this extension is prepared.
        inputs[str(path.relative_to(ROOT))]=sha(path)
        groups.append(dict(condition=condition,tokens=tokens,rows=rows))
    require([g['condition']['id'] for g in groups]==['a1_L0','a1_L2','a1_L3'],'Unexpected conditions')
    return groups,inputs


def all_types(groups):
    kinds=set()
    for group in groups:
        for row in group['tokens']:
            for n in range(1,9):
                prefix=row[:n].tolist();kinds.add(frequency_type(prefix))
                for label in list(dict.fromkeys(prefix))+[next(x for x in range(64) if x not in prefix)]:
                    kinds.add(frequency_type(prefix+[label]))
    return sorted(kinds,key=lambda k:(1+k[0]+sum(k[1]),k))


def run(stage):
    out=OUT/stage;out.mkdir(parents=True,exist_ok=True)
    bankfolder=BANKS/stage;original=json.loads((bankfolder/'plan.json').read_text())
    summary=json.loads((bankfolder/'summary.json').read_text())
    require(summary['complete'] and summary['plan_sha256']==sha(bankfolder/'plan.json'),'Original calibration changed')
    for name,digest in {**original['code_sha256'],**summary['scoring_inputs_sha256']}.items():
        require(sha(ROOT/name)==digest,'Original calibrated input changed: '+name)
    groups,inputs=load_inputs()
    with np.load(bankfolder/'bank.npz',allow_pickle=False) as z:q=z['q'].copy();nu=z['nu'].copy()
    require(sha(bankfolder/'bank.npz')==summary['reference']['bank_sha256'],'Source bank changed')
    banks,weights=bank_weights(original['settings'])
    require(q.shape==(banks*len(weights),64) and nu.shape==q.shape,'Prior bank layout changed')
    kinds=all_types(groups)
    plan=dict(schema='copying-frequency-extension-v1',stage=stage,times=list(range(1,9)),
        source_bank_plan_sha256=sha(bankfolder/'plan.json'),source_bank_sha256=sha(bankfolder/'bank.npz'),
        original_calibration_summary_sha256=sha(bankfolder/'summary.json'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'code/transformer/stationary_frequency_moments.py')},
        inputs_sha256=inputs,original_scoring_inputs_sha256=summary['scoring_inputs_sha256'],
        runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__),frequency_types=[key(k) for k in kinds],
        maximum_worker_seconds=1800,maximum_attempts=1,finite_shuffle_exact=False,scientific_timing_validated=False)
    require(not (out/'plan.json').exists(),'One attempt only; preserve the existing extension')
    write(out/'plan.json',plan);began=time.monotonic();joint={}
    for index,kind in enumerate(kinds):
        if time.monotonic()-began>1800:
            write(out/'status.json',dict(complete=False,status='time_limit',types_completed=len(joint),types_total=len(kinds)))
            return 75
        values=stationary_joint_type(kind,q,nu,block_size=1024)
        joint[key(kind)]=(values.reshape(banks,-1)@weights).tolist()
        if index%10==0 or index==len(kinds)-1:
            write(out/'moments.json',joint)
            event=dict(complete=False,status='running',types_completed=len(joint),types_total=len(kinds),elapsed_seconds=time.monotonic()-began)
            write(out/'status.json',event);print(json.dumps(event),flush=True)
    maximum_normalization=0.;minimum_bank_weight_ess=float(banks);results=[]
    def probs(prefix):
        nonlocal maximum_normalization,minimum_bank_weight_ess
        denominator=np.asarray(joint[key(frequency_type(prefix))]);total=denominator.sum()
        minimum_bank_weight_ess=min(minimum_bank_weight_ess,float(total**2/np.sum(denominator**2)))
        mean=np.empty(64);loo=np.empty((banks,64))
        for label in range(64):
            numerator=np.asarray(joint[key(frequency_type(prefix+[label]))])
            mean[label]=numerator.sum()/total
            loo[:,label]=(numerator.sum()-numerator)/(total-denominator)
        error=max(abs(mean.sum()-1),float(np.max(abs(loo.sum(1)-1))))
        maximum_normalization=max(maximum_normalization,error)
        require(error<1e-10 and np.all(mean>0) and np.all(loo>0),'Raw prediction normalization failed')
        return mean,loo
    for group in groups:
        name=group['condition']['id'];loss=np.empty((256,8));loss_loo=np.empty((banks,256,8))
        for i,row in enumerate(group['tokens']):
            for n in range(1,9):
                mean,loo=probs(row[:n].tolist());oracle=group['rows'][i,row[n-1]]
                loss[i,n-1]=-oracle@np.log2(mean)
                loss_loo[:,i,n-1]=-np.log2(loo)@oracle
        comparisons={}
        for seed in (0,1):
            # Resolve the same score files as the frozen early calibration,
            # using its own recorded input inventory rather than guessed paths.
            matches=[ROOT/p for p in summary['scoring_inputs_sha256']
                     if p.endswith('.npz') and 'shuffled' in p and ('seed'+str(seed)) in p and name in p]
            require(len(matches)==1,'Cannot identify the exact shuffled-model score input')
            with np.load(matches[0],allow_pickle=False) as z:model=z['expected_loss_bits'][:,:8]
            delta=loss-model;x=(loss_loo-model).mean(1);center=x.mean(0)
            comparisons[str(seed)]=dict(mean=delta.mean(0).tolist(),paired_sequence_se=(delta.std(0,ddof=1)/16).tolist(),
                prior_bank_mcse=np.sqrt((banks-1)/banks*np.sum((x-center)**2,axis=0)).tolist())
        old=next(g for g in summary['groups'] if g['id']==name)
        for seed in (0,1):
            expected=old['comparisons'][f'stationary_minus_shuffled_seed{seed}']['mean']
            require(np.max(abs(np.asarray(comparisons[str(seed)]['mean'][:4])-expected))<1e-11,'Independent early moments do not replay')
        np.savez_compressed(out/(name+'.npz'),stationary_loss=loss,stationary_leave_one_bank_out_loss=loss_loo)
        results.append(dict(id=name,comparisons=comparisons))
    report=dict(schema='copying-frequency-extension-v1',complete=True,plan_sha256=sha(out/'plan.json'),
        groups=results,times=list(range(1,9)),maximum_normalization_error=maximum_normalization,
        minimum_prior_bank_weight_ess=minimum_bank_weight_ess,
        elapsed_seconds=time.monotonic()-began,scientific_timing_validated=False,finite_shuffle_exact=False,
        interpretation='Stationary iid surrogate calibrated on reused ordered histories; no Bayesian payoff-time forecast.')
    write(out/'summary.json',report);write(out/'status.json',dict(complete=True,status='complete',elapsed_seconds=report['elapsed_seconds']))
    print(json.dumps(dict(complete=True,stage=stage,elapsed_seconds=report['elapsed_seconds'])),flush=True)
    return 0


def bounded_run(stage):
    out=OUT/stage;out.mkdir(parents=True,exist_ok=True)
    with (out/'.invocation.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (out/'plan.json').exists() and not (out/'attempt.json').exists(),
                'One attempt only; existing results remain unchanged')
        write(out/'attempt.json',dict(started_unix=time.time(),maximum_worker_seconds=1800))
        def timeout(signum,frame):
            raise TimeoutError('Fixed1800-second stage allocation exhausted')
        previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
        try:
            return run(stage)
        except TimeoutError as error:
            write(out/'status.json',dict(complete=False,status='time_limit',error=str(error)))
            return 75
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['stratified','refined'],required=True)
    raise SystemExit(bounded_run(parser.parse_args().stage))

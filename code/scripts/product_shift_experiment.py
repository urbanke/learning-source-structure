"""Resumable local product-predictor tests under relabeling and parameter changes.

Benchmark validates new histories before the full fixed cohort is evaluated.
All predictions retain the original training prior and unbroken history.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
import conditional_product_accuracy as previous
import product_context_confirmation as confirmation
from product_shift_source import generate_product_shift_cases

SCHEMA = 'product-shift-experiment-v1'
CONDITIONS = ['stationary', 'relabeled', 'switch_one', 'switch_both']
PAIRS = [[1, 1], [4, 4]]
TIMES = [1, 2, 4, 8, 16, 32, 64, 128, 192, 224, 240, 248, 255,
         256, 257, 260, 264, 272, 288, 320, 384, 448, 512, 640, 768, 896, 1024]
existing = previous.existing
write = confirmation.write
require = previous.require
_STATE = None


def configure():
    old = confirmation.read(ROOT / 'output/product_context_confirmation_2026-09-17/plan.json')
    for key, field in [('LSA_STORE', 'lsa_store'), ('PMM_SRC', 'pmm_src')]:
        os.environ.setdefault(key, old['numerical_reference'][field])
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    return confirmation.backend_environment()


def numerical_args(out):
    return confirmation.numerical_args(out, TIMES, seed=2026091809, torch_threads=1)


def prepare(out, n_seq):
    env = configure()
    args = numerical_args(out)
    case = existing._case('product', args)
    files = [Path(__file__), ROOT/'code/transformer/product_shift_source.py',
             Path(previous.__file__), Path(confirmation.__file__),
             ROOT/'code/transformer/shared_depth.py', ROOT/'code/transformer/lsa_seq.py',
             ROOT/'code/transformer/phi_source.py', ROOT/'code/transformer/source.py',
             ROOT/'code/transformer/families.py', ROOT/'code/transformer/model.py',
             ROOT/'code/transformer/eval_shared_depth.py', case['checkpoint'], case['config_path']]
    spec = dict(schema=SCHEMA, n_seq=n_seq, times=TIMES, conditions=CONDITIONS,
        pairs=PAIRS, switch_observed=256, n_tokens=1025, seed=args.seed,
        source='Fresh uniform-base LSA product rows at fixed depths; independent source draws',
        relabeling='Independent arbitrary joint-label permutation per source, constant within its sequence',
        switching='One or both matrices redrawn; continue actual state; new law first predicts token index256',
        predictor_information='Original product training prior; no true depth, permutation, change flag or reset',
        reference=previous.reference_spec(case,args), backend_environment=env,
        model=existing._model_spec(case,args.prefixes), runtime=existing._runtime_versions(),
        inputs={str(p.resolve()):existing._sha_file(p) for p in files},
        uncertainty='Independent source draws, paired conditions; one fixed trained model seed only',
        primary='Full next-token KL to actual current law and predictive agreement',
        recovery=dict(prechange_times=[192,224,240,248,255], allowance_bits=.05,
            definition='First tested post-switch time within own prechange mean risk plus allowance at all remaining tested times',
            nonattainment='Retained as not observed by1024 tokens; not assigned a finite recovery time'),
        stopping='Complete all fixed1024 source draws per depth, independent of outcomes')
    require(spec['reference']['store_manifest_sha256'] and spec['reference']['backend_sha256'],
            'Sealed numerical table store missing')
    spec['plan_id'] = existing._fingerprint(spec)
    path = out/'plan.json'
    if path.exists():
        require(confirmation.read(path) == spec, 'Plan/input changed; use a new output directory')
    else:
        write(path,spec)
    return spec


def load(out):
    configure()
    p = confirmation.read(out/'plan.json')
    require(p['plan_id'] == existing._fingerprint({k:v for k,v in p.items() if k != 'plan_id'}),
            'Frozen plan altered')
    for name,digest in p['inputs'].items():
        require(existing._sha_file(name) == digest, 'Frozen input changed: '+name)
    args = numerical_args(out)
    case = existing._case('product',args)
    require(previous.reference_spec(case,args)==p['reference'], 'Numerical reference settings changed')
    return p,args,case


def initialize(out):
    global _STATE
    out = Path(out)
    p,args,case = load(out)
    ref = previous.make_reference(args)
    _STATE = dict(out=out,plan=p,args=args,case=case,reference=ref,model=None)


def network():
    import torch
    from model import GPT
    if _STATE['model'] is None:
        torch.set_num_threads(1)
        ck = torch.load(_STATE['case']['checkpoint'],map_location='cpu',weights_only=True)
        cfg = ck['cfg']['model']
        require(ck['step']==100000 and cfg==_STATE['case']['config']['model'], 'Checkpoint identity failed')
        m = GPT(cfg['vocab'],cfg['n_ctx'],cfg['n_layer'],cfg['n_head'],cfg['d_model'],
                mlp=cfg['mlp'],dropout=cfg.get('dropout',0.))
        m.load_state_dict(ck['model']); m.eval()
        _STATE['model'] = m
    return _STATE['model']


def record_path(out,pair,index):
    return out/'records'/f'L{pair[0]}_{pair[1]}'/f'seq_{index:06d}.npz'


def cache_id(plan,pair,index):
    return existing._fingerprint(dict(plan_id=plan['plan_id'],pair=pair,index=index))


def evaluate(task):
    import torch
    pair,index = task
    s = _STATE; p = s['plan']; out = s['out']
    path = record_path(out,pair,index)
    ident = cache_id(p,pair,index)
    if existing._read_cache(path,ident) is not None:
        return dict(pair=pair,index=index,reused=True,seconds=0.)
    start=time.monotonic()
    cases = generate_product_shift_cases(index,depths=tuple(pair),seed=p['seed'],
                    n_tokens=p['n_tokens'],switch_observed=p['switch_observed'])
    tokens = np.stack([cases[name].tokens for name in CONDITIONS])
    true = np.stack([cases[name].true_probs(np.asarray(TIMES)) for name in CONDITIONS])
    predictions=[]; losses=[]
    ref_start=time.monotonic()
    for x in tokens:
        z = s['reference'].run(x,full_probs=True,positions=np.asarray(TIMES))
        predictions.append(z['probs']); losses.append(z['loss'][1:])
    reference_seconds=time.monotonic()-ref_start
    bayes=np.stack(predictions)
    m=network(); model_start=time.monotonic()
    with torch.inference_mode():
        x=torch.from_numpy(tokens.astype(np.int64))
        lp=torch.log_softmax(m(x[:,:-1]).float(),dim=-1)
        selected=lp[:,np.asarray(TIMES)-1,:].numpy().astype(float)
        model_loss=-lp.gather(2,x[:,1:,None])[:,:,0].numpy().astype(float)/np.log(2.)
    transformer=np.exp(selected)
    norms={name:existing._normalization_error(a.reshape(-1,64),name,tol)
           for name,a,tol in [('true',true,2e-12),('bayes',bayes,2e-6),('transformer',transformer,2e-6)]}
    require(np.min(bayes)>0 and np.min(transformer)>0,'Missing predictor support')
    # Changed futures must not affect any predictions based on the common prefix.
    before=np.asarray(TIMES)<=p['switch_observed']
    for j in (2,3):
        require(np.array_equal(tokens[0,:256],tokens[j,:256]), 'Prechange history differs')
        require(np.max(np.abs(bayes[0,before]-bayes[j,before]))<2e-12, 'Bayes causal alignment failed')
        require(np.max(np.abs(transformer[0,before]-transformer[j,before]))<2e-6, 'Transformer causal alignment failed')
    arrays=dict(cache_id=ident,plan_id=p['plan_id'],index=index,depth_pair=pair,
        times=TIMES,conditions=CONDITIONS,tokens=tokens,true_probs=true,bayes_probs=bayes,
        transformer_probs=transformer,bayes_loss=np.stack(losses),transformer_loss=model_loss,
        pre_matrices=np.stack([cases[name].pre_matrices for name in CONDITIONS]),
        post_matrices=np.stack([cases[name].post_matrices for name in CONDITIONS]),
        permutations=np.stack([cases[name].permutation for name in CONDITIONS]),
        reference_seconds=reference_seconds,model_seconds=time.monotonic()-model_start,
        elapsed_seconds=time.monotonic()-start,max_normalization_errors=list(norms.values()))
    existing._atomic_npz(path,**arrays)
    return dict(pair=pair,index=index,reused=False,seconds=arrays['elapsed_seconds'])


def validate(out):
    p,args,_=load(out)
    wide=previous.make_reference(args,wide=True)
    exact=previous.make_reference(args,depths=[1])
    maximum=dict(wide_probability=0.,wide_risk_bits=0.,dirichlet_probability=0.)
    for pair in PAIRS:
        for index in (0,3):
            record=existing._read_cache(record_path(out,pair,index),cache_id(p,pair,index))
            require(record is not None,'Benchmark record missing')
            for j,name in enumerate(CONDITIONS):
                x=record['tokens'][j].astype(int)
                alternate=wide.run(x,full_probs=True,positions=np.asarray(TIMES))['probs']
                existing._normalization_error(alternate,'wide Bayes',2e-6)
                b=record['bayes_probs'][j]; true=record['true_probs'][j]
                maximum['wide_probability']=max(maximum['wide_probability'],float(np.max(np.abs(b-alternate))))
                maximum['wide_risk_bits']=max(maximum['wide_risk_bits'],float(np.max(np.abs(np.sum(true*np.log2(b/alternate),axis=-1)))))
                e=exact.run(x,full_probs=True,positions=np.asarray(TIMES))['probs']
                c=np.zeros((2,8,8),dtype=int); manual=[]
                for t in range(1,len(x)):
                    a,b0=divmod(int(x[t-1]),8)
                    if t in TIMES:
                        manual.append(np.outer((c[0,a]+1)/(c[0,a].sum()+8),(c[1,b0]+1)/(c[1,b0].sum()+8)).ravel())
                    u,v=divmod(int(x[t]),8); c[0,a,u]+=1; c[1,b0,v]+=1
                maximum['dirichlet_probability']=max(maximum['dirichlet_probability'],float(np.max(np.abs(e-np.asarray(manual)))))
    require(maximum['wide_probability']<2e-6 and maximum['wide_risk_bits']<2e-6
            and maximum['dirichlet_probability']<2e-12, 'Independent numerical validation failed')
    return dict(passed=True,maximum_errors=maximum,source_indices=[0,3],conditions=CONDITIONS,
                scope='Selected histories at every new condition; raw normalization and common-prefix causality checked for every record')


def benchmark(out):
    start=time.monotonic(); initialize(str(out))
    timings=[evaluate((pair,i)) for pair in PAIRS for i in range(4)]
    validation=validate(out)
    elapsed=time.monotonic()-start
    fresh=[t['seconds'] for t in timings if not t['reused']]
    report=dict(passed=True,plan_id=_STATE['plan']['plan_id'],validation=validation,
        source_draws=8,trajectories=32,elapsed_seconds=elapsed,
        mean_seconds_per_four_condition_draw=float(np.mean(fresh)) if fresh else None,
        projected_serial_minutes=float(np.mean(fresh))*2*_STATE['plan']['n_seq']/60 if fresh else None,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        projection_scope='Measured serial evaluation; parallel throughput still needs measurement')
    write(out/'benchmark.json',report); print(json.dumps(report),flush=True)


def run(out,workers):
    p,_,_=load(out)
    b=confirmation.read(out/'benchmark.json')
    require(b['passed'] and b['plan_id']==p['plan_id'], 'Valid benchmark required')
    all_tasks=[(pair,i) for i in range(p['n_seq']) for pair in PAIRS]
    pending=[t for t in all_tasks if existing._read_cache(record_path(out,*t),cache_id(p,*t)) is None]
    completed=len(all_tasks)-len(pending); start=time.monotonic()
    def status(state):
        payload=dict(status=state,completed_source_draws=completed,total_source_draws=len(all_tasks),
            completed_trajectories=4*completed,workers=workers,updated_utc=confirmation.utc(),
            elapsed_seconds=time.monotonic()-start,plan_id=p['plan_id'])
        write(out/'status.json',payload);print(json.dumps(payload),flush=True)
    status('running')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(str(out),)) as pool:
        futures={pool.submit(evaluate,t):t for t in pending}
        try:
            for f in as_completed(futures):
                f.result();completed+=1
                if completed%16==0 or completed==len(all_tasks):status('running')
        except BaseException:
            status('failed');
            for f in futures:f.cancel()
            raise
    status('completed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'output/product_shift_2026-09-18')
    ap.add_argument('--stage',choices=['prepare','benchmark','run'],required=True)
    ap.add_argument('--n-seq',type=int,default=1024)
    ap.add_argument('--workers',type=int,default=4)
    cli=ap.parse_args();cli.out=cli.out.resolve();cli.out.mkdir(parents=True,exist_ok=True)
    require(cli.n_seq>=4 and 1<=cli.workers<=8,'Invalid cohort or worker count')
    with (cli.out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if cli.stage=='prepare':
            plan=prepare(cli.out,cli.n_seq);print(json.dumps(dict(prepared=True,plan_id=plan['plan_id'])),flush=True)
        elif cli.stage=='benchmark':benchmark(cli.out)
        else:run(cli.out,cli.workers)


if __name__=='__main__':main()

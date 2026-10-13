"""Independent, paired-alphabet checks of one tilted distribution's Bayes cost.

OPENBLAS_NUM_THREADS=1 python code/scripts/iid_alpha15_followup.py \
  --alpha 1.5 --ds 4096 16384 65536 --reps 512 \
  --out output/iid_alpha15_followup/alpha15.json

Weights and sample uniforms share prefixes across alphabets. This makes
alphabet differences paired; the random stream is independent of the previous
campaign. Raw per-draw costs and normalizers stay beside --out in ignored output.
Source/sample draws are reproducible from (seed, L, repetition); no full dense
collection of weights or counts is retained. No chain is generated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from iid_layered_tilt_check import scalar_sum
from tilted_row_reference import DEFAULT_STORE, TiltedRowReference, zipf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--ds', type=int, nargs='+', default=[4096,16384,65536])
    ap.add_argument('--ks', type=int, nargs='+', default=[256,1024,4096,16384])
    ap.add_argument('--L', type=int, default=3, choices=range(1,54))
    ap.add_argument('--skip-original-prediction', action='store_true',
                    help='Measure costs without the archived L=1,3 scalar table.')
    ap.add_argument('--reps', type=int, default=512)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=202609068)
    ap.add_argument('--step', type=float, default=1/64)
    ap.add_argument('--store', type=Path, default=DEFAULT_STORE)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    ds, ks = sorted(set(args.ds)), sorted(set(args.ks))
    assert args.reps >= 2 and min(ds) >= 2 and min(ks) >= 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    master_path = Path('results/mechanism/master_curves.json')
    master = json.loads(master_path.read_text())
    q = [zipf(d,args.alpha) for d in ds]
    references = []
    start = time.monotonic()
    for d, qq in zip(ds,q):
        references.append(TiltedRowReference(qq,args.L,store=args.store,step=args.step))
        print(f'alpha={args.alpha} d={d}: reference ready at {time.monotonic()-start:.1f}s',flush=True)
    shape = (args.reps,len(ds),len(ks))
    redundancy, oracle, bayes, numeric = [np.zeros(shape) for _ in range(4)]
    normalizers = np.zeros((args.reps,len(ds)))
    distinct = np.zeros(shape,dtype=np.int32)
    if args.L not in (1,3) and not args.skip_original_prediction:
        ap.error('For other depths, use --skip-original-prediction and the independent depth prediction script.')
    prediction = None if args.skip_original_prediction else np.array(
        [[scalar_sum(qq,k,args.L,master)[0] for k in ks] for qq in q])

    def save(done):
        vals = redundancy[:done]
        cells=[]
        for j,d in enumerate(ds):
            mean=vals[:,j].mean(0); se=vals[:,j].std(0,ddof=1)/math.sqrt(done)
            cells.append(dict(d=d,L=args.L,alpha=args.alpha,k=ks,reps=done,
                redundancy_mean=mean.tolist(),redundancy_se=se.tolist(),
                original_scalar_sum_bits=None if prediction is None else prediction[j].tolist(),
                relative_to_original=None if prediction is None else (mean/prediction[j]-1).tolist(),
                known_source_cost_mean=oracle[:done,j].mean(0).tolist(),
                bayes_cost_mean=bayes[:done,j].mean(0).tolist(),
                distinct_symbols_mean=distinct[:done,j].mean(0).tolist(),
                normalizer_mean=float(normalizers[:done,j].mean()),
                normalizer_variance=float(normalizers[:done,j].var(ddof=1)),
                exact_normalizer_variance=float((2**args.L-1)*(q[j]@q[j])),
                maximum_outer_refinement_difference_bits=float(numeric[:done,j].max())))
        paired=[]
        for j in range(1,len(ds)):
            diff=vals[:,j]-vals[:,0]
            paired.append(dict(d_small=ds[0],d_large=ds[j],k=ks,
                mean_difference_bits=diff.mean(0).tolist(),
                difference_se_bits=(diff.std(0,ddof=1)/math.sqrt(done)).tolist()))
        result=dict(schema_version=1,complete=done==args.reps,cells=cells,
            paired_alphabet_differences=paired,se_ddof=1,seed=args.seed,
            repetition_start=args.rep_start,
            seed_protocol='Weights: default_rng([seed,L,rep,0]).exponential(size=(max(ds),L)).prod(1); uniforms: independent default_rng([seed,L,rep,1]); common prefixes across d and alpha',
            target='Expected Bayes learning cost of iid normalized tilted rows with q,L known',
            qualifications=['Alpha and alphabet comparisons share randomness; checkpoints share prefixes.',
                'Sample mean standard errors are separate from numerical integration checks.',
                'Original scalar sum is an uncorrected comparison above alpha=1.',
                'No finite-alphabet correction or leading coefficient is fitted to these measurements.'],
            master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
            evaluator_sha256=hashlib.sha256(Path(__file__).with_name('tilted_row_reference.py').read_bytes()).hexdigest(),
            command_argv=sys.argv,elapsed_seconds=time.monotonic()-start)
        tmp=args.out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.out)
        return result

    try:
        for i in range(args.reps):
            rep=args.rep_start+i
            w=np.random.default_rng([args.seed,args.L,rep,0]).exponential(size=(max(ds),args.L)).prod(1)
            uniforms=np.random.default_rng([args.seed,args.L,rep,1]).random(max(ks))
            for j,(d,qq,ref) in enumerate(zip(ds,q,references)):
                s=float(qq@w[:d]);normalizers[i,j]=s
                p=qq*w[:d]/s;cdf=np.cumsum(p);cdf[-1]=1.
                labels=np.searchsorted(cdf,uniforms,side='right')
                counts=np.zeros(d,dtype=np.int64);previous=0
                for t,k in enumerate(ks):
                    counts+=np.bincount(labels[previous:k],minlength=d);previous=k
                    lm,diag=ref.log_probability(counts)
                    observed=np.flatnonzero(counts)
                    true=float(counts[observed]@np.log(p[observed]))
                    oracle[i,j,t]=-true/math.log(2);bayes[i,j,t]=-lm/math.log(2)
                    redundancy[i,j,t]=(true-lm)/math.log(2)
                    numeric[i,j,t]=diag['coarse_fine_difference_bits']
                    distinct[i,j,t]=len(observed)
            if (i+1)%32==0 or i+1==args.reps:
                result=save(i+1)
                last=result['cells'][-1]
                print(f'alpha={args.alpha} draws={i+1}, d={last["d"]}, '
                      f'k={ks[-1]}: {last["redundancy_mean"][-1]:.3f} +/- '
                      f'{last["redundancy_se"][-1]:.3f} SE; '
                      f'{result["elapsed_seconds"]:.1f}s',flush=True)
    finally:
        for ref in references:ref.close()
    np.savez_compressed(args.out.with_suffix('.npz'),redundancy=redundancy,
        known_source_cost=oracle,bayes_cost=bayes,normalizers=normalizers,
        distinct=distinct,numerical_error=numeric,ds=ds,ks=ks)


if __name__=='__main__':
    main()

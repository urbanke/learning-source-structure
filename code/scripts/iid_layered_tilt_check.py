"""Test the single tilted-distribution Bayes cost at fixed source depth.

The measured target knows q and L. Compare it with the original scalar
sum over symbols, not with the cost of a different depth-mixture code.
Each repetition draws fresh normalized product weights and an iid path.
The full weights, counts, and per-draw results stay in ignored output/.
Small aggregate files are reproducible using the command recorded in JSON.

Example:
  python code/scripts/iid_layered_tilt_check.py --d 1024 --alpha .75 --L 3 \
    --reps 128 --ks 16,64,256,1024,4096 --out output/iid_tilt_L3/d1024_a075.json
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

from tilted_row_reference import DEFAULT_STORE, TiltedRowReference, zipf


def scalar_sum(q, k, L, master):
    """The same numerical master-curve convention as the existing figure."""
    x = k * q
    vals = np.interp(np.log(x), np.log(master['x']), master[f'r{L}_bits'])
    hi = x > master['x'][-1]
    lo = x < master['x'][0]
    vals[hi] = .5 * np.log2(x[hi]) + master['c_bits'][str(L)]
    vals[lo] = master['slope0_bits_per_sample'][str(L)] * x[lo]
    return float(vals.sum()), int(hi.sum()), int(lo.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--L', type=int, default=3, choices=[1, 3])
    ap.add_argument('--reps', type=int, default=128)
    ap.add_argument('--ks', default='16,64,256,1024,4096')
    ap.add_argument('--step', type=float, default=1/64)
    ap.add_argument('--store', type=Path, default=DEFAULT_STORE)
    ap.add_argument('--master', type=Path, default=Path('results/mechanism/master_curves.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.reps < 2:
        ap.error('at least two independent repetitions are needed')
    ks = sorted(set(int(x) for x in args.ks.split(',')))
    if min(ks) < 1:
        ap.error('sample counts must be positive')
    q = zipf(args.d, args.alpha)
    master = json.loads(args.master.read_text())
    start = time.monotonic()
    reference = TiltedRowReference(q, args.L, store=args.store, step=args.step)
    redundancy = np.zeros((args.reps, len(ks)))
    source_weights = np.zeros((args.reps, args.d))
    all_counts = np.zeros((args.reps, len(ks), args.d), dtype=np.int32)
    normalizers = np.zeros(args.reps)
    oracle = np.zeros_like(redundancy)
    bayes = np.zeros_like(redundancy)
    numeric_error = np.zeros_like(redundancy)
    predictions = [scalar_sum(q, k, args.L, master) for k in ks]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save(done, complete=False):
        r = redundancy[:done]
        se = r.std(axis=0, ddof=1)/math.sqrt(done) if done > 1 else np.zeros(len(ks))
        pred = np.array([x[0] for x in predictions])
        result = {
            'd': args.d, 'alpha': args.alpha, 'L': args.L, 'k': ks,
            'reps': done, 'requested_reps': args.reps, 'complete': complete,
            'se_ddof': 1,
            'seed_protocol': 'default_rng([20260906, 927, L, repetition]); same uniforms and weight prefixes at a common d across alpha',
            'target': 'Bayes sequence code for one iid normalized tilted row; q and source depth L known',
            'prediction': 'sum_b r_L(k q_b), using the unchanged saved master table and its original interpolation/tail convention',
            'prediction_scope': 'diffuse-row approximation for alpha<=1 at k=O(d); uncorrected comparison only for alpha>1',
            'redundancy_mean': r.mean(axis=0).tolist(),
            'redundancy_se': se.tolist(),
            'poisson_comparison_bits': pred.tolist(),
            'difference_bits': (r.mean(axis=0)-pred).tolist(),
            'relative_difference': (r.mean(axis=0)/pred-1).tolist(),
            'known_source_cost_mean': oracle[:done].mean(axis=0).tolist(),
            'bayes_code_cost_mean': bayes[:done].mean(axis=0).tolist(),
            'max_coarse_fine_difference_bits': float(numeric_error[:done].max()),
            'normalizer_mean': float(normalizers[:done].mean()),
            'normalizer_variance': float(normalizers[:done].var(ddof=1)) if done > 1 else None,
            'exact_normalizer_variance': float((2**args.L-1)*(q@q)),
            'sum_q_squared': float(q@q),
            'large_x_extrapolated_symbols': [x[1] for x in predictions],
            'small_x_extrapolated_symbols': [x[2] for x in predictions],
            'master_sha256': hashlib.sha256(args.master.read_bytes()).hexdigest(),
            'evaluator_sha256': hashlib.sha256(Path(__file__).with_name('tilted_row_reference.py').read_bytes()).hexdigest(),
            'command_argv': sys.argv,
            'elapsed_seconds': time.monotonic()-start,
            'qualification': 'Independent rows; checkpoints share samples. Numerical refinement checks are not certified total quadrature error bounds. No chain or crossing was simulated.',
        }
        tmp = args.out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result, indent=2)+'\n')
        tmp.replace(args.out)
        return result

    try:
        for rep in range(args.reps):
            rng = np.random.default_rng([20260906, 927, args.L, rep])
            w = rng.exponential(size=(args.L, args.d)).prod(axis=0)
            source_weights[rep] = w
            z = float(q@w)
            normalizers[rep] = z
            p = q*w/z
            cdf = np.cumsum(p)
            cdf[-1] = 1.
            labels = np.searchsorted(cdf, rng.random(max(ks)), side='right')
            counts = np.zeros(args.d, dtype=np.int64)
            prev = 0
            for j, k in enumerate(ks):
                counts += np.bincount(labels[prev:k], minlength=args.d)
                prev = k
                all_counts[rep, j] = counts
                lm, diag = reference.log_probability(counts)
                known = float(counts@np.log(p))
                redundancy[rep, j] = (known-lm)/math.log(2)
                oracle[rep, j] = -known/math.log(2)
                bayes[rep, j] = -lm/math.log(2)
                numeric_error[rep, j] = diag['coarse_fine_difference_bits']
            if (rep+1) % 16 == 0 or rep+1 == args.reps:
                result = save(rep+1, rep+1 == args.reps)
                print(f'd={args.d} alpha={args.alpha} L={args.L} draws={rep+1}: '
                      f'last={result["redundancy_mean"][-1]:.3f} +/- '
                      f'{result["redundancy_se"][-1]:.3f} SE; '
                      f'comparison={predictions[-1][0]:.3f}; '
                      f'{result["elapsed_seconds"]:.1f}s', flush=True)
    finally:
        reference.close()
    np.savez_compressed(args.out.with_suffix('.npz'), source_weights=source_weights,
                        counts=all_counts, redundancy=redundancy, normalizers=normalizers,
                        known_source_cost=oracle, bayes_code_cost=bayes,
                        numerical_refinement_error=numeric_error)


if __name__ == '__main__':
    main()

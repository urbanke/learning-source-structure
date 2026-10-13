"""Finite-alphabet, random-normalizer diagnostics for one depth-three iid row.

    python code/scripts/iid_tilt_finite_prediction.py \
        --out results/mechanism/iid_tilt_finite_prediction.json
    python code/scripts/iid_tilt_finite_prediction.py --smoke --out /tmp/finite-smoke.json

No Bayes-cost samples are used to fit this prediction. Independent Monte Carlo
draws generate Z_d=sum_{b<=d} b^-alpha W_b, with W_b a product of three Exp(1)
variables. The diagnostic is E F_d(k/Z_d), F_d(t)=sum_{b<=d} r3(t*b^-alpha).
It treats the random denominator as an independent common rate scale; it is
not an identity for the normalized row's Bayes redundancy. In particular,
conditioning on its actual denominator changes the law of the individual W_b.

For alpha>1 we also compute E F_infinity(k/Z_d) and E J(k/Z_d)^(1/alpha).
Their differences separate the missing rare-symbol tail and replacing a
discrete sum by its leading integral, within this diagnostic model. The
independent infinite-normalizer coefficient is evaluated by the Laplace
integral used in iid_tilt_tail_check.py. Alpha=1 has only finite predictions.

The original master table, interpolation, and tail conventions are unchanged.
Power-law rank intervals let every scalar sum be evaluated in O(table size),
without building an array of size draws times alphabet size for each sample
count. Normalizers use bounded batches. No Markov chain is simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import digamma, expit, gammaln, zeta

from iid_tilt_tail_check import integrate_master, log_product_density

ROOT = Path(__file__).resolve().parents[2]
SEED = [20260906, 1937, 3]


def rank_sum(t, alpha, d, master):
    """Exactly sum the original piecewise-linear-in-log scalar convention."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x = np.asarray(master['x']); logx = np.log(x)
    values = np.asarray(master['r3_bits'])
    slope = np.diff(values)/np.diff(logx)
    logt = np.log(t)
    ranks = np.floor(np.exp((logt[:, None]-logx)/alpha))
    if d is not None:
        ranks = np.minimum(ranks, d)
    assert ranks.max() < 2**52
    counts = ranks[:, :-1]-ranks[:, 1:]
    logfactorial = gammaln(ranks+1)
    result = np.sum(counts*(values[:-1]+slope*(logt[:, None]-logx[:-1]))
                    -alpha*slope*(logfactorial[:, :-1]-logfactorial[:, 1:]), axis=1)
    high = ranks[:, -1]
    result += high*(.5*logt/math.log(2)+master['c_bits']['3'])
    result -= .5*alpha/math.log(2)*logfactorial[:, -1]
    first = ranks[:, 0]+1
    if alpha == 1:
        assert d is not None
        missing = digamma(d+1)-digamma(first)
    else:
        missing = zeta(alpha, first)
        if d is not None:
            missing -= zeta(alpha, d+1)
    result += master['slope0_bits_per_sample']['3']*t*missing
    return result


def direct_sum(t, alpha, d, master):
    x = t*np.arange(1, d+1, dtype=float)**(-alpha)
    values = np.interp(np.log(x), np.log(master['x']), master['r3_bits'])
    lo, hi = x < master['x'][0], x > master['x'][-1]
    values[lo] = master['slope0_bits_per_sample']['3']*x[lo]
    values[hi] = .5*np.log2(x[hi])+master['c_bits']['3']
    return float(values.sum())


def validate(master):
    checks = []
    for alpha in (1., 1.4, 1.5, 1.6, 2.):
        for d in (16, 4096, 65536):
            for t in (1., 100., 10000., 1000000.):
                fast = float(rank_sum([t], alpha, d, master)[0])
                direct = direct_sum(t, alpha, d, master)
                assert abs(fast-direct) < 2e-7, (alpha, d, t, fast-direct)
                checks.append(dict(alpha=alpha, d=d, t=t, error_bits=fast-direct))
    return dict(direct_sum_checks=checks,
                maximum_error_bits=max(abs(r['error_bits']) for r in checks))


def infinite_moments(alphas, step=1/128, head=8192):
    """Independent deterministic E Z_infinity^-beta; no cost observations."""
    _, _, grid, product = log_product_density(step)
    u = np.arange(-9, 14+step, step)
    lp = []
    for i in range(0, len(u), 64):
        lp.extend(np.log(expit(-(u[i:i+64, None]+grid))@product))
    spline = CubicSpline(u, lp)

    def log_phi(v):
        small = v < -9; out = np.empty_like(v); x = np.exp(v[small])
        out[small] = np.log1p(-x+4*x*x-36*x**3+576*x**4)
        out[~small] = spline(v[~small])
        return out

    output = []
    for alpha in alphas:
        if alpha <= 1:
            continue
        beta = 1/alpha
        loga = -alpha*np.log(np.arange(1, head+1, dtype=float))
        tails = [float(zeta(j*alpha, head+1)) for j in range(1, 5)]
        def prefix(v):
            return beta*v+log_phi(v+loga).sum()
        def kernel(v):
            x = math.exp(v)
            return (prefix(v)-x*tails[0]+3.5*x*x*tails[1]
                    -97/3*x**3*tails[2]+2143/4*x**4*tails[3])
        hi = 0.
        while prefix(hi) > -48:
            hi += .5
        assert math.exp(hi)*(head+1)**(-alpha) < .005
        val, error = quad(lambda v: math.exp(kernel(v)), -60/beta, hi,
                          epsabs=2e-11, epsrel=2e-11, limit=200)
        output.append(dict(alpha=alpha, beta=beta, moment=val/math.gamma(beta),
                           quadrature_error=error/math.gamma(beta), step=step, head=head))
    return output


def sample_normalizers(ds, alphas, draws, batch):
    # Separate rank-block streams preserve the original <=65536 predictions
    # when extending this campaign to larger alphabets.
    block = 65536
    streams = [np.random.default_rng(SEED if j == 0 else SEED+[j])
               for j in range((max(ds)+block-1)//block)]
    weights = {a: np.arange(1, max(ds)+1, dtype=float)**(-a) for a in alphas}
    output = {(a, d): np.empty(draws) for a in alphas for d in ds}
    for start in range(0, draws, batch):
        stop = min(draws, start+batch)
        totals = {a: np.zeros(stop-start) for a in alphas}
        for j, rng in enumerate(streams):
            first, last = j*block, min((j+1)*block, max(ds))
            g = rng.exponential(size=(stop-start, last-first))
            g *= rng.exponential(size=g.shape)
            g *= rng.exponential(size=g.shape)
            for alpha in alphas:
                for d in ds:
                    if first < d <= last:
                        output[alpha, d][start:stop] = (totals[alpha]
                            +g[:, :d-first]@weights[alpha][first:d])
                totals[alpha] += g@weights[alpha][first:last]
    return output


def summarize(values):
    return dict(mean_bits=float(np.mean(values)),
                SE_bits=float(np.std(values, ddof=1)/math.sqrt(len(values))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--master', type=Path, default=ROOT/'results/mechanism/master_curves.json')
    ap.add_argument('--ds', nargs='+', type=int, default=[4096, 16384, 65536, 262144])
    ap.add_argument('--alphas', nargs='+', type=float, default=[1., 1.4, 1.5, 1.6, 2.])
    ap.add_argument('--ks', nargs='+', type=int, default=[1024, 4096, 16384])
    ap.add_argument('--draws', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.ds, args.alphas, args.ks, args.draws = [128, 512], [1., 1.5, 2.], [64, 256], 64
    assert args.draws >= 2 and min(args.ds) >= 2 and min(args.ks) > 0 and min(args.alphas) >= 1
    started = time.monotonic()
    master = json.loads(args.master.read_text())
    validation = validate(master)
    moments = infinite_moments(args.alphas, 1/64 if args.smoke else 1/128,
                               4096 if args.smoke else 8192)
    for item in moments:
        item['J_bits'] = integrate_master(master, item['alpha'], master['x'][-1])['J_bits']
        item['coefficient_bits'] = item['J_bits']*item['moment']
    normalizers = sample_normalizers(args.ds, args.alphas, args.draws, args.batch)
    rows = []
    finite_draws = {}
    for alpha in args.alphas:
        for d in args.ds:
            z = normalizers[alpha, d]
            h = float(np.sum(np.arange(1, d+1, dtype=float)**(-alpha)))
            for k in args.ks:
                t = k/z
                finite = rank_sum(t, alpha, d, master)
                finite_draws[alpha, d, k] = finite
                original = float(rank_sum([k/h], alpha, d, master)[0])
                rec = dict(alpha=alpha, d=d, k=k, L=3, draws=args.draws,
                    original_sum_bits=original, finite_random_scale=summarize(finite),
                    random_scale_change_from_original=summarize(finite-original),
                    normalizer_mean=float(z.mean()), exact_normalizer_mean=h,
                    normalizer_mean_SE=float(z.std(ddof=1)/math.sqrt(args.draws)))
                if alpha > 1:
                    c = next(m for m in moments if m['alpha'] == alpha)
                    infinite = rank_sum(t, alpha, None, master)
                    leading = c['J_bits']*t**c['beta']
                    rec.update(infinite_sum_same_random_scale=summarize(infinite),
                        integral_same_random_scale=summarize(leading),
                        omitted_rare_symbol_tail=summarize(infinite-finite),
                        sum_to_integral_difference=summarize(leading-infinite),
                        infinite_normalizer_leading_bits=c['coefficient_bits']*k**c['beta'],
                        finite_normalizer_change_in_leading=summarize(leading-c['coefficient_bits']*k**c['beta']),
                        k_over_d_alpha=k/d**alpha)
                rows.append(rec)
            last = rows[-1]
            print('alpha',alpha,'d',d,'k',last['k'],'finite prediction',last['finite_random_scale'],
                  'old',last['original_sum_bits'],flush=True)
    alphabet_differences = []
    ordered_ds = sorted(args.ds)
    for alpha in args.alphas:
        for k in args.ks:
            for d0, d1 in zip(ordered_ds[:-1], ordered_ds[1:]):
                alphabet_differences.append(dict(alpha=alpha, k=k, L=3,
                    d_from=d0, d_to=d1,
                    prediction_change=summarize(finite_draws[alpha, d1, k]
                                                -finite_draws[alpha, d0, k])))
    result = dict(schema_version=1, source_depth=3, seed=SEED, batch=args.batch,
        normalizer_rank_block_size=65536,
        normalizer_seed_protocol='Rank block 0 uses seed; block j>0 uses seed+[j]. Within each batch, three independent exponential arrays are multiplied.',
        draws=args.draws, ds=args.ds, alphas=args.alphas, ks=args.ks, smoke=args.smoke,
        master_sha256=hashlib.sha256(args.master.read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,
        estimand='E sum_{b<=d} r3(k b^-alpha/Z_d), where Z_d is independently sampled from the true prior normalizer distribution.',
        qualifications=[
            'This is a random-rate diagnostic, not an exact finite-sample Bayes redundancy. It neglects the dependence between the denominator and the coordinate weights in the true row experiment.',
            'No Bayes-cost observations are fitted or used to choose a correction coefficient.',
            'The same independent product-weight prefixes are used across alphabets and exponents. Within a cell all sample counts share the same normalizer draws.',
            'SEs describe independent normalizer Monte Carlo draws; differences within a row use paired draws.',
            'The original master interpolation and small/large-x continuations are retained. Numerical integration and master-table errors are separate from the reported Monte Carlo SEs.',
            'For alpha>1, omitted tail is the difference of two scalar sums at exactly the same random scale; the sum-to-integral difference is recorded separately.',
            'For alpha=1 the infinite scalar sum and its power-law leading integral diverge, so only the finite prediction is reported.',
            'Only iid-row predictions are computed. No Markov chain is simulated.'],
        validation=validation, infinite_leading_constants=moments, rows=rows,
        alphabet_differences=alphabet_differences)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('wrote',args.out,'seconds',result['elapsed_seconds'],flush=True)


if __name__ == '__main__':
    main()

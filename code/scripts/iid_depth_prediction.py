"""Independent scalar information and finite iid-row predictions at fixed depth.

    OPENBLAS_NUM_THREADS=1 python code/scripts/iid_depth_prediction.py --out results/mechanism/iid_depth_prediction.json
    OPENBLAS_NUM_THREADS=1 python code/scripts/iid_depth_prediction.py --smoke --out /tmp/iid-depth-smoke.json

Integrating the final exponential makes the count law a mixture of geometric
laws. Its entropy is evaluated by direct small-count summation and a smooth
count-tail integral with Euler--Maclaurin correction. Conditional Poisson
entropy is integrated separately over the product-weight law. No phi store
or observed learning costs are read. Refinements are numerical checks, not
certified total error bounds. The original master table is only a validation
input and is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
from scipy.special import gammaln, logsumexp, expit

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)
GAMMA = 0.5772156649015329


def product_log_density(depth, step, relative_cut=1e-15):
    if depth == 0:
        return np.array([0.]), np.array([1.])
    lo = -45.
    s = np.arange(lo, 9., step)
    f = np.exp(s-np.exp(s))*step
    out = f.copy()
    for _ in range(depth-1):
        out = fftconvolve(out, f)
    grid = depth*lo+step*np.arange(len(out))
    # Trim numerical tails below the relative cutoff. The mathematical
    # product-log density has full real support; this is a truncation.
    keep = out > out.max()*relative_cut
    first, last = np.flatnonzero(keep)[[0, -1]]
    out = np.maximum(out[first:last+1], 0.)
    out /= out.sum()
    return grid[first:last+1], out


class ScalarInformation:
    def __init__(self, depth, *, step=1/64, count_start=128, tail_order=192,
                 relative_cut=1e-15):
        self.depth, self.step = depth, step
        self.count_start, self.tail_order = count_start, tail_order
        self.relative_cut = relative_cut
        self.sh, self.mh = product_log_density(depth-1, step, relative_cut)
        self.sw, self.mw = product_log_density(depth, step, relative_cut)
        self.logmh = np.log(self.mh)
        self.nodes, self.weights = leggauss(tail_order)
        # Exact finite Poisson sum on a dense log-rate grid; the remaining
        # tail at the largest rate is negligible. Large rates use the
        # standard asymptotic entropy expansion through inverse fourth order.
        v = np.arange(-30., math.log(500.)+1/256, 1/256)
        h = np.empty_like(v)
        n = np.arange(1200, dtype=float)
        for start in range(0, len(v), 128):
            vv = v[start:start+128]
            lp = n[None, :]*vv[:, None]-np.exp(vv[:, None])-gammaln(n+1)
            h[start:start+len(vv)] = -np.sum(np.exp(lp)*lp, axis=1)
        self.poi_entropy = CubicSpline(v, h)

    def poisson_entropy(self, v):
        v = np.asarray(v)
        out = np.empty_like(v)
        lo, hi = v < -30., v > math.log(500.)
        out[lo] = np.exp(v[lo])*(1-v[lo])
        out[~(lo|hi)] = self.poi_entropy(v[~(lo|hi)])
        inv = np.exp(-v[hi])
        out[hi] = .5*(math.log(2*math.pi*math.e)+v[hi])-inv/12-inv**2/24-19*inv**3/360-3*inv**4/16
        return out

    def __call__(self, x, *, geometric_closed_form=True):
        v = math.log(x)
        conditional = float(self.mw@self.poisson_entropy(v+self.sw))
        if self.depth == 1 and geometric_closed_form:
            marginal = math.log1p(x)+x*math.log1p(1/x)
            return (marginal-conditional)/LN2
        logq = -np.logaddexp(0., -v-self.sh)
        logone = -np.logaddexp(0., v+self.sh)
        logbase = self.logmh+logone

        def count_entropy(n, derivative=False):
            n = np.atleast_1d(n)
            lp = logbase[None, :]+n[:, None]*logq
            logp = logsumexp(lp, axis=1)
            p = np.exp(logp)
            if derivative:
                weights = np.exp(lp-logp[:, None])
                m1 = weights@logq
                m2 = weights@(logq**2)
                m3 = weights@(logq**3)
                f1 = -p*m1*(logp+1)
                f3 = -p*(m3*(logp+1)+3*m1*m2-m1**3)
                return f1, f3
            return -p*logp

        n0 = self.count_start
        small = float(count_entropy(np.arange(n0)).sum())
        # Beyond this endpoint every geometric component has exponentially
        # small count probability. This uses the retained density support.
        lower = math.log(n0)
        upper = max(lower+1., v+float(self.sh[-1])+math.log(80.))
        z = (upper+lower)/2+(upper-lower)/2*self.nodes
        n = np.exp(z)
        tail = float(self.weights@(count_entropy(n)*n))*(upper-lower)/2
        f1, f3 = count_entropy([n0], True)
        tail += .5*float(count_entropy([n0])[0])-float(f1[0])/12+float(f3[0])/720
        return (small+tail-conditional)/LN2

    def diagnostics(self):
        return dict(depth=self.depth, log_weight_grid_step=self.step,
            count_start=self.count_start, tail_quadrature_order=self.tail_order,
            density_relative_cutoff=self.relative_cut,
            weight_mass=float(self.mw.sum()), mean_weight=float(self.mw@np.exp(self.sw)),
            mean_log_weight=float(self.mw@self.sw), exact_mean_log_weight=-self.depth*GAMMA)

    def log_count_probability(self, x, n):
        v = math.log(x)
        return float(logsumexp(self.logmh-np.logaddexp(0., v+self.sh)
                              -n*np.logaddexp(0., -v-self.sh)))

    def tail_constant(self):
        hlog = -float(self.mw@np.log(self.mw/self.step))
        return (hlog+.5*float(self.mw@self.sw)-.5*math.log(2*math.pi*math.e))/LN2


def scalar_validation(master, depths):
    from iid_tilt_coverage_check import scalar_r1
    checks = []
    for depth in depths:
        calc = ScalarInformation(depth)
        refined = ScalarInformation(depth, step=1/128, count_start=256, tail_order=256,
                                    relative_cut=1e-14)
        xs = [.000001, .001, .1, 1., 10., 100., 1000., 100000., 10000000.]
        exact_one = scalar_r1(np.array(xs), 256) if depth == 1 else None
        for i, x in enumerate(xs):
            a, b = calc(x), refined(x)
            check = dict(L=depth, x=x, value_bits=a, refined_bits=b,
                         refinement_difference_bits=b-a)
            if exact_one is not None:
                check['independent_depth_one_integral_difference_bits'] = a-float(exact_one[i])
                check['geometric_entropy_sum_difference_bits'] = calc(x, geometric_closed_form=False)-a
                assert abs(check['geometric_entropy_sum_difference_bits']) < 2e-8
            assert abs(a-b) < 2e-8, check
            checks.append(check)
        if depth in [1, 3]:
            for i in [0, 20, 40, 60, 80, 100, 120, 140]:
                x = master['x'][i]
                checks.append(dict(L=depth, x=x,
                    original_master_knot_difference_bits=calc(x)-master[f'r{depth}_bits'][i]))
    return checks


def sealed_probability_validation(depths):
    from tilted_row_reference import configure_store, DEFAULT_STORE
    configure_store(DEFAULT_STORE)
    from lsa.universal_tables import UniversalTables
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    checks = []
    try:
        for depth in depths:
            if depth == 1:
                continue
            calc = ScalarInformation(depth)
            for x in [.001, .1, 1., 10., 1000.]:
                for n in [0, 1, 5, 64, 512]:
                    a = calc.log_count_probability(x, n)
                    b = float(tables.log_phi(depth, n, np.array([math.log(x)]))[0]
                              +n*math.log(x)-gammaln(n+1))
                    probability_error = math.exp(a)-math.exp(b)
                    check = dict(L=depth, x=x, n=n, sealed_probability=math.exp(b),
                                 log_probability_difference_nats=a-b,
                                 probability_difference=probability_error)
                    # Tiny probabilities beyond the retained density support
                    # need not have accurate relative error; their absolute
                    # contribution is tested, and is negligible for entropy.
                    assert abs(probability_error) < 2e-11, check
                    if b > math.log(1e-12):
                        assert abs(a-b) < 2e-5, check
                    checks.append(check)
    finally:
        tables.close()
    return checks


def sum_curve(t, alpha, d, logx, values, slope0, tail_constant):
    """Sum the interpolated curve over finite power-law ranks, in batches."""
    t = np.atleast_1d(t)
    harmonic = np.r_[0., np.cumsum(np.arange(1, d+1, dtype=float)**(-alpha))]
    slope = np.diff(values)/np.diff(logx)
    result = np.empty(len(t))
    for start in range(0, len(t), 128):
        tt = t[start:start+128]
        logt = np.log(tt)
        ranks = np.minimum(np.floor(np.exp(np.minimum((logt[:, None]-logx)/alpha, 50.))), d)
        counts = ranks[:, :-1]-ranks[:, 1:]
        lf = gammaln(ranks+1)
        r = np.sum(counts*(values[:-1]+slope*(logt[:, None]-logx[:-1]))
                   -alpha*slope*(lf[:, :-1]-lf[:, 1:]), axis=1)
        r += ranks[:, -1]*(.5*logt/LN2+tail_constant)-.5*alpha/LN2*lf[:, -1]
        r += slope0*tt*(harmonic[d]-harmonic[ranks[:, 0].astype(int)])
        result[start:start+len(tt)] = r
    return result


def statistics(values):
    return dict(mean_bits=float(np.mean(values)),
                SE_bits=float(np.std(values, ddof=1)/math.sqrt(len(values))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--depths', nargs='+', type=int, default=[1, 3, 6])
    ap.add_argument('--ds', nargs='+', type=int, default=[256, 4096])
    ap.add_argument('--alphas', nargs='+', type=float, default=[.5, 1., 1.5, 2.])
    ap.add_argument('--ks', nargs='+', type=int, default=[16, 64, 256, 1024, 4096, 16384])
    ap.add_argument('--draws', type=int, default=8192)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.ds, args.ks, args.draws = [64, 256], [16, 64], 64
    assert min(args.depths) >= 1 and min(args.alphas) > 0 and args.draws >= 2
    start = time.monotonic()
    master_path = ROOT/'results/mechanism/master_curves.json'
    master = json.loads(master_path.read_text())
    validation = scalar_validation(master, args.depths)
    sealed = sealed_probability_validation(args.depths)
    rows, curves, sum_checks = [], {}, []
    xs = np.geomspace(1e-8, 1e8, 1025)
    logx = np.log(xs)
    for depth in args.depths:
        calc = ScalarInformation(depth)
        rr = np.array([calc(float(x)) for x in xs])
        assert np.all(np.isfinite(rr)) and np.all(rr > 0) and np.all(np.diff(rr) > 0)
        slope0, tail_constant = depth*(1-GAMMA)/LN2, calc.tail_constant()
        curves[str(depth)] = dict(x=xs.tolist(), r_bits=rr.tolist(),
            slope0_bits=slope0, tail_constant_bits=tail_constant,
            diagnostics=calc.diagnostics())
        print('scalar curve complete L',depth,'seconds',time.monotonic()-start,flush=True)
        rng = np.random.default_rng([20260906, 3017, depth])
        w = np.ones((args.draws, max(args.ds)))
        for _ in range(depth):
            w *= rng.exponential(size=w.shape)
        for alpha in args.alphas:
            a = np.arange(1, max(args.ds)+1, dtype=float)**(-alpha)
            for d in args.ds:
                z = w[:, :d]@a[:d]
                h = float(a[:d].sum())
                for k in args.ks:
                    t = k/z
                    fine = sum_curve(t, alpha, d, logx, rr, slope0, tail_constant)
                    coarse = sum_curve(t, alpha, d, logx[::2], rr[::2], slope0, tail_constant)
                    original = float(sum_curve([k/h], alpha, d, logx, rr, slope0, tail_constant)[0])
                    direct = float(np.interp(np.log(k/h*a[:d]), logx, rr).sum())
                    assert abs(original-direct) < 2e-6, (depth, d, alpha, k, original-direct)
                    sum_checks.append(dict(L=depth,d=d,alpha=alpha,k=k,error_bits=original-direct))
                    rows.append(dict(L=depth,d=d,alpha=alpha,k=k,draws=args.draws,
                        finite_random_scale=statistics(fine),original_sum_bits=original,
                        interpolation_refinement=statistics(fine-coarse),
                        normalizer_mean=float(z.mean()),
                        normalizer_mean_SE=float(z.std(ddof=1)/math.sqrt(args.draws)),
                        exact_normalizer_mean=h))
                print('L',depth,'alpha',alpha,'d',d,'k',args.ks[-1],
                      rows[-1]['finite_random_scale'],flush=True)
        del w
    result = dict(schema_version=1,source_depths=args.depths,ds=args.ds,
        alphas=args.alphas,ks=args.ks,draws=args.draws,smoke=args.smoke,
        seed_protocol='default_rng([20260906,3017,L]); multiply L exponential arrays of shape(draws,max(ds)). Independent of cost draws; common weight prefixes across d and alpha at each depth.',
        estimand='T_d(k)=E sum_{b<=d} r_L(k b^-alpha/Z_d), Z_d=sum_{b<=d} b^-alpha W_(L,b).',
        qualifications=[
            'The finite prediction is an approximation, not an exact finite-sample Bayes identity. It has a uniform additive asymptotic justification for fixed alpha>1 and fixed L; alpha<=1 comparisons remain numerical tests here.',
            'No observed Bayes costs are read or fitted. Prediction SEs are from independent prior-normalizer Monte Carlo draws.',
            'Each numerical scalar curve is evaluated independently by entropy integration; no phi table is built. The read-only sealed store is used only for probability validation.',
            'Scalar integration refinement, interpolation refinement, and Monte Carlo standard errors measure distinct sources of numerical uncertainty. None is a certified total error bound.',
            'The original master table is unchanged; original_sum_bits means the uncorrected sum using this newly evaluated scalar curve, not interpolation of the older sparse table.',
            'No Markov chain or crossing is simulated.'],
        original_master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
        scalar_validation=validation,sealed_probability_validation=sealed,
        sum_validation=sum_checks,scalar_curves=curves,rows=rows,
        elapsed_seconds=time.monotonic()-start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('wrote',args.out,'seconds',result['elapsed_seconds'],flush=True)


if __name__ == '__main__':
    main()

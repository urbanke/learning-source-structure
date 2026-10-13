"""Compare saved L3 iid-row data with the above-one leading asymptotic.

No source rows are simulated and no parameters are fitted to the observations.
The inputs are small committed JSON files; the raw sample archive is unnecessary.

    python code/scripts/iid_tilt_tail_check.py \
        --out results/mechanism/iid_tilt_tail_check.json
    python code/scripts/iid_tilt_tail_check.py --smoke --out /tmp/tail-smoke.json

For beta=1/alpha, the leading coefficient is J*E[Z_infinity^-beta],
where J=integral_0^infinity r3(t^-alpha)dt and
Z_infinity=sum_b b^-alpha W_b, with W a product of three Exp(1) variables.
J uses the original master table and its original head/tail continuation.
The independent inverse moment uses
 E[Z^-beta] = integral_0^infinity s^(beta-1) prod_b phi3(s*b^-alpha)ds/Gamma(beta).
Here phi3(s)=E[1/(1+s*E1*E2)]; log-exponential convolution evaluates it.
The log-Laplace expansion sums the infinite tail beyond an explicit head.
Grid, head, and master-tail refinement checks report numerical sensitivity;
they are not certified total error bounds. All information quantities are bits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
from scipy.special import expit, zeta

ROOT = Path(__file__).resolve().parents[2]


def log_product_density(step):
    s = np.arange(-40, 8, step)
    mass = np.exp(s-np.exp(s)); mass /= mass.sum()
    product = np.maximum(fftconvolve(mass, mass), 0)
    product /= product.sum()
    grid = 2*s[0]+step*np.arange(len(product))
    return s, mass, grid, product


def inverse_moments(step, head):
    _, _, grid, product = log_product_density(step)
    u = np.arange(-9, 14+step, step)
    log_values = []
    for i in range(0, len(u), 64):
        log_values.extend(np.log(expit(-(u[i:i+64, None]+grid))@product))
    spline = CubicSpline(u, log_values)

    def log_phi(v):
        v = np.asarray(v); small = v < -9; result = np.empty_like(v)
        x = np.exp(v[small])
        # E[W^j]/j! = (j!)^2 for the product of three exponentials.
        result[small] = np.log1p(-x+4*x*x-36*x**3+576*x**4)
        result[~small] = spline(v[~small])
        return result

    results = []
    for alpha in (1.5, 2.):
        beta = 1/alpha
        log_weights = -alpha*np.log(np.arange(1, head+1, dtype=float))
        tails = [float(zeta(j*alpha, head+1)) for j in range(1, 5)]

        def prefix(v):
            return beta*v+log_phi(v+log_weights).sum()

        def kernel(v):
            x = math.exp(v)
            # log(phi3(x))=-x+7x^2/2-97x^3/3+2143x^4/4+O(x^5).
            tail = (-x*tails[0]+3.5*x*x*tails[1]
                    -97/3*x**3*tails[2]+2143/4*x**4*tails[3])
            return prefix(v)+tail

        hi = 0.
        while prefix(hi) > -48:
            hi += .5
        assert math.exp(hi)*(head+1)**(-alpha) < .005
        value, error = quad(lambda v: math.exp(kernel(v)), -60/beta, hi,
                            epsabs=2e-11, epsrel=2e-11, limit=200)
        results.append(dict(alpha=alpha, beta=beta, moment=value/math.gamma(beta),
            quad_error=error/math.gamma(beta), head=head, grid_step=step, hi=hi,
            maximum_tail_argument=math.exp(hi)*(head+1)**(-alpha)))
    return results


def validate_laplace(step):
    s, mass, grid2, product2 = log_product_density(step)
    product3 = np.maximum(fftconvolve(product2, mass), 0)
    product3 /= product3.sum()
    grid3 = 3*s[0]+step*np.arange(len(product3))
    checks = []
    for x in (.001, .1, 1., 10., 100., 1000.):
        a = float(product2@expit(-(math.log(x)+grid2)))
        b = float(product3@np.exp(-np.exp(math.log(x)+grid3)))
        assert abs(a-b) < 1e-12
        checks.append(dict(laplace_argument=x, final_exponential_integral=a,
            full_logweight_integral=b, absolute_difference=abs(a-b)))
    return checks


def integrate_master(master, alpha, cutoff):
    x = np.asarray(master['x']); r = np.asarray(master['r3_bits'])
    c = master['c_bits']['3']; a = master['slope0_bits_per_sample']['3']
    beta = 1/alpha
    xs = np.r_[x[x < cutoff], cutoff]
    rs = np.interp(np.log(xs), np.log(x), r); v = np.log(xs)
    head = beta*a*xs[0]**(1-beta)/(1-beta)
    middle = 0.
    # beta*integral r(exp(v))*exp(-beta*v)dv, exactly on each linear segment.
    for i in range(len(v)-1):
        slope = (rs[i+1]-rs[i])/(v[i+1]-v[i])
        intercept = rs[i]-slope*v[i]
        def primitive(t):
            return -math.exp(-beta*t)*(intercept+slope*t+slope/beta)
        middle += primitive(v[i+1])-primitive(v[i])
    tail = cutoff**(-beta)*(.5*math.log2(cutoff)+c+.5/(beta*math.log(2)))
    return dict(cutoff=cutoff, J_bits=head+middle+tail,
                head_part=head, middle_part=middle, tail_part=tail)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--master', type=Path, default=ROOT/'results/mechanism/master_curves.json')
    ap.add_argument('--measurements', type=Path, default=ROOT/'results/mechanism/iid_layered_tilt_check.json')
    ap.add_argument('--normalizers', type=Path, default=ROOT/'results/mechanism/context_concentrated_check.json')
    args = ap.parse_args()
    master = json.loads(args.master.read_text())
    measured = json.loads(args.measurements.read_text())['cells']
    archived = json.loads(args.normalizers.read_text())['normalizers']
    coarse = inverse_moments(1/32 if args.smoke else 1/64, 1024 if args.smoke else 4096)
    fine = inverse_moments(1/64 if args.smoke else 1/128, 4096 if args.smoke else 8192)
    constants, comparisons = [], []
    for alpha in (1.5, 2.):
        beta = 1/alpha
        js = [integrate_master(master, alpha, cut) for cut in (75, 150, 300)]
        j = js[-1]['J_bits']
        inv = next(r for r in fine if r['alpha'] == alpha)
        previous = next(r for r in coarse if r['alpha'] == alpha)
        old = next(r for r in archived if r['alpha'] == alpha and r['L'] == 3 and r['d'] == 2048)
        coefficient = j*inv['moment']
        assert abs(previous['moment']-inv['moment']) < 1e-7
        constants.append(dict(alpha=alpha, beta=beta, J_from_original_master=js,
            inverse_normalizer=inv, inverse_moment_coarse_fine_difference=previous['moment']-inv['moment'],
            coefficient_bits=coefficient,
            normalizer_correction_to_leading_coefficient=inv['moment']*float(zeta(alpha, 1))**beta,
            archived_finite2048_inverse_moment=old['unnormalized_inverse_moment_mean'],
            archived_finite2048_inverse_moment_SE=old['unnormalized_inverse_moment_se']))
        data = next(r for r in measured if r['d'] == 4096 and r['alpha'] == alpha and r['L'] == 3)
        assert data['complete']
        d = data['d']
        for k in (4096, 16384):
            idx = data['k'].index(k)
            mean, se = data['redundancy_mean'][idx], data['redundancy_se'][idx]
            original = data['poisson_comparison_bits'][idx]; prediction = coefficient*k**beta
            comparisons.append(dict(alpha=alpha, d=d, k=k, L=3, reps=data['reps'],
                measured_mean_bits=mean, measured_SE_bits=se, original_sum_r3_bits=original,
                original_relative_to_mean=original/mean-1,
                normalizer_aware_infinite_leading_bits=prediction,
                leading_relative_to_mean=prediction/mean-1,
                leading_difference_in_empirical_SE=(prediction-mean)/se,
                leading_using_archived_finite2048_inverse_moment_bits=j*old['unnormalized_inverse_moment_mean']*k**beta,
                k_over_d_alpha=k/d**alpha,
                finite_cutoff_order_relative_to_k_beta=(k/d**alpha)**(1-beta),
                excluded_base_weight_mass=float(zeta(alpha, d+1)),
                excluded_mass_times_k=k*float(zeta(alpha, d+1))))
    result = dict(schema_version=1, source_depth=3, smoke=args.smoke,
        master_sha256=digest(args.master), measurements_sha256=digest(args.measurements),
        normalizer_archive_sha256=digest(args.normalizers),
        method='No parameters fitted to redundancy measurements. J uses the unchanged original master table with log-x linear interpolation, linear small-x continuation, and half-log large-x continuation. Independent E Z_infinity^-beta uses its Laplace transform, final-exponential integration, log-density convolution, and summable log-Laplace tails.',
        qualifications=[
            'The leading expression is an infinite-alphabet/growing-sample asymptotic, not an exact finite-d,k correction.',
            'Master-table tail and interpolation errors are not certified by the displayed cutoff sensitivity.',
            'The inverse moment is a numerical deterministic integral; grid/head refinements measure stability rather than certified total error.',
            'The finite-cutoff relative quantity records the order (k/d^alpha)^(1-1/alpha), without asserting a unit constant or equating it to actual error.',
            'The two checkpoints share source/sample draws. Reported SEs concern sample means, not simultaneous confidence intervals.',
            'No new source rows or chains were simulated for this comparison.'],
        constants=constants, comparisons=comparisons,
        independent_L3_Laplace_validation=validate_laplace(1/64 if args.smoke else 1/128))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()

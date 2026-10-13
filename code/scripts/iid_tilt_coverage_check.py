"""Check the genuine normalized depth-one iid row against the scalar sum.

No Markov chain is generated. Each replication draws independent exponentials,
normalizes q*E exactly, and draws one iid sample sequence. Its Bayes sequence
mixture is evaluated by direct adaptive quadrature, retaining the normalizer.
The original diffuse approximation is sum_b r1(k*q_b), including outside its
proved regime as an explicitly labelled diagnostic. For alpha>1 a separate
infinite-alphabet leading term retains E[Z_infinity**(-1/alpha)].

python code/scripts/iid_tilt_coverage_check.py --workers 2 \
    --out results/mechanism/iid_tilt_coverage_check.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, gammaln, roots_laguerre, roots_legendre, zeta

from context_row_check import log_mixture_adaptive
from necessary_examples_figure import r_star_exact_bits

LN2 = math.log(2.)
EG = .5772156649015329


def scalar_r1(x, order=128):
    """Stable exact integral, with Gauss-Laguerre quadrature after y=1-exp(-z)."""
    x = np.asarray(x, dtype=float)
    z, w = roots_laguerre(order)
    y = -np.expm1(-z)
    cy = (1/z + .5 - 1/y)/y
    remainder = w*(cy + 1/12 + y/24)
    lx = np.log(x)
    log1 = np.log1p(x)
    inv = 1/np.maximum(x, .1)
    j0 = 1-inv*log1
    j1 = .5-inv*j0
    small = x < .1
    xs = np.minimum(x, .1)
    s0, s1, power = np.zeros_like(x), np.zeros_like(x), xs.copy()
    for j in range(1, 20):
        s0 += power/(j+1)
        s1 += power/(j+2)
        power *= -xs
    j0, j1 = np.where(small, s0, j0), np.where(small, s1, j1)
    correction = expit(lx[..., None]+np.log(y))@remainder
    return (.5*log1-j0/12-j1/24+correction)/LN2


def independent_log_mixture(counts, q, order=384):
    """Direct Gauss-Legendre integration of the exact log-scale kernel."""
    k = int(sum(counts))
    if not k:
        return 0.
    logq = np.log(q)
    mult = np.asarray(counts)+1
    def kernel(u):
        return k*u-float(mult@np.logaddexp(0, u+logq))
    def slope(u):
        return k-float(mult@expit(u+logq))
    mode = brentq(slope, -40, 40)
    peak = kernel(mode)
    lo, hi = mode-1, mode+1
    while kernel(lo)-peak > -48:
        lo -= 1
    while kernel(hi)-peak > -48:
        hi += 1
    z, w = roots_legendre(order)
    u = (lo+hi)/2+(hi-lo)*z/2
    logf = k*u-np.logaddexp(0, u[:, None]+logq)@mult
    integral = (hi-lo)/2*float(w@np.exp(logf-peak))
    return float(np.asarray(counts)@logq+gammaln(np.asarray(counts)+1).sum()
                 -gammaln(k)+peak+math.log(integral))


def normalizer_coefficient(alpha, head=2048, terms=8):
    """Infinite-row leading coefficient, using a bounded Laplace-product tail.

    E Z^-beta = integral s^(beta-1) prod_b(1+s*b^-alpha)^-1 ds/Gamma(beta).
    The omitted product is expanded in alternating powers of s*b^-alpha;
    the first omitted term bounds its log error throughout the interval.
    For J=integral_0^infinity r1(t^-alpha) dt, insert the exact scalar
    integral and use integral_0^infinity dt/(t^alpha+y)
    =pi*beta*y^(beta-1)/sin(pi*beta). Substituting y=z^(1/beta)
    leaves the nonsingular integral for J evaluated below.
    """
    beta = 1/alpha
    a = np.arange(1, head+1, dtype=float)**(-alpha)
    zz = [float(zeta(alpha*j, head+1)) for j in range(1, terms+2)]
    def tail(s):
        return sum((-1)**(j+1)*s**j*zz[j-1]/j for j in range(1, terms+1))
    def kernel(u):
        s = math.exp(u)
        return beta*u-float(np.log1p(s*a).sum())-tail(s)
    hi = 0.
    while beta*hi-float(np.log1p(math.exp(hi)*a).sum()) > -48:
        hi += .5
    assert math.exp(hi)*(head+1)**(-alpha) < .5
    lo = -60/beta
    value, error = quad(lambda u: math.exp(kernel(u)), lo, hi,
                        epsabs=2e-11, epsrel=2e-11, limit=200)
    prefix_slope = beta-float((math.exp(hi)*a/(1+math.exp(hi)*a)).sum())
    upper_tail = math.exp(beta*hi-float(np.log1p(math.exp(hi)*a).sum()))/(-prefix_slope)
    lower_tail = math.exp(beta*lo)/beta
    log_tail_error = math.exp(hi*(terms+1))*zz[terms]/(terms+1)
    moment = value/math.gamma(beta)
    def bracket(y):
        if y < 1e-4:
            return .5-y/12-y*y/24-19*y**3/720
        if y >= 1:
            return 0.
        return 1/(-math.log1p(-y))-(1-y)/y
    integral, jerror = quad(lambda t: bracket(t**(1/beta)), 0, 1,
                            epsabs=2e-12, epsrel=2e-12, limit=200)
    J = math.pi*integral/(math.sin(math.pi*beta)*LN2)
    return dict(alpha=alpha, beta=beta, J_bits=J,
                inverse_normalizer_moment=moment, coefficient_bits=J*moment,
                omitted_product_log_error_bound=log_tail_error,
                inverse_moment_quadrature_error=error/math.gamma(beta),
                inverse_moment_tail_bound=(upper_tail+lower_tail)/math.gamma(beta),
                J_quadrature_error=jerror*math.pi/(math.sin(math.pi*beta)*LN2))


def validate():
    rng = np.random.default_rng(202609061)
    rows = []
    for d in (3, 16, 256, 1024, 4096):
        for k in (1, 16, 256, 4096):
            q = np.full(d, 1/d)
            counts = rng.multinomial(k, q)
            exact = float(gammaln(d)+gammaln(counts+1).sum()-gammaln(k+d))
            actual, diag = log_mixture_adaptive(counts, q)
            assert abs(actual-exact) < 5e-9
            rows.append(dict(d=d,k=k,uniform_closed_form_error_nats=actual-exact,**diag))
    direct = []
    for d in (16, 256, 1024, 4096):
        for alpha in (.75, 1., 1.5, 2.):
            q = np.arange(1,d+1,dtype=float)**(-alpha); q /= q.sum()
            for k in (16, 4096):
                p = q*rng.exponential(size=d); p /= p.sum()
                counts = rng.multinomial(k,p)
                actual, diag = log_mixture_adaptive(counts,q)
                refined, _ = log_mixture_adaptive(counts,q,epsabs=2e-13,epsrel=2e-13)
                other = independent_log_mixture(counts,q)
                assert abs(actual-other) < 1e-8
                assert abs(actual-refined) < 1e-8
                direct.append(dict(d=d,alpha=alpha,k=k,
                    legendre_difference_nats=actual-other,tolerance_difference_nats=actual-refined))
    xs = np.array([1e-5,.01,.1,1,16,64,4096])
    from master_curve import r1_geometric
    a = scalar_r1(xs); b = np.array([r1_geometric(float(x)) for x in xs])
    assert np.max(abs(a-b)) < 3e-9
    return dict(uniform_closed_form=rows,independent_quadrature=direct,
                scalar_geometric_maximum_error_bits=float(np.max(abs(a-b))))


def run_cell(d, alpha, ks, reps):
    start = time.monotonic()
    q = np.arange(1,d+1,dtype=float)**(-alpha); q /= q.sum()
    values = np.zeros((reps,len(ks)))
    normalizers = []
    max_error, max_tail = 0., 0.
    for rep in range(reps):
        rng = np.random.default_rng([202609062,d,int(round(alpha*100)),rep,1])
        weighted = q*rng.exponential(size=d)
        normalizers.append(float(weighted.sum()))
        p = weighted/weighted.sum()
        labels = rng.choice(d,size=max(ks),p=p)
        counts = np.zeros(d,dtype=np.int64)
        prev = 0
        for j,k in enumerate(ks):
            counts += np.bincount(labels[prev:k],minlength=d); prev = k
            lm, diag = log_mixture_adaptive(counts,q)
            values[rep,j] = (float(counts@np.log(p))-lm)/LN2
            max_error = max(max_error,diag['relative_quadrature_error'])
            max_tail = max(max_tail,diag['relative_tail_bound'])
    pred = np.array([float(scalar_r1(k*q).sum()) for k in ks])
    refined = np.array([float(scalar_r1(k*q,256).sum()) for k in ks])
    assert max(abs(pred-refined)) < 2e-7
    mean = values.mean(0); se = values.std(0,ddof=1)/math.sqrt(reps)
    return dict(d=d,alpha=alpha,L=1,k=ks,repetitions=reps,
        redundancy_mean_bits=mean.tolist(),redundancy_se_bits=se.tolist(),
        original_sum_r1_bits=pred.tolist(),difference_bits=(mean-pred).tolist(),
        relative_difference=(mean/pred-1).tolist(),difference_in_sample_SE=((mean-pred)/se).tolist(),
        scalar_quadrature_refinement_maximum_bits=float(max(abs(pred-refined))),
        sum_q_squared=float(q@q),normalizer_mean=float(np.mean(normalizers)),
        normalizer_variance=float(np.var(normalizers,ddof=1)),
        exact_normalizer_variance=float(q@q),
        maximum_relative_quadrature_error=max_error,maximum_relative_tail_bound=max_tail,
        k_over_d_to_alpha=[k/d**alpha for k in ks],elapsed_seconds=time.monotonic()-start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--reps',type=int,default=128)
    ap.add_argument('--ds',type=int,nargs='+',default=[256,1024,4096])
    ap.add_argument('--alphas',type=float,nargs='+',default=[.5,.75,.9,1.,1.25,1.5,2.])
    ap.add_argument('--ks',type=int,nargs='+',default=[16,64,256,1024,4096])
    ap.add_argument('--smoke',action='store_true')
    args = ap.parse_args()
    if args.smoke: args.ds,args.alphas,args.ks,args.reps=[16],[.75,2.],[1,16],8
    assert args.reps>=2 and sorted(set(args.ks))==args.ks
    start = time.monotonic(); validation = validate()
    coefficients = [normalizer_coefficient(a) for a in args.alphas if a>1]
    for rec in coefficients:
        alternative = normalizer_coefficient(rec['alpha'],head=4096)
        rec['head_refinement_coefficient_difference_bits'] = rec['coefficient_bits']-alternative['coefficient_bits']
        assert abs(rec['head_refinement_coefficient_difference_bits'])<2e-8
        if rec['alpha']==2:
            rec['exact_inverse_normalizer_moment'] = math.sqrt(math.pi)/2
            assert abs(rec['inverse_normalizer_moment']-math.sqrt(math.pi)/2)<1e-9
    rows=[]
    checkpoint=Path('/private/tmp/iid_tilt_coverage_progress.json')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(run_cell,d,a,args.ks,args.reps) for d in args.ds for a in args.alphas]
        for future in as_completed(pending):
            rec=future.result();rows.append(rec)
            checkpoint.write_text(json.dumps(rows,indent=2)+'\n')
            print('cell',rec['d'],rec['alpha'],'mean',rec['redundancy_mean_bits'][-1],
                  'SE',rec['redundancy_se_bits'][-1],'old',rec['original_sum_r1_bits'][-1],flush=True)
    rows.sort(key=lambda r:(r['d'],r['alpha']))
    for row in rows:
        if row['alpha']>1:
            c=next(c for c in coefficients if c['alpha']==row['alpha'])
            row['normalized_infinite_row_leading_bits']=[c['coefficient_bits']*k**c['beta'] for k in args.ks]
        if row['alpha']==0:
            row['uniform_exact_expected_bits']=[r_star_exact_bits(k,row['d']) for k in args.ks]
    old=[]
    for path in ['results/mechanism/context_row_check.json','results/mechanism/context_concentrated_check.json']:
        raw=Path(path).read_bytes();old.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest()))
    result=dict(schema_version=1,source_depth=1,se_ddof=1,elapsed_seconds=time.monotonic()-start,
        seed_protocol='default_rng([202609062,d,round(100*alpha),replication,1])',
        scope='Normalized iid rows only. No Markov visits, stationary law, depth-three code, or crossing prediction is tested.',
        estimand='Prior-averaged redundancy of the Bayes sequence mixture under P_b=q_b E_b/sum(q E).',
        comparisons={'original':'sum_b r1(k q_b); diffuse large-d approximation, also displayed outside its claimed range.',
         'above_one':'J_alpha E[Z_infinity^(-1/alpha)] k^(1/alpha), with Z_infinity=sum_b b^(-alpha) E_b. Infinite-alphabet leading term, not an exact finite-d/k prediction.'},
        uncertainty='Sample standard errors across independent source/sample draws; no simultaneous confidence claim. Numerical error estimates are separate.',
        existing_data_reused=old,validation=validation,infinite_row_coefficients=coefficients,rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('wrote',args.out,'seconds',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()

"""Expected-profile bounds and a covariance correction for the actual LSA code.

OPENBLAS_NUM_THREADS=1 python code/scripts/lsa_expected_profile.py --smoke --out /tmp/lsa_profile_smoke.json
OPENBLAS_NUM_THREADS=1 python code/scripts/lsa_expected_profile.py --out results/mechanism/lsa_expected_profile.json

This is a calculation of a fixed predictor's cost, not a new predictor.
The finite depth mixture remains inside the logarithm. Source probabilities
determine multinomial count laws only. Exact binomial marginals give the
mean profile; independent cheap count draws estimate its covariance.
Separate draws evaluate the existing production code for comparison.
The Jensen estimate is an upper bound before numerical integration; its
second-order covariance correction is an approximation, not a bound.
Only existing sealed moment tables are read; no tables are built.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

from tilted_row_reference import DEFAULT_LSA, DEFAULT_STORE, configure_store

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def count_profile_mean(p, k):
    """E[number of symbols observed r times], including unseen symbols."""
    out = binom.pmf(np.arange(k+1)[:, None], k, np.asarray(p)[None, :]).sum(1)
    assert np.isclose(out.sum(), len(p), rtol=1e-12)
    assert np.isclose(out @ np.arange(k+1), k, rtol=1e-12)
    return out


def sampled_profiles(p, k, repetitions, rng):
    counts = rng.multinomial(k, p, size=repetitions)
    h = np.array([np.bincount(z, minlength=k+1) for z in counts], dtype=float)
    return counts, h


def fractional_cost(nu, k, l_max, tables, step=.025, covariance=None):
    """Evaluate Jensen and delta-method predictions for one fixed sample size.

    The covariance is for the full multiplicity vector. Projection removes
    its two deterministic directions: number of symbols and sample size.
    Every depth is retained, even if its posterior weight is tiny.
    """
    nu = np.asarray(nu, dtype=float)
    rs = np.arange(len(nu))
    assert len(nu) == k+1 and abs(nu @ rs-k) < 1e-8
    d = nu.sum()
    grid = np.r_[np.arange(-220., -75., 4*step), np.arange(-75., 35.+step/2, step)]
    widths = np.empty_like(grid)
    widths[1:-1] = (grid[2:]-grid[:-2])/2
    widths[0], widths[-1] = (grid[1]-grid[0])/2, (grid[-1]-grid[-2])/2
    logweights = np.log(widths)
    kernels, projected, gradients = [], [], []
    max_refinement = 0.
    eigenfactor = None
    if covariance is not None:
        eig, vec = np.linalg.eigh(covariance[2:, 2:])
        live = eig > max(1e-13, eig.max(initial=0)*1e-12)
        eigenfactor = vec[:, live]*np.sqrt(np.maximum(eig[live], 0.))
    first = None
    for j in range(1, l_max+1):
        if j == 1:
            lp = gammaln(rs[:, None]+1)-(rs[:, None]+1)*np.logaddexp(0., grid)
        else:
            lp = tables.log_phi_matrix(j, list(map(int, rs)), grid)
        kernel = k*grid+nu @ lp
        peak = kernel.max()
        assert np.isfinite(kernel).all() and max(kernel[0], kernel[-1]) < peak-35, (j, peak, kernel[[0,-1]])
        logz = logsumexp(kernel+logweights)
        if j == 1:
            first = (gammaln(k)-logz)/LN2
            closed = (gammaln(k+d)-gammaln(d)-nu @ gammaln(rs+1))/LN2
            assert abs(first-closed) < 2e-6, (first, closed)
        # Nested trapezoid check on the same piecewise grid.
        ix = np.unique(np.r_[np.arange(0, len(grid), 2), len(grid)-1])
        coarse = np.trapezoid(np.exp(kernel[ix]-peak), grid[ix])
        max_refinement = max(max_refinement, abs(logz-(math.log(coarse)+peak))/LN2)
        kernels.append(kernel+logweights)
        r = rs[2:, None]
        b = lp[2:]-r*lp[1]+(r-1)*lp[0]
        gradients.append(b @ np.exp(kernel+logweights-logz))
        if eigenfactor is not None:
            projected.append(eigenfactor.T @ b)
    logz = logsumexp(np.array(kernels))-math.log(l_max)
    upper = (gammaln(k)-logz)/LN2
    depth_weights = np.exp(logsumexp(np.array(kernels),axis=1)-logsumexp(np.array(kernels)))
    gradient = sum(w*g for w,g in zip(depth_weights,gradients))/LN2
    correction = None
    if eigenfactor is not None:
        weights = np.exp(np.array(kernels)-logsumexp(np.array(kernels)))
        mean = sum(b @ w for b, w in zip(projected, weights))
        second = sum((b*b).sum(0) @ w for b, w in zip(projected, weights))
        correction = max(0., float(second-mean @ mean))/(2*LN2)
    return dict(jensen_bits=float(upper), covariance_correction_bits=correction,
                corrected_bits=None if correction is None else float(upper-correction),
                depth_one_bits=float(first), maximum_nested_grid_difference_bits=float(max_refinement),
                integration_step=step, projected_gradient_bits=gradient.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    configure_store(DEFAULT_STORE)
    if str(DEFAULT_LSA) not in sys.path:
        sys.path.insert(0, str(DEFAULT_LSA))
    from lsa.universal_tables import UniversalTables
    from lsa.codelength import default_l_max, depth_averaged_codelength_profiles, profile_of
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    started = time.monotonic()
    settings = [(64, .5, 2, 16), (64, 1.5, 6, 64)] if args.smoke else [
        (d, alpha, L, k) for d in (64, 256) for alpha, L in ((.5,2),(1.5,6)) for k in (16,64,128)]
    cases = []
    for d, alpha, L, k in settings:
        rng = np.random.default_rng([20260907, 4811, d, round(alpha*100), L])
        w = rng.exponential(size=(d,L)).prod(1)*np.arange(1,d+1,dtype=float)**(-alpha)
        p = w/w.sum()
        nu = count_profile_mean(p,k)
        covreps, reps = (2048,128) if args.smoke else (16384,1024)
        _, h = sampled_profiles(p,k,covreps,np.random.default_rng([20260907,4812,d,k,L,round(alpha*100)]))
        covariance = np.cov(h, rowvar=False, ddof=1)
        upper = fractional_cost(nu,k,default_l_max(d),tables,covariance=covariance)
        counts, measured_h = sampled_profiles(p,k,reps,np.random.default_rng([20260907,4813,d,k,L,round(alpha*100)]))
        profiles = [profile_of(z) for z in counts]
        evaluated = depth_averaged_codelength_profiles({v:v for v in set(profiles)},d=d,
            l_max=default_l_max(d),jobs=1,tables_source='universal',universal_path=DEFAULT_STORE)
        costs = np.array([-evaluated[v].log2_q_avg for v in profiles])
        mean, se = float(costs.mean()), float(costs.std(ddof=1)/math.sqrt(reps))
        # Exact zero-mean control variate removes the linear count fluctuations.
        # Each adjusted value equals the Jensen bound minus a nonnegative
        # log-partition Bregman divergence. No observed cost is fitted.
        adjusted=costs+(measured_h[:,2:]-nu[2:]) @ np.array(upper['projected_gradient_bits'])
        adjusted_mean=float(adjusted.mean())
        adjusted_se=float(adjusted.std(ddof=1)/math.sqrt(reps))
        assert np.max(adjusted-upper['jensen_bits']) < 3e-4
        # Compare integer-histogram integration with the independent production evaluator.
        z=counts[0]; integer=np.bincount(z,minlength=k+1).astype(float)
        direct=fractional_cost(integer,k,default_l_max(d),tables)
        assert abs(direct['jensen_bits']-costs[0]) < 2e-4, (direct,costs[0])
        cases.append(dict(d=d,alpha=alpha,source_depth=L,k=k,coding_depth_cap=default_l_max(d),
            source_probabilities=p.tolist(),prediction=upper,
            comparison=dict(repetitions=reps,mean_bits=mean,SE_bits=se,
                controlled_mean_bits=adjusted_mean,controlled_SE_bits=adjusted_se),
            covariance_repetitions=covreps,integer_profile_agreement_bits=direct['jensen_bits']-float(costs[0]),
            jensen_minus_measured_bits=upper['jensen_bits']-adjusted_mean,
            corrected_minus_measured_bits=upper['corrected_bits']-adjusted_mean))
        result=dict(schema_version=1,complete=len(cases)==len(settings),target='Actual LSA expected iid cost at fixed source p',
            qualifications=['Source probabilities are inputs to the cost analysis only.',
                'Jensen upper bound is exact before numerical integration; covariance correction is not a bound.',
                'Covariance count draws and measured-code draws are independent.',
                'This does not test the Markov count approximation or locate a crossing.'],
            cells=cases,elapsed_seconds=time.monotonic()-started)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({key:cases[-1][key] for key in ('d','alpha','source_depth','k','jensen_minus_measured_bits','corrected_minus_measured_bits')}),flush=True)
    tables.close()


if __name__=='__main__':
    main()

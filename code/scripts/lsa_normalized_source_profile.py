"""Compute exact normalized-source iid count profiles without source draws.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_normalized_source_profile.py --smoke --out /tmp/lsa_normalized_profile_smoke.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_normalized_source_profile.py --out results/mechanism/lsa_normalized_source_profile_check.json

The formula is exact; scalar quadrature and sealed moment interpolation are
numerical. Positive polynomial prefix/suffix products compute the required
coefficients. No high numerical derivatives, source draws, or moment-table
builds are used. The default source is d=128, alpha=.75, source depth2;
fixed integer sample counts are16,32,64. Full-mixture costs at mean profiles
are compared, not asserted equal to mean costs. The unnormalized Poisson
profile comparison is a separate diagnostic with a random sample total.
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
from scipy.special import gammaln, logsumexp

from tilted_row_reference import DEFAULT_STORE, configure_store

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mixed_poisson(depth, log_intensities, max_count, tables):
    """Rows are counts and columns are intensities; no pmf renormalization."""
    u = np.asarray(log_intensities, dtype=float).ravel()
    r = np.arange(max_count+1)
    factorial = gammaln(r+1)[:, None]
    if depth == 0:
        logp = r[:, None]*u-np.exp(u)-factorial
    elif depth == 1:
        logp = r[:, None]*u-(r[:, None]+1)*np.logaddexp(0., u)
    else:
        phi = tables.log_phi_matrix(depth, r.tolist(), u)
        logp = phi+r[:, None]*u-factorial
    return np.exp(logp)


def normalized_source_profile(q, source_depth, k, tables, *, step=.125,
                              lower=-32., upper=20.):
    """Return the normalized iid mean profile and raw quadrature diagnostics.

    h_r=k integral sum_b p_b(r;s)[z^(k-r)]prod_(c!=b)B_c(s,z) dlog(s).
    Polynomials only need degrees0..k; that truncation is exact.
    The returned profile is divided by the scalar total-count mass, whose
    exact value is1. Raw errors are retained and checked before doing so.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or len(q) < 2 or np.any(q <= 0) or not np.isclose(q.sum(), 1.):
        raise ValueError('q must be a positive probability vector')
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError('k must be a nonnegative integer')
    d = len(q)
    if k == 0:
        return np.array([float(d)]), dict(total_count_mass=1., raw_symbol_error=0.,
                                         raw_count_error=0., step=step, k=0)
    u = np.arange(lower, upper+step/2, step)
    query = u[:, None]+np.log(q)[None, :]
    pmf = mixed_poisson(source_depth, query.ravel(), k, tables)
    pmf = pmf.reshape(k+1, len(u), d).transpose(1, 2, 0)
    integrand = np.empty((len(u), k+1))
    mass_integrand = np.empty(len(u))
    for j, probabilities in enumerate(pmf):
        prefix = np.zeros((d+1, k+1))
        suffix = np.zeros((d+1, k+1))
        prefix[0, 0] = suffix[d, 0] = 1.
        for b in range(d):
            prefix[b+1] = np.convolve(prefix[b], probabilities[b])[:k+1]
        for b in range(d-1, -1, -1):
            suffix[b] = np.convolve(probabilities[b], suffix[b+1])[:k+1]
        marginal_sum = np.zeros(k+1)
        for b in range(d):
            others = np.convolve(prefix[b], suffix[b+1])[:k+1]
            marginal_sum += probabilities[b]*others[::-1]
        integrand[j] = k*marginal_sum
        mass_integrand[j] = k*prefix[d, k]
    raw = np.trapezoid(integrand, u, axis=0)
    mass = float(np.trapezoid(mass_integrand, u))
    if not np.isfinite(mass) or abs(mass-1.) > 2e-6:
        raise ArithmeticError(f'Total-count quadrature mass differs from1: {mass}')
    profile = raw/mass
    symbol_error = float(raw.sum()-d)
    count_error = float(raw @ np.arange(k+1)-k)
    if abs(profile.sum()-d) > 2e-9 or abs(profile @ np.arange(k+1)-k) > 2e-9:
        raise ArithmeticError('Polynomial coefficient constraints failed')
    peak = float(mass_integrand.max())
    diagnostic = dict(k=k, step=step, lower=lower, upper=upper,
        quadrature_points=len(u), total_count_mass=mass,
        raw_symbol_error=symbol_error, raw_count_error=count_error,
        profile_divided_by_total_count_mass=True,
        maximum_change_from_mass_normalization=float(np.max(abs(profile-raw))),
        normalized_symbol_error=float(profile.sum()-d),
        normalized_count_error=float(profile @ np.arange(k+1)-k),
        endpoint_integrand_relative_to_peak=float(max(mass_integrand[0],mass_integrand[-1])/peak),
        minimum_profile_entry=float(profile.min()))
    return profile, diagnostic


def uniform_dirichlet_profile(d, k):
    r = np.arange(k+1)
    # d*choose(k-r+d-2,d-2)/choose(k+d-1,d-1).
    logp = (math.log(d)+gammaln(k-r+d-1)-gammaln(d-1)-gammaln(k-r+1)
            -gammaln(k+d)+gammaln(d)+gammaln(k+1))
    return np.exp(logp)


def mean_profile_costs(profiles, sample_counts, coding_cap, tables, *, step=.1):
    """Full coding mixture at fractional profiles, keeping all gamma factors.

    All profiles are supplied already; no source or count expectation is
    performed here. A Poisson profile with random total is not a Jensen
    bound for a fixed-k iid experiment. No missing-count term is fabricated.
    """
    max_count = max(len(h) for h in profiles)-1
    nu = np.zeros((len(profiles), max_count+1))
    for i, h in enumerate(profiles):
        nu[i, :len(h)] = h
    k = np.asarray(sample_counts, dtype=float)
    r = np.arange(max_count+1)
    u = np.r_[np.arange(-220., -75., 4*step), np.arange(-75., 35.+step/2, step)]
    widths = np.empty(len(u))
    widths[1:-1] = (u[2:]-u[:-2])/2
    widths[0], widths[-1] = (u[1]-u[0])/2, (u[-1]-u[-2])/2
    logweights = np.log(widths)
    results = []
    worst_edge = np.full(len(profiles), -np.inf)
    depth_one = None
    for depth in range(1, coding_cap+1):
        if depth == 1:
            logphi = gammaln(r[:, None]+1)-(r[:, None]+1)*np.logaddexp(0., u)
        else:
            logphi = tables.log_phi_matrix(depth, r.tolist(), u)
        exponent = k[:, None]*u+nu @ logphi
        peak = np.max(exponent, axis=1)
        worst_edge = np.maximum(worst_edge, np.max(exponent[:, [0,-1]],axis=1)-peak)
        logz = logsumexp(exponent+logweights, axis=1)
        results.append(logz)
        if depth == 1:
            actual = (gammaln(k)-logz)/LN2
            d = nu.sum(axis=1)
            closed = (gammaln(k+d)-gammaln(d)-nu @ gammaln(r+1))/LN2
            depth_one = actual-closed
    if np.any(worst_edge > -35):
        raise ArithmeticError(f'Coding integral does not decay at grid edges: {worst_edge}')
    values = (gammaln(k)-logsumexp(results,axis=0)+math.log(coding_cap))/LN2
    return [dict(mean_profile_cost_bits=float(values[i]),
        depth_one_identity_discrepancy_bits=float(depth_one[i]),
        worst_log_integrand_edge_relative_to_peak=float(worst_edge[i]),
        step=step, maximum_profile_count=max_count)
        for i in range(len(profiles))]


def matched_saved_comparison(metadata, tables):
    """One previously saved source-averaged case; no new source/path draws."""
    parameters = (256, .75, 3, 67)
    cell = next(c for c in metadata['comparisons'] if
        (c['d'],c['alpha'],c['source_depth'],c['k']) == parameters)
    source = next(s for s in metadata['sources'] if
        (s['d'],s['alpha'],s['source_depth']) == parameters[:3])
    d, alpha, depth, k = parameters
    cap = round(2*math.log(d)/(1-0.5772156649015329))
    assert cell['coding_depth'] == source['coding_depth'] == cap
    q = np.arange(1,d+1,dtype=float)**(-alpha)
    q /= q.sum()
    coarse, dc = normalized_source_profile(q,depth,k,tables,step=.125)
    fine, df = normalized_source_profile(q,depth,k,tables,step=.0625)
    difference = float(np.max(abs(fine-coarse)))
    assert difference < 2e-7
    cc = mean_profile_costs([coarse],[k],cap,tables,step=.1)[0]
    cf = mean_profile_costs([fine],[k],cap,tables,step=.05)[0]
    saved = ROOT/source['saved_path']
    assert sha256(saved) == source['saved_sha256']
    archive = json.loads(saved.read_text())
    ix = archive['k'].index(k)
    assert abs(archive['rho_mean'][ix]-cell['saved_rho_mean']) < 1e-10
    entropies = np.asarray(source['source_entropy_bits'])
    probabilities = np.asarray(source['source_probabilities'])
    assert np.max(abs(-(probabilities*np.log2(probabilities)).sum(axis=1)-entropies)) < 2e-12
    rows = len(entropies)
    assert rows == source['rows'] == cell['rows'] == archive['rows']
    entropy_mean = float(entropies.mean())
    entropy_se = float(entropies.std(ddof=1)/math.sqrt(rows))
    total_mean = float(cell['saved_rho_mean']+k*entropy_mean)
    rho_se = float(cell['saved_rho_SE_converted_to_ddof1'])
    result = dict(d=d,alpha=alpha,source_depth=depth,k=k,coding_cap=cap,
        source_repetitions=rows,normalized_profile=fine.tolist(),
        coarse_diagnostics=dc,fine_diagnostics=df,maximum_grid_profile_difference=difference,
        normalized_mean_profile_cost=cf,
        profile_and_coding_grid_difference_bits=abs(cf['mean_profile_cost_bits']-cc['mean_profile_cost_bits']),
        saved_source_path=str(saved.relative_to(ROOT)),saved_source_sha256=sha256(saved),
        saved_excess_mean_bits=float(cell['saved_rho_mean']),
        saved_source_entropy_mean_bits=entropy_mean,
        reconstructed_saved_total_cost_mean_bits=total_mean,
        normalized_upper_minus_saved_mean_bits=cf['mean_profile_cost_bits']-total_mean,
        saved_excess_SE_bits_ddof1=rho_se,source_entropy_SE_bits_ddof1=entropy_se,
        total_cost_SE_bits=None,
        triangle_upper_bound_on_sample_total_cost_SE_bits=rho_se+k*entropy_se,
        saved_conditional_profile_upper_mean_total_bits=cell['uncorrected_upper']['mean']+k*entropy_mean,
        qualifications=[
            'The saved statistic is excess codelength; adding k times the same32-source mean entropy reconstructs its mean total cost.',
            'The raw per-source excess/entropy covariance is not stored, so the total-cost SE is not reconstructed.',
            'The triangle bound is a bound on the sample SE, not a confidence interval.',
            'This normalized unconditional mean-profile cost is a mathematical upper bound on the population mean row cost before numerical integration.',
            'The difference from a32-draw saved mean is not an exact averaging bias or a crossing error.'])
    print(json.dumps({key:result[key] for key in ('d','source_depth','k',
        'reconstructed_saved_total_cost_mean_bits','normalized_upper_minus_saved_mean_bits')}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=.75)
    parser.add_argument('--source-depth', type=int, default=2)
    parser.add_argument('--out', type=Path, default=ROOT/'results/mechanism/lsa_normalized_source_profile_check.json')
    args = parser.parse_args()
    if args.d < 2 or not 0 <= args.alpha or not 1 <= args.source_depth < 54:
        raise ValueError('Require d>=2, alpha>=0, and source depth1..53')
    configure_store(DEFAULT_STORE)
    from lsa.universal_tables import UniversalTables
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    started = time.monotonic()
    try:
        q = np.arange(1,args.d+1,dtype=float)**(-args.alpha)
        q /= q.sum()
        sample_counts = [16] if args.smoke else [16,32,64]
        cap = round(2*math.log(args.d)/(1-0.5772156649015329))
        checks = []
        for k in [1,16] if args.smoke else [1,16,32,64]:
            uniform = np.full(args.d,1/args.d)
            h, diag = normalized_source_profile(uniform,1,k,None,step=.0625)
            expected = uniform_dirichlet_profile(args.d,k)
            error = float(np.max(abs(h-expected)))
            assert error < 2e-8, (k,error)
            checks.append(dict(check='uniform_source_depth1_Dirichlet_multinomial', k=k,
                               maximum_profile_error=error,diagnostics=diag))
        h1, diag1 = normalized_source_profile(q,args.source_depth,1,tables,step=.0625)
        first_error = float(np.max(abs(h1-np.array([args.d-1,1.]))))
        assert first_error < 2e-8
        checks.append(dict(check='normalized_source_k1',maximum_profile_error=first_error,diagnostics=diag1))
        first_cost = mean_profile_costs([h1],[1],cap,tables,step=.05)[0]
        first_cost_error = abs(first_cost['mean_profile_cost_bits']-math.log2(args.d))
        assert first_cost_error < 2e-6
        checks.append(dict(check='all_coding_depths_first_token_uniform',
            cost_error_bits=first_cost_error,diagnostics=first_cost))
        cells, normalized, poisson = [], [], []
        for k in sample_counts:
            coarse, dc = normalized_source_profile(q,args.source_depth,k,tables,step=.125)
            fine, df = normalized_source_profile(q,args.source_depth,k,tables,step=.0625)
            difference = float(np.max(abs(coarse-fine)))
            assert difference < 2e-7, (k,difference)
            raw_poisson = mixed_poisson(args.source_depth,np.log(k*q),2048,tables).sum(axis=1)
            missed_symbols = float(args.d-raw_poisson.sum())
            missed_counts = float(k-raw_poisson @ np.arange(len(raw_poisson)))
            if abs(missed_symbols)>2e-7 or abs(missed_counts)>2e-7:
                raise ArithmeticError(f'Poisson comparison tail too large: {missed_symbols}, {missed_counts}')
            normalized.append(fine)
            poisson.append(raw_poisson)
            cells.append(dict(d=args.d,alpha=args.alpha,source_depth=args.source_depth,k=k,
                coding_cap=cap, normalized_profile=fine.tolist(),
                coarse_diagnostics=dc,fine_diagnostics=df,maximum_grid_profile_difference=difference,
                normalized_expected_distinct_symbols=float(args.d-fine[0]),
                poisson_expected_distinct_symbols=float(args.d-raw_poisson[0]),
                poisson_profile_tail_beyond_fixed_k=float(raw_poisson[k+1:].sum()),
                poisson_profile_missing_symbols=missed_symbols,poisson_profile_missing_counts=missed_counts,
                poisson_count_cutoff=len(raw_poisson)-1))
            print(json.dumps({key:cells[-1][key] for key in ('k','maximum_grid_profile_difference',
                'normalized_expected_distinct_symbols','poisson_expected_distinct_symbols')}),flush=True)
        all_profiles = normalized+poisson
        all_counts = sample_counts+sample_counts
        costs_coarse = mean_profile_costs(all_profiles,all_counts,cap,tables,step=.1)
        costs_fine = mean_profile_costs(all_profiles,all_counts,cap,tables,step=.05)
        for i, cell in enumerate(cells):
            j = i+len(cells)
            cell['normalized_mean_profile_cost']=costs_fine[i]
            cell['poisson_mean_profile_cost']=costs_fine[j]
            cell['normalized_minus_poisson_mean_profile_bits']=(costs_fine[i]['mean_profile_cost_bits']-
                                                              costs_fine[j]['mean_profile_cost_bits'])
            cell['normalized_cost_grid_difference_bits']=abs(costs_fine[i]['mean_profile_cost_bits']-
                                                             costs_coarse[i]['mean_profile_cost_bits'])
            cell['poisson_cost_grid_difference_bits']=abs(costs_fine[j]['mean_profile_cost_bits']-
                                                          costs_coarse[j]['mean_profile_cost_bits'])
        if max(c['normalized_cost_grid_difference_bits'] for c in cells)>2e-5:
            raise ArithmeticError('Normalized mean-profile coding quadrature needs refinement')
        if max(abs(c['normalized_mean_profile_cost']['depth_one_identity_discrepancy_bits'])
               for c in cells)>2e-6:
            raise ArithmeticError('Normalized-profile depth-one identity failed')
        existing = ROOT/'results/mechanism/lsa_single_source_validation.json'
        metadata = json.loads(existing.read_text())
        matched = None if args.smoke else matched_saved_comparison(metadata,tables)
        result = dict(schema_version=1,complete=True,formula='Equation38 in notes/lsa_parameter_gap_derivation_2026-09-08.md',
            target='Source-averaged iid count profile with full finite normalization, at fixed integer k',
            source_draws=0,script_sha256=sha256(__file__),
            moment_store=str(DEFAULT_STORE),moment_manifest_sha256=sha256(DEFAULT_STORE/'manifest.json'),
            moment_anchors_sha256=sha256(DEFAULT_STORE/'anchors.json'),
            moment_api_sha256=sha256(sys.modules['lsa.universal_tables'].__file__),
            configure_helper_sha256=sha256(ROOT/'code/scripts/tilted_row_reference.py'),
            command_argv=sys.argv,parameters=dict(d=args.d,alpha=args.alpha,source_depth=args.source_depth),
            checks=checks,cells=cells,matched_saved_comparison=matched,
            existing_cost_validation=dict(path=str(existing.relative_to(ROOT)),sha256=sha256(existing),
                status='The main d128 source-depth2 profile grid has no selected matched cost sample; a separate d256 source-depth3 k67 case is matched in the full run.',
                saved_target=metadata.get('target')),
            qualifications=[
                'The normalized-profile formula and polynomial truncation at degree k are exact.',
                'Reported integration and moment-interpolation errors are numerical checks, not certified intervals.',
                'Raw normalization errors are reported before dividing by the scalar total-count mass, whose exact value is1.',
                'The normalized fixed-k mean-profile cost is a mathematical Jensen upper bound before numerical error.',
                'The unnormalized Poisson mean-profile cost is a diagnostic, not a bound for the normalized fixed-k experiment.',
                'No expected full-mixture row cost or actual-chain crossing is computed here.'],
            elapsed_seconds=time.monotonic()-started)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(complete=True,elapsed_seconds=result['elapsed_seconds'],
                             output=str(args.out))),flush=True)
    finally:
        tables.close()


if __name__=='__main__':
    main()

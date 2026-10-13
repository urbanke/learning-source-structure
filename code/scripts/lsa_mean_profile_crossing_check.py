"""Check deterministic LSA mean-profile upper costs on retained source tables.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_mean_profile_crossing_check.py

Uses the first four retained d=512 tables at five saved checkpoints:
L=10, alpha=1.5 near 40,80,160 tokens; L=6, alpha=1.5 near 423;
L=6, alpha=1 near 550. No source tables or Markov paths are sampled.
Independent multinomial counts and initial tokens estimate the same auxiliary
experiment as the analytic binomial profiles. Both costs use the unchanged
finite LSA depth mixture. Their difference is not an upper bound on the gap.
The original all-positive-row Jensen calculation is retained. A tighter
version averages the true LSA costs exactly over all count profiles at row
totals 1..8 and applies Jensen only conditional on more than eight visits.
Only sealed moment tables are read. No new moment columns or caches are built.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import binom

from tilted_row_reference import DEFAULT_LSA, DEFAULT_STORE, configure_store

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)
SOURCE = ROOT / "output/crossing_lsa_extension/refined256/L10_a1.5_d512.json"
EXACT_ROW_CAP = 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_array(path, name, first=None):
    """Read only the requested leading rows of a compressed numeric NPY member."""
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported array header {version}")
        if fortran or dtype.hasobject:
            raise ValueError("Expected C-order numeric source arrays")
        target = shape if first is None else (min(first, shape[0]),) + shape[1:]
        count = math.prod(target)
        data = stream.read(count * dtype.itemsize)
        if len(data) != count * dtype.itemsize:
            raise ValueError("Truncated source array")
        return np.frombuffer(data, dtype=dtype).reshape(target).copy()


def moments(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()),
                SE=float(values.std(ddof=1) / math.sqrt(len(values))))


def partitions(k, largest=None):
    if k == 0:
        yield ()
        return
    for first in range(min(k, k if largest is None else largest), 0, -1):
        for rest in partitions(k - first, first):
            yield (first,) + rest


def exact_profile_probabilities(probabilities, cap=EXACT_ROW_CAP):
    """Probabilities of every iid count profile, using ordered injective sums.

    Power-sum recursion is exact algebraically. Extended precision reduces
    subtraction error for concentrated rows; clipped negative roundoff and
    the probability mass error are measured explicitly, not hidden by
    renormalizing the resulting profile probabilities.
    """
    p = np.atleast_2d(np.asarray(probabilities, dtype=np.longdouble)).copy()
    assert np.all(p >= 0) and np.all(p.sum(axis=1) > 0)
    p /= p.sum(axis=1)[:, None]
    powers = {j: np.sum(p**j, axis=1) for j in range(1, cap + 1)}
    memo = {(): np.ones(len(p), dtype=np.longdouble)}

    def injective(profile):
        profile = tuple(sorted(profile, reverse=True))
        if profile not in memo:
            last, prefix = profile[-1], profile[:-1]
            value = powers[last] * injective(prefix)
            for j in range(len(prefix)):
                merged = prefix[:j] + (prefix[j] + last,) + prefix[j+1:]
                value = value - injective(merged)
            memo[profile] = value
        return memo[profile]

    out = {(): np.ones(len(p))}
    maximum_mass_error, maximum_negative_mass = 0., 0.
    for k in range(1, cap + 1):
        total = np.zeros(len(p), dtype=np.longdouble)
        negative = np.zeros(len(p), dtype=np.longdouble)
        for profile in partitions(k):
            denominator = math.prod(math.factorial(r) for r in profile)
            denominator *= math.prod(math.factorial(h) for h in Counter(profile).values())
            probability = injective(profile) * np.longdouble(math.factorial(k)) / denominator
            if len(profile) > p.shape[1]:
                # An injective assignment to fewer labels is impossible exactly.
                probability = np.zeros(len(p), dtype=np.longdouble)
            negative += np.maximum(-probability, 0.)
            probability = np.maximum(probability, 0.)
            total += probability
            out[profile] = np.asarray(probability, dtype=float)
        maximum_mass_error = max(maximum_mass_error, float(np.max(abs(total - 1))))
        maximum_negative_mass = max(maximum_negative_mass, float(negative.max()))
    s2, s3 = np.asarray(powers[2], dtype=float), np.asarray(powers[3], dtype=float)
    checks = [np.max(abs(out[(2,)] - s2)), np.max(abs(out[(1, 1)] - (1-s2))),
              np.max(abs(out[(3,)] - s3)), np.max(abs(out[(2, 1)] - 3*(s2-s3))),
              np.max(abs(out[(1, 1, 1)] - (1-3*s2+2*s3)))]
    assert maximum_mass_error < 1e-9 and maximum_negative_mass < 1e-9
    assert max(checks) < 1e-12
    return out, dict(maximum_profile_probability_mass_error=maximum_mass_error,
        maximum_clipped_negative_probability_mass=maximum_negative_mass,
        maximum_two_three_sample_formula_error=float(max(checks)),
        arithmetic_dtype=str(p.dtype), profile_cache_count=len(out))


def profile_checks():
    """Independent multinomial enumeration at d=2..4 through eight samples."""
    def compositions(k, d):
        if d == 1:
            yield (k,)
        else:
            for first in range(k + 1):
                for rest in compositions(k-first, d-1):
                    yield (first,) + rest
    maximum_error = 0.
    rows = [[.8, .2], [.6, .3, .1], [.999999, .0000008, .0000002], [.4, .3, .2, .1]]
    for values in rows:
        p = np.asarray(values)
        probabilities, _ = exact_profile_probabilities(p)
        for k in range(1, EXACT_ROW_CAP + 1):
            direct = Counter()
            for m in compositions(k, len(p)):
                profile = tuple(sorted((v for v in m if v), reverse=True))
                coefficient = math.factorial(k) / math.prod(math.factorial(v) for v in m)
                direct[profile] += coefficient * math.prod(float(x)**r for x, r in zip(p, m))
            for profile in partitions(k):
                maximum_error = max(maximum_error, abs(probabilities[profile][0]-direct[profile]))
    assert maximum_error < 1e-10, maximum_error
    T = np.array([[.7, .2, .1], [.02, .97, .01], [.4, .3, .3]])
    r = np.array([.2, .3, .5])
    n = 17
    recurrence, tail_recurrence = positive_binomial_profiles(T, r, n)
    direct = binom.pmf(np.arange(1, n)[None, None, :], n-1, (r[:, None]*T)[:, :, None]).sum(axis=1)
    recurrence_error = float(np.max(abs(recurrence-direct)))
    nu, _, _, tail, live, _ = row_tail_profiles(T, r, n, positive=tail_recurrence)
    by_total = np.zeros_like(nu)
    for j, a in enumerate(live):
        for k in range(EXACT_ROW_CAP+1, n):
            by_total[j] += (binom.pmf(k, n-1, r[a])/tail[a]
                * binom.pmf(np.arange(n+1)[:, None], k, T[a][None, :]).sum(axis=1))
    tail_error = float(np.max(abs(nu-by_total)))
    assert recurrence_error < 1e-12 and tail_error < 1e-11
    return dict(maximum_independent_count_enumeration_error=maximum_error,
                source_probability_vectors=rows, maximum_sample_count=EXACT_ROW_CAP,
                integer_profiles_including_empty=sum(1 for k in range(EXACT_ROW_CAP+1) for _ in partitions(k)),
                maximum_binomial_recurrence_direct_pmf_error=recurrence_error,
                maximum_tail_profile_conditional_total_error=tail_error)


def positive_binomial_profiles(T, r, n, cap=EXACT_ROW_CAP):
    """Sum binomial marginals over outputs; reuse them for both row conditions.

    Starting at zero is safe for the retained cells (checked explicitly).
    A direct-pmf fallback also handles more concentrated hypothetical cells.
    No count orders are truncated.
    """
    d, t = len(r), n-1
    pair = r[:, None]*T
    other = r[:, None]*(1-T)/(1-pair)
    logzero = t*np.log1p(-pair)
    use_recurrence = float(logzero.min()) > math.log(np.finfo(float).tiny)+20
    current = np.exp(logzero)
    odds = pair/(1-pair)
    positive, tail = np.zeros((d, t)), np.zeros((d, t))
    for m in range(1, t+1):
        if use_recurrence:
            current *= odds*((t-m+1)/m)
        else:
            current = binom.pmf(m, t, pair)
        positive[:, m-1] = current.sum(axis=1)
        if m <= cap:
            tail[:, m-1] = (current*binom.sf(cap-m, t-m, other)).sum(axis=1)
        else:
            tail[:, m-1] = positive[:, m-1]
    return positive, tail


def row_tail_profiles(T, r, n, cap=EXACT_ROW_CAP, positive=None):
    """E Phi and E log Gamma(K), conditional on K>cap, using exact marginals."""
    d, t = len(r), n-1
    tail = binom.sf(cap, t, r)
    live = np.flatnonzero(tail > 0.)
    if not len(live):
        return np.empty((0, n+1)), np.empty(0), np.empty(0), tail, live, dict(
            maximum_tail_profile_mass_error=0., maximum_tail_profile_count_error=0.,
            omitted_tail_worst_case_cost_bits=0., rows_with_positive_tail=0)
    rs, ks = np.arange(1, t+1), np.arange(cap+1, t+1)
    kpmf = binom.pmf(ks[None, :], t, r[live, None])
    nbar = (kpmf @ ks) / tail[live]
    gamma = (kpmf @ gammaln(ks)) / tail[live]
    nu = np.zeros((len(live), n+1))
    if positive is None:
        _, positive = positive_binomial_profiles(T, r, n, cap)
    nu[:, 1:t+1] = positive[live] / tail[live, None]
    nu[:, 0] = d-nu[:, 1:].sum(axis=1)
    mass_error = float(np.max(abs(nu.sum(axis=1)-d)))
    count_error = float(np.max(abs(nu @ np.arange(n+1)-nbar)))
    assert np.all(nu >= 0) and mass_error < 2e-9 and count_error < 2e-9
    assert np.all(nbar >= cap+1-1e-10)
    # At these checkpoints every mathematically positive tail is representable.
    # No tail cutoff or omitted-cost approximation is used.
    assert t <= cap or len(live) == d
    return nu, nbar, gamma, tail, live, dict(
        maximum_tail_profile_mass_error=mass_error, maximum_tail_profile_count_error=count_error,
        omitted_tail_worst_case_cost_bits=0., rows_with_positive_tail=len(live),
        minimum_positive_tail_probability=float(tail[live].min()))


def binomial_profiles(T, q, r, n, positive=None):
    """Full-support profiles, conditional on a nonempty row; frequency last."""
    d, t = len(q), n - 1
    counts = np.arange(n + 1)
    nu = np.zeros((d + 1, n + 1))
    active = -np.expm1(t * np.log1p(-r))
    assert np.all(active > 0.)
    gamma_mean = np.zeros(d + 1)
    nbar = np.r_[t * r / active, n]
    exact_one = np.zeros(d + 1)
    ks = np.arange(t + 1)
    kpmf = binom.pmf(ks[None, :], t, r[:, None])
    gamma_mean[:d] = (kpmf[:, 1:] @ gammaln(ks[1:])) / active
    if positive is None:
        positive, _ = positive_binomial_profiles(T, r, n)
    nu[:d, 1:t+1] = positive / active[:, None]
    exact_one[:d] = (kpmf @ gammaln(ks+d) - gammaln(d)
                     - positive @ gammaln(np.arange(1, t+1)+1)) / LN2
    nu[:d, 0] = d - nu[:d, 1:].sum(axis=1)
    v = r @ T
    nu[d] = ((1 - q)[None, :] * binom.pmf(counts[:, None], t, v[None, :])
             + q[None, :] * binom.pmf(counts[:, None] - 1, t, v[None, :])).sum(axis=1)
    gamma_mean[d] = gammaln(n)
    exact_one[d] = (gammaln(n + d) - gammaln(d)
                    - nu[d] @ gammaln(counts + 1)) / LN2
    mass_error = float(np.max(abs(nu.sum(axis=1) - d)))
    count_error = float(np.max(abs(nu @ counts - nbar)))
    assert mass_error < 2e-9 and count_error < 2e-9 and nu.min() >= 0.
    return nu, nbar, gamma_mean, active, exact_one, dict(
        maximum_profile_mass_error=mass_error,
        maximum_profile_count_error=count_error,
        minimum_row_active_probability=float(active.min()))


def control_checks():
    """Check unbiased hybrid controls by exhaustive binary auxiliary counts.

    All 572 matrix/initial outcomes at t=10 are enumerated. This check uses
    the depth-one gamma formula and its analytic gradient, not quadrature.
    """
    d, t, n = 2, 10, 11
    T, q, r = np.array([[.7, .3], [.2, .8]]), np.array([.8, .2]), np.array([.6, .4])
    nu, _, _, _, exact_one, _ = binomial_profiles(T, q, r, n)
    tail_nu, nk, gk, tail, live, _ = row_tail_profiles(T, r, n)
    rs = np.arange(n+1)
    def l1(counts):
        k = int(np.sum(counts))
        return float((gammaln(k+d)-gammaln(d)-gammaln(np.asarray(counts)+1).sum())/LN2)
    def mean_cost(h, mean_k, mean_gamma):
        return float((mean_gamma-gammaln(mean_k)+gammaln(mean_k+d)-gammaln(d)
                      -h@gammaln(rs+1))/LN2)
    def gradient(mean_k):
        return (gammaln(rs+1)+rs*(digamma(mean_k)-digamma(mean_k+d)))/LN2
    exact_prob, _ = exact_profile_probabilities(T)
    small = sum(float(binom.pmf(k,t,r) @ sum(exact_prob[p]*l1(p+(0,)*(d-len(p)))
                for p in partitions(k) if len(p)<=d)) for k in range(1,EXACT_ROW_CAP+1))
    ut = np.array([mean_cost(h, k, g) for h,k,g in zip(tail_nu,nk,gk)])
    gradients = np.array([gradient(k) for k in nk])
    uf = exact_one[d]
    uc = math.log2(d)+small+tail[live]@ut
    gf = gradient(n)
    sums = np.zeros(5)
    maximum_upper_excess = 0.
    outcomes = 0
    pair = (r[:,None]*T).ravel()
    for a in range(t+1):
        for b in range(t-a+1):
            for c in range(t-a-b+1):
                flat = np.array([a,b,c,t-a-b-c])
                matrix_probability = math.factorial(t)/math.prod(math.factorial(int(v)) for v in flat)
                matrix_probability *= float(np.prod(pair**flat))
                M = flat.reshape(d,d)
                cv_c = uc
                for j,row in enumerate(M):
                    k = int(row.sum())
                    if k > EXACT_ROW_CAP:
                        h = np.bincount(row,minlength=n+1)
                        divergence = (gammaln(k)-gk[j])/LN2-l1(row)+ut[j]-gradients[j]@(h-tail_nu[j])
                        cv_c -= divergence
                actual_c = math.log2(d)+sum(l1(row) for row in M)
                for initial in range(d):
                    N = M.sum(axis=0).copy(); N[initial] += 1
                    actual_f = l1(N)
                    cv_f = actual_f+gf@(np.bincount(N,minlength=n+1)-nu[d])
                    probability = matrix_probability*q[initial]
                    sums += probability*np.array([1.,actual_f,cv_f,actual_c,cv_c])
                    maximum_upper_excess = max(maximum_upper_excess,cv_f-uf,cv_c-uc)
                    outcomes += 1
    error = max(abs(sums[0]-1),abs(sums[1]-sums[2]),abs(sums[3]-sums[4]),
                abs(sums[1]-exact_one[d]),abs(sums[3]-math.log2(d)-exact_one[:d].sum()))
    assert error < 1e-10 and maximum_upper_excess < 1e-10, (error,maximum_upper_excess)
    return dict(enumerated_matrix_and_initial_outcomes=outcomes,
        maximum_unbiasedness_and_exact_cost_error_bits=float(error),
        maximum_controlled_cost_above_upper_bits=float(maximum_upper_excess))


def fractional_batch(nu, nbar, gamma_mean, l_max, tables, step, gradient_indices=()):
    """One shared scale grid, all rows vectorized, mixture inside the logarithm."""
    rs = np.arange(nu.shape[1])
    grid = np.r_[np.arange(-220., -75., 4 * step),
                 np.arange(-75., 35. + step / 2, step)]
    widths = np.empty_like(grid)
    widths[1:-1] = (grid[2:] - grid[:-2]) / 2
    widths[0], widths[-1] = (grid[1] - grid[0]) / 2, (grid[-1] - grid[-2]) / 2
    logwidths = np.log(widths)
    ix = np.unique(np.r_[np.arange(0, len(grid), 2), len(grid) - 1])
    depths, maximum_error, minimum_drop = [], 0., math.inf
    gi = np.asarray(gradient_indices,dtype=int)
    depth_gradients = []
    gradient_error = 0.
    depth_one = None
    depth_start = time.monotonic()
    for L in range(1, l_max + 1):
        if L == 1:
            phi = gammaln(rs[:, None] + 1) - (rs[:, None] + 1) * np.logaddexp(0., grid)
        else:
            phi = tables.log_phi_matrix(L, list(map(int, rs)), grid)
        kernel = nbar[:, None] * grid + nu @ phi
        peak = kernel.max(axis=1)
        drop = peak - np.maximum(kernel[:, 0], kernel[:, -1])
        minimum_drop = min(minimum_drop, float(drop.min()))
        assert np.isfinite(kernel).all() and drop.min() > 35, (L, drop.min())
        logz = logsumexp(kernel + logwidths, axis=1)
        if len(gi):
            weights = np.exp(kernel[gi]+logwidths-logz[gi,None])
            # Remove the deterministic total-symbol direction, keeping K random.
            centered = phi[1:]-phi[0]+rs[1:,None]*grid
            g = weights @ centered.T
            depth_gradients.append(g)
            if L == 1:
                closed_gradient = (gammaln(rs[1:]+1)
                    +(digamma(nbar[gi])-digamma(nbar[gi]+nu[gi].sum(axis=1)))[:,None]*rs[1:])
                gradient_error = float(np.max(abs(g-closed_gradient))/LN2)
                assert gradient_error < 2e-6, gradient_error
        coarse = np.trapezoid(np.exp(kernel[:, ix] - peak[:, None]), grid[ix], axis=1)
        error = np.abs(logz - np.log(coarse) - peak) / LN2
        maximum_error = max(maximum_error, float(error.max()))
        if L == 1:
            depth_one = (gamma_mean - logz) / LN2
            closed = (gamma_mean - gammaln(nbar) + gammaln(nbar + nu.sum(axis=1))
                      - gammaln(nu.sum(axis=1)) - nu @ gammaln(rs + 1)) / LN2
            closed_error = float(np.max(abs(depth_one - closed)))
            assert closed_error < 2e-6, closed_error
        depths.append(logz)
        if L % 5 == 0 or L == l_max:
            print(f"fractional depths 1..{L}: {time.monotonic()-depth_start:.2f}s", flush=True)
    logz = logsumexp(np.asarray(depths), axis=0) - math.log(l_max)
    costs = (gamma_mean - logz) / LN2
    gradients = {}
    if len(gi):
        depth_weights = np.exp(np.asarray(depths)[:,gi]-logsumexp(np.asarray(depths)[:,gi],axis=0))
        combined = np.einsum('jm,jmr->mr',depth_weights,np.asarray(depth_gradients))/LN2
        gradients = {int(i):np.r_[0.,g] for i,g in zip(gi,combined)}
    return costs, depth_one, dict(
        integration_step=step, scale_log_interval=[float(grid[0]), float(grid[-1])],
        scale_grid_points=len(grid), coding_depth_cap=l_max,
        maximum_nested_grid_difference_bits=maximum_error,
        minimum_endpoint_drop_nats=minimum_drop,
        maximum_depth_one_closed_form_difference_bits=closed_error,
        maximum_depth_one_projected_gradient_error_bits=gradient_error,
        gradients_evaluated=len(gi),
        qualification="Nested grids and endpoint drops are numerical checks, not certified total integration bounds."), gradients


def run_case(args, source_path, near_tokens, profile_validation):
    started = time.monotonic()
    source = json.loads(source_path.read_text())
    assert source["d"] == 512 and (source["L"], source["alpha"]) in ((10, 1.5), (6, 1.5), (6, 1.))
    raw = ROOT / source["raw_path"]
    assert sha(raw) == source["raw_sha256"]
    ns = np.asarray(source["tokens"])
    checkpoint = int(np.argmin(abs(ns - near_tokens)))
    n, d = int(ns[checkpoint]), source["d"]
    t = n - 1
    q = prefix_array(raw, "q")
    Ts = prefix_array(raw, "transition_matrices", args.tables)
    occupations = prefix_array(raw, "expected_occupation_laws", args.tables)[:, checkpoint]
    old_costs = {key: prefix_array(raw, key, args.tables)[:, checkpoint] for key in (
        "count_frequency_codelength", "count_context_codelength", "chain_frequency_codelength", "chain_context_codelength")}
    configure_store(DEFAULT_STORE)
    from lsa.universal_tables import UniversalTables
    from lsa.codelength import default_l_max, depth_averaged_codelength_profiles, profile_of
    cap = default_l_max(d)
    assert cap == source["l_max"]
    fractional, lengths, gammas, cell_data = [], [], [], []
    tails, tail_lengths, tail_gammas = [], [], []
    samples, integer_examples = [], []
    small_profiles = [p for k in range(EXACT_ROW_CAP+1) for p in partitions(k)]
    profiles = set(small_profiles)
    for table_index, (T, r) in enumerate(zip(Ts, occupations)):
        assert np.max(abs(T.sum(axis=1) - 1)) < 1e-12
        state, accumulated = q.copy(), np.zeros(d)
        for _ in range(t):
            accumulated += state
            state = state @ T
        occupation_error = float(np.max(abs(accumulated / t - r)))
        assert occupation_error < 1e-12
        positive, tail_positive = positive_binomial_profiles(T, r, n)
        nu, nbar, gmean, active, exact_one, diagnostics = binomial_profiles(T, q, r, n, positive=positive)
        fractional.append(nu); lengths.append(nbar); gammas.append(gmean)
        tnu, tnbar, tgamma, tail, live, tdiagnostics = row_tail_profiles(T, r, n, positive=tail_positive)
        tails.append(tnu); tail_lengths.append(tnbar); tail_gammas.append(tgamma)
        profile_probabilities, pdiagnostics = exact_profile_probabilities(T)
        pair = (r[:, None] * T).ravel()
        assert abs(pair.sum() - 1) < 1e-12
        pair /= pair.sum()
        rng = np.random.default_rng([20260907, 5961, table_index, n])
        recorded = []
        for repetition in range(args.draws):
            M = rng.multinomial(t, pair).reshape(d, d)
            initial = int(rng.choice(d, p=q))
            N = M.sum(axis=0)
            N[initial] += 1
            fp = profile_of(N)
            row_totals = M.sum(axis=1)
            rows = Counter(profile_of(row) for row in M[row_totals > 0])
            profiles.add(fp); profiles.update(rows)
            tail_rows = tuple((int(a),profile_of(M[a])) for a in np.flatnonzero(row_totals>EXACT_ROW_CAP))
            recorded.append((fp, rows, tail_rows))
            if repetition == 0:
                for z in (N, M[int(np.argmax(row_totals))]):
                    integer_examples.append((np.bincount(z, minlength=n + 1), int(z.sum()), profile_of(z)))
        samples.append(recorded)
        v = r @ T
        entropy_rows = -(T * np.log2(T)).sum(axis=1)
        lower = dict(frequency=float(-q @ np.log2(q) - t * v @ np.log2(v)),
                     context=float(math.log2(d) + t * r @ entropy_rows))
        cell_data.append(dict(table_index=table_index, active=active, exact_one=exact_one,
                              tail=tail, tail_live=live, small_profile_probabilities=profile_probabilities,
                              row_probability=r, lower=lower, validation=dict(**diagnostics, **tdiagnostics,
                              **pdiagnostics,
                              maximum_direct_occupation_error=occupation_error)))
    preparation_seconds = time.monotonic() - started
    print(f"Prepared {args.tables} retained tables at n={n}: {len(profiles)} production profiles, "
          f"{preparation_seconds:.2f}s", flush=True)
    tail_start = args.tables * (d + 1)
    integer_start = tail_start + sum(len(v) for v in tails)
    nu_all = np.vstack(fractional + tails + [h[None, :] for h, _, _ in integer_examples])
    nbar_all = np.r_[np.concatenate(lengths + tail_lengths), [k for _, k, _ in integer_examples]]
    gamma_all = np.r_[np.concatenate(gammas + tail_gammas), [gammaln(k) for _, k, _ in integer_examples]]
    gradient_indices = [i*(d+1)+d for i in range(args.tables)]
    cursor = tail_start
    for entry, draws in zip(cell_data,samples):
        live = entry['tail_live']
        queried = {a for _,_,tail_rows in draws for a,_ in tail_rows}
        gradient_indices += [cursor+int(np.searchsorted(live,a)) for a in sorted(queried)]
        cursor += len(live)
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    integration_start = time.monotonic()
    try:
        refinements = []
        for refinement in range(3):
            step = args.step / 2**refinement
            costs, one, integration, gradients = fractional_batch(nu_all, nbar_all, gamma_all, cap, tables, step,
                                                                  gradient_indices=gradient_indices)
            refinements.append(dict(step=step, difference_bits=integration['maximum_nested_grid_difference_bits']))
            if integration['maximum_nested_grid_difference_bits'] < 2e-5:
                break
            print(f"Refine scale grid after difference {refinements[-1]['difference_bits']:.3g} bits", flush=True)
        else:
            raise ValueError(f"Scale integration remains unresolved: {refinements}")
        integration['refinement_attempts'] = refinements
    finally:
        tables.close()
    integration_seconds = time.monotonic() - integration_start
    evaluation_start = time.monotonic()
    evaluated = depth_averaged_codelength_profiles({p: p for p in profiles}, d=d,
        l_max=cap, jobs=1, tables_source="universal", universal_path=DEFAULT_STORE)
    evaluation_seconds = time.monotonic() - evaluation_start
    error = max(abs(costs[integer_start + j] + evaluated[p].log2_q_avg)
                for j, (_, _, p) in enumerate(integer_examples))
    assert error < 2e-4, error
    assert abs(evaluated[()].log2_q_avg) < 1e-8
    assert abs(evaluated[(1,)].log2_q_avg + math.log2(d)) < 1e-8
    two_mass = d * 2**evaluated[(2,)].log2_q_avg + d*(d-1) * 2**evaluated[(1, 1)].log2_q_avg
    assert abs(two_mass - 1) < 1e-8
    cells, all_f, all_c, all_controlled_f, all_controlled_c = [], [], [], [], []
    tail_cursor = tail_start
    for i, (entry, draws) in enumerate(zip(cell_data, samples)):
        first = i * (d + 1)
        active = entry.pop("active")
        exact_one = entry.pop("exact_one")
        tail = entry.pop("tail")
        live = entry.pop("tail_live")
        profile_probabilities = entry.pop("small_profile_probabilities")
        row_probability = entry.pop("row_probability")
        upper_f = float(costs[first + d])
        upper_c = float(math.log2(d) + active @ costs[first:first + d])
        upper_gap = upper_f - upper_c
        small_cost = 0.
        by_k = []
        for k in range(1, min(t, EXACT_ROW_CAP)+1):
            expected = sum(profile_probabilities[p] * (-evaluated[p].log2_q_avg) for p in partitions(k))
            contribution = float(binom.pmf(k, t, row_probability) @ expected)
            small_cost += contribution
            by_k.append(dict(row_total=k, expected_context_cost_contribution_bits=contribution))
        tail_base = tail_cursor
        tail_cost = float(tail[live] @ costs[tail_base:tail_base+len(live)])
        tail_cursor += len(live)
        hybrid_c = math.log2(d) + small_cost + tail_cost
        assert hybrid_c <= upper_c + 2e-6
        one_f = float(exact_one[d])
        one_c = float(math.log2(d) + exact_one[:d].sum())
        # A second exact upper bound follows from the depth-one mixture component.
        depth_one_upper_f = one_f + math.log2(cap)
        depth_one_upper_c = one_c + active.sum() * math.log2(cap)
        assert np.min(active * one[first:first+d] - exact_one[:d]) > -2e-6
        f = np.array([-evaluated[fp].log2_q_avg for fp, _, _ in draws])
        c = np.array([math.log2(d) - math.fsum(mult*evaluated[p].log2_q_avg
                     for p, mult in rows.items()) for _, rows, _ in draws])
        gap = f - c
        all_f.append(f); all_c.append(c)
        controlled_f, controlled_c = np.empty_like(f), np.empty_like(c)
        minimum_tail_divergence = math.inf
        for draw_index,(fp,_,tail_rows) in enumerate(draws):
            h = np.bincount(fp,minlength=n+1).astype(float)
            h[0] = d-len(fp)
            controlled_f[draw_index] = f[draw_index]+gradients[first+d]@(h-fractional[i][d])
            divergence_sum = 0.
            for a,profile in tail_rows:
                j = int(np.searchsorted(live,a))
                assert live[j] == a
                h = np.bincount(profile,minlength=n+1).astype(float)
                h[0] = d-len(profile)
                k = sum(profile)
                # log Z(observed)-log Z(mean)-gradient*(Phi-E Phi).
                divergence = ((gammaln(k)-tail_gammas[i][j])/LN2
                    +evaluated[profile].log2_q_avg+costs[tail_base+j]
                    -gradients[tail_base+j]@(h-tails[i][j]))
                minimum_tail_divergence = min(minimum_tail_divergence,float(divergence))
                divergence_sum += divergence
            controlled_c[draw_index] = hybrid_c-divergence_sum
        maximum_controlled_upper_excess = max(float(np.max(controlled_f-upper_f)),
                                              float(np.max(controlled_c-hybrid_c)))
        assert maximum_controlled_upper_excess < 3e-4
        assert minimum_tail_divergence > -2e-4
        controlled_gap = controlled_f-controlled_c
        all_controlled_f.append(controlled_f); all_controlled_c.append(controlled_c)
        cell = dict(**entry, tokens=n, transitions=t,
            mean_profile_upper_bits=dict(frequency=upper_f, context=upper_c),
            difference_of_upper_costs_bits=upper_gap,
            depth_one_expected_cost_bits=dict(frequency=one_f, context=one_c),
            depth_one_mixture_upper_bits=dict(frequency=depth_one_upper_f, context=depth_one_upper_c),
            auxiliary_samples=dict(draws=args.draws, frequency=moments(f), context=moments(c), gap=moments(gap),
                frequency_bits=f.tolist(), context_bits=c.tolist(), gap_bits=gap.tolist()),
            upper_minus_auxiliary_mean_bits=dict(frequency=upper_f-float(f.mean()), context=upper_c-float(c.mean())),
            upper_over_auxiliary_mean_minus_one=dict(frequency=upper_f/float(f.mean())-1, context=upper_c/float(c.mean())-1),
            upper_cost_gap_minus_auxiliary_gap_bits=upper_gap-float(gap.mean()),
            hybrid=dict(exact_row_total_cap=EXACT_ROW_CAP,
                context_upper_bits=hybrid_c, difference_of_upper_costs_bits=upper_f-hybrid_c,
                context_upper_minus_auxiliary_mean_bits=hybrid_c-float(c.mean()),
                context_upper_over_auxiliary_mean_minus_one=hybrid_c/float(c.mean())-1,
                upper_cost_gap_minus_auxiliary_gap_bits=upper_f-hybrid_c-float(gap.mean()),
                tightening_of_context_upper_bits=upper_c-hybrid_c,
                exact_small_row_cost_bits=small_cost, large_row_tail_upper_bits=tail_cost,
                expected_rows_above_exact_cap=float(tail.sum()),
                small_row_cost_by_total=by_k),
            variance_reduced_auxiliary_samples=dict(
                frequency=moments(controlled_f), context=moments(controlled_c), gap=moments(controlled_gap),
                frequency_bits=controlled_f.tolist(), context_bits=controlled_c.tolist(), gap_bits=controlled_gap.tolist(),
                hybrid_upper_minus_mean_bits=dict(frequency=upper_f-float(controlled_f.mean()),
                                                  context=hybrid_c-float(controlled_c.mean())),
                hybrid_upper_over_mean_minus_one=dict(frequency=upper_f/float(controlled_f.mean())-1,
                                                       context=hybrid_c/float(controlled_c.mean())-1),
                hybrid_gap_error_with_SE=dict(mean=upper_f-hybrid_c-float(controlled_gap.mean()),
                                              SE=moments(controlled_gap)['SE']),
                maximum_controlled_cost_above_upper_bits=maximum_controlled_upper_excess,
                minimum_observed_tail_divergence_bits=(minimum_tail_divergence if math.isfinite(minimum_tail_divergence) else None)),
            saved_single_draw_costs={key:float(value[i]) for key,value in old_costs.items()})
        cells.append(cell)
    all_f, all_c = np.array(all_f), np.array(all_c)
    all_controlled_f,all_controlled_c = np.array(all_controlled_f),np.array(all_controlled_c)
    # Monte Carlo uncertainty is conditional on these four fixed source tables.
    # Different tables need not have equal conditional variances.
    def conditional_average(values):
        return dict(mean=float(values.mean()),
            SE=float(np.sqrt(np.sum(np.var(values, axis=1, ddof=1)/args.draws)) / args.tables))
    uf = float(np.mean([x["mean_profile_upper_bits"]["frequency"] for x in cells]))
    uc = float(np.mean([x["mean_profile_upper_bits"]["context"] for x in cells]))
    hc = float(np.mean([x["hybrid"]["context_upper_bits"] for x in cells]))
    aggregation = dict(mean_profile_upper_bits=dict(frequency=uf, context=uc),
        difference_of_upper_costs_bits=uf-uc,
        auxiliary_conditional_mean=dict(frequency=conditional_average(all_f), context=conditional_average(all_c),
                                       gap=conditional_average(all_f-all_c)),
        upper_minus_auxiliary_mean_bits=dict(frequency=uf-float(all_f.mean()), context=uc-float(all_c.mean())),
        upper_over_auxiliary_mean_minus_one=dict(frequency=uf/float(all_f.mean())-1, context=uc/float(all_c.mean())-1),
        upper_cost_gap_minus_auxiliary_gap_bits=uf-uc-float((all_f-all_c).mean()),
        hybrid=dict(context_upper_bits=hc, difference_of_upper_costs_bits=uf-hc,
            context_upper_minus_auxiliary_mean_bits=hc-float(all_c.mean()),
            context_upper_over_auxiliary_mean_minus_one=hc/float(all_c.mean())-1,
            upper_cost_gap_minus_auxiliary_gap_bits=uf-hc-float((all_f-all_c).mean()),
            tightening_of_context_upper_bits=uc-hc,
            expected_rows_above_exact_cap=float(np.mean([x['hybrid']['expected_rows_above_exact_cap'] for x in cells]))))
    cf,cc,cgap = (conditional_average(values) for values in (all_controlled_f,all_controlled_c,all_controlled_f-all_controlled_c))
    aggregation['variance_reduced'] = dict(auxiliary_conditional_mean=dict(frequency=cf,context=cc,gap=cgap),
        hybrid_upper_minus_mean_bits=dict(frequency=uf-cf['mean'],context=hc-cc['mean']),
        hybrid_upper_over_mean_minus_one=dict(frequency=uf/cf['mean']-1,context=hc/cc['mean']-1),
        hybrid_gap_error_with_SE=dict(mean=uf-hc-cgap['mean'],SE=cgap['SE']),
        standard_error_reduction_factors={key:(aggregation['auxiliary_conditional_mean'][key]['SE']/v['SE']
            if v['SE'] else None) for key,v in [('frequency',cf),('context',cc),('gap',cgap)]})
    result = dict(schema_version=4, complete=True, d=d, alpha=source['alpha'], source_depth=source['L'],
        coding_depth_cap=cap, tokens=n, transitions=t, retained_tables=args.tables,
        selected_table_indices=list(range(args.tables)), auxiliary_draws_per_table=args.draws, se_ddof=1,
        source_summary=str(source_path.relative_to(ROOT)), source_summary_sha256=sha(source_path),
        source_raw=str(raw.relative_to(ROOT)), source_raw_sha256=source["raw_sha256"],
        producer_sha256=sha(Path(__file__)),
        expected_profile_helper_sha256=sha(Path(__file__).with_name("lsa_expected_profile.py")),
        evaluator_sha256=sha(DEFAULT_LSA / "lsa/codelength.py"),
        moment_anchor_sha256=sha(DEFAULT_STORE / "anchors.json"),
        seed_protocol="default_rng([20260907,5961,table_index,n]); per draw multinomial(t,vec(r_t*T)) then fresh categorical(q) initial token",
        checkpoint_selection="Saved token length minimizing its absolute distance from requested_near_tokens; no coefficient fitting",
        requested_near_tokens=near_tokens,
        integration=integration, validation=dict(maximum_integer_profile_production_difference_bits=float(error),
             two_token_probability=two_mass, production_profile_count=len(profiles),
             exact_profile_enumeration=profile_validation),
        exact_small_profile_costs=[dict(profile=list(p), cost_bits=float(-evaluated[p].log2_q_avg)) for p in small_profiles],
        cells=cells, aggregate=aggregation,
        timings_seconds=dict(preparation=preparation_seconds, fractional_integration=integration_seconds,
            production_profile_evaluation=evaluation_seconds, total=time.monotonic()-started),
        qualifications=[
            "The Jensen bound is exact before numerical integration. Grid and endpoint diagnostics are not certified integration bounds.",
            "Only saved source tables are reused; all new count and initial-token draws are independent of the archived draws.",
            "Frequency and copying costs have separate upper bounds. Their difference is neither an upper nor a lower bound on the true gap.",
            "This single checkpoint does not estimate a crossing displacement. The reported gap displacement is in bits.",
            "Standard errors describe new count/initial-token sampling conditional on the selected fixed tables, not population uncertainty over source tables.",
            "Every binomial count from zero through its exact finite support is included; no histogram tail is truncated.",
            "The hybrid copier upper bound uses exact profile probabilities for K=1..8 and Jensen only for K>8. No positive row tail is dropped.",
            "The injective power-sum recursion is exact algebraically; floating-point probability mass and independent enumeration checks are recorded.",
            "The variance-reduced estimates use exact zero-mean profile, log-Gamma, and row-tail-occupancy controls; no coefficient is fitted.",
            "Frequency control uses the exact mean profile. The copying control integrates small row totals exactly and samples only the tail log-partition divergence.",
            "The same count draws underlie raw and controlled costs. Their paired standard errors use the corresponding paired values.",
            "No source-table, chain, predictor, coding-depth range, or moment store was changed."],
        command_argv=sys.argv)
    print(json.dumps(dict(source_depth=source['L'], alpha=source['alpha'], tokens=n, tables=args.tables, aggregate=aggregation,
                         timings=result["timings_seconds"], validation=result["validation"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--near-tokens", type=int, default=80)
    parser.add_argument("--single", action="store_true", help="Only the original source at --near-tokens")
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results/mechanism/lsa_mean_profile_crossing_check.json")
    args = parser.parse_args()
    if args.draws < 2 or args.step <= 0:
        parser.error("At least two draws and a positive integration step are required")
    profile_validation = profile_checks()
    profile_validation['exact_gradient_control'] = control_checks()
    if args.self_test:
        print(json.dumps(profile_validation))
        return
    grid = [(SOURCE, args.near_tokens)]
    if not args.single:
        grid += [(SOURCE, 40), (SOURCE, 160),
                 (SOURCE.with_name('L6_a1.5_d512.json'), 423),
                 (SOURCE.with_name('L6_a1_d512.json'), 550)]
    cases = []
    started = time.monotonic()
    for path, near in grid:
        case = run_case(args, path, near, profile_validation)
        cases.append(case)
        # Preserve the original checkpoint fields for consumers of the first check.
        result = dict(cases[0])
        result.update(complete=len(cases)==len(grid), cases=cases,
                      requested_cases=[dict(source_summary=str(p.relative_to(ROOT)), near_tokens=v) for p, v in grid],
                      completed_cases=len(cases), total_elapsed_seconds=time.monotonic()-started)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

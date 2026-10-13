"""Independent checks of the matched-Bayes main-text figure.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/optimal_context_figure_audit.py
python code/scripts/optimal_context_figure_audit.py --self-test

No new source tables or Markov paths are sampled. The audit replays saved
source draws, reconstructs counts, and independently computes occupations,
cost identities and summaries. Binary depth-two row probabilities use an
explicit log-ratio density, without the production LSA moment tables.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from tilted_row_reference import TiltedRowReference


ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=3e-8, rtol=3e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), (a, b)
    return float(np.max(abs(a-b))) if a.size else 0.


def moments(x):
    x = np.asarray(x)
    means = np.array([math.fsum(col)/len(col) for col in x.T])
    errors = np.array([math.sqrt(math.fsum((col-m)**2)/(len(col)-1)/len(col))
                       for col, m in zip(x.T, means)])
    return dict(mean=means.tolist(), SE=errors.tolist())


def log_ratio_density(x, L):
    """Density of log(G_1/G_2), for G_i products of L exponentials.

    For L=1 it is logistic. For L=2 the convolution equals
    [x*coth(x/2)-2]/[4*sinh(x/2)^2]. The series avoids cancellation.
    """
    a = abs(x)
    z = math.exp(-a)
    if L == 1:
        return z/(1+z)**2
    assert L == 2
    if a < .05:
        return 1/6-a*a/60+a**4/1008-a**6/21600
    return z*(a*(1+z)/(1-z)-2)/(1-z)**2


def binary_row_probability(m, k, q, L):
    if not m+k:
        return 1.
    shift = math.log(q[0]/q[1])
    value, error = quad(lambda x: log_ratio_density(x, L)*expit(shift+x)**m
                        *expit(-shift-x)**k, -100, 100, epsabs=2e-12, epsrel=2e-12)
    assert error < 2e-10 and value > 0
    return value


def tiny_paths():
    checks = []
    for L, q0 in itertools.product((1, 2), (.5, .7)):
        q = np.array([q0, 1-q0])
        ref = TiltedRowReference(q, L, step=1/128)
        cache, maximum_row_error = {}, 0.
        try:
            def row_probability(row):
                nonlocal maximum_row_error
                key = tuple(int(x) for x in row)
                if key not in cache:
                    p = binary_row_probability(*key, q, L)
                    log_production, _ = ref.log_probability(np.array(key))
                    maximum_row_error = max(maximum_row_error,
                                            abs(log_production-math.log(p))/LN2)
                    cache[key] = p
                return cache[key]

            for n in range(1, 7):
                pc, pf = [], []
                for path in itertools.product(range(2), repeat=n):
                    counts = np.zeros((2, 2), dtype=int)
                    for a, b in zip(path[:-1], path[1:]):
                        counts[a, b] += 1
                    # Both proper codes use q for the first token. The
                    # frequency alternative shares one unknown iid row.
                    pc.append(q[path[0]]*math.prod(row_probability(row) for row in counts))
                    pf.append(q[path[0]]*row_probability(counts.sum(0)))
                pc, pf = np.array(pc), np.array(pf)
                close(pc.sum(), 1., atol=2e-10, rtol=0)
                close(pf.sum(), 1., atol=2e-10, rtol=0)
                hc = -math.fsum(pc*np.log2(pc))
                hf = -math.fsum(pc*np.log2(pf))
                kl = math.fsum(pc*np.log2(pc/pf))
                close(hf-hc, kl, atol=1e-12, rtol=0)
                assert kl >= -1e-10
                checks.append(dict(L=L, q=q.tolist(), tokens=n, paths=2**n,
                    context_mass=float(pc.sum()), frequency_mass=float(pf.sum()),
                    expected_frequency_minus_bayes_bits=hf-hc, KL_bits=kl,
                    maximum_row_probability_difference_bits=maximum_row_error))
            assert maximum_row_error < 1e-9
        finally:
            ref.close()
    return checks


def occupations(T, q, ts):
    """Direct iteration, with independently solved stationary tail completion."""
    d = len(q)
    A, b = T.T-np.eye(d), np.zeros(d)
    A[-1], b[-1] = 1., 1.
    pi = np.linalg.solve(A, b)
    assert pi.min() > 0
    state, total, j, result = q.copy(), np.zeros(d), 0, []
    for t in ts:
        while j < t and abs(state-pi).sum() >= 2e-14:
            total += state
            state = state@T
            j += 1
        result.append((total+(t-j)*pi)/t)
    residual = float(abs(pi@T-pi).sum())
    epsilon = float(T.min(0).sum())
    diagnostic = (float(abs(state-pi).sum())/min(ts)+residual)/epsilon
    return np.array(result), pi, dict(direct_steps=j, stationary_residual_l1=residual,
        minorization_mass=epsilon, tail_completion_diagnostic=diagnostic)


def scalar_values(x, record):
    x = np.asarray(x)
    knots = np.asarray(record['x'])
    result = np.interp(np.log(x), np.log(knots), record['r_bits'])
    low, high = x < knots[0], x > knots[-1]
    result[low] = record['slope0_bits']*x[low]
    result[high] = .5*np.log2(x[high])+record['tail_constant_bits']
    return result


def audit(source, tiny):
    data = json.loads(source.read_text())
    assert data['complete'] and (data['L'], data['alpha'], data['d'], data['repetitions']) == (2, 1., 256, 16)
    assert data['se_ddof'] == 1 and data['source_coding_cutoff'] == 26
    paths = {key:ROOT/data[key] for key in ('source', 'source_summary', 'raw_path')}
    for key, hash_key in [('source', 'source_sha256'), ('source_summary', 'source_summary_sha256'), ('raw_path', 'raw_sha256')]:
        assert digest(paths[key]) == data[hash_key]
    for filename, field in [('optimal_context_figure_check.py', 'script_sha256'),
                            ('tilted_row_reference.py', 'row_evaluator_sha256'),
                            ('iid_depth_prediction.py', 'scalar_evaluator_sha256')]:
        assert digest(Path(__file__).with_name(filename)) == data[field]
    old_summary = json.loads(paths['source_summary'].read_text())
    with np.load(paths['source']) as z:
        old = {k:z[k] for k in z.files}
    with np.load(paths['raw_path']) as z:
        raw = {k:z[k] for k in z.files}
    ns, ts, q = raw['tokens'], raw['transitions'], raw['q']
    close(ns, old['tokens'], atol=0, rtol=0)
    close(ts, ns-1, atol=0, rtol=0)
    target_q = 1/np.arange(1, 257, dtype=float)
    target_q /= target_q.sum()
    close(q, target_q, atol=1e-14, rtol=0)
    Hq = -math.fsum(q*np.log2(q))
    expected, realized = np.zeros((16, 10)), np.zeros((16, 10))
    jfinite, jstationary = np.zeros((16, 10)), np.zeros((16, 10))
    occupation_checks, source_error, occupation_error = [], 0., 0.
    for rep in range(16):
        prefix = [old_summary['seed'], 256, 2, 100, rep]
        G = np.random.default_rng(prefix+[77]).exponential(size=(256, 256, 2)).prod(2)
        S = G@q
        T = G*q[None, :]/S[:, None]
        source_error = max(source_error, close(S, old['normalized_row_sums'][rep]),
                           close(T, old['transition_matrices'][rep]))
        initial = int(np.random.default_rng(prefix+[78]).choice(256, p=q))
        assert initial == old['initial_states'][rep]
        initial_cost = -math.log2(q[initial])
        close(initial_cost, raw['initial_bits'][rep])
        r, pi, diagnostic = occupations(T, q, ts)
        occupation_error = max(occupation_error, close(r, raw['expected_occupation_laws'][rep]))
        close(pi, old['stationary_laws'][rep])
        occupation_checks.append(diagnostic)
        h = -np.sum(T*np.log2(T), axis=1)
        close(h, raw['row_entropies'][rep])
        expected[rep] = Hq+ts*(r@h)
        path = old['chain_paths'][rep]
        assert path[0] == initial
        for j, t in enumerate(ts):
            counts = np.bincount(path[:t].astype(np.int64)*256+path[1:t+1], minlength=256**2).reshape(256, 256)
            close(counts, old['chain_transition_counts'][rep, j], atol=0, rtol=0)
            realized[rep, j] = initial_cost-math.fsum((counts*np.log2(T)).ravel())
            jfinite[rep, j] = math.fsum(scalar_values(t*r[j, :, None]*q[None, :]/S[:, None], data['scalar_curve']).ravel())
            jstationary[rep, j] = math.fsum(scalar_values(t*pi[:, None]*q[None, :]/S[:, None], data['scalar_curve']).ravel())
    close(expected, raw['known_table_expected_full_cost'])
    close(realized, raw['known_table_realized_full_cost'])
    close(jfinite, raw['J_finite_occupation'])
    close(jstationary, raw['J_stationary'])
    bayes = raw['initial_bits'][:, None]+raw['bayes_row_costs'].sum(2)
    redundancy = bayes-realized
    close(bayes, raw['bayes_full_cost'])
    close(redundancy, raw['bayes_redundancy'])
    close(expected+redundancy, raw['bayes_control_variate_full_cost'])
    close(expected+jfinite, raw['predicted_finite_full_cost'])
    close(expected+jstationary, raw['predicted_stationary_full_cost'])
    close(old['chain_frequency_codelength'], raw['lsa_frequency_full_cost'], atol=0, rtol=0)
    close(old['chain_context_codelength'], raw['lsa_context_full_cost'], atol=0, rtol=0)
    for key, record in data['costs'].items():
        check = moments(raw[key])
        close(check['mean'], record['mean'])
        close(check['SE'], record['SE'])
    contrasts = {'redundancy_minus_J_finite_occupation':redundancy-jfinite,
        'redundancy_minus_J_stationary':redundancy-jstationary,
        'bayes_raw_minus_finite_prediction':bayes-expected-jfinite,
        'lsa_frequency_minus_bayes':raw['lsa_frequency_full_cost']-bayes,
        'lsa_context_minus_bayes':raw['lsa_context_full_cost']-bayes,
        'lsa_frequency_minus_bayes_control_variate':raw['lsa_frequency_full_cost']-expected-redundancy,
        'lsa_context_minus_bayes_control_variate':raw['lsa_context_full_cost']-expected-redundancy}
    for name, value in contrasts.items():
        check = moments(value)
        close(check['mean'], data['paired_differences'][name]['mean'])
        close(check['SE'], data['paired_differences'][name]['SE'])
    close(redundancy.mean(0)/jfinite.mean(0)-1, data['measured_redundancy_over_finite_prediction_minus_one'])
    # Re-evaluate saved d=256 rows on a different outer grid and wider range.
    refined_rows = []
    ref = TiltedRowReference(q, 2, step=1/256, lower=-68., upper=34.)
    try:
        for rep in (0, 5, 10, 15):
            for j in (0, 4, 9):
                counts = old['chain_transition_counts'][rep, j]
                k = counts.sum(1)
                occupied = np.flatnonzero(k)
                for a in set([int(np.argmax(k)), int(occupied[len(occupied)//2])]):
                    value, _ = ref.log_probability(counts[a])
                    error = -value/LN2-raw['bayes_row_costs'][rep, j, a]
                    assert abs(error) < 1e-7
                    refined_rows.append(dict(repetition=rep, checkpoint=j, context=a,
                        observations=int(k[a]), difference_bits=float(error)))
    finally:
        ref.close()
    return dict(status='passed', source=str(source.relative_to(ROOT)), source_sha256=digest(source),
        auditor_sha256=digest(Path(__file__)), raw_sha256=data['raw_sha256'],
        checked_tables=16, checked_checkpoint_means=10, checked_paths_and_count_matrices=160,
        maximum_source_replay_error=source_error, maximum_occupation_difference=occupation_error,
        occupation_checks=occupation_checks, independent_binary_path_checks=tiny,
        refined_saved_rows=refined_rows,
        redundancy_relative_errors=(redundancy.mean(0)/jfinite.mean(0)-1).tolist(),
        scope=['No new source tables or Markov paths.',
            'Independent exact binary L1/L2 row integrals, path normalization and expected-cost/KL identity.',
            'Every saved source, count, initial charge, conditional expected oracle cost, scalar sum and summary checked.',
            'A selected set of d256 Bayes rows is reevaluated on a finer outer grid; production moment tables are shared for these large-row checks.',
            'Scalar values use the independently computed production scalar curve; this audit checks their arguments and sums, not a new derivation of that scalar curve.',
            'Refinement and stationary tail-completion diagnostics are floating-point checks, not certified error bounds.',
            'No assertion that finite empirical mean gaps certify Bayes optimality; the exact KL identity supplies that statement.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/mechanism/optimal_context_figure_check.json')
    parser.add_argument('--out', type=Path, default=ROOT/'results/mechanism/optimal_context_figure_audit.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    tiny = tiny_paths()
    if args.self_test:
        print(json.dumps(dict(status='passed', checks=tiny), indent=2))
        return
    result = audit(args.source, tiny)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status', 'checked_tables', 'checked_checkpoint_means',
        'maximum_source_replay_error', 'maximum_occupation_difference']}, indent=2))


if __name__ == '__main__':
    main()

"""Deterministic full-LSA gap on retained source tables, without new draws.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_gap_deterministic.py

Start with one retained d512, source-depth10, alpha1.5 table at n49.
All row totals1..8 are averaged exactly. Larger row-total terms use the
single-distribution mean-profile integral and full covariance correction.
Omitted tiny terms carry an explicit summed upper bound on their true cost.
The first token is included using exact shifted-multinomial multiplicities.
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
from scipy.special import gammaln, xlogy
from scipy.stats import binom

from lsa_expected_profile import fractional_cost
from lsa_single_covariance import count_multiplicity_moments, compositions
from lsa_single_validation import direct_trace_correction
from lsa_mean_profile_crossing_check import (
    prefix_array, exact_profile_probabilities, partitions, EXACT_ROW_CAP)
from tilted_row_reference import DEFAULT_STORE, configure_store

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shifted_moments(p, q, t, memory_mb=32.):
    """Multiplicity moments of Y+e_I, Y~Mult(t,p), I~q independent."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    assert p.shape == q.shape and np.all(p >= 0) and np.all(q >= 0)
    assert abs(math.fsum(p)-1) < 1e-12 and abs(math.fsum(q)-1) < 1e-12
    p, q = p/math.fsum(p), q/math.fsum(q)
    d, n = len(p), t+1
    all_r = np.arange(n+1)
    nu = ((1-q)*binom.pmf(all_r[:, None], t, p)+
          q*binom.pmf(all_r[:, None]-1, t, p)).sum(1)
    core = np.arange(2, n+1)
    covariance = np.zeros((n+1, n+1))
    if len(core):
        ri, si = np.meshgrid(np.arange(len(core)), np.arange(len(core)), indexing="ij")
        live = core[ri]+core[si] <= n
        ri, si = ri[live], si[live]
        rr, ss = core[ri], core[si]
        ai, bi = np.triu_indices(d, 1)
        rest = np.asarray([math.fsum(p[j] for j in range(d) if j != a and j != b)
                           for a, b in zip(ai, bi)])
        batch = max(1, int(memory_mb*1024**2)//max(1, 5*len(rr)*8))
        summed = np.zeros(len(rr), dtype=np.longdouble)
        for start in range(0, len(ai), batch):
            stop = min(start+batch, len(ai))
            a, b = ai[start:stop], bi[start:stop]
            total = np.zeros((len(a), len(rr)))
            for shift_a, shift_b, weight in [
                    (0, 0, 1-q[a]-q[b]), (1, 0, q[a]), (0, 1, q[b])]:
                r, s = rr-shift_a, ss-shift_b
                remaining = t-r-s
                valid = (r >= 0) & (s >= 0) & (remaining >= 0)
                logprob = (gammaln(t+1)-gammaln(r[valid]+1)-gammaln(s[valid]+1)
                           -gammaln(remaining[valid]+1))[None, :]
                logprob = (logprob+xlogy(r[valid], p[a, None])+xlogy(s[valid], p[b, None])
                           +xlogy(remaining[valid], rest[start:stop, None]))
                total[:, valid] += weight[:, None]*np.exp(logprob)
            summed += total.sum(0, dtype=np.longdouble)
        one_order = np.zeros((len(core), len(core)), dtype=np.longdouble)
        one_order[ri, si] = summed
        means = np.asarray(nu[core], dtype=np.longdouble)
        second = one_order+one_order.T+np.diag(means)
        cov_core = second-means[:, None]*means[None, :]
        transform = np.zeros((n+1, len(core)), dtype=np.longdouble)
        transform[core, np.arange(len(core))] = 1
        transform[1] = -core
        transform[0] = core-1
        covariance = np.asarray(transform @ cov_core @ transform.T, dtype=float)
        covariance = (covariance+covariance.T)/2
    return nu, covariance


def check_shifted_moments():
    """Independent finite enumeration of the newly needed initial-token law."""
    error_mean = error_cov = 0.
    outcomes = 0
    for p, q in [([.6, .3, .1], [.1, .2, .7]), ([.8, .2, 0.], [0., .3, .7])]:
        for t in range(6):
            records = []
            for z in compositions(t, len(p)):
                probability = math.factorial(t)/math.prod(math.factorial(k) for k in z)
                probability *= math.prod(x**k for x, k in zip(p, z))
                for initial, weight in enumerate(q):
                    counts = np.asarray(z).copy(); counts[initial] += 1
                    profile = np.bincount(counts, minlength=t+2)
                    records.append((probability*weight, profile))
                    outcomes += 1
            mean = sum(probability*profile for probability, profile in records)
            covariance = sum(probability*np.outer(profile-mean, profile-mean)
                             for probability, profile in records)
            nu, cov = shifted_moments(p, q, t)
            error_mean = max(error_mean, float(abs(nu-mean).max()))
            error_cov = max(error_cov, float(abs(cov-covariance).max()))
    assert error_mean < 1e-11 and error_cov < 1e-10
    return dict(cases=12, enumerated_outcomes=outcomes,
                maximum_mean_error=error_mean, maximum_covariance_error=error_cov)


class KernelCache:
    def __init__(self, tables, largest_count):
        self.tables, self.largest_count = tables, largest_count
        self.values = {}

    def log_phi_matrix(self, depth, counts, grid):
        assert list(counts) == list(range(len(counts)))
        key = (depth, grid.tobytes())
        if key not in self.values:
            self.values[key] = self.tables.log_phi_matrix(depth,
                list(range(self.largest_count+1)), grid)
        return self.values[key][:len(counts)]


def corrected_cost(nu, cov, k, cap, cache):
    upper = fractional_cost(nu, k, cap, cache, covariance=None)
    correction = direct_trace_correction(nu, cov, k, cap, cache)
    assert correction > -1e-6
    return dict(upper_bits=upper["jensen_bits"], correction_bits=correction,
        corrected_bits=upper["jensen_bits"]-correction,
        nested_grid_difference_bits=upper["maximum_nested_grid_difference_bits"])


def sample_summary(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()),
                SE=float(values.std(ddof=1)/math.sqrt(len(values))), draws=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near", type=int, nargs="+", default=[40])
    parser.add_argument("--table-index", type=int, default=0)
    parser.add_argument("--tail-cost-budget", type=float, default=1e-5)
    parser.add_argument("--out", type=Path,
        default=ROOT/"results/mechanism/lsa_gap_deterministic_check.json")
    args = parser.parse_args()
    started = time.monotonic()
    validation = check_shifted_moments()
    source_path = ROOT/"output/crossing_lsa_extension/refined256/L10_a1.5_d512.json"
    source = json.loads(source_path.read_text())
    raw = ROOT/source["raw_path"]
    assert sha(raw) == source["raw_sha256"]
    saved_path = ROOT/"results/mechanism/lsa_mean_profile_crossing_check.json"
    saved = json.loads(saved_path.read_text())
    index = args.table_index
    T = prefix_array(raw, "transition_matrices", index+1)[index]
    q = prefix_array(raw, "q")
    d, cap = source["d"], source["l_max"]
    configure_store(DEFAULT_STORE)
    from lsa.universal_tables import UniversalTables
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    profile_probabilities, probability_checks = exact_profile_probabilities(T)
    result = dict(schema_version=1, complete=False, target="Conditional-on-one-retained-T auxiliary LSA gap",
        d=d, alpha=source["alpha"], source_depth=source["L"], coding_depth_cap=cap,
        table_index=index, requested_near_tokens=args.near,
        source_path=str(source_path.relative_to(ROOT)), source_sha256=sha(source_path),
        raw_path=str(raw.relative_to(ROOT)), raw_sha256=source["raw_sha256"],
        saved_auxiliary_path=str(saved_path.relative_to(ROOT)), saved_auxiliary_sha256=sha(saved_path),
        producer_sha256=sha(__file__),
        helper_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ["lsa_single_covariance.py", "lsa_single_validation.py", "lsa_expected_profile.py", "lsa_mean_profile_crossing_check.py"]},
        command_argv=sys.argv, shifted_moment_validation=validation,
        exact_small_profile_probability_checks=probability_checks, cases=[],
        qualifications=[
            "No new source, chain, token, or auxiliary count draws.",
            "This conditions on one saved source table; it is not a parameter-only source average.",
            "Full normalized probabilities and the original finite coding-depth mixture are retained.",
            "Row totals1..8 are exact finite profile averages; larger retained totals use exact covariance and a full Hessian trace.",
            "Omitted binomial terms have an explicit summed upper bound on their TRUE expected code cost; no rare probability is silently discarded.",
            "The curvature approximation error is separate and not bounded by the tail-cost budget.",
            "The saved auxiliary comparison uses32 draws conditional on the same T; the saved chain cost is one path, so has no conditional SE.",
            "Nested quadrature checks are not certified integral-error bounds.",
        ])
    for near in args.near:
        checkpoint = int(np.argmin(abs(np.asarray(source["tokens"])-near)))
        n, t = source["tokens"][checkpoint], source["tokens"][checkpoint]-1
        old = next(case for case in saved["cases"] if case["source_depth"] == 10 and case["tokens"] == n)
        cell = old["cells"][index]
        costs = {tuple(x["profile"]):x["cost_bits"] for x in old["exact_small_profile_costs"]}
        r = prefix_array(raw, "expected_occupation_laws", index+1)[index, checkpoint]
        state, total = q.copy(), np.zeros(d)
        for _ in range(t):
            total += state; state = state @ T
        occupation_error = float(abs(total/t-r).max()); assert occupation_error < 1e-12
        v = r @ T
        cache = KernelCache(tables, n)
        nu, covariance = shifted_moments(v, q, t)
        frequency = corrected_cost(nu, covariance, n, cap, cache)
        small_cost = 0.
        for k in range(1, min(t, EXACT_ROW_CAP)+1):
            expected = sum(profile_probabilities[p]*costs[p] for p in partitions(k))
            small_cost += float(binom.pmf(k, t, r) @ expected)
        terms = []
        for k in range(EXACT_ROW_CAP+1, t+1):
            weight = binom.pmf(k, t, r)
            cost_bound = (gammaln(k+d)-gammaln(d))/LN2+math.log2(cap)
            terms.extend((float(w*cost_bound), a, k, float(w)) for a, w in enumerate(weight))
        terms.sort()
        omitted = 0.
        boundary = 0
        while boundary < len(terms) and omitted+terms[boundary][0] <= args.tail_cost_budget:
            omitted += terms[boundary][0]; boundary += 1
        selected = terms[boundary:]
        print(f"n={n}, table={index}: {len(selected)} retained row-total terms; "
              f"omitted true-cost bound {omitted:.3g} bits", flush=True)
        row_results, tail_corrected, tail_upper = [], 0., 0.
        for j, (_, a, k, weight) in enumerate(selected):
            moment = count_multiplicity_moments(T[a], k)
            evaluation = corrected_cost(moment.mean, moment.covariance, k, cap, cache)
            tail_corrected += weight*evaluation["corrected_bits"]
            tail_upper += weight*evaluation["upper_bits"]
            row_results.append(dict(context=a, k=k, binomial_weight=weight, **evaluation))
            if (j+1) % 25 == 0:
                print(f"n={n}: {j+1}/{len(selected)} terms, elapsed {time.monotonic()-started:.1f}s", flush=True)
        copying = math.log2(d)+small_cost+tail_corrected
        predicted_gap = frequency["corrected_bits"]-copying
        auxiliary = cell["variance_reduced_auxiliary_samples"]
        saved_gap = sample_summary(auxiliary["gap_bits"])
        saved_frequency = sample_summary(auxiliary["frequency_bits"])
        saved_copying = sample_summary(auxiliary["context_bits"])
        copying_upper = math.log2(d)+small_cost+tail_upper
        chain_f = float(prefix_array(raw, "chain_frequency_codelength", index+1)[index, checkpoint])
        chain_c = float(prefix_array(raw, "chain_context_codelength", index+1)[index, checkpoint])
        result["cases"].append(dict(n=n, transitions=t, maximum_occupation_error=occupation_error,
            unigram=frequency, exact_small_row_cost_bits=small_cost,
            corrected_large_row_cost_bits=tail_corrected,
            upper_large_row_cost_bits=tail_upper,
            copying_corrected_bits=copying, predicted_gap_bits=predicted_gap,
            copying_uncorrected_bits=copying_upper,
            uncorrected_gap_bits=frequency["upper_bits"]-copying_upper,
            gap_curvature_adjustment_bits=predicted_gap-(frequency["upper_bits"]-copying_upper),
            omitted_true_copying_cost_interval_bits=[0., omitted],
            omitted_gap_contribution_interval_bits=[-omitted, 0.],
            omitted_row_total_terms=boundary, retained_row_total_terms=len(selected),
            saved_auxiliary_gap=saved_gap,
            saved_auxiliary_unigram=saved_frequency, saved_auxiliary_copying=saved_copying,
            unigram_corrected_minus_saved_mean_bits=frequency["corrected_bits"]-saved_frequency["mean"],
            copying_corrected_minus_saved_mean_bits=copying-saved_copying["mean"],
            predicted_minus_saved_auxiliary_gap_bits=predicted_gap-saved_gap["mean"],
            saved_chain_gap_bits=chain_f-chain_c,
            predicted_minus_saved_chain_gap_bits=predicted_gap-(chain_f-chain_c),
            saved_chain_unigram_bits=chain_f, saved_chain_copying_bits=chain_c,
            retained_term_evaluations=row_results, elapsed_seconds=time.monotonic()-started))
        result["complete"] = len(result["cases"]) == len(args.near)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key:value for key,value in result["cases"][-1].items()
                         if key != "retained_term_evaluations"}), flush=True)
    tables.close()


if __name__ == "__main__":
    main()

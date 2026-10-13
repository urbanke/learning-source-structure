"""Compare finite-row expressions with the proper Bayes cost of actual chains.

OPENBLAS_NUM_THREADS=1 python code/scripts/context_finite_prediction_check.py
python code/scripts/context_finite_prediction_check.py --self-test

The initial state has law q independently of the random transition table.
The initial q factor cancels from the known-table/Bayes log ratio. All
comparisons use the sampled table's own stationary law and normalizers;
the actual-visit comparison uses the observed departure counts as well.
These are empirical tests of proposed chain approximations, not an
assumption that a fixed-sample iid row formula applies at stopping times.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln

from context_stationary_check import stationary_checked
from tilted_row_reference import DEFAULT_STORE, TiltedRowReference


ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.)
PREDICTIONS = ("old_stationary", "joint_stationary", "joint_actual_visits")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(array):
    return dict(mean=np.mean(array, axis=0).tolist(),
                SE=(np.std(array, axis=0, ddof=1) / math.sqrt(len(array))).tolist())


class ScalarCurve:
    def __init__(self, record):
        self.x = np.asarray(record["x"])
        self.logx = np.log(self.x)
        self.y = np.asarray(record["r_bits"])
        self.slope0 = record["slope0_bits"]
        self.tail = record["tail_constant_bits"]

    def __call__(self, rates):
        rates = np.asarray(rates, dtype=float)
        assert np.all(rates >= 0) and np.isfinite(rates).all()
        values = np.zeros_like(rates)
        positive = rates > 0
        values[positive] = np.interp(np.log(rates[positive]), self.logx, self.y)
        low, high = (rates > 0) & (rates < self.x[0]), rates > self.x[-1]
        values[low] = self.slope0 * rates[low]
        values[high] = .5 * np.log2(rates[high]) + self.tail
        return values


def self_test(store):
    # Independent closed-form integration for uniform depth-one row priors.
    q = np.array([.5, .5])
    ref = TiltedRowReference(q, 1, store=store, step=1/128)
    maximum_log_error, maximum_mass_error = 0., 0.
    cache = {}
    try:
        for t in range(5):
            total_mass = 0.
            for states in itertools.product(range(2), repeat=t+1):
                counts = np.zeros((2, 2), dtype=np.int64)
                for a, b in zip(states[:-1], states[1:]):
                    counts[a, b] += 1
                log_bayes = math.log(q[states[0]])
                exact_log_bayes = log_bayes
                for row in counts:
                    key = tuple(row)
                    if key not in cache:
                        cache[key] = ref.log_probability(row)[0]
                    log_bayes += cache[key]
                    exact_log_bayes += float(gammaln(row+1).sum()-gammaln(row.sum()+2))
                maximum_log_error = max(maximum_log_error, abs(log_bayes-exact_log_bayes)/LN2)
                total_mass += math.exp(log_bayes)
                flow = np.bincount([states[0]], minlength=2)-np.bincount([states[-1]], minlength=2)
                assert np.array_equal(counts.sum(1)-counts.sum(0), flow)
            maximum_mass_error = max(maximum_mass_error, abs(total_mass-1))
    finally:
        ref.close()
    assert maximum_log_error < 1e-10 and maximum_mass_error < 1e-10
    return dict(maximum_closed_form_log_error_bits=maximum_log_error,
                maximum_total_path_probability_error=maximum_mass_error,
                scope="All binary paths through four transitions, uniform depth-one row priors, independent initial q.")


def run_cell(args, L, alpha, d, curve, scalar_hash):
    path = args.raw_dir / f"L{L}_a{alpha:g}_d{d}.json"
    npz_path = path.with_suffix(".npz")
    ts = np.asarray([d*d//4, d*d, 4*d*d], dtype=np.int64)
    expected = dict(L=L, alpha=alpha, d=d, repetitions=args.reps,
                    seed=args.seed, step=args.step, transitions=ts.tolist(),
                    scalar_sha256=scalar_hash, experiment_sha256=digest(Path(__file__)))
    if path.exists() and npz_path.exists():
        previous = json.loads(path.read_text())
        if previous.get("complete") and previous.get("configuration") == expected:
            return previous
    start = time.monotonic()
    weights = np.arange(1, d+1, dtype=float) ** (-alpha)
    q = weights / weights.sum()
    initial_cdf = np.cumsum(q)
    initial_cdf[-1] = 1.
    shape = (args.reps, len(ts))
    costs = {name: np.zeros(shape) for name in ("redundancy", "oracle", "bayes", *PREDICTIONS)}
    counts_saved = np.zeros((args.reps, len(ts), d, d), dtype=np.int32)
    matrices = np.zeros((args.reps, d, d))
    normalizers = np.zeros((args.reps, d))
    pis = np.zeros_like(normalizers)
    visit_counts = np.zeros((args.reps, len(ts), d), dtype=np.int64)
    initials = np.zeros(args.reps, dtype=np.int32)
    final_states = np.zeros(shape, dtype=np.int32)
    initial_cost = np.zeros(args.reps)
    visits_pi_tv = np.zeros(shape)
    visits_q_tv = np.zeros(shape)
    pi_q_tv = np.zeros(args.reps)
    information = np.zeros(args.reps)
    refinement = np.zeros(shape)
    max_row_refinement = np.zeros(shape)
    stationarity = []
    ref = TiltedRowReference(q, L, store=args.store, step=args.step)
    try:
        for rep in range(args.reps):
            table_rng = np.random.default_rng([args.seed, L, d, rep, 0])
            sample_rng = np.random.default_rng([args.seed, L, d, rep, 1])
            G = table_rng.exponential(size=(d, d, L)).prod(axis=2)
            Z = G @ weights
            T = G * weights[None, :] / Z[:, None]
            assert np.all(T > 0) and np.max(abs(T.sum(1)-1)) < 1e-12
            pi, diagnostic = stationary_checked(T)
            stationarity.append(diagnostic)
            matrices[rep], normalizers[rep], pis[rep] = T, Z, pi
            pi_q_tv[rep] = .5*abs(pi-q).sum()
            positive_pi = pi > 0
            information[rep] = float(-np.sum(pi[positive_pi]*np.log2(pi[positive_pi]))
                                     + np.sum(pi[:, None]*T*np.log2(T)))
            cdf = np.cumsum(T, axis=1)
            cdf[:, -1] = 1.
            uniforms = sample_rng.random(int(ts[-1])+1)
            x = int(np.searchsorted(initial_cdf, uniforms[0], side="right"))
            initials[rep] = x
            initial_cost[rep] = -math.log2(q[x])
            origin = np.empty(int(ts[-1]), dtype=np.int32)
            destination = np.empty_like(origin)
            for i, uniform in enumerate(uniforms[1:]):
                y = int(np.searchsorted(cdf[x], uniform, side="right"))
                origin[i], destination[i] = x, y
                x = y
            counts = np.zeros((d, d), dtype=np.int64)
            previous = 0
            for j, t in enumerate(ts):
                flat = origin[previous:t].astype(np.int64)*d+destination[previous:t]
                counts += np.bincount(flat, minlength=d*d).reshape(d, d)
                previous = int(t)
                K = counts.sum(1)
                assert K.sum() == t
                terminal = int(destination[t-1])
                flow = np.bincount([initials[rep]], minlength=d)-np.bincount([terminal], minlength=d)
                assert np.array_equal(K-counts.sum(0), flow)
                total_log_mixture = 0.
                errors = []
                for a in np.flatnonzero(K):
                    value, diag = ref.log_probability(counts[a])
                    total_log_mixture += value
                    errors.append(diag["coarse_fine_difference_bits"])
                true_log_probability = float(np.sum(counts*np.log(T)))
                costs["oracle"][rep, j] = -true_log_probability/LN2
                costs["bayes"][rep, j] = -total_log_mixture/LN2
                costs["redundancy"][rep, j] = (true_log_probability-total_log_mixture)/LN2
                costs["old_stationary"][rep, j] = float(curve(t*pi[:, None]*q[None, :]).sum())
                scale = weights[None, :]/Z[:, None]
                costs["joint_stationary"][rep, j] = float(curve(t*pi[:, None]*scale).sum())
                costs["joint_actual_visits"][rep, j] = float(curve(K[:, None]*scale).sum())
                refinement[rep, j] = sum(errors)
                max_row_refinement[rep, j] = max(errors, default=0.)
                counts_saved[rep, j], visit_counts[rep, j] = counts, K
                final_states[rep, j] = terminal
                visits_pi_tv[rep, j] = .5*abs(K/t-pi).sum()
                visits_q_tv[rep, j] = .5*abs(K/t-q).sum()
    finally:
        ref.close()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **costs, transition_counts=counts_saved,
        transition_matrices=matrices, unnormalized_row_sums=normalizers,
        normalized_row_sums=normalizers/weights.sum(), stationary_laws=pis,
        visit_counts=visit_counts, initial_states=initials, terminal_states=final_states,
        initial_cost=initial_cost, visits_to_pi_TV=visits_pi_tv,
        visits_to_q_TV=visits_q_tv, pi_to_q_TV=pi_q_tv,
        stationary_information_bits=information, refinement_sum_bits=refinement,
        maximum_row_refinement_bits=max_row_refinement, q=q, transitions=ts)
    comparisons = {}
    for name in PREDICTIONS:
        r, p = costs["redundancy"], costs[name]
        mean_r, mean_p = r.mean(0), p.mean(0)
        contrast = r-p
        ratio_influence = (r-(mean_r/mean_p)*p)/mean_p
        comparisons[name] = dict(prediction=moments(p), difference=moments(contrast),
            measured_over_prediction_minus_one=(mean_r/mean_p-1).tolist(),
            relative_difference_delta_method_SE=(ratio_influence.std(0, ddof=1)/math.sqrt(args.reps)).tolist(),
            measured_prediction_sample_covariance=[float(np.cov(r[:, j], p[:, j], ddof=1)[0, 1]) for j in range(len(ts))])
    result = dict(complete=True, configuration=expected,
        initialization="X0 is drawn from q independently of the transition table; initial q costs cancel.",
        code="Matched-prior Bayes path probability q(X0) times the product of exact tilted row integrals.",
        seed_protocol="Weights default_rng([seed,L,d,rep,0]); initial/sample uniforms use independent stream ending in1. Alpha settings share weights/uniforms at a given L,d; different d use different streams.",
        se_ddof=1, transitions=ts.tolist(), redundancy=moments(costs["redundancy"]),
        transition_oracle_cost=moments(costs["oracle"]), transition_bayes_cost=moments(costs["bayes"]),
        comparisons=comparisons, mean_pi_to_q_TV=float(pi_q_tv.mean()),
        visits_to_pi_TV=moments(visits_pi_tv), visits_to_q_TV=moments(visits_q_tv),
        stationary_information_bits=moments(information),
        maximum_stationarity_residual_l1=max(x["residual_l1"] for x in stationarity),
        maximum_stationary_iterations=max(x["iterations"] for x in stationarity),
        maximum_sum_of_row_refinement_differences_bits=float(refinement.max()),
        maximum_single_row_refinement_difference_bits=float(max_row_refinement.max()),
        raw_path=str(npz_path.relative_to(ROOT)) if npz_path.is_relative_to(ROOT) else str(npz_path),
        raw_sha256=digest(npz_path), evaluator_sha256=digest(Path(__file__).with_name("tilted_row_reference.py")),
        command_argv=sys.argv, elapsed_seconds=time.monotonic()-start)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"L={L} alpha={alpha:g} d={d} N={args.reps}: {result['elapsed_seconds']:.1f}s; "
          f"R={result['redundancy']['mean'][-1]:.2f}; errors%="
          + ",".join(f"{name}:{100*comparisons[name]['measured_over_prediction_minus_one'][-1]:+.2f}" for name in PREDICTIONS), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 3, 6])
    parser.add_argument("--alphas", nargs="+", type=float, default=[.5, 1., 1.5, 2.])
    parser.add_argument("--ds", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=202609070)
    parser.add_argument("--step", type=float, default=1/128)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--scalar", type=Path, default=ROOT/"results/mechanism/iid_depth_prediction.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"output/context_finite_prediction")
    parser.add_argument("--out", type=Path, default=ROOT/"results/mechanism/context_finite_prediction_check.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert args.reps >= 2 and all(d >= 2 and d % 2 == 0 for d in args.ds)
    args.raw_dir = args.raw_dir.resolve()
    validation = self_test(args.store)
    if args.self_test:
        print(json.dumps(validation, indent=2))
        return
    scalar = json.loads(args.scalar.read_text())
    assert all(str(L) in scalar["scalar_curves"] for L in args.depths)
    scalar_hash = digest(args.scalar)
    start = time.monotonic()
    cells = []
    for L, alpha, d in itertools.product(args.depths, args.alphas, args.ds):
        cells.append(run_cell(args, L, alpha, d, ScalarCurve(scalar["scalar_curves"][str(L)]), scalar_hash))
        result = dict(schema_version=1, complete=len(cells)==len(args.depths)*len(args.alphas)*len(args.ds),
            target="Expected Bayes learning cost I(T;X0,...,Xt) of an actual Markov chain with independent initial law q.",
            formulas={"old_stationary":"sum_ab r_L(t*pi_a*q_b)",
                "joint_stationary":"sum_ab r_L(t*pi_a*b^(-alpha)/Z_a), Z_a=sum_b b^(-alpha)*G_ab",
                "joint_actual_visits":"sum_ab r_L(K_a*b^(-alpha)/Z_a), K_a is the actual departure count"},
            qualifications=["The corrected comparisons preserve the joint distribution of each table's stationary law and row normalizers.",
                "Inserting actual visits is an empirical comparison, not conditioning iid rows on independent sample counts.",
                "Prediction differences and their uncertainty are paired across the same tables and paths; covariance is retained.",
                "Source exponents describe q, not a fitted power law for the actual stationary distribution.",
                "Numerical grid refinement is separate from sampling uncertainty and is not a certified total integration error bound.",
                "No crossing length or asymptotic exponent is estimated."],
            scalar_path=str(args.scalar.relative_to(ROOT)) if args.scalar.is_relative_to(ROOT) else str(args.scalar),
            scalar_sha256=scalar_hash, validation=validation, cells=cells, command_argv=sys.argv,
            elapsed_seconds=time.monotonic()-start)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Completed {len(cells)} settings in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

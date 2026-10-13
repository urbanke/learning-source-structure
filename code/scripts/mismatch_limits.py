"""Deterministic stationary product-Markov projection of saved relabeled sources.

Run with:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MKL_NUM_THREADS=1 /private/tmp/layered-shared-depth-env/bin/python \
code/scripts/mismatch_limits.py

No trajectories are generated and no posterior/network is evaluated. For the
visible state Y=(U,V), the best coordinate-product Markov kernel is
Q*((u',v')|(u,v))=P(U'=u'|U=u) P(V'=v'|V=v), with all probabilities under
the stationary joint law of the true relabeled 64-state Markov chain.
The minimum stationary expected KL is H(U'|U)+H(V'|V)-H(Y'|Y).

This is the stationary infinite-context Bayes prediction limit subject to
posterior concentration on the pseudo-true product parameters; the script
does not prove concentration or verify future transformer predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import json
import os
from pathlib import Path

# Must happen before NumPy is imported.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.special import betaln, expit, xlogy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/product_shift_2026-09-18"
OUT = SOURCE / "mismatch_limit"
TOL = 2e-11


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def stationary(kernel):
    system = kernel.T - np.eye(len(kernel))
    rhs = np.zeros(len(kernel))
    system[-1] = 1.0
    rhs[-1] = 1.0
    # The normalization equation is part of the solve. No clipping or
    # normalization is applied to the returned probabilities.
    return np.linalg.solve(system, rhs)


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))),
                min=float(a.min()), max=float(a.max()))


def run_depth(depth, plan_id, limit=None):
    paths = sorted((SOURCE / "records" / f"L{depth}_{depth}").glob("seq_*.npz"))
    require(len(paths) == 1024, f"Expected 1024 source records at depth {depth}")
    require([p.name for p in paths] == [f"seq_{i:06d}.npz" for i in range(1024)], "Record indices differ")
    if limit is not None:
        paths = paths[:limit]
    errors = {}
    minimum_support = {}
    records = []
    fingerprint = hashlib.sha256()

    def check(name, value):
        error = float(np.max(np.abs(value)))
        errors[name] = max(errors.get(name, 0.0), error)
        require(np.isfinite(error) and error < TOL, f"{name}: {error}")

    def positive(name, a):
        value = float(np.min(a))
        minimum_support[name] = min(minimum_support.get(name, np.inf), value)
        require(np.isfinite(a).all() and value > 0.0, f"Nonpositive/nonfinite {name}: {value}")

    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as saved:
            require(str(saved["plan_id"]) == plan_id, "Frozen plan mismatch")
            require(int(saved["index"]) == index, "Source index mismatch")
            require(np.array_equal(saved["depth_pair"], [depth, depth]), "Depth mismatch")
            require(str(saved["conditions"][1]) == "relabeled", "Condition order changed")
            factors = saved["pre_matrices"][1]
            perm = saved["permutations"][1]
            times = saved["times"]
            tokens = saved["tokens"][1]
            saved_true = saved["true_probs"][1]
        require(factors.shape == (2, 8, 8), "Factor shape mismatch")
        require(np.array_equal(np.sort(perm), np.arange(64)), "Invalid permutation")
        fingerprint.update(path.name.encode())
        fingerprint.update(factors.tobytes())
        fingerprint.update(perm.tobytes())
        positive("factor_transition", factors)
        check("factor_row_normalization", factors.sum(axis=-1) - 1)
        factor_pi = np.stack([stationary(t) for t in factors])
        positive("factor_stationary", factor_pi)
        check("factor_stationary_normalization", factor_pi.sum(axis=-1) - 1)
        for p, t in zip(factor_pi, factors):
            check("factor_stationarity", p @ t - p)

        # perm[canonical] is its visible state. Independently construct the
        # permutation using assignment and verify with inverse indexing.
        canonical_t = np.kron(*factors)
        true_t = np.empty((64, 64))
        true_t[np.ix_(perm, perm)] = canonical_t
        inverse = np.argsort(perm)
        check("permutation_reconstruction", true_t - canonical_t[np.ix_(inverse, inverse)])
        pi = np.empty(64)
        pi[perm] = np.kron(*factor_pi)
        positive("true_transition", true_t)
        positive("joint_stationary", pi)
        check("true_row_normalization", true_t.sum(axis=-1) - 1)
        check("joint_stationary_normalization", pi.sum() - 1)
        check("joint_stationarity", pi @ true_t - pi)
        pi_direct = stationary(true_t)
        positive("direct_joint_stationary", pi_direct)
        check("stationary_factor_vs_full_solve", pi - pi_direct)
        check("saved_true_row_reconstruction", true_t[tokens[times - 1]] - saved_true)

        joint = pi[:, None] * true_t
        check("stationary_pair_normalization", joint.sum() - 1)
        check("stationary_pair_row_marginal", joint.sum(axis=1) - pi)
        check("stationary_pair_column_marginal", joint.sum(axis=0) - pi)
        j4 = joint.reshape(8, 8, 8, 8)
        ju = j4.sum(axis=(1, 3))
        jv = j4.sum(axis=(0, 2))
        pu = pi.reshape(8, 8).sum(axis=1)
        pv = pi.reshape(8, 8).sum(axis=0)
        a = ju / pu[:, None]
        b = jv / pv[:, None]
        positive("projected_factor_transition", np.stack([a, b]))
        check("projected_factor_row_normalization", np.stack([a, b]).sum(axis=-1) - 1)
        projected = np.kron(a, b)
        positive("projected_joint_transition", projected)
        check("projected_joint_row_normalization", projected.sum(axis=-1) - 1)

        h_true = -float(np.sum(joint * np.log2(true_t)))
        h_factor = -float(np.sum(factor_pi[:, :, None] * factors * np.log2(factors)))
        hu = -float(np.sum(ju * np.log2(a)))
        hv = -float(np.sum(jv * np.log2(b)))
        projected_ce = hu + hv
        risk = projected_ce - h_true
        full_kl = float(np.sum(joint * (np.log2(true_t) - np.log2(projected))))
        full_ce = -float(np.sum(joint * np.log2(projected)))
        check("entropy_visible_vs_canonical_bits", h_true - h_factor)
        check("entropy_formula_vs_full_kl_bits", risk - full_kl)
        check("coordinate_vs_full_crossentropy_bits", projected_ce - full_ce)
        require(risk >= -TOL, "Negative minimum KL")
        records.append(dict(index=index, stationary_kl_bits=risk,
                            entropy_rate_bits=h_true,
                            projected_crossentropy_bits=projected_ce,
                            conditional_entropy_u_bits=hu,
                            conditional_entropy_v_bits=hv))

    quantities = {key: stats([r[key] for r in records]) for key in records[0] if key != "index"}
    return dict(depth_pair=[depth, depth], n_sources=len(records),
                input_arrays_sha256=fingerprint.hexdigest(), quantities=quantities,
                maximum_absolute_errors=errors, minimum_positive_support=minimum_support,
                per_source=records)


def coin_probability(n, s, a=0.2, prior="two_point"):
    s = np.asarray(s)
    if prior == "uniform":
        return (s + 1) / (n + 2)
    return a + (1 - 2*a) * expit((2*s - n) * math.log((1-a)/a))


def binary_kl(p, q):
    return (xlogy(p, p/q) + xlogy(1-p, (1-p)/(1-q))) / math.log(2)


def coin_risk(n, p, prior):
    counts = np.arange(n + 1)
    return float(binom.pmf(counts, n, p) @ binary_kl(p, coin_probability(n, counts, prior=prior)))


def coin_analysis():
    maximum_error = 0.0
    for prior in ("two_point", "uniform"):
        for n in range(11):
            for p in (0.1, 0.4, 0.5, 0.6, 0.9):
                enumeration = 0.0
                for seq in itertools.product((0, 1), repeat=n):
                    s = sum(seq)
                    if prior == "two_point":
                        la = .2**s * .8**(n-s)
                        lb = .8**s * .2**(n-s)
                        pred = (.2*la + .8*lb) / (la+lb)
                    else:
                        pred = math.exp(betaln(s+2, n-s+1) - betaln(s+1, n-s+1))
                    enumeration += p**s * (1-p)**(n-s) * binary_kl(p, pred)
                maximum_error = max(maximum_error, abs(enumeration - coin_risk(n, p, prior)))
    require(maximum_error < 5e-14, "Coin finite-sum and sequence enumeration disagree")
    times = list(range(257))
    return dict(schema="mismatched-coin-analytic-v1", a=.2, times=times,
                scope="Exact finite sums; no transformer evaluated; n observations predict token n+1",
                validation=dict(enumeration_max_n=10, maximum_absolute_error_bits=maximum_error),
                two_point_limits={str(p): float(binary_kl(p, .2 if p <= .5 else .8))
                                  for p in (.1, .2, .4, .5, .6, .8, .9)},
                curves={prior: {str(p): [coin_risk(n, p, prior) for n in times]
                                for p in (.1, .2, .4, .5, .6, .8, .9)}
                        for prior in ("two_point", "uniform")})


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use four saved sources per depth")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    OUT = args.out or SOURCE / ("mismatch_limit_smoke" if args.smoke else "mismatch_limit")
    OUT.mkdir(parents=True, exist_ok=True)
    coin = coin_analysis()
    (OUT / "coin_summary.json").write_text(json.dumps(coin, indent=2, allow_nan=False) + "\n")
    plan = json.loads((SOURCE / "plan.json").read_text())
    require(plan["n_seq"] == 1024, "Wrong source count")
    summary = dict(schema="stationary-relabeled-product-projection-v1",
                   source_plan_id=plan["plan_id"],
                   smoke=args.smoke,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   method="Exact stationary 64-state source law, KL projection onto visible-coordinate 8x8 product Markov kernels",
                   risk_formula="H(U_next|U)+H(V_next|V)-H(Y_next|Y), in bits",
                   uncertainty="Sample standard deviation / sqrt(n_sources) across independent saved source draws; deterministic calculation for each source",
                   no_posthoc_renormalization=True,
                   limitations="Stationary infinite-context Bayes limit subject to concentration on pseudo-true product parameters. No Bayes convergence experiment or future transformer prediction verification; current transformer context ends at 1024.",
                   validation_tolerance=TOL,
                   runtime=dict(numpy=np.__version__, single_thread_blas=True),
                   depths={str(depth): run_depth(depth, plan["plan_id"], limit=4 if args.smoke else None) for depth in [1, 4]})
    (OUT / "projection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for depth, result in summary["depths"].items():
        print(json.dumps(dict(depth=depth, n_sources=result["n_sources"],
                              quantities=result["quantities"],
                              maximum_absolute_errors=result["maximum_absolute_errors"],
                              minimum_positive_support=result["minimum_positive_support"]), indent=2))


if __name__ == "__main__":
    main()

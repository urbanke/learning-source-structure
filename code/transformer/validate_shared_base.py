"""Bounded independent validation of the small-alphabet shared-base Bayes rule.

The independent depth-one reference integrates over a transformed uniform
simplex, not over the Laplace transform used by the weighted row kernel.
Depth zero is checked against a separate iid Zipf mixture calculation.
Full-depth checks use all depths 0..8, exact permutation enumeration, and
refine the exponent quadrature. No transformer is trained or evaluated.

    python code/transformer/validate_shared_base.py --smoke --out /private/tmp/shared-base-smoke.json
    python code/transformer/validate_shared_base.py --out results/transformer/shared_base/validation.json

Set LSA_STORE and PMM_SRC to the read-only anchor store and its package source.
Every numerical check records its tolerance and measured error in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import gammaln, logsumexp, roots_jacobi, roots_legendre

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)


def zipf(d, alpha):
    weights = np.arange(1, d + 1, dtype=np.float64) ** (-float(alpha))
    return weights / math.fsum(weights)


def simplex_l1_log_evidence(counts, q, order=48):
    """Independent d2/d3 integral against the transformed Dirichlet density.

If p_i=q_i*u_i/sum(q*u), u uniform on the simplex, its density is
Gamma(d)/[prod(q_i)*(sum(p_i/q_i))**d]. Absorb prod(p_i**m_i)
into a Dirichlet(m+1) probability measure, then integrate its independent
Beta stick-breaking variables using normalized Gauss-Jacobi rules.
"""
    m = np.asarray(counts, dtype=int)
    q = np.asarray(q, dtype=np.float64)
    d = len(q)
    if d not in (2, 3) or m.shape != q.shape or (m < 0).any() or (q <= 0).any():
        raise ValueError("Independent simplex reference requires d=2 or d=3 and positive weights")
    a = m + 1.0
    abscissae, weights = [], []
    for i in range(d - 1):
        x, w = roots_jacobi(order, a[i+1:].sum() - 1, a[i] - 1)
        abscissae.append((x + 1) / 2)
        weights.append(w / math.fsum(w))
    if d == 2:
        p = np.stack((abscissae[0], 1 - abscissae[0]), axis=-1)
        w = weights[0]
    else:
        x, y = np.meshgrid(*abscissae, indexing="ij")
        p = np.stack((x, (1-x)*y, (1-x)*(1-y)), axis=-1)
        w = weights[0][:, None] * weights[1][None, :]
    expectation = np.sum(w * np.sum(p/q, axis=-1) ** (-d))
    return float(gammaln(d) + gammaln(a).sum() - gammaln(a.sum())
                 - np.log(q).sum() + np.log(expectation))


def independent_iid_log_joint(x, d, order=64):
    """Depth-zero law: integrate iid likelihood over alpha and label permutations."""
    x = np.asarray(x, dtype=int)
    counts = np.bincount(x, minlength=d)
    nodes, weights = roots_legendre(order)
    alphas = 1.0 + 0.5*nodes
    weights = weights / math.fsum(weights)
    terms = []
    for alpha, weight in zip(alphas, weights):
        for q in itertools.permutations(zipf(d, alpha)):
            terms.append(math.log(weight) - math.lgamma(d + 1) + float(counts @ np.log(q)))
    return float(logsumexp(terms))


def representative_path(d, length):
    motif = [0, 0, 1, 0, 2, 0, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0]
    if d == 4:
        motif = [0, 0, 1, 0, 2, 0, 3, 3, 0, 1, 2, 3, 0, 0, 2, 1]
    return np.resize(np.asarray(motif, dtype=int), length)


def source_likelihood_monte_carlo(paths, samples, seed):
    """Average latent-source likelihoods, independently of the phi integrals.

Uses the actual source's parameter/matrix construction, then an independent
uniform alphabet permutation. Averaging q(X1)*prod T(Xt,Xt+1) avoids sampled
trajectory event noise. This is a model check, not a precision certificate.
"""
    from source import Source
    d = 3
    generator = Source(d, 4, alpha=(0.5, 1.5), ls=(0, 8), permute=True)
    rng = np.random.default_rng(seed)
    alpha, depth = generator.draw_params(samples, rng)
    q, transitions = generator.matrices(alpha, depth, rng)
    possible = np.asarray(list(itertools.permutations(range(d))), dtype=int)
    permutation = possible[rng.integers(len(possible), size=samples)]
    inverse = np.argsort(permutation, axis=1)
    row = np.arange(samples)
    results = []
    for path in paths:
        original_labels = inverse[:, np.asarray(path, dtype=int)]
        likelihood = q[row, original_labels[:, 0]].copy()
        for t in range(1, len(path)):
            likelihood *= transitions[row, original_labels[:, t-1], original_labels[:, t]]
        mean = math.fsum(likelihood) / samples
        se = math.sqrt(math.fsum((likelihood-mean)**2) / (samples*(samples-1)))
        results.append(dict(path=path, mean=mean, se=se, samples=samples, seed=seed))
    return results


class Checks:
    def __init__(self):
        self.records = []
        self.timings = []

    def close(self, name, actual, expected, tolerance, **details):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if details.get("units") == "nats":
            actual, expected, tolerance = actual / LN2, expected / LN2, tolerance / LN2
            details["units"] = "bits"
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{name}: nonfinite or incompatible arrays")
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
        self.records.append(dict(name=name, error=error, tolerance=tolerance, passed=error <= tolerance, **details))
        if error > tolerance:
            raise AssertionError(f"{name}: {error:.6g} exceeds {tolerance:.6g}")
        print(f"PASS {name}: {error:.3g} <= {tolerance:.3g}", flush=True)
        return error

    def probability(self, name, p, tolerance=2e-7, **details):
        p = np.asarray(p, dtype=float)
        if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
            raise AssertionError(f"{name}: invalid probabilities")
        self.close(name, p.sum(axis=1), np.ones(len(p)), tolerance, **details)

    def timed(self, name, function, **details):
        start = time.monotonic()
        result = function()
        elapsed = time.monotonic() - start
        self.timings.append(dict(name=name, seconds=elapsed, **details))
        print(f"TIME {name}: {elapsed:.3f}s", flush=True)
        return result


def make_kernel(depths, args, *, wider=False, finer=False):
    from weighted_lsa import WeightedLSARowEvidence
    return WeightedLSARowEvidence(depths=list(depths),
                                  u_lo=-36.0 if wider else args.u_lo,
                                  u_hi=34.0 if wider else args.u_hi,
                                  step=args.step / 2 if finer else args.step)


def run_checks(args, checks):
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA
    from shared_base import SharedBaseLSA

    singleton = make_kernel([1], args)
    for d, count_vectors in [(2, [[1, 0], [0, 1], [16, 16], [32, 0]]),
                             (3, [[1, 0, 0], [10, 11, 11], [32, 0, 0]])]:
        q = zipf(d, 1.5)
        for counts in count_vectors:
            independent32 = simplex_l1_log_evidence(counts, q, 32)
            independent64 = simplex_l1_log_evidence(counts, q, 64)
            label = f"d{d}_m{'-'.join(map(str, counts))}"
            checks.close(f"simplex_quadrature_{label}", independent32, independent64, 1e-10, units="nats")
            checks.close(f"weighted_L1_vs_simplex_{label}", singleton.log_evidence(counts, q)[0],
                         independent64, 2e-8, units="nats")

    base = make_kernel(range(9), args)
    dimensions = [3] if args.smoke else [3, 4]
    length = 8 if args.smoke else 32
    for d in dimensions:
        path = representative_path(d, length)
        weighted = SharedBaseLSA(d, base, known_q=np.full(d, 1/d))
        # The existing independent implementation uses step 1/64 internally.
        uniform = SharedDepthLSA(SeqLSA(d, depths=list(range(9)), rmax=length+1,
                                       u_lo=args.u_lo, u_hi=args.u_hi, table_dir=str(HERE)))
        a = weighted.run(path, full_probs=True)
        b = uniform.run(path, full_probs=True)
        checks.close(f"uniform_limit_d{d}_loss", a["loss"], b["loss"], 2e-7, units="bits")
        checks.close(f"uniform_limit_d{d}_probs", a["probs"], b["probs"], 2e-7)

    zero = make_kernel([0], args)
    for d in dimensions:
        path = representative_path(d, length)
        iid = SharedBaseLSA(d, zero, alpha_order=32)
        result = iid.run(path, full_probs=True)
        for n in sorted(set([1, 2, min(8, length), length])):
            expected = independent_iid_log_joint(path[:n], d, 64)
            checks.close(f"depth0_independent_joint_d{d}_n{n}", iid.log_joint(path[:n]), expected, 2e-8, units="nats")
            if n < length:
                p = np.asarray([math.exp(independent_iid_log_joint(np.append(path[:n], b), d, 64)-expected)
                                for b in range(d)])
                checks.close(f"depth0_independent_predictive_d{d}_prefix{n}", result["probs"][n], p, 2e-8)
        checks.close(f"depth0_firsttoken_d{d}", result["probs"][0], np.full(d, 1/d), 2e-10)
        if result["probs"][1, path[0]] <= 1/d + 0.001:
            raise AssertionError("First observation failed to update the shared base posterior")

    orders = [8, 16] if args.smoke else [8, 16, 32]
    full_results = {}
    for d in dimensions:
        path = representative_path(d, length)
        for order in orders:
            predictor = checks.timed(f"construct_d{d}_alpha{order}",
                                    lambda d=d, order=order: SharedBaseLSA(d, base, alpha_order=order))
            if not args.smoke:
                checks.timed(f"run_d{d}_alpha{order}_n16", lambda: predictor.run(path[:16], full_probs=True), tokens=16)
            result = checks.timed(f"run_d{d}_alpha{order}_n{length}",
                                  lambda: predictor.run(path, full_probs=True), tokens=length)
            checks.probability(f"full_prior_predictive_normalization_d{d}_alpha{order}", result["probs"])
            checks.close(f"joint_sequential_d{d}_alpha{order}",
                         predictor.log_joint(path), -LN2 * np.sum(result["loss"]), 2e-7, units="nats")
            if order == orders[-1]:
                perm = np.roll(np.arange(d), 1)
                relabeled = predictor.run(perm[path], full_probs=True)
                checks.close(f"label_equivariance_d{d}", relabeled["probs"][:, perm], result["probs"], 2e-8)
                checks.close(f"label_invariant_joint_d{d}", predictor.log_joint(perm[path]), predictor.log_joint(path), 2e-8, units="nats")
                for prefix in [1, min(4, length-1), length-1]:
                    current_joint = predictor.log_joint(path[:prefix])
                    extensions = np.asarray([math.exp(predictor.log_joint(np.append(path[:prefix], b))-current_joint)
                                             for b in range(d)])
                    checks.close(f"joint_extension_d{d}_prefix{prefix}", extensions, result["probs"][prefix], 2e-7)
                enumerate_length = 2 if args.smoke or d == 4 else 3
                total = math.fsum(math.exp(predictor.log_joint(np.asarray(x, dtype=int)))
                                  for x in itertools.product(range(d), repeat=enumerate_length))
                checks.close(f"full_prior_joint_normalization_d{d}_n{enumerate_length}", total, 1.0, 2e-7)
            full_results[d, order] = result
        for lower, higher in zip(orders[:-1], orders[1:]):
            a, b = full_results[d, lower], full_results[d, higher]
            checks.close(f"alpha_refinement_d{d}_{lower}_to_{higher}_logjoint",
                         a["log_evidence"][-1], b["log_evidence"][-1], 2e-7, units="nats")
            checks.close(f"alpha_refinement_d{d}_{lower}_to_{higher}_probs", a["probs"], b["probs"], 2e-7)

    # Row-level refinements isolate integration-bound and grid errors from the
    # separate exponent quadrature. They cover every retained depth, not just L8.
    wide = make_kernel(range(9), args, wider=True)
    fine = make_kernel(range(9), args, wider=True, finer=True)
    for d in dimensions:
        q = zipf(d, 1.5)
        balanced = [32//d] * d
        for i in range(32 % d):
            balanced[i] += 1
        for counts in [[1] + [0]*(d-1), [32] + [0]*(d-1), balanced]:
            normal = base.log_evidence(counts, q)
            expanded = wide.log_evidence(counts, q)
            refined = fine.log_evidence(counts, q)
            label = f"d{d}_m{'-'.join(map(str, counts))}"
            checks.close(f"row_bounds_refinement_{label}", normal, expanded, 2e-8, units="nats")
            checks.close(f"row_grid_refinement_{label}", expanded, refined, 2e-8, units="nats")
    if not args.smoke:
        for d in dimensions:
            path = representative_path(d, length)
            refined_predictor = SharedBaseLSA(d, fine, alpha_order=16)
            refined = checks.timed(f"full_refined_grid_d{d}_alpha16_n{length}",
                                  lambda: refined_predictor.run(path, full_probs=True), tokens=length)
            previous = full_results[d, 16]
            checks.close(f"full_grid_refinement_d{d}_logjoint", refined["log_evidence"][-1], previous["log_evidence"][-1], 3e-7, units="nats")
            checks.close(f"full_grid_refinement_d{d}_probs", refined["probs"], previous["probs"], 3e-7)

    mc_paths = [[0, 0, 0, 0], [0, 1, 0, 2]]
    mc_samples = 16384 if args.smoke else 131072
    mc = checks.timed("independent_source_likelihood_monte_carlo",
                      lambda: source_likelihood_monte_carlo(mc_paths, mc_samples, 20260915), samples=mc_samples)
    predictor = SharedBaseLSA(3, base, alpha_order=32)
    for result in mc:
        analytic = math.exp(predictor.log_joint(np.asarray(result["path"], dtype=int)))
        label = "".join(map(str, result["path"]))
        checks.close(f"actual_source_MC_joint_d3_{label}", analytic, result["mean"],
                     5*result["se"] + 2e-8, analytic_probability=analytic,
                     monte_carlo=result, units="probability",
                     interpretation="five standard errors plus numerical slack; source-model check, not numerical precision certificate")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--u-lo", type=float, default=-32.0)
    parser.add_argument("--u-hi", type=float, default=32.0)
    parser.add_argument("--step", type=float, default=1/32)
    args = parser.parse_args()
    if args.u_lo >= args.u_hi or args.step <= 0:
        parser.error("Require increasing bounds and a positive grid step")
    checks = Checks()
    start = time.monotonic()
    report = dict(schema="shared_base_validation_v1", smoke=args.smoke,
                  settings=dict(u_lo=args.u_lo, u_hi=args.u_hi, step=args.step,
                                depths=list(range(9)), alpha_bounds=[0.5, 1.5]),
                  runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                  store=os.environ.get("LSA_STORE"), backend_source=os.environ.get("PMM_SRC"),
                  code_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ["weighted_lsa.py", "shared_base.py", "shared_depth.py", "source.py", "phi_source.py", "validate_shared_base.py"]})
    failure = None
    try:
        run_checks(args, checks)
    except Exception as exc:
        failure = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(passed=failure is None, elapsed_seconds=time.monotonic()-start,
                  checks=checks.records, timings=checks.timings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    rows = ["name\terror\ttolerance\tunits\tpassed"]
    rows += [f"{item['name']}\t{item['error']:.12g}\t{item['tolerance']:.12g}\t{item.get('units', 'probability')}\t{int(item['passed'])}" for item in checks.records]
    args.out.with_suffix(".tsv").write_text("\n".join(rows)+"\n")
    if failure is not None:
        raise failure
    print(f"All {len(checks.records)} checks passed in {report['elapsed_seconds']:.2f}s; {args.out}", flush=True)


if __name__ == "__main__":
    main()

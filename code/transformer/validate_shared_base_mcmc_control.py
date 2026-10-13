"""Exact d=64 symmetry controls for the shared-base posterior sampler.

With just one observed label, all unseen labels remain exchangeable.
The reference explicitly sums that label's 64 possible ranks, the shared
depth, and exponent quadrature. It uses WeightedLSARowEvidence separately
from the sampler's cached RankedLSARowEvidence implementation.

These repeated-label prefixes test d=64 without factorial enumeration.
They do not establish mixing on typical prefixes with many seen labels.
Set LSA_STORE and PMM_SRC to the existing read-only anchor store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy
from scipy.special import logsumexp

from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from validate_shared_base_mcmc import check_probability, json_safe, table_provenance
from weighted_lsa import WeightedLSARowEvidence


HERE = Path(__file__).resolve().parent
LN2 = math.log(2)


def exponent_rule(order):
    abscissae, weights = leggauss(order)
    return 1.0 + 0.5 * abscissae, weights / math.fsum(weights)


def base_law(d, alpha):
    weights = np.arange(1, d + 1, dtype=float) ** (-float(alpha))
    return weights / math.fsum(weights)


def repeated_label_reference(length, kernel, order=16, d=64):
    """Joint and repeat prediction, summed over all relevant latent states."""
    nodes, weights = exponent_rule(order)
    depths = np.asarray(kernel.depths)
    log_prior_depth = -math.log(len(depths))
    log_before = np.empty((order, d, len(depths)))
    log_after = np.empty_like(log_before)
    for a, (alpha, weight) in enumerate(zip(nodes, weights)):
        q = base_law(d, alpha)
        for rank in range(d):
            counts = np.zeros(d, dtype=np.int64)
            counts[rank] = length - 1
            before = kernel.log_evidence(counts, q)
            counts[rank] += 1
            after = kernel.log_evidence(counts, q)
            # Uniform ranks under a uniform permutation; x1 contributes q.
            shared = math.log(weight) - math.log(d) + math.log(q[rank]) + log_prior_depth
            log_before[a, rank] = shared + before
            log_after[a, rank] = shared + after
    log_joint = float(logsumexp(log_before))
    repeat = float(np.exp(logsumexp(log_after) - log_joint))
    posterior = np.exp(log_before - log_joint)
    probabilities = np.full(d, (1.0 - repeat) / (d - 1))
    probabilities[0] = repeat
    return dict(predictive_probs=probabilities,
                posterior_alpha=posterior.sum(axis=(1, 2)),
                posterior_depth=posterior.sum(axis=(0, 1)),
                posterior_observed_rank=posterior.sum(axis=(0, 2)),
                negative_log_joint_bits=-log_joint / LN2,
                alpha_nodes=nodes, alpha_weights=weights,
                explanation="The unseen-label probability follows exact exchangeability and total probability; independent raw row-normalization checks are reported separately.")


def independent_row_checks(length, kernel, d=64):
    """Check every possible next label for three representative latent states."""
    nodes, _ = exponent_rule(16)
    results = []
    for alpha_index, rank in ((0, 0), (8, 31), (15, 63)):
        q = base_law(d, nodes[alpha_index])
        counts = np.zeros(d, dtype=np.int64)
        counts[rank] = length - 1
        before = kernel.log_evidence(counts, q)
        probabilities = np.empty((len(kernel.depths), d))
        for candidate in range(d):
            counts[candidate] += 1
            probabilities[:, candidate] = np.exp(kernel.log_evidence(counts, q) - before)
            counts[candidate] -= 1
        error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
        valid = bool(np.isfinite(probabilities).all() and (probabilities >= 0).all())
        results.append(dict(alpha_index=alpha_index, alpha=float(nodes[alpha_index]),
                            observed_rank_one_based=rank + 1,
                            max_raw_probability_sum_error=error,
                            passed=valid and error <= 1e-7, tolerance=1e-7))
    return results


def run_control(length, args, reference_kernel, sampler):
    start = time.monotonic()
    exact = repeated_label_reference(length, reference_kernel, order=16)
    exact_seconds = time.monotonic() - start
    start = time.monotonic()
    refined = repeated_label_reference(length, reference_kernel, order=32)
    refined_seconds = time.monotonic() - start
    raw_rows = independent_row_checks(length, reference_kernel)
    print(json.dumps(dict(prefix_length=length, stage="exact_reference",
                          seconds=exact_seconds, refined_seconds=refined_seconds,
                          repeat_probability=float(exact["predictive_probs"][0]))), flush=True)
    estimate = sampler.sample(np.zeros(length, dtype=np.int64), chains=4,
                              burn=args.burn, draws=args.draws, thin=args.thin,
                              seed=args.seed + length)
    p = np.asarray(estimate["predictive_probs"])
    se = np.asarray(estimate["mcse_probs"])
    error = np.abs(p - exact["predictive_probs"])
    quadrature_error = float(np.max(np.abs(exact["predictive_probs"] - refined["predictive_probs"])))
    joint_error = abs(exact["negative_log_joint_bits"] - refined["negative_log_joint_bits"])
    cap = 0.03 if args.smoke else 0.01
    finite_se = bool(se.shape == (64,) and np.isfinite(se).all() and (se >= 0).all())
    checks = dict(
        exact_probability=check_probability(exact["predictive_probs"], 64),
        sampled_probability=check_probability(p, 64),
        sampled_depth=check_probability(estimate["posterior_depth"], 9),
        sampled_alpha=check_probability(estimate["posterior_alpha"], 16),
        exponent_refinement_prediction=dict(passed=quadrature_error <= 2e-7,
                                            max_absolute_error=quadrature_error, tolerance=2e-7),
        exponent_refinement_joint_bits=dict(passed=joint_error <= 1e-6,
                                            absolute_error_bits=joint_error, tolerance_bits=1e-6),
        raw_conditional_rows=dict(passed=all(row["passed"] for row in raw_rows), cases=raw_rows),
        probability_absolute_cap=dict(passed=bool(np.all(error <= cap)),
                                      max_absolute_error=float(error.max()), tolerance=cap),
        probability_mcse=dict(passed=finite_se and bool(np.all(error <= 5 * se + 0.0002)),
                              absolute_error=error, mcse=se, tolerance=5 * se + 0.0002),
        unseen_symmetry=dict(passed=bool(np.max(np.abs(p[1:] - p[1])) <= 1e-15),
                             max_absolute_error=float(np.max(np.abs(p[1:] - p[1]))), tolerance=1e-15),
    )
    for quantity, se_name in (("posterior_depth", "mcse_depth"),
                             ("posterior_alpha", "mcse_alpha")):
        value = np.asarray(estimate[quantity])
        uncertainty = np.asarray(estimate[se_name])
        discrepancy = np.abs(value - exact[quantity])
        valid = bool(value.shape == uncertainty.shape == exact[quantity].shape
                     and np.isfinite(uncertainty).all() and (uncertainty >= 0).all())
        checks[quantity + "_mcse"] = dict(passed=valid and bool(np.all(discrepancy <= 5 * uncertainty + 0.0002)),
                                         absolute_error=discrepancy, mcse=uncertainty,
                                         tolerance=5 * uncertainty + 0.0002)
    if length == 1:
        alpha_error = float(np.max(np.abs(exact["posterior_alpha"] - exact["alpha_weights"])))
        depth_error = float(np.max(np.abs(exact["posterior_depth"] - 1 / 9)))
        initial_error = abs(exact["negative_log_joint_bits"] - 6.0)
        checks["first_token_prior_identities"] = dict(
            passed=max(alpha_error, depth_error, initial_error) <= 1e-12,
            alpha_error=alpha_error, depth_error=depth_error,
            first_token_code_bits_error=initial_error, tolerance=1e-12)
    result = dict(
        prefix_length=length, tokens=[0] * length,
        control="only one label has been observed",
        passed=all(check["passed"] for check in checks.values()),
        exact=exact, refined=refined,
        estimate={name: estimate[name] for name in ("predictive_probs", "mcse_probs", "posterior_depth",
                                                   "mcse_depth", "posterior_alpha", "mcse_alpha")},
        diagnostics=estimate["diagnostics"], checks=checks,
        elapsed_seconds=dict(exact=exact_seconds, refined=refined_seconds,
                             sampler=estimate["elapsed_seconds"]),
    )
    print(json.dumps(dict(prefix_length=length, stage="comparison", passed=result["passed"],
                          max_probability_error=float(error.max()),
                          max_probability_mcse=float(se.max()),
                          sampler_seconds=estimate["elapsed_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    args.burn = args.burn if args.burn is not None else (256 if args.smoke else 2048)
    args.draws = args.draws if args.draws is not None else (256 if args.smoke else 2048)
    if args.burn < 0 or args.draws < 32 or args.thin < 1:
        parser.error("require burn >= 0, draws >= 32, and thin >= 1")
    nodes, weights = exponent_rule(16)
    reference_kernel = WeightedLSARowEvidence()
    sampler_kernel = RankedLSARowEvidence(64, nodes)
    sampler_kernel._get_backend()
    sampler = SharedBaseMCMC(64, sampler_kernel, weights)
    start = time.monotonic()
    records = [run_control(length, args, reference_kernel, sampler)
               for length in ([1] if args.smoke else [1, 32])]
    code_names = ("ranked_lsa.py", "shared_base_mcmc.py", "weighted_lsa.py", "lsa_seq.py", "phi_source.py",
                  "validate_shared_base_mcmc.py", "validate_shared_base_mcmc_control.py")
    report = dict(
        schema="shared_base_mcmc_d64_control_v1",
        passed=all(row["passed"] for row in records), d=64,
        settings=dict(smoke=args.smoke, chains=4, burn=args.burn, draws=args.draws,
                      thin=args.thin, base_seed=args.seed, per_case_seed="base_seed + prefix_length",
                      alpha_orders=[16, 32], alpha_interval=[0.5, 1.5],
                      depth_values=list(range(9)), depth_prior="uniform and shared across rows",
                      rank_prior="uniform permutation"),
        limitations=[
            "A repeated-label prefix permits exact reduction to the observed label's rank; this is a symmetry control.",
            "These controls do not establish MCMC mixing for typical d=64 prefixes with many observed labels.",
            "MCSE and split Rhat are diagnostics, not convergence certificates or error bounds.",
            "The independent weighted and ranked row implementations share the same anchor backend; row-grid accuracy is a separate numerical question.",
            "The 32-node reference refines the continuous exponent integration only on these specified prefixes.",
        ],
        cases=records, elapsed_seconds=time.monotonic() - start,
        caches=dict(reference=reference_kernel.cache_info(), sampler=sampler_kernel.cache_info()),
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        provenance=dict(code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                     for name in code_names},
                        **table_provenance()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    rows = ["prefix_tokens\texact_repeat\testimated_repeat\trepeat_mcse\tabsolute_error\talpha_refinement_error\tsampler_seconds\tpassed"]
    for row in records:
        rows.append("\t".join(str(value) for value in (
            row["prefix_length"], row["exact"]["predictive_probs"][0],
            row["estimate"]["predictive_probs"][0], row["estimate"]["mcse_probs"][0],
            row["checks"]["probability_absolute_cap"]["max_absolute_error"],
            row["checks"]["exponent_refinement_prediction"]["max_absolute_error"],
            row["elapsed_seconds"]["sampler"], row["passed"])))
    args.out.with_suffix(".tsv").write_text("\n".join(rows) + "\n")
    print(json.dumps(dict(out=str(args.out), passed=report["passed"],
                          cases=len(records), seconds=report["elapsed_seconds"])), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

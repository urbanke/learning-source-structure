"""Independent long-count controls for shared-base Bayesian prediction.

The d=3 case enumerates every label permutation and shared depth using
batch transition counts. The d=64 repeated-label control instead sums the
observed label's rank exactly. Exponent quadrature is refined from 16 to
32 nodes, and representative latent states use a wider/finer row grid.
The independent weighted row kernel and the sampler's ranked row kernel
read the existing sealed production store; no tables are built.

These are two numerical controls, not independent d=64 source replications.
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
import scipy
from scipy.special import logsumexp

from ranked_lsa import RankedLSARowEvidence
from shared_base import SharedBaseLSA
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC
from source import Source
from validate_shared_base_mcmc import check_probability, json_safe, table_provenance
from validate_shared_base_mcmc_control import (
    exponent_rule, repeated_label_reference,
)
from weighted_lsa import WeightedLSARowEvidence


HERE = Path(__file__).resolve().parent
LN2 = math.log(2)


def progress(**values):
    print(json.dumps(json_safe(values), allow_nan=False), flush=True)


def batch_prediction(prefix, reference):
    """Enumerated posterior and extensions without any sequential loop."""
    d = reference.d
    counts = np.zeros((d, d), dtype=np.int64)
    np.add.at(counts, (prefix[:-1], prefix[1:]), 1)
    log_terms = reference.log_prior + np.log(reference.base_laws[:, prefix[0]])[:, None]
    for row in counts:
        if np.any(row):
            log_terms += reference._row_evidence(row)
    log_joint = float(logsumexp(log_terms))
    posterior = np.exp(log_terms - log_joint)
    current = counts[prefix[-1]]
    before = reference._row_evidence(current)
    probability = np.empty(d)
    for candidate in range(d):
        after = current.copy()
        after[candidate] += 1
        probability[candidate] = np.exp(logsumexp(
            log_terms + reference._row_evidence(after) - before) - log_joint)
    # Independent API recomputes all rows for the joint and each extension.
    direct_log_joint = reference.log_joint(prefix)
    direct_probability = np.array([
        np.exp(reference.log_joint(np.append(prefix, y)) - direct_log_joint)
        for y in range(d)
    ])
    return dict(
        predictive_probs=probability,
        posterior_depth=posterior.sum(axis=0),
        posterior_alpha=posterior.sum(axis=1).reshape(
            len(reference.alpha_nodes), math.factorial(d)).sum(axis=1),
        negative_log_joint_bits=-log_joint / LN2,
        alpha_nodes=reference.alpha_nodes, alpha_weights=reference.alpha_weights,
        batch_joint_api_error_bits=abs(log_joint - direct_log_joint) / LN2,
        batch_extension_api_max_error=float(np.max(np.abs(probability - direct_probability))),
    )


def fixed_state_checks(prefix, sampler, estimate):
    """Compare both kernels and row grids on dispersed and final states."""
    kernel = sampler.kernel
    nodes = kernel.alpha_nodes
    refined = WeightedLSARowEvidence(u_lo=-36, u_hi=34, step=1 / 64,
                                     term_cache_size=256)
    baseline = WeightedLSARowEvidence(term_cache_size=256)
    model = PrefixPosterior(sampler, prefix)
    counts = np.bincount(prefix, minlength=sampler.d)
    frequency_ranks = np.empty(sampler.d, dtype=np.int64)
    frequency_ranks[np.argsort(-counts, kind="stable")] = np.arange(sampler.d)
    states = [("frequency_low_alpha", 0, frequency_ranks),
              ("reverse_middle_alpha", len(nodes) // 2, sampler.d - 1 - frequency_ranks),
              ("identity_high_alpha", len(nodes) - 1, np.arange(sampler.d))]
    if estimate is not None:
        states.extend((f"final_chain_{j}", item["alpha_index"], np.asarray(item["ranks"]))
                      for j, item in enumerate(estimate["diagnostics"]["final_states"]))
    checks = []
    for name, alpha_index, ranks in states:
        state = model.evaluate(alpha_index, ranks)
        ranked_prediction, _, ranked_normalization = model.predict(state)
        q = kernel.base_laws[alpha_index, ranks]
        outputs = []
        for independent in (baseline, refined):
            depth_terms = sampler.log_depth_prior.copy()
            for row in model.counts:
                depth_terms += independent.log_evidence(row, q)
            likelihood = float(np.log(q[prefix[0]]) + logsumexp(depth_terms))
            current_evidence = independent.log_evidence(model.current_row, q)
            conditional = np.empty((len(sampler.depths), sampler.d))
            for candidate in range(sampler.d):
                after = model.current_row.copy()
                after[candidate] += 1
                conditional[:, candidate] = np.exp(
                    independent.log_evidence(after, q) - current_evidence)
            prediction = np.exp(depth_terms - logsumexp(depth_terms)) @ conditional
            outputs.append((likelihood, prediction,
                            float(np.max(np.abs(conditional.sum(axis=1) - 1)))))
        coarse, fine = outputs
        # Unseen labels are exactly averaged by the ranked sampler. Apply the
        # same symmetry to each independent state's prediction only for this
        # comparison; the conditional-row normalization remains unmodified.
        for output in outputs:
            if len(model.unseen):
                output[1][model.unseen] = output[1][model.unseen].mean()
        kernel_ll = abs(coarse[0] - state.log_likelihood) / LN2
        kernel_p = float(np.max(np.abs(coarse[1] - ranked_prediction)))
        grid_ll = abs(coarse[0] - fine[0]) / LN2
        grid_p = float(np.max(np.abs(coarse[1] - fine[1])))
        error = max(ranked_normalization, coarse[2], fine[2])
        checks.append(dict(
            name=name, alpha_index=alpha_index, alpha=float(nodes[alpha_index]),
            ranks=ranks, kernel_log_likelihood_difference_bits=kernel_ll,
            kernel_max_prediction_difference=kernel_p,
            grid_log_likelihood_difference_bits=grid_ll,
            grid_max_prediction_difference=grid_p,
            max_raw_conditional_normalization_error=error,
            passed=bool(kernel_ll <= 1e-5 and kernel_p <= 1e-7
                        and grid_ll <= 1e-5 and grid_p <= 1e-7 and error <= 1e-7),
        ))
    return checks


def run_case(d, args):
    length = 32 if args.smoke else 1024
    if d == 3:
        # Always generate the full 1025-token path, including in smoke mode,
        # so changing the evaluated prefix never changes this source draw.
        values, parameters = Source(3, 1025, alpha=(.5, 1.5), ls=(0, 8),
                                    permute=True).batch(1, np.random.default_rng(args.source_seed))
        prefix = values[0, :length]
        source = dict(seed=args.source_seed, generated_tokens=1025,
                      generated_latent_parameters={k: v[0].tolist() for k, v in parameters.items()},
                      note="Latents are recorded for reproducibility and never passed to predictors.")
    else:
        prefix = np.zeros(length, dtype=np.int64)
        source = dict(control="one repeated label; exact observed-rank reduction")
    nodes, weights = exponent_rule(16)
    kernel = RankedLSARowEvidence(d, nodes, term_cache_size=1024)
    kernel._get_backend()  # Lock production before the independent import.
    sampler = SharedBaseMCMC(d, kernel, weights)
    exact_kernel = WeightedLSARowEvidence(term_cache_size=2048)
    exact, times = {}, {}
    for order in (16, 32):
        progress(case=f"d{d}", prefix_tokens=length, stage="start_exact", alpha_order=order)
        start = time.monotonic()
        exact[order] = (batch_prediction(prefix, SharedBaseLSA(d, exact_kernel, alpha_order=order))
                        if d == 3 else repeated_label_reference(length, exact_kernel, order, d))
        times[f"exact_alpha{order}"] = time.monotonic() - start
        progress(case=f"d{d}", stage="exact_complete", alpha_order=order,
                 seconds=times[f"exact_alpha{order}"],
                 leading_probabilities=exact[order]["predictive_probs"][:3])
    # The exact kernels are no longer needed while MCMC builds its caches.
    exact_kernel.clear_cache()
    progress(case=f"d{d}", stage="start_sampler", chains=4,
             burn=args.burn, draws=args.draws, thin=args.thin)
    estimate = sampler.sample(prefix, chains=4, burn=args.burn, draws=args.draws,
                              thin=args.thin, seed=args.seed + d)
    times["sampler"] = estimate["elapsed_seconds"]
    p, se = np.asarray(estimate["predictive_probs"]), np.asarray(estimate["mcse_probs"])
    diagnostic = estimate["diagnostics"]
    rhat = diagnostic["rhat_probs"] + diagnostic["rhat_depth"] + diagnostic["rhat_alpha_and_loglik"]
    thresholds = dict(split_rhat_below_1_05=all(v is not None and v < 1.05 for v in rhat),
                      probability_ess_at_least_200=min(diagnostic["ess_probs"]) >= 200,
                      maximum_probability_mcse_at_most_0_005=float(se.max()) <= .005)
    progress(case=f"d{d}", stage="sampler_complete", seconds=times["sampler"],
             max_mcse=float(se.max()), max_error=float(np.max(np.abs(p-exact[16]["predictive_probs"]))),
             diagnostics=thresholds)
    start = time.monotonic()
    grids = fixed_state_checks(prefix, sampler, estimate)
    times["independent_kernel_and_grid_checks"] = time.monotonic() - start
    probability_error = np.abs(p - exact[16]["predictive_probs"])
    refine_p = float(np.max(np.abs(exact[16]["predictive_probs"] - exact[32]["predictive_probs"])))
    refine_joint = abs(exact[16]["negative_log_joint_bits"] - exact[32]["negative_log_joint_bits"])
    valid_se = bool(np.isfinite(se).all() and (se >= 0).all())
    checks = dict(
        exact16_probability=check_probability(exact[16]["predictive_probs"], d),
        exact32_probability=check_probability(exact[32]["predictive_probs"], d),
        sampled_probability=check_probability(p, d),
        exponent_prediction_refinement=dict(passed=refine_p <= 2e-7, max_absolute_error=refine_p, tolerance=2e-7),
        exponent_joint_refinement=dict(passed=refine_joint <= 1e-6, absolute_error_bits=refine_joint, tolerance_bits=1e-6),
        independent_kernels_and_row_grids=dict(passed=all(c["passed"] for c in grids), cases=grids),
        sampled_prediction_absolute_cap=dict(passed=bool(probability_error.max() <= (.03 if args.smoke else .01)),
                                            max_absolute_error=float(probability_error.max()), tolerance=.03 if args.smoke else .01),
        sampled_prediction_mcse=dict(passed=valid_se and bool(np.all(probability_error <= 5 * se + .0002)),
                                    absolute_error=probability_error, mcse=se, tolerance=5 * se + .0002),
    )
    for quantity, mcse in (("posterior_depth", "mcse_depth"), ("posterior_alpha", "mcse_alpha")):
        actual, uncertainty = np.asarray(estimate[quantity]), np.asarray(estimate[mcse])
        discrepancy = np.abs(actual - exact[16][quantity])
        checks[quantity + "_normalized"] = check_probability(actual, len(actual))
        checks[quantity + "_mcse"] = dict(
            passed=bool(np.isfinite(uncertainty).all() and (uncertainty >= 0).all()
                        and np.all(discrepancy <= 5 * uncertainty + .0002)),
            absolute_error=discrepancy, mcse=uncertainty, tolerance=5 * uncertainty + .0002)
    if d == 3:
        checks["independent_batch_api"] = dict(
            passed=all(exact[o]["batch_joint_api_error_bits"] <= 1e-10
                       and exact[o]["batch_extension_api_max_error"] <= 1e-10 for o in (16, 32)))
    else:
        checks["unseen_label_symmetry"] = dict(passed=bool(np.max(np.abs(p[1:] - p[1])) <= 1e-15))
    passed = all(c["passed"] for c in checks.values())
    result = dict(name=f"d{d}_long_control", d=d, prefix_length=length, tokens=prefix,
                  source=source, estimate=estimate, exact=exact[16], refined=exact[32],
                  checks=checks, passed=passed, diagnostic_threshold_checks=thresholds,
                  diagnostic_thresholds_passed=all(thresholds.values()), elapsed_seconds=times,
                  token_sha256=hashlib.sha256(np.asarray(prefix, dtype="<i8").tobytes()).hexdigest())
    progress(case=f"d{d}", stage="complete", passed=passed,
             diagnostics_passed=result["diagnostic_thresholds_passed"],
             alpha_refinement_error=refine_p, seconds=sum(times.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=("both", "d3", "d64"), default="both")
    parser.add_argument("--source-seed", type=int, default=20260923)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--thin", type=int, default=2)
    args = parser.parse_args()
    # Even the 32-token symmetry control has rare posterior exponent nodes;
    # a few hundred draws can miss them and report a misleading zero MCSE.
    args.burn = args.burn if args.burn is not None else 4096
    args.draws = args.draws if args.draws is not None else (16384 if args.smoke else 8192)
    if args.burn < 0 or args.draws < 32 or args.thin < 1:
        parser.error("require burn>=0, draws>=32 and thin>=1")
    start = time.monotonic()
    records = [run_case(d, args) for d in ([3, 64] if args.case == "both" else [int(args.case[1:])])]
    names = ("validate_shared_base_long.py", "shared_base.py", "shared_base_mcmc.py",
             "ranked_lsa.py", "weighted_lsa.py", "phi_source.py", "lsa_seq.py", "source.py",
             "validate_shared_base_mcmc.py", "validate_shared_base_mcmc_control.py")
    report = dict(
        schema="shared_base_long_validation_v1", passed=all(r["passed"] for r in records),
        diagnostic_thresholds_passed=all(r["diagnostic_thresholds_passed"] for r in records),
        settings=dict(smoke=args.smoke, case=args.case, prefix_tokens=32 if args.smoke else 1024,
                      alpha_orders=[16, 32], alpha_interval=[.5, 1.5], depths=list(range(9)),
                      depth_prior="one shared uniform depth", rank_prior="uniform permutation",
                      chains=4, burn=args.burn, draws=args.draws, thin=args.thin,
                      base_sampler_seed=args.seed, sampler_seed_rule="base_sampler_seed + d",
                      source_seed=args.source_seed, row_grid=[-32, 32, 1/32],
                      refined_row_grid=[-36, 34, 1/64]),
        cases=records, elapsed_seconds=time.monotonic() - start,
        limitations=["Two fixed numerical controls, not a representative source ensemble.",
                     "MCMC diagnostic thresholds are reported separately from exact-reference accuracy checks.",
                     "Shared backend tables are read only; the two row evaluators are not independent of their shared phi backend.",
                     "Row-grid checks cover specified latent states only.",
                     "Exponent quadrature refinement covers these prefixes only.",
                     "No transformer predictions or cumulative sequential codelengths are evaluated."],
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        provenance=dict(code_sha256={n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
                        **table_provenance()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    rows = ["case\tprefix_tokens\tmax_probability_error\tmax_mcse\talpha_refinement_error\tmax_grid_error_bits\tsampler_seconds\taccuracy_passed\tdiagnostics_passed"]
    for r in records:
        rows.append("\t".join(str(v) for v in (
            r["name"], r["prefix_length"], r["checks"]["sampled_prediction_absolute_cap"]["max_absolute_error"],
            max(r["estimate"]["mcse_probs"]), r["checks"]["exponent_prediction_refinement"]["max_absolute_error"],
            max(g["grid_log_likelihood_difference_bits"] for g in r["checks"]["independent_kernels_and_row_grids"]["cases"]),
            r["elapsed_seconds"]["sampler"], r["passed"], r["diagnostic_thresholds_passed"])))
    args.out.with_suffix(".tsv").write_text("\n".join(rows) + "\n")
    progress(out=str(args.out), passed=report["passed"],
             diagnostics_passed=report["diagnostic_thresholds_passed"], seconds=report["elapsed_seconds"])
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

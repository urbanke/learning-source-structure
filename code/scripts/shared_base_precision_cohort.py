"""Numerical precision for a predeclared mean of full-family Bayes regrets.

The scientific estimand is the mean over every declared sequence. Numerical
Monte Carlo errors are independent across sequences; Jensen bias in log of a
sampled predictive mean does not cancel. This module creates a distinct
cohort-validity decision and never relabels an old per-sequence precision flag.
Its curvature and jackknife checks are local bias diagnostics, not rigorous
bounds on nonconvergence or undiscovered posterior modes.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from stable_chain_statistics import (stable_predictive_moments, _mean as stable_mean,
    _variance as stable_variance, _covariance_of_mean as stable_covariance_of_mean)

LN2 = math.log(2)


def _law(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum()-1) > 1e-7:
        raise ValueError("invalid probability law")
    return p


def _loss(p, r):
    mask = r > 0
    return float(-r[mask] @ np.log2(p[mask]))


def _split_rhat(values):
    half = values.shape[1]//2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    within = float(stable_mean(stable_variance(split, 1), 0))
    between = float(half*stable_variance(stable_mean(split, 1), 0))
    if within < 1e-30:
        return 1.0 if between < 1e-30 else None
    return float(np.sqrt(((half-1)*within+between)/(half*within)))


def case_precision(samples, oracle):
    """Return loss precision from raw (chains, draws, tokens) predictions.

    No oracle information enters posterior inference: it enters the derivative
    of the scored loss here, after the predictions have already been computed.
    At least two chains and 16 draws are needed. Production uses four chains.
    """
    values = np.asarray(samples, dtype=float)
    r = _law(oracle)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 16
            or values.shape[2] != len(r) or not np.isfinite(values).all()
            or np.any(values <= 0) or np.max(abs(values.sum(axis=2)-1)) > 1e-7):
        raise ValueError("invalid posterior predictive samples")
    chains, draws, d = values.shape
    stable = stable_predictive_moments(values)
    p = stable["mean"]
    cm = stable["chain_means"]
    base = stable["batch_length"]
    covariances = {"between": stable["cov_chain"], f"batch_{base}": stable["cov_batch"]}
    for length in sorted({base, 2*base, 4*base, 8*base}):
        if length == base:
            continue
        n = draws//length
        if n < 8 and length != base:
            continue
        batch = stable_mean(values[:, :n*length].reshape(chains*n, length, d), 1)
        covariances[f"batch_{length}"] = np.asarray(stable_covariance_of_mean(batch), dtype=float)
    g = r/(p*LN2)
    g -= g.mean()
    projected = values@g
    variances = {name: max(0., float(g@c@g)) for name, c in covariances.items()}
    variance = max(variances.values())
    diagonal_envelope = np.maximum.reduce([np.maximum(np.diag(c), 0) for c in covariances.values()])
    jensen = float(np.sum(r*diagonal_envelope/(p*p))/(2*LN2))
    full_loss = _loss(p, r)
    jackknife = max(0., float((chains-1)*(np.mean([
        _loss(np.delete(cm, j, axis=0).mean(axis=0), r) for j in range(chains)])-full_loss)))
    marginal_variance = float(stable_variance(projected.ravel(), 0))
    ess = float(min(values.shape[0]*values.shape[1], marginal_variance/variance)) if variance > 1e-30 else float(chains*draws)
    relative = float(np.max(np.sqrt(diagonal_envelope)/p))
    scalar_rhat = _split_rhat(projected)
    return dict(predictive_probs=p.tolist(), bayes_cross_entropy_bits=full_loss,
        oracle_entropy_bits=_loss(r, r), bayes_regret_bits=full_loss-_loss(r, r),
        mc_variance_bits2=variance, mcse_bits=math.sqrt(variance),
        variance_by_estimator_bits2=variances,
        jensen_curvature_scale_bits=jensen, chain_jackknife_bias_bits=jackknife,
        jensen_proxy_bits=max(jensen, jackknife),
        max_relative_probability_mcse=relative,
        scalar_split_rhat=scalar_rhat, scalar_ess=ess,
        scalar_mixing_passed=scalar_rhat is not None and scalar_rhat < 1.05 and ess >= 200,
        jensen_local_check_passed=relative <= .1 and jackknife <= 2*jensen+1e-12,
        chains=chains, draws_per_chain=draws, deterministic=False)


def deterministic_precision(prediction, oracle):
    """No sampling error; quadrature and provenance must still be validated."""
    p, r = _law(prediction), _law(oracle)
    if len(p) != len(r) or np.any(p <= 0):
        raise ValueError("incompatible deterministic prediction")
    return dict(predictive_probs=p.tolist(), bayes_cross_entropy_bits=_loss(p,r),
        oracle_entropy_bits=_loss(r,r), bayes_regret_bits=_loss(p,r)-_loss(r,r),
        mc_variance_bits2=0., mcse_bits=0., variance_by_estimator_bits2={},
        jensen_curvature_scale_bits=0., chain_jackknife_bias_bits=0., jensen_proxy_bits=0.,
        max_relative_probability_mcse=0., scalar_split_rhat=1., scalar_ess=None,
        scalar_mixing_passed=True, jensen_local_check_passed=True,
        chains=0, draws_per_chain=0, deterministic=True)


def validate_cohort(coarse, refined, expected_ids, *, resolution_bits=.01,
                    numerical_fraction=1/3, relative_fraction=.25,
                    mc_multiplier=2., bias_multiplier=2.):
    """Validate both exponent rules for every predeclared case, without selection.

    Input lists contain records with keys case_id, sampling_seed (None only for
    a deterministic reference), precision (case_precision result), and boolean
    full_prior_valid, provenance_valid, mixing_passed, row_grid_passed,
    per_sequence_precision_passed. The old last flag is reported, never gated
    or overwritten. Incomplete or malformed cohorts raise ValueError.

    All uncertainty diagnostics remain conditional on MCMC reaching its target.
    The two-times-bias allowance is a declared diagnostic margin, not a bound.
    """
    expected = list(expected_ids)
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ValueError("need at least two distinct predeclared cases")
    if min(resolution_bits, numerical_fraction, relative_fraction, mc_multiplier, bias_multiplier) <= 0:
        raise ValueError("precision-budget parameters must be positive")
    required = ("full_prior_valid", "provenance_valid", "mixing_passed", "row_grid_passed")
    lookups = []
    for records in (coarse, refined):
        lookup = {v["case_id"]: v for v in records}
        if len(lookup) != len(records) or set(lookup) != set(expected):
            raise ValueError("every declared case must occur exactly once in both exponent rules")
        for item in records:
            if any(type(item[k]) is not bool for k in required+("per_sequence_precision_passed",)):
                raise ValueError("all validity flags must be explicit booleans")
            m = item["precision"]
            for key in ("bayes_regret_bits", "mc_variance_bits2", "jensen_proxy_bits"):
                if not np.isfinite(m[key]) or (key != "bayes_regret_bits" and m[key] < 0):
                    raise ValueError("invalid precision summary")
        lookups.append(lookup)
    stochastic = [r for lookup in lookups for r in lookup.values() if not r["precision"]["deterministic"]]
    seeds = [r["sampling_seed"] for r in stochastic]
    if any(s is None for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("all stochastic references require independent, distinct sampling seeds")
    n = len(expected)
    summaries = []
    checks = []
    for name, lookup in zip(("coarse", "refined"), lookups):
        records = [lookup[key] for key in expected]
        metrics = [r["precision"] for r in records]
        regrets = np.array([m["bayes_regret_bits"] for m in metrics])
        variances = np.array([m["mc_variance_bits2"] for m in metrics])
        source_sample_variance = float(regrets.var(ddof=1))
        observed_se = math.sqrt(source_sample_variance/n)
        denoised_se = math.sqrt(max(0., source_sample_variance-float(variances.mean()))/n)
        numerical_se = math.sqrt(float(variances.sum()))/n
        jensen_mean = float(np.mean([m["jensen_proxy_bits"] for m in metrics]))
        error_allowance = mc_multiplier*numerical_se+bias_multiplier*jensen_mean
        budget = min(resolution_bits*numerical_fraction, relative_fraction*denoised_se)
        all_basic = all(all(r[key] for key in required) for r in records)
        all_scalar = all(m["scalar_mixing_passed"] for m in metrics)
        all_jensen = all(m["jensen_local_check_passed"] for m in metrics)
        passed = all_basic and all_scalar and all_jensen and error_allowance <= budget+1e-14
        checks.extend([dict(name=name+":all_basic_checks", passed=all_basic),
                       dict(name=name+":all_scalar_mixing", passed=all_scalar),
                       dict(name=name+":all_jensen_local_checks", passed=all_jensen),
                       dict(name=name+":cohort_numerical_budget", passed=error_allowance <= budget+1e-14)])
        # Do not release a scientific mean if any required case check fails.
        summaries.append(dict(rule=name, n_sequences=n, candidate_mean_bayes_regret_bits=float(regrets.mean()),
            observed_sequence_se_bits=observed_se, denoised_sequence_se_bits=denoised_se,
            numerical_mcse_bits=numerical_se, mean_jensen_proxy_bits=jensen_mean,
            numerical_error_allowance_bits=error_allowance, numerical_budget_bits=budget,
            preliminary_passed=passed,
            per_sequence_precision_status={str(r["case_id"]): r["per_sequence_precision_passed"] for r in records}))
    differences = np.array([lookups[0][i]["precision"]["bayes_regret_bits"]-
                            lookups[1][i]["precision"]["bayes_regret_bits"] for i in expected])
    per_case_refinement = []
    for i, difference in zip(expected, differences):
        a, b = (lookup[i]["precision"] for lookup in lookups)
        sigma = math.sqrt(a["mc_variance_bits2"]+b["mc_variance_bits2"])
        allowance = 5*sigma+bias_multiplier*(a["jensen_proxy_bits"]+b["jensen_proxy_bits"])+.0002
        per_case_refinement.append(dict(case_id=i, absolute_difference_bits=abs(float(difference)),
                                        allowance_bits=allowance, passed=abs(difference) <= allowance))
    combined_mcse = math.hypot(*[s["numerical_mcse_bits"] for s in summaries])
    combined_bias = bias_multiplier*sum(s["mean_jensen_proxy_bits"] for s in summaries)
    mean_difference = abs(float(differences.mean()))
    resolution_limit = resolution_bits*numerical_fraction
    stochastic_limit = 5*combined_mcse+combined_bias+.0001
    checks.extend([dict(name="all_case_refinement_checks", passed=all(r["passed"] for r in per_case_refinement)),
                   dict(name="aggregate_refinement_resolution", passed=mean_difference <= resolution_limit),
                   dict(name="aggregate_refinement_stochastic_agreement", passed=mean_difference <= stochastic_limit)])
    passed = all(c["passed"] for c in checks)
    result = dict(schema="shared-base-cohort-precision-v1", passed=passed,
        estimand="mean true next-token regret over all predeclared cases at one setting and time",
        n_sequences=n, expected_ids=expected, checks=checks, rules=summaries,
        refinement=dict(absolute_mean_difference_bits=mean_difference,
                        resolution_limit_bits=resolution_limit,
                        stochastic_agreement_limit_bits=stochastic_limit,
                        per_case=per_case_refinement),
        budget=dict(resolution_bits=resolution_bits, numerical_fraction=numerical_fraction,
                    relative_fraction=relative_fraction, mc_multiplier=mc_multiplier, bias_multiplier=bias_multiplier),
        reported_mean_bayes_regret_bits=summaries[1]["candidate_mean_bayes_regret_bits"] if passed else None,
        note="No failed-case subset average. Old individual precision flags remain unchanged. Numerical bias allowance is a local diagnostic margin, not a rigorous bound or convergence certificate.")
    return result

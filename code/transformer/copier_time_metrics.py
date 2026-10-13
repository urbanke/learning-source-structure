"""Paired natural-sequence copying observables, in probability units and bits.

At array index j the prefix is tokens[:j+1] and the target is tokens[j+1].
The successor set uses only transitions completed inside that prefix.
No predictor is fitted, clipped, or renormalized by this module.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import xlogy


CONDITIONS = ("intact", "copier_deleted", "control_deleted", "bayes")
BAYES_TOLERANCE = 1e-7
PROBABILITY_TOLERANCE = 1e-12
_LOG2 = np.log(2.0)


def _probabilities(value, shape, name, tolerance, *, positive):
    """Validate the raw array; conversion to float64 does not repair it."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric probability array")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} contains invalid probabilities")
    if positive and np.any(array == 0):
        raise ValueError(f"{name} has a zero probability; finite log-loss evaluation requires positive predictions")
    error = float(np.max(np.abs(array.sum(axis=-1) - 1.0)))
    if error > tolerance:
        raise ValueError(f"{name} normalization error {error:.12g} exceeds {tolerance:.12g}")
    return array, error


def _kl_bits(first, second):
    # xlogy handles zero source probabilities exactly, without a floor.
    return np.sum(xlogy(first, first) - xlogy(first, second), axis=-1) / _LOG2


def natural_metrics(tokens, predictions, transition):
    """Return ``(flat_arrays, checks)`` for one complete independent sequence.

    ``tokens`` is an integer vector of length n+1. ``predictions`` contains
    exactly CONDITIONS, each with shape (n,d), and ``transition`` is the
    true d-by-d source transition matrix in the same token labels.
    Predictions must be strictly positive; source rows may contain zeros,
    but the realized sequence must have positive source probability.
    All supplied probabilities are checked as received and never repaired.

    ``benefit_copier_bits`` is deleted-head oracle loss minus intact loss;
    ``benefit_bayes_bits`` is uniform oracle loss minus Bayesian loss.
    These are different comparisons, not estimates of mixture coefficients.
    """
    raw_tokens = np.asarray(tokens)
    if raw_tokens.ndim != 1 or len(raw_tokens) < 2 or raw_tokens.dtype.kind not in "iu":
        raise ValueError("tokens must be an integer vector containing at least one prefix and its target")
    raw_transition = np.asarray(transition)
    if raw_transition.ndim != 2 or raw_transition.shape[0] != raw_transition.shape[1] or raw_transition.shape[0] < 2:
        raise ValueError("transition must be a square matrix with at least two symbols")
    d = raw_transition.shape[0]
    if np.any(raw_tokens < 0) or np.any(raw_tokens >= d):
        raise ValueError("tokens lie outside the transition matrix alphabet")
    tokens = raw_tokens.astype(np.int64, copy=False)
    n = len(tokens) - 1
    if not isinstance(predictions, Mapping) or set(predictions) != set(CONDITIONS):
        raise ValueError(f"predictions must contain exactly {CONDITIONS}")

    source, source_error = _probabilities(
        transition, (d, d), "transition", PROBABILITY_TOLERANCE, positive=False)
    probabilities, normalization = {}, {"transition": source_error}
    for name in CONDITIONS:
        tolerance = BAYES_TOLERANCE if name == "bayes" else PROBABILITY_TOLERANCE
        probabilities[name], normalization[name] = _probabilities(
            predictions[name], (n, d), name, tolerance, positive=True)

    counts = np.zeros((d, d), dtype=np.int64)
    row_counts = np.empty((n, d), dtype=np.int64)
    for j in range(n):
        row_counts[j] = counts[tokens[j]]
        # Record the target only AFTER the current prefix's statistic.
        counts[tokens[j], tokens[j+1]] += 1
    mask = row_counts > 0
    sizes = mask.sum(axis=-1)
    oracle = source[tokens[:-1]]
    targets = tokens[1:]
    target_source_probs = oracle[np.arange(n), targets]
    if np.any(target_source_probs == 0):
        raise ValueError("The realized sequence contains a transition with zero source probability")
    entropy = -np.sum(xlogy(oracle, oracle), axis=-1) / _LOG2
    uniform = np.full((n, d), 1.0 / d, dtype=np.float64)
    out = {
        "prefix_lengths": np.arange(1, n + 1, dtype=np.int64),
        "row_counts": row_counts,
        "row_totals": row_counts.sum(axis=-1),
        "successor_mask": mask,
        "support_sizes": sizes,
        "uniform_mass": sizes.astype(np.float64) / d,
        "source_entropy_bits": entropy,
        "source_observed_loss_bits": -np.log2(target_source_probs),
    }
    identity_errors = {}
    for name, probs in (*probabilities.items(), ("uniform", uniform)):
        ce = -np.sum(xlogy(oracle, probs), axis=-1) / _LOG2
        regret = ce - entropy
        out["oracle_ce_" + name] = ce
        out["oracle_regret_" + name] = regret
        out["observed_loss_" + name] = -np.log2(probs[np.arange(n), targets])
        identity_errors[name] = float(np.max(np.abs(regret - _kl_bits(oracle, probs))))
        if name != "uniform":
            mass = np.sum(probs * mask, axis=-1)
            out["mass_" + name] = mass
            out["excess_" + name] = mass - out["uniform_mass"]

    for label in ("copier", "control"):
        deleted = label + "_deleted"
        out[label + "_causal_mass"] = out["mass_intact"] - out["mass_" + deleted]
        benefit = out["oracle_ce_" + deleted] - out["oracle_ce_intact"]
        out["benefit_" + label + "_bits"] = benefit
        out["cumulative_benefit_" + label + "_bits"] = np.cumsum(benefit)
        out[deleted + "_to_uniform_kl_bits"] = _kl_bits(probabilities[deleted], uniform)
        out["uniform_to_" + deleted + "_kl_bits"] = _kl_bits(uniform, probabilities[deleted])
    out["benefit_bayes_bits"] = out["oracle_ce_uniform"] - out["oracle_ce_bayes"]
    out["cumulative_benefit_bayes_bits"] = np.cumsum(out["benefit_bayes_bits"])
    out["intact_bayes_kl_bits"] = _kl_bits(probabilities["intact"], probabilities["bayes"])

    if any(not np.isfinite(array).all() for array in out.values()):
        raise FloatingPointError("A natural-sequence metric is not finite; no clipping or normalization was applied")
    if max(identity_errors.values()) > 1e-10:
        raise FloatingPointError(f"Oracle regret/KL identity failed: {identity_errors}")
    checks = {
        "normalization_error": normalization,
        "oracle_regret_kl_error_bits": identity_errors,
        "empty_support_prefixes": int(np.count_nonzero(sizes == 0)),
        "full_support_prefixes": int(np.count_nonzero(sizes == d)),
        "n_prefixes": n,
        "d": d,
        "finite_metrics": True,
        "probability_repair": "none",
    }
    return out, checks

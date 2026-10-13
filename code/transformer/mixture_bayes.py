"""Combine family Bayes references when their sequence evidence is available.

All input logarithms are natural; reported losses are in bits. A family
evidence value is log B_f(x[:t]), with every within-family parameter already
integrated under its normalized prior. It is NOT a posterior weight, a
likelihood at fitted parameters, or an evidence known only up to a constant.

This module performs no integration or sampling. In particular, the current
shared-base posterior predictions at selected times do not supply the
sequence marginal likelihoods required here. Long-context shared-base
evidence remains a separate numerical task.

Declarations, ranges, the empty-prefix anchor, and (when supplied) child
evidence consistency are checked. These checks cannot certify the numerical
integration or detect every falsely labelled evidence array; the producing
reference must independently validate its normalization and prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

LN2 = np.log(2.0)
EVIDENCE_KIND = "normalized_sequence_marginal"
TOLERANCE = 1e-10


@dataclass(frozen=True, kw_only=True)
class SequenceEvidence:
    """One observed prefix, with one log marginal per family in fixed order.

    ``evidence_kind`` is deliberately required, so callers must explicitly
    declare absolute sequence marginals rather than pass posterior outputs.
    ``-inf`` means that the prefix is impossible under that family. The
    marginal of the empty prefix must be exactly one for every family.
    """

    prefix_length: int
    log_marginals: np.ndarray
    evidence_kind: str

    def __post_init__(self):
        if self.evidence_kind != EVIDENCE_KIND:
            raise ValueError("absolute normalized sequence marginals are required")
        if (isinstance(self.prefix_length, (bool, np.bool_))
                or not isinstance(self.prefix_length, (int, np.integer))
                or self.prefix_length < 0):
            raise ValueError("prefix_length must be a nonnegative integer")
        values = np.array(self.log_marginals, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("log_marginals must be a nonempty family vector")
        _check_log_marginals(values)
        if self.prefix_length == 0 and not np.all(values == 0):
            raise ValueError("every empty-prefix log marginal must be zero")
        values.setflags(write=False)
        object.__setattr__(self, "log_marginals", values)


def _check_log_marginals(values):
    if np.isnan(values).any() or np.any(values > 0):
        raise ValueError("sequence log marginals must be <= 0, with no NaNs")


def _check_evidence(evidence):
    if not isinstance(evidence, SequenceEvidence):
        raise TypeError("supply an explicit SequenceEvidence, not posterior predictions")


def _log_prior(evidence, prior_weights):
    _check_evidence(evidence)
    prior = np.asarray(prior_weights, dtype=np.float64)
    if (prior.shape != evidence.log_marginals.shape or not np.isfinite(prior).all()
            or np.any(prior < 0) or np.any(prior > 1)
            or not np.isclose(prior.sum(), 1.0, rtol=0, atol=TOLERANCE)):
        raise ValueError("prior weights must match the families and sum to one")
    logs = np.full(prior.shape, -np.inf)
    np.log(prior, out=logs, where=prior > 0)
    return logs


def _joint_and_total(evidence, prior_weights):
    log_prior = _log_prior(evidence, prior_weights)
    joint = evidence.log_marginals + log_prior
    if not np.isfinite(joint).any():
        raise ValueError("prefix is impossible under every positive-prior family")
    return joint, float(logsumexp(joint))


def posterior_family_weights(evidence, prior_weights):
    """Return P(F=f | x[:t]) from absolute marginal likelihoods and priors.

    Zero-prior or impossible families receive zero weight. An observation
    impossible under the entire mixture raises, since no posterior exists.
    """
    joint, _ = _joint_and_total(evidence, prior_weights)
    # Shift before exponentiating, avoiding underflow of the evidence itself.
    scaled = np.exp(joint - np.max(joint))
    return scaled / scaled.sum()


def mixture_prediction(evidence, prior_weights, component_probabilities, *,
                       successor_log_marginals=None):
    """Combine next-token distributions using posterior family weights.

    ``component_probabilities`` has shape (families, alphabet). Each active
    row must be a normalized probability distribution; raw rows are never
    clipped or renormalized. A zero-prior or impossible family may use an
    all-NaN row to mark its unused/undefined prediction.

    Optional ``successor_log_marginals[f, a]`` means log B_f(x[:t] + [a]).
    If supplied, all children must sum to their parent evidence, and their
    evidence ratios must match the component predictions. This detects,
    among other errors, a posterior vector reused as child evidence or a
    missing normalization factor at the current step. It does not replace
    an independent audit of the supplied marginal likelihoods.
    """
    joint, _ = _joint_and_total(evidence, prior_weights)
    active = np.isfinite(joint)
    probabilities = np.asarray(component_probabilities, dtype=np.float64)
    k = len(evidence.log_marginals)
    if probabilities.ndim != 2 or probabilities.shape[0] != k or probabilities.shape[1] == 0:
        raise ValueError("component probabilities must have shape (families, alphabet)")
    omitted = np.isnan(probabilities).all(axis=1)
    if np.any(omitted & active):
        raise ValueError("active families require a next-token distribution")
    supplied = probabilities[~omitted]
    if (not np.isfinite(supplied).all() or np.any(supplied < 0)
            or np.any(supplied > 1)
            or not np.allclose(supplied.sum(axis=1), 1.0, rtol=0, atol=TOLERANCE)):
        raise ValueError("component probability rows must be normalized")

    if successor_log_marginals is not None:
        children = np.asarray(successor_log_marginals, dtype=np.float64)
        if children.shape != probabilities.shape:
            raise ValueError("successor evidence must match the component probability shape")
        _check_log_marginals(children)
        parent = evidence.log_marginals
        finite = np.isfinite(parent)
        if not np.isneginf(children[~finite]).all():
            raise ValueError("an impossible prefix cannot have a possible continuation")
        child_totals = logsumexp(children[finite], axis=1)
        if not np.allclose(child_totals, parent[finite], rtol=0, atol=TOLERANCE):
            raise ValueError("successor evidence is not normalized to its parent marginal")
        compared = finite & ~omitted
        ratios = np.exp(children[compared] - parent[compared, None])
        if not np.allclose(ratios, probabilities[compared], rtol=0, atol=TOLERANCE):
            raise ValueError("component probabilities disagree with evidence ratios")

    weights = posterior_family_weights(evidence, prior_weights)
    return weights[active] @ probabilities[active]


def cumulative_mixture_losses(evidence, prior_weights, *, first_token_evidence=None):
    """Return mixture/component cumulative losses and differences, in bits.

    Normally all ``prefix_length`` tokens are scored. To omit the first
    observed token, supply its ``SequenceEvidence``. It must have length one
    and the same finite log marginal in every family. The shared first-token
    loss is then removed from both losses, preserving the mixture penalty
    log2(1 / prior_weight). This function deliberately rejects unequal
    first-token marginals; that case needs conditional prior weights.

    ``mixture_minus_component_bits[f]`` can be negative on an individual
    sequence, and is -inf when that family gives the sequence zero mass.
    ``upper_bound_bits[f]`` is infinite for a zero-prior family. No posterior
    or loss is returned for a sequence impossible under the whole mixture.
    """
    _, log_mixture = _joint_and_total(evidence, prior_weights)
    common = 0.0
    scored_tokens = evidence.prefix_length
    if first_token_evidence is not None:
        _check_evidence(first_token_evidence)
        first = first_token_evidence.log_marginals
        if (evidence.prefix_length < 1 or first_token_evidence.prefix_length != 1
                or first.shape != evidence.log_marginals.shape
                or not np.isfinite(first).all()
                or not np.allclose(first, first[0], rtol=0, atol=TOLERANCE)):
            raise ValueError("omitting the first token requires one common finite marginal")
        if np.any(evidence.log_marginals > first + TOLERANCE):
            raise ValueError("sequence evidence cannot exceed its first-token marginal")
        if evidence.prefix_length == 1 and not np.allclose(
                evidence.log_marginals, first, rtol=0, atol=TOLERANCE):
            raise ValueError("the two first-token evidence arrays must agree")
        common = float(first[0])
        scored_tokens -= 1
    return {
        "scored_tokens": scored_tokens,
        "mixture_loss_bits": (-log_mixture + common) / LN2,
        "component_loss_bits": (-evidence.log_marginals + common) / LN2,
        "mixture_minus_component_bits": (evidence.log_marginals - log_mixture) / LN2,
        "upper_bound_bits": -_log_prior(evidence, prior_weights) / LN2,
    }

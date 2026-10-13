"""Experimental normalized evidence for shared-base LSA, using existing kernels.

This is a finite-particle estimate, not an exact Bayesian reference. Start
from the *normalized* prior conditional on the first observed token, whose
marginal is exactly 1/d. Anneal the collapsed transition likelihood; keep
the alpha prior, rank prior and first-token likelihood untempered. Fixed
temperatures and unbiased resampling give the standard SMC normalizing-
constant estimate for the supplied quadrature target. Its logarithm has
Monte Carlo bias. Independent repetitions and refinement are required.

No source parameters or fitted token frequencies initialize the particles.
The implementation reuses PrefixPosterior and either the original
SharedBaseMCMC moves or the existing corrected Dirichlet-row block moves.
The sealed-store RankedLSARowEvidence and existing samplers are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import time

import numpy as np
from scipy.special import logsumexp

from shared_base_mcmc import PrefixPosterior
from shared_depth import _tokens


class TemperedTransitionTarget:
    """Existing posterior target with only transition evidence tempered."""

    def __init__(self, target, beta):
        self.target, self.beta = target, float(beta)
        self.seen, self.d = target.seen, target.d
        # The row-block auxiliary uses these observed sufficient statistics;
        # they never change the tempered target or supply true parameters.
        self.x, self.counts = target.x, target.counts

    def transition_loglik(self, state):
        first = self.target.sampler.kernel.log_base_laws[
            state.alpha_index, state.ranks[self.target.x[0]]]
        return state.log_likelihood - first

    def retarget(self, state):
        sampler = self.target.sampler
        first = sampler.kernel.log_base_laws[
            state.alpha_index, state.ranks[self.target.x[0]]]
        value = (first + sampler.log_alpha_weights[state.alpha_index]
                 + self.beta * self.transition_loglik(state))
        return replace(state, log_target=float(value))

    def evaluate(self, alpha_index, ranks):
        return self.retarget(self.target.evaluate(alpha_index, ranks))

    def swap(self, state, i, j):
        return self.retarget(self.target.swap(state, i, j))


def systematic_resample(weights, rng):
    """Unbiased offspring counts, with one independent random offset."""
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # only roundoff in a normalized resampling CDF
    points = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(cumulative, points, side="right")


def estimate_log_evidence(sampler, tokens, *, particles=128, temperatures=64,
                          moves=4, seed=2026091603, resample_fraction=.75):
    """One SMC replicate; natural-log marginal includes the first token.

    The original SharedBaseMCMC remains supported unchanged. The exact
    SharedBaseEfficiencyRowMCMC class adds its existing reversible auxiliary
    block proposal, with the full endpoint Hastings correction. Its auxiliary
    stays untempered; the acceptance target is this routine's current beta.
    All other subclasses are rejected pending a separate invariance audit.
    """
    from shared_base_mcmc import SharedBaseMCMC
    from shared_base_efficiency import SharedBaseEfficiencyRowMCMC
    if type(sampler) not in (SharedBaseMCMC, SharedBaseEfficiencyRowMCMC):
        raise TypeError("use the audited SharedBaseMCMC or SharedBaseEfficiencyRowMCMC class")
    if type(sampler) is SharedBaseMCMC:
        move_names = ["seen_label_swap", "head_weighted_rank_swap",
                      "alpha_prior_proposal", "alpha_adjacent_proposal"]
        proposal_spec = dict(id="original-shared-base-mcmc-four-moves",
                             outer_probabilities=[.4, .4, .1, .1])
    else:
        move_names = ["dirichlet_row_surrogate_block",
                      "equal_mixture_seen_label_and_head_weighted_rank_swap",
                      "alpha_prior_proposal", "alpha_adjacent_proposal"]
        proposal_spec = deepcopy(sampler.proposal_spec)
    for name, value, minimum in (("particles", particles, 2),
                                 ("temperatures", temperatures, 1),
                                 ("moves", moves, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 < resample_fraction <= 1:
        raise ValueError("resample_fraction must lie in (0,1]")
    x = _tokens(tokens, sampler.d)
    if len(x) <= 1:
        return dict(log_evidence=0. if not len(x) else -float(np.log(sampler.d)),
                    exact_anchor=True, n_tokens=len(x), elapsed_seconds=0.,
                    sampler_class=type(sampler).__name__, move_names=move_names,
                    proposal_spec=proposal_spec)
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    target = PrefixPosterior(sampler, x)
    tempered = TemperedTransitionTarget(target, 0.)
    states = []
    for _ in range(particles):
        ai = int(rng.choice(len(sampler.alpha_weights), p=sampler.alpha_weights))
        first_rank = int(rng.choice(sampler.d, p=sampler.kernel.base_laws[ai]))
        ranks = rng.permutation(sampler.d)
        other_label = int(np.flatnonzero(ranks == first_rank)[0])
        ranks[x[0]], ranks[other_label] = ranks[other_label], ranks[x[0]]
        states.append(tempered.evaluate(ai, ranks))
    betas = np.linspace(0., 1., temperatures + 1) ** 3
    log_weights = np.full(particles, -np.log(particles))
    log_evidence = -float(np.log(sampler.d))
    attempts = np.zeros(len(move_names), dtype=np.int64)
    accepts = np.zeros(len(move_names), dtype=np.int64)
    ess_history, log_increments, resampled = [], [], []
    ancestry = np.arange(particles)
    ancestry_history = [particles]
    n_leading = min(4, sampler.d)
    leading_changes = np.zeros(n_leading, dtype=np.int64)
    leading_distinct_labels, leading_distinct_assignments = [], []

    def record_leading_diversity():
        assignments = np.asarray([np.argsort(s.ranks)[:n_leading] for s in states])
        leading_distinct_labels.append([len(np.unique(assignments[:, i])) for i in range(n_leading)])
        leading_distinct_assignments.append(len(np.unique(assignments, axis=0)))
        return assignments

    record_leading_diversity()
    for previous, beta in zip(betas[:-1], betas[1:]):
        loglik = np.array([tempered.transition_loglik(s) for s in states])
        incremental = log_weights + (beta - previous) * loglik
        normalizer = float(logsumexp(incremental))
        log_evidence += normalizer
        log_increments.append(normalizer)
        log_weights = incremental - normalizer
        weights = np.exp(log_weights)
        ess = float(1. / (weights @ weights))
        ess_history.append(ess)
        should_resample = ess < resample_fraction * particles
        resampled.append(should_resample)
        if should_resample:
            indices = systematic_resample(weights, rng)
            states = [states[i] for i in indices]
            ancestry = ancestry[indices]
            log_weights.fill(-np.log(particles))
        ancestry_history.append(len(np.unique(ancestry)))
        tempered.beta = float(beta)
        for index, state in enumerate(states):
            state = tempered.retarget(state)
            leading_before = np.argsort(state.ranks)[:n_leading]
            for _ in range(moves):
                state = sampler._step(tempered, state, rng, attempts, accepts)
                leading_after = np.argsort(state.ranks)[:n_leading]
                leading_changes += leading_after != leading_before
                leading_before = leading_after
            states[index] = state
        final_leading = record_leading_diversity()
    final_weights = np.exp(log_weights)
    leading_probabilities = np.asarray([
        np.bincount(final_leading[:, i], weights=final_weights, minlength=sampler.d)
        for i in range(n_leading)])
    return dict(log_evidence=float(log_evidence), exact_anchor=False,
                n_tokens=len(x), particles=particles, temperatures=temperatures,
                beta_schedule="(k/K)^3; fixed before sampling", moves=moves,
                seed=seed, resample_fraction=resample_fraction,
                effective_sample_sizes=ess_history, resampled=resampled,
                log_normalizer_increments=log_increments,
                move_attempts=attempts.tolist(), move_accepts=accepts.tolist(),
                sampler_class=type(sampler).__name__, move_names=move_names,
                proposal_spec=proposal_spec,
                initial_particle_ancestry_counts=ancestry_history,
                leading_rank_diagnostics=dict(
                    tracked_ranks=list(range(n_leading)),
                    label_changes_during_mh=leading_changes.tolist(),
                    distinct_labels_by_temperature=leading_distinct_labels,
                    distinct_assignments_by_temperature=leading_distinct_assignments,
                    final_weighted_label_probabilities=leading_probabilities.tolist(),
                    change_count_convention="Holder changes between outer MH endpoints; excludes resampling and transient auxiliary states.",
                    interpretation="Diversity and movement only; neither proves posterior exploration or evidence accuracy."),
                elapsed_seconds=time.monotonic() - started,
                scientific_reference_validated=False,
                note="One SMC estimate. Particle/temperature/alpha/row-grid refinement and independent repeats are required.")

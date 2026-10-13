"""Reversible inexpensive proposals with the full shared-base MH correction.

The auxiliary distribution is the depth-zero likelihood of every observed
token, including the first, times the exponent quadrature weights. It is
used only to propose; the accepted state targets the existing joint rank /
exponent posterior with *all* depths marginalized by PrefixPosterior.

One auxiliary step randomly selects exponent Gibbs or a random matching
of disjoint label-pair Gibbs updates. The latter commute because the
auxiliary log density is additive across labels. Each matching kernel is
reversible; choosing a matching independently of the state preserves this.
The random-scan mixture and any fixed power of it are therefore reversible.
Consequently the endpoint proposal ratio is the reverse auxiliary-density
ratio, even though its transition probabilities are not computed.

The other outer moves retain global rank and exponent proposals. No row
integral, latent depth, numerical target, or conditional prediction is
replaced by the auxiliary distribution.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from shared_base_mcmc import SharedBaseMCMC


MOVE_NAMES = [
    "depth_zero_surrogate_block",
    "equal_mixture_seen_label_and_head_weighted_rank_swap",
    "alpha_prior_proposal",
    "alpha_adjacent_proposal",
]
OUTER_PROBABILITIES = [0.5, 0.4, 0.05, 0.05]
ALPHA_GIBBS_PROBABILITY = 0.2
PROPOSAL_SPEC = dict(
    id="shared-base-depth-zero-surrogate-v1",
    surrogate_steps=64,
    outer_probabilities=list(OUTER_PROBABILITIES),
    surrogate_alpha_gibbs_probability=ALPHA_GIBBS_PROBABILITY,
    matching="uniform random permutation, adjacent disjoint label pairs; odd last label untouched",
    auxiliary_target="alpha quadrature weight times product of q over every prefix token, including the first",
    acceptance="full collapsed log-target difference minus auxiliary log-target difference",
    rank_bucket="equal mixture of original seen-label and head-weighted-rank swaps",
    counters="four outer buckets; accepted blocks include unchanged endpoints",
)


class SharedBaseSurrogateMCMC(SharedBaseMCMC):
    """Full collapsed posterior with corrected multi-step auxiliary moves."""

    def __init__(self, d, row_kernel, alpha_weights, depth_prior=None,
                 prob_tol=1e-7, *, surrogate_steps=64):
        super().__init__(d, row_kernel, alpha_weights, depth_prior, prob_tol)
        if (isinstance(surrogate_steps, (bool, np.bool_))
                or not isinstance(surrogate_steps, (int, np.integer))
                or surrogate_steps < 1):
            raise ValueError("surrogate_steps must be a positive integer")
        self.surrogate_steps = int(surrogate_steps)
        self.proposal_spec = deepcopy(PROPOSAL_SPEC)
        self.proposal_spec["surrogate_steps"] = self.surrogate_steps

    def surrogate_log_target(self, counts, alpha_index, ranks):
        """Unnormalized auxiliary log density; counts include the first token."""
        return float(self.log_alpha_weights[alpha_index]
                     + counts @ self.kernel.log_base_laws[alpha_index, ranks])

    def _surrogate_step(self, counts, alpha_index, ranks, rng):
        """Mutate a private rank vector; return the new exponent index."""
        if rng.random() < ALPHA_GIBBS_PROBABILITY:
            terms = (self.log_alpha_weights
                     + self.kernel.log_base_laws[:, ranks] @ counts)
            return int(rng.choice(len(self.alpha_weights),
                                  p=np.exp(terms - logsumexp(terms))))
        permutation = rng.permutation(self.d)
        pairs = permutation[:2 * (self.d // 2)].reshape(-1, 2)
        i, j = pairs.T
        logq = self.kernel.log_base_laws[alpha_index]
        delta = (counts[i] - counts[j]) * (logq[ranks[j]] - logq[ranks[i]])
        chosen = rng.random(len(pairs)) < expit(delta)
        i, j = i[chosen], j[chosen]
        # Disjoint pairs allow a simultaneous Gibbs update without conflicts.
        ranks[i], ranks[j] = ranks[j].copy(), ranks[i].copy()
        return alpha_index

    def surrogate_proposal(self, target, state, rng):
        counts = np.bincount(target.x, minlength=self.d)
        alpha_index, ranks = state.alpha_index, state.ranks.copy()
        before = self.surrogate_log_target(counts, alpha_index, ranks)
        for _ in range(self.surrogate_steps):
            alpha_index = self._surrogate_step(counts, alpha_index, ranks, rng)
        after = self.surrogate_log_target(counts, alpha_index, ranks)
        return alpha_index, ranks, before - after

    def _step(self, target, state, rng, attempts, accepts):
        move = int(rng.choice(4, p=OUTER_PROBABILITIES))
        attempts[move] += 1
        correction = 0.0
        if move == 0:
            index, ranks, correction = self.surrogate_proposal(target, state, rng)
            proposed = (state if index == state.alpha_index
                        and np.array_equal(ranks, state.ranks)
                        else target.evaluate(index, ranks))
        elif move == 1:
            if rng.integers(2) == 0:
                i, j = int(rng.choice(target.seen)), int(rng.integers(self.d))
            else:
                r = int(rng.choice(self.d, p=self.rank_proposal_weights))
                s = int(rng.integers(self.d))
                labels = np.argsort(state.ranks)
                i, j = int(labels[r]), int(labels[s])
            proposed = target.swap(state, i, j)
        elif move == 2:
            index = int(rng.choice(len(self.alpha_weights), p=self.alpha_weights))
            proposed = target.evaluate(index, state.ranks) if index != state.alpha_index else state
            correction = self.log_alpha_weights[state.alpha_index] - self.log_alpha_weights[index]
        else:
            index = state.alpha_index + int(rng.choice([-1, 1]))
            proposed = target.evaluate(index, state.ranks) if 0 <= index < len(self.alpha_weights) else state
        if np.log(rng.random()) < min(0.0, proposed.log_target - state.log_target + correction):
            accepts[move] += 1
            return proposed
        return state

    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        diagnostic = result["diagnostics"]
        if "move_names" in diagnostic:
            diagnostic["move_names"] = list(MOVE_NAMES)
        diagnostic["proposal_spec"] = deepcopy(self.proposal_spec)
        return result


def proposal_descriptor(sampler, extra_code_files=()):
    """Bind proposal settings and implementation to a new runner's identity."""
    if type(sampler) is not SharedBaseSurrogateMCMC:
        raise TypeError("proposal descriptor requires SharedBaseSurrogateMCMC")
    here = Path(__file__).resolve().parent
    names = ("shared_base_surrogate.py", *extra_code_files)
    return dict(schema="shared-base-surrogate-proposal-v1",
                sampler_class="shared_base_surrogate.SharedBaseSurrogateMCMC",
                proposal_spec=deepcopy(sampler.proposal_spec),
                code_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                             for name in names})

"""Bounded prototype: local rank swaps for the existing collapsed posterior.

The target, conditional predictions, depth marginalization and all numerical
kernels are inherited unchanged. Half of the original head-weighted proposal
bucket instead exchanges the labels occupying a uniformly chosen adjacent
pair of ranks. Global swaps remain, preserving the original accessibility.

For an adjacent rank pair (r,r+1), each direction proposes that same pair
with probability 0.2/(d-1). Swapping its labels leaves the unordered pair of
ranks unchanged, so no Hastings correction is needed. The other proposals
retain their existing corrections. A mixture of these reversible kernels
is reversible with respect to the same collapsed posterior.

This module is not adopted by the production runner. Its code hash and
PROPOSAL_SPEC must enter any future checkpoint fingerprint: the inherited
ChainRun fingerprint currently hashes only the original sampler files, and
its report hard-codes the original bucket labels.
"""
from __future__ import annotations

import numpy as np

from shared_base_mcmc import SharedBaseMCMC


PROPOSAL_SPEC = {
    "id": "shared-base-local-rank-v1",
    "elementary_probabilities": {
        "seen_label_swap": 0.4,
        "head_weighted_rank_swap": 0.2,
        "uniform_adjacent_rank_swap": 0.2,
        "alpha_prior_proposal": 0.1,
        "alpha_adjacent_proposal": 0.1,
    },
    "adjacent_rank_selection": "uniform over zero-based r=0,...,d-2; exchange ranks r and r+1",
    "bucket_probabilities": [0.4, 0.4, 0.1, 0.1],
    "mixed_bucket_conditional_probabilities": [0.5, 0.5],
}

MOVE_NAMES = [
    "seen_label_swap",
    "equal_mixture_head_weighted_and_uniform_adjacent_rank_swap",
    "alpha_prior_proposal",
    "alpha_adjacent_proposal",
]


class SharedBaseLocalRankMCMC(SharedBaseMCMC):
    """Existing posterior sampler with one explicitly reported mixed bucket."""

    def _step(self, target, state, rng, attempts, accepts):
        move = int(rng.choice(4, p=[0.4, 0.4, 0.1, 0.1]))
        attempts[move] += 1
        correction = 0.0
        if move == 0:
            i, j = int(rng.choice(target.seen)), int(rng.integers(self.d))
            proposed = target.swap(state, i, j)
        elif move == 1:
            adjacent = bool(rng.integers(2))
            labels = np.argsort(state.ranks)
            if adjacent:
                r = int(rng.integers(self.d - 1))
                i, j = int(labels[r]), int(labels[r + 1])
            else:
                r = int(rng.choice(self.d, p=self.rank_proposal_weights))
                s = int(rng.integers(self.d))
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
        diagnostic["proposal_spec"] = PROPOSAL_SPEC
        return result

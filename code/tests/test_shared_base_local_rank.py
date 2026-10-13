"""Finite-state checks for the prototype's actual mixed MH transitions."""
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_local_rank import MOVE_NAMES, PROPOSAL_SPEC, SharedBaseLocalRankMCMC
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC


def sampler(d=3):
    nodes, weights = leggauss(3)
    return SharedBaseLocalRankMCMC(d, RankedLSARowEvidence(d, 1 + nodes/2, (0,)), weights/2)


class ScriptedRNG:
    def __init__(self, choices, integers, uniform):
        self.choices, self.values, self.uniform = list(choices), list(integers), uniform
        self.first_choice = True

    def choice(self, values, p=None):
        if self.first_choice:
            assert values == 4
            np.testing.assert_array_equal(p, [.4, .4, .1, .1])
            self.first_choice = False
        if not self.choices:
            raise AssertionError("unexpected choice call")
        return self.choices.pop(0)

    def integers(self, high):
        value = self.values.pop(0)
        assert 0 <= value < high
        return value

    def random(self):
        assert not self.choices and not self.values
        return self.uniform


def exchanged(ranks, i, j):
    values = list(ranks)
    values[i], values[j] = values[j], values[i]
    return tuple(values)


@pytest.mark.parametrize("x", [[0], [0, 1], [0, 0, 1, 2, 1]])
def test_actual_transition_acceptance_detailed_balance_and_stationarity(x):
    mc = sampler()
    target = PrefixPosterior(mc, x)
    states = [(a, ranks) for a in range(3) for ranks in permutations(range(3))]
    indices = {state: i for i, state in enumerate(states)}
    # This target is computed directly from the iid depth-zero likelihood,
    # independently of PrefixPosterior's collapsed-row implementation.
    counts = np.bincount(x, minlength=3)
    logpi = np.array([np.log(mc.alpha_weights[a]) + counts @ mc.kernel.log_base_laws[a, list(r)]
                      for a, r in states])
    pi = np.exp(logpi - logsumexp(logpi))
    transition = np.zeros((len(states), len(states)))
    adjacent_proposal = np.zeros_like(transition)
    for s, (a, ranks) in enumerate(states):
        current = target.evaluate(a, ranks)
        proposals = []
        for i in target.seen:
            for j in range(3):
                proposals.append(([0, int(i)], [j], (a, exchanged(ranks, i, j)),
                                  .4 / len(target.seen) / 3, 0.0))
        weights = 1/np.arange(1, 4, dtype=float)
        weights /= weights.sum()
        for r in range(3):
            for q in range(3):
                changed = exchanged(ranks, ranks.index(r), ranks.index(q))
                proposals.append(([1, r], [0, q], (a, changed), .2*weights[r]/3, 0.0))
        for r in range(2):
            changed = exchanged(ranks, ranks.index(r), ranks.index(r+1))
            destination = (a, changed)
            proposals.append(([1], [1, r], destination, .2/2, 0.0))
            adjacent_proposal[s, indices[destination]] += .2/2
        for b in range(3):
            proposals.append(([2, b], [], (b, ranks), .1*mc.alpha_weights[b],
                              np.log(mc.alpha_weights[a]/mc.alpha_weights[b])))
        for direction in (-1, 1):
            b = a + direction
            b = b if 0 <= b < 3 else a
            proposals.append(([3, direction], [], (b, ranks), .1/2, 0.0))
        assert sum(p[3] for p in proposals) == pytest.approx(1, abs=1e-15)
        for choices, integers, destination, probability, correction in proposals:
            t = indices[destination]
            acceptance = min(1, np.exp(logpi[t] - logpi[s] + correction))
            for u in (acceptance/2, (1+acceptance)/2):
                attempts, accepts = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
                observed = mc._step(target, current, ScriptedRNG(choices, integers, u), attempts, accepts)
                expected = destination if u < acceptance else (a, ranks)
                assert (observed.alpha_index, tuple(observed.ranks)) == expected
                assert attempts.sum() == 1 and attempts[choices[0]] == 1
                assert accepts.sum() == int(u < acceptance)
            transition[s, t] += probability*acceptance
            transition[s, s] += probability*(1-acceptance)
    # Actual adjacent proposal probabilities agree in both directions.
    np.testing.assert_array_equal(adjacent_proposal, adjacent_proposal.T)
    np.testing.assert_allclose(adjacent_proposal.sum(axis=1), .2, atol=1e-15)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=2e-15)
    flow = pi[:, None]*transition
    np.testing.assert_allclose(flow, flow.T, atol=2e-16)
    np.testing.assert_allclose(pi @ transition, pi, atol=2e-16)
    assert np.all(np.diag(transition) > 0)


def test_adjacent_move_uses_rank_order_and_reverses():
    mc = sampler(4)
    target = PrefixPosterior(mc, [0, 1])
    initial = target.evaluate(1, [2, 0, 3, 1])
    # Ranks 1 and 2 occupy labels 3 and 0, not labels 1 and 2.
    moved = mc._step(target, initial, ScriptedRNG([1], [1, 1], 1e-100),
                     np.zeros(4, int), np.zeros(4, int))
    np.testing.assert_array_equal(moved.ranks, [1, 0, 3, 2])
    restored = mc._step(target, moved, ScriptedRNG([1], [1, 1], 1e-100),
                        np.zeros(4, int), np.zeros(4, int))
    np.testing.assert_array_equal(restored.ranks, initial.ranks)


def test_sample_identifies_proposal_and_retains_four_buckets():
    mc = sampler()
    output = mc.sample([0, 1, 0, 2], chains=2, burn=32, draws=64, thin=2,
                       seed=27, retain_samples=True)
    second = mc.sample([0, 1, 0, 2], chains=2, burn=32, draws=64, thin=2,
                       seed=27, retain_samples=True)
    diagnostic = output["diagnostics"]
    assert diagnostic["move_names"] == MOVE_NAMES
    assert diagnostic["proposal_spec"] == PROPOSAL_SPEC
    assert np.asarray(diagnostic["move_attempts"]).shape == (2, 4)
    np.testing.assert_array_equal(np.sum(diagnostic["move_attempts"], axis=1), [128, 128])
    np.testing.assert_array_equal(output["samples"]["predictive_probs"], second["samples"]["predictive_probs"])
    assert output["samples"]["predictive_probs"].shape == (2, 64, 3)
    baseline = SharedBaseMCMC(mc.d, mc.kernel, mc.alpha_weights)
    original = baseline.sample([0, 1], chains=2, burn=16, draws=16, thin=1, seed=2)
    assert original["diagnostics"]["move_names"][1] == "head_weighted_rank_swap"

"""Independent targets and finite-state detailed balance for posterior moves."""
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base import SharedBaseLSA
from shared_base_mcmc import PrefixPosterior, SharedBaseMCMC, chain_summary
from test_shared_base import ElementaryRows


def sampler(d=3, depths=(0,), order=3):
    nodes, weights = leggauss(order)
    return SharedBaseMCMC(d, RankedLSARowEvidence(d, 1 + nodes / 2, depths), weights / 2)


def test_collapsed_target_and_predictions_against_independent_simplex():
    mc = sampler(2, (0, 1))
    x = [0, 0, 1, 0, 1, 1]
    target = PrefixPosterior(mc, x)
    for a in range(3):
        for ranks in permutations(range(2)):
            q = mc.kernel.base_laws[a, list(ranks)]
            exact = SharedBaseLSA(2, ElementaryRows(), known_q=q)
            state = target.evaluate(a, ranks)
            assert state.log_likelihood == pytest.approx(exact.log_joint(x), abs=2e-12)
            result = exact.run(x + [0], full_probs=True, positions=[len(x)], posteriors=True)
            p, depth, error = target.predict(state)
            np.testing.assert_allclose(p, result['probs'][0], atol=2e-12)
            np.testing.assert_allclose(depth, result['depth_posterior'][-1], atol=2e-12)
            assert error < 2e-12
            swapped = target.swap(state, 0, 1)
            rebuilt = target.evaluate(a, state.ranks[::-1])
            np.testing.assert_allclose(swapped.rows, rebuilt.rows, atol=1e-13)
            assert swapped.log_target == pytest.approx(rebuilt.log_target, abs=1e-13)


class ScriptedRNG:
    def __init__(self, move, choice, integer, u):
        self.move, self.next_choice, self.integer, self.u = move, choice, integer, u
        self.first = True

    def choice(self, *args, **kwargs):
        if self.first:
            self.first = False
            return self.move
        return self.next_choice

    def integers(self, *args):
        return self.integer

    def random(self):
        return self.u


@pytest.mark.parametrize('x', [[0], [0, 1], [0, 0, 1, 2, 1]])
def test_actual_moves_match_independent_mh_acceptance_and_detailed_balance(x):
    mc = sampler()
    target = PrefixPosterior(mc, x)
    states = [(a, r) for a in range(3) for r in permutations(range(3))]
    indices = {s: i for i, s in enumerate(states)}
    # Independent iid target at L=0, with nonuniform quadrature weights.
    frequencies = np.bincount(x, minlength=3)
    logpi = np.array([np.log(mc.alpha_weights[a]) + frequencies @ mc.kernel.log_base_laws[a, list(r)]
                      for a, r in states])
    pi = np.exp(logpi - logsumexp(logpi))
    transition = np.zeros((len(states), len(states)))
    for s, (a, ranks) in enumerate(states):
        state = target.evaluate(a, ranks)
        proposals = []
        for i in target.seen:
            for j in range(3):
                changed = list(ranks)
                changed[i], changed[j] = changed[j], changed[i]
                proposals.append((0, int(i), j, (a, tuple(changed)), .4 / len(target.seen) / 3, 0.0))
        rank_weights = 1 / np.arange(1, 4, dtype=float)
        rank_weights /= rank_weights.sum()
        for r in range(3):
            for t in range(3):
                i, j = ranks.index(r), ranks.index(t)
                changed = list(ranks)
                changed[i], changed[j] = changed[j], changed[i]
                proposals.append((1, r, t, (a, tuple(changed)), .4 * rank_weights[r] / 3, 0.0))
        for b in range(3):
            correction = np.log(mc.alpha_weights[a] / mc.alpha_weights[b])
            proposals.append((2, b, None, (b, ranks), .1 * mc.alpha_weights[b], correction))
        for direction in (-1, 1):
            b = a + direction
            b = b if 0 <= b < 3 else a
            proposals.append((3, direction, None, (b, ranks), .1 / 2, 0.0))
        for move, choice, integer, destination, probability, correction in proposals:
            t = indices[destination]
            acceptance = min(1, np.exp(logpi[t] - logpi[s] + correction))
            for u in (acceptance / 2, (1 + acceptance) / 2):
                actual = mc._step(target, state, ScriptedRNG(move, choice, integer, u),
                                  np.zeros(4, dtype=int), np.zeros(4, dtype=int))
                wanted = destination if u < acceptance else (a, ranks)
                assert (actual.alpha_index, tuple(actual.ranks)) == wanted
            transition[s, t] += probability * acceptance
            transition[s, s] += probability * (1 - acceptance)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=2e-15)
    flow = pi[:, None] * transition
    np.testing.assert_allclose(flow, flow.T, atol=2e-16)
    np.testing.assert_allclose(pi @ transition, pi, atol=2e-16)
    assert np.all(np.diag(transition) > 0)  # no periodic transposition-only chain


def test_unseen_predictions_are_conditional_average_over_completions():
    mc = sampler(4)
    target = PrefixPosterior(mc, [2, 2, 2])
    state = target.evaluate(1, [0, 2, 1, 3])
    p, _, _ = target.predict(state)
    assert p[2] == pytest.approx(mc.kernel.base_laws[1, 1])
    expected = mc.kernel.base_laws[1, [0, 2, 3]].mean()
    np.testing.assert_allclose(p[[0, 1, 3]], expected, atol=1e-15)


def test_sampling_is_reproducible_and_does_not_change_input():
    mc = sampler()
    x = np.array([0, 1, 0, 2])
    first = mc.sample(x, chains=2, burn=32, draws=64, thin=1, seed=17)
    second = mc.sample(x, chains=2, burn=32, draws=64, thin=1, seed=17)
    np.testing.assert_array_equal(x, [0, 1, 0, 2])
    np.testing.assert_array_equal(first['predictive_probs'], second['predictive_probs'])
    assert first['diagnostics'] == second['diagnostics']
    assert sum(first['predictive_probs']) == pytest.approx(1, abs=2e-15)


def test_diagnostics_detect_separated_constant_chains():
    values = np.zeros((4, 64, 2))
    values[2:, :, 0] = 1
    out = chain_summary(values)
    assert np.isinf(out['rhat'][0])
    assert out['mcse'][0] > .2
    assert out['mcse'][1] == 0
    assert out['rhat'][1] == 1


@pytest.mark.parametrize('kwargs', [dict(chains=1), dict(draws=15), dict(burn=-1), dict(thin=0), dict(chains=True)])
def test_invalid_sampling_settings(kwargs):
    with pytest.raises(ValueError):
        sampler().sample([0], **kwargs)

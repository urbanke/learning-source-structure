"""Enumerate auxiliary and full-target transition laws independently."""
from itertools import permutations, product
from math import factorial
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base import SharedBaseLSA
from shared_base_mcmc import PrefixPosterior
from shared_base_surrogate import (ALPHA_GIBBS_PROBABILITY, MOVE_NAMES,
    SharedBaseSurrogateMCMC, proposal_descriptor)
from test_shared_base import ElementaryRows


def sampler(d=3, depths=(0,), steps=3):
    z, w = leggauss(3)
    return SharedBaseSurrogateMCMC(d, RankedLSARowEvidence(d, 1+z/2, depths),
                                   w/2, surrogate_steps=steps)


def enumeration(mc, x):
    states = [(a, r) for a in range(len(mc.alpha_weights))
              for r in permutations(range(mc.d))]
    lookup = {state: i for i, state in enumerate(states)}
    c = np.bincount(x, minlength=mc.d)
    # Reconstruct rank power laws, independent of production kernel arrays.
    q = np.arange(1, mc.d+1, dtype=float)[None, :] ** -mc.kernel.alpha_nodes[:, None]
    q /= q.sum(axis=1, keepdims=True)
    logaux = np.array([np.log(mc.alpha_weights[a]) + c @ np.log(q[a, list(r)])
                       for a, r in states])
    return states, lookup, c, q, logaux


class AuxiliaryRNG:
    def __init__(self, *, destination=None, probabilities=None, permutation=None,
                 uniforms=None):
        self.destination, self.probabilities = destination, probabilities
        self.labels, self.uniforms = permutation, uniforms
        self.calls = 0

    def random(self, size=None):
        self.calls += 1
        if size is None:
            assert self.calls == 1
            return 0.0 if self.destination is not None else 0.9
        assert self.calls == 2 and len(self.uniforms) == size
        return np.asarray(self.uniforms)

    def choice(self, count, p):
        np.testing.assert_allclose(p, self.probabilities, atol=2e-15)
        assert 0 <= self.destination < count
        return self.destination

    def permutation(self, d):
        assert len(self.labels) == d
        return np.asarray(self.labels)


def actual_auxiliary_transition(mc, x):
    states, lookup, counts, q, logaux = enumeration(mc, x)
    transition = np.zeros((len(states), len(states)))
    for s, (a, ranks) in enumerate(states):
        # Exponent Gibbs conditional reconstructed directly for fixed ranks.
        terms = np.array([logaux[lookup[b, ranks]] for b in range(len(mc.alpha_weights))])
        probabilities = np.exp(terms-logsumexp(terms))
        for b, p in enumerate(probabilities):
            r = np.array(ranks)
            observed = mc._surrogate_step(counts, a, r,
                AuxiliaryRNG(destination=b, probabilities=probabilities))
            assert observed == b and tuple(r) == ranks
            transition[s, lookup[b, ranks]] += ALPHA_GIBBS_PROBABILITY*p
        for labels in permutations(range(mc.d)):
            pairs = list(zip(labels[::2], labels[1::2]))
            probabilities = []
            for i, j in pairs:
                swapped = list(ranks)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                delta = logaux[lookup[a, tuple(swapped)]]-logaux[s]
                probabilities.append(np.exp(delta-logsumexp([0., delta])))
            for bits in product((False, True), repeat=len(pairs)):
                r = np.array(ranks)
                expected = np.array(ranks)
                event_probability = 1.0
                uniforms = []
                for (i, j), p, swapped in zip(pairs, probabilities, bits):
                    event_probability *= p if swapped else 1-p
                    uniforms.append(p/2 if swapped else (1+p)/2)
                    if swapped:
                        expected[i], expected[j] = expected[j], expected[i]
                observed = mc._surrogate_step(counts, a, r,
                    AuxiliaryRNG(permutation=labels, uniforms=uniforms))
                assert observed == a
                np.testing.assert_array_equal(r, expected)
                transition[s, lookup[a, tuple(r)]] += (
                    (1-ALPHA_GIBBS_PROBABILITY)*event_probability/factorial(mc.d))
    return states, lookup, transition, logaux


@pytest.mark.parametrize("d,x", [(2, [0, 0, 1]), (3, [0, 1, 0, 0]),
                                  (4, [3, 0, 1, 0, 0])])
def test_actual_auxiliary_matching_and_alpha_steps_are_reversible(d, x):
    mc = sampler(d)
    _, _, transition, logaux = actual_auxiliary_transition(mc, x)
    pi = np.exp(logaux-logsumexp(logaux))
    for power in (1, mc.surrogate_steps):
        block = np.linalg.matrix_power(transition, power)
        np.testing.assert_allclose(block.sum(axis=1), 1, atol=3e-15)
        np.testing.assert_allclose(pi[:, None]*block, (pi[:, None]*block).T, atol=2e-16)
        np.testing.assert_allclose(pi @ block, pi, atol=2e-16)


class OuterRNG:
    def __init__(self, move, choices, integers, uniform):
        self.move, self.choices, self.integers_values, self.uniform = (
            move, list(choices), list(integers), uniform)
        self.first = True

    def choice(self, values, p=None):
        if self.first:
            self.first = False
            assert values == 4
            np.testing.assert_array_equal(p, [.5, .4, .05, .05])
            return self.move
        return self.choices.pop(0)

    def integers(self, high):
        value = self.integers_values.pop(0)
        assert 0 <= value < high
        return value

    def random(self):
        assert not self.choices and not self.integers_values
        return self.uniform


def test_full_mixture_actual_mh_correction_preserves_nonzero_depth_target():
    mc = sampler(2, (0, 1), steps=3)
    x = [0, 0, 1, 1, 1, 0]
    target = PrefixPosterior(mc, x)
    states, lookup, auxiliary, logaux = actual_auxiliary_transition(mc, x)
    _, _, _, q, _ = enumeration(mc, x)
    exact = [SharedBaseLSA(2, ElementaryRows(), known_q=q[a, list(r)])
             for a, r in states]
    logpi = np.array([model.log_joint(x)+np.log(mc.alpha_weights[a])
                      for model, (a, _) in zip(exact, states)])
    pi = np.exp(logpi-logsumexp(logpi))
    assert np.max(abs(pi-np.exp(logaux-logsumexp(logaux)))) > 1e-3
    auxiliary = np.linalg.matrix_power(auxiliary, mc.surrogate_steps)
    transition = np.zeros_like(auxiliary)
    for s, (a, ranks) in enumerate(states):
        current = target.evaluate(a, ranks)
        assert current.log_target == pytest.approx(logpi[s], abs=3e-12)
        reference = exact[s].run(x+[0], full_probs=True, positions=[len(x)], posteriors=True)
        probs, depth, _ = target.predict(current)
        np.testing.assert_allclose(probs, reference['probs'][0], atol=3e-12)
        np.testing.assert_allclose(depth, reference['depth_posterior'][-1], atol=3e-12)
        assert np.all(depth > 0)
        for t, (b, destination) in enumerate(states):
            correction = logaux[s]-logaux[t]
            # Isolate the actual outer correction from the separately
            # enumerated, production-checked endpoint proposal kernel.
            mc.surrogate_proposal = lambda target, state, rng, b=b, r=destination, c=correction: (
                b, np.array(r), c)
            acceptance = min(1., np.exp(logpi[t]-logpi[s]+correction))
            for u in (acceptance/2, (1+acceptance)/2):
                attempts, accepts = np.zeros(4, int), np.zeros(4, int)
                actual = mc._step(target, current, OuterRNG(0, [], [], u), attempts, accepts)
                wanted = (b, destination) if u < acceptance else (a, ranks)
                assert (actual.alpha_index, tuple(actual.ranks)) == wanted
                assert attempts[0] == 1 and accepts[0] == int(u < acceptance)
            transition[s, t] += .5*auxiliary[s, t]*acceptance
            transition[s, s] += .5*auxiliary[s, t]*(1-acceptance)
        proposals = []
        for i in target.seen:
            for j in range(mc.d):
                r = list(ranks)
                r[i], r[j] = r[j], r[i]
                proposals.append((1, [int(i)], [0, j], (a, tuple(r)),
                                  .2/len(target.seen)/mc.d, 0.))
        for r in range(mc.d):
            for v in range(mc.d):
                i, j = ranks.index(r), ranks.index(v)
                changed = list(ranks)
                changed[i], changed[j] = changed[j], changed[i]
                proposals.append((1, [r], [1, v], (a, tuple(changed)),
                                  .2*mc.rank_proposal_weights[r]/mc.d, 0.))
        for b in range(len(mc.alpha_weights)):
            proposals.append((2, [b], [], (b, ranks), .05*mc.alpha_weights[b],
                              np.log(mc.alpha_weights[a]/mc.alpha_weights[b])))
        for sign in (-1, 1):
            b = a+sign
            b = b if 0 <= b < len(mc.alpha_weights) else a
            proposals.append((3, [sign], [], (b, ranks), .025, 0.))
        for move, choices, integers, destination, weight, correction in proposals:
            t = lookup[destination]
            acceptance = min(1., np.exp(logpi[t]-logpi[s]+correction))
            for u in (acceptance/2, (1+acceptance)/2):
                attempts, accepts = np.zeros(4, int), np.zeros(4, int)
                actual = mc._step(target, current,
                    OuterRNG(move, choices, integers, u), attempts, accepts)
                wanted = destination if u < acceptance else (a, ranks)
                assert (actual.alpha_index, tuple(actual.ranks)) == wanted
                assert attempts[move] == 1 and accepts[move] == int(u < acceptance)
            transition[s, t] += weight*acceptance
            transition[s, s] += weight*(1-acceptance)
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=3e-15)
    flow = pi[:, None]*transition
    np.testing.assert_allclose(flow, flow.T, atol=3e-16)
    np.testing.assert_allclose(pi @ transition, pi, atol=3e-16)


def test_proposal_counts_first_token_preserves_state_and_repeats_fixed_steps():
    mc = sampler()
    target = PrefixPosterior(mc, [2, 0, 1, 0])
    state = target.evaluate(1, [1, 2, 0])
    saved = state.ranks.copy()
    calls = []
    def step(counts, alpha_index, ranks, rng):
        calls.append(counts.copy())
        ranks[0], ranks[1] = ranks[1], ranks[0]
        return 2
    mc._surrogate_step = step
    a, ranks, correction = mc.surrogate_proposal(target, state, None)
    assert len(calls) == mc.surrogate_steps
    np.testing.assert_array_equal(calls, [[2, 1, 1]]*mc.surrogate_steps)
    np.testing.assert_array_equal(state.ranks, saved)
    assert a == 2
    expected = (np.log(mc.alpha_weights[1])+np.log(mc.kernel.base_laws[1, saved]) @ calls[0]
                -np.log(mc.alpha_weights[a])-np.log(mc.kernel.base_laws[a, ranks]) @ calls[0])
    assert correction == pytest.approx(expected, abs=2e-15)


def test_sampler_reproducibility_metadata_and_descriptor():
    mc = sampler()
    kwargs = dict(chains=2, burn=16, draws=32, thin=2, seed=15, retain_samples=True)
    a = mc.sample([0, 1, 0, 2], **kwargs)
    b = mc.sample([0, 1, 0, 2], **kwargs)
    np.testing.assert_array_equal(a['samples']['predictive_probs'], b['samples']['predictive_probs'])
    assert a['diagnostics']['move_names'] == MOVE_NAMES
    assert a['diagnostics']['proposal_spec'] == mc.proposal_spec
    np.testing.assert_array_equal(np.sum(a['diagnostics']['move_attempts'], axis=1), [64, 64])
    descriptor = proposal_descriptor(mc)
    assert descriptor['proposal_spec']['surrogate_steps'] == 3
    assert len(descriptor['code_sha256']['shared_base_surrogate.py']) == 64
    assert descriptor != proposal_descriptor(sampler(steps=4))


@pytest.mark.parametrize('steps', [True, 0, -1, 1.5])
def test_invalid_step_count(steps):
    with pytest.raises(ValueError):
        sampler(steps=steps)

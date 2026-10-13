"""Independent finite-state verification for the proposed shorter blocks."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
from shared_base import SharedBaseLSA
from shared_base_mcmc import PrefixPosterior
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from test_shared_base import ElementaryRows
from test_shared_base_surrogate import (sampler, actual_auxiliary_transition,
                                       enumeration, AuxiliaryRNG, OuterRNG)


@pytest.mark.parametrize('steps', [1, 4, 16, 64])
def test_shorter_blocks_have_identical_nonzero_depth_posterior(steps):
    mc = sampler(2, (0, 1), steps=steps)
    x = [0, 0, 1, 1, 1, 0]
    states, _, auxiliary, logaux = actual_auxiliary_transition(mc, x)
    _, _, _, q, _ = enumeration(mc, x)
    target = PrefixPosterior(mc, x)
    exact = [SharedBaseLSA(2, ElementaryRows(), known_q=q[a, list(r)])
             for a, r in states]
    independent_logpi = np.array([model.log_joint(x)+np.log(mc.alpha_weights[a])
                                 for model, (a, _) in zip(exact, states)])
    target_logpi = np.array([target.evaluate(a, r).log_target for a, r in states])
    np.testing.assert_allclose(target_logpi, independent_logpi, atol=3e-12)
    pi = np.exp(independent_logpi-logsumexp(independent_logpi))
    proposal = np.linalg.matrix_power(auxiliary, steps)
    log_ratio = (target_logpi[None, :]-target_logpi[:, None]
                 + logaux[:, None]-logaux[None, :])
    accepted = proposal*np.exp(np.minimum(log_ratio, 0.0))
    transition = accepted + np.diag(1-accepted.sum(axis=1))
    np.testing.assert_allclose(transition.sum(axis=1), 1, atol=3e-15)
    np.testing.assert_allclose(pi[:, None]*transition, (pi[:, None]*transition).T, atol=3e-16)
    np.testing.assert_allclose(pi @ transition, pi, atol=3e-16)
    # Predictions still marginalize the full depth prior and rank posterior.
    production_prediction = sum(weight*target.predict(target.evaluate(a, r))[0]
                                for weight, (a, r) in zip(pi, states))
    reference_prediction = sum(weight*model.run(x+[0], full_probs=True,
                                               positions=[len(x)])['probs'][0]
                               for weight, model in zip(pi, exact))
    np.testing.assert_allclose(production_prediction, reference_prediction, atol=3e-12)


@pytest.mark.parametrize('power', [.25, .5, 1.0, 'row'])
def test_actual_auxiliary_steps_and_full_target_correction(power):
    base = sampler(2, (0, 1), steps=4)
    mc = (SharedBaseEfficiencyRowMCMC(2, base.kernel, base.alpha_weights,
                                     surrogate_steps=4, concentration=1.0)
          if power == 'row' else
          SharedBaseEfficiencyMCMC(2, base.kernel, base.alpha_weights,
                                   surrogate_steps=4, auxiliary_power=power))
    x = [0, 0, 1, 1, 1, 0]
    target = PrefixPosterior(mc, x)
    states, lookup, counts, q, _ = enumeration(mc, x)
    if power == 'row':
        # Construct each rising factorial directly, independently of the
        # proposal's gamma-function and count-profile implementation.
        logaux = np.array([
            np.log(mc.alpha_weights[a])+np.log(q[a, ranks[x[0]]])
            +sum(np.log(mc.concentration*q[a, ranks[b]]+k)
                 for row in target.counts for b in range(mc.d)
                 for k in range(int(row[b])))
            for a, ranks in states])
        counts = mc.auxiliary_table(target)
    else:
        logaux = np.array([np.log(mc.alpha_weights[a])+power*counts@np.log(q[a, list(r)])
                           for a, r in states])
    auxiliary = np.zeros((len(states), len(states)))
    for s, (a, ranks) in enumerate(states):
        assert mc.surrogate_log_target(counts, a, np.array(ranks)) == pytest.approx(logaux[s], abs=3e-15)
        terms = np.array([logaux[lookup[b, ranks]] for b in range(len(mc.alpha_weights))])
        probs = np.exp(terms-logsumexp(terms))
        for b, p in enumerate(probs):
            r = np.array(ranks)
            actual = mc._surrogate_step(counts, a, r,
                AuxiliaryRNG(destination=b, probabilities=probs))
            assert actual == b and tuple(r) == ranks
            auxiliary[s, lookup[b, ranks]] += .2*p
        destination = tuple(reversed(ranks))
        delta = logaux[lookup[a, destination]]-logaux[s]
        p = np.exp(delta-logsumexp([0., delta]))
        for swapped, probability, u in [(True, p, p/2), (False, 1-p, (1+p)/2)]:
            r = np.array(ranks)
            actual = mc._surrogate_step(counts, a, r,
                AuxiliaryRNG(permutation=[0, 1], uniforms=[u]))
            expected = destination if swapped else ranks
            assert actual == a and tuple(r) == expected
            auxiliary[s, lookup[a, expected]] += .8*probability
    pi_aux = np.exp(logaux-logsumexp(logaux))
    np.testing.assert_allclose(pi_aux[:, None]*auxiliary,
                               (pi_aux[:, None]*auxiliary).T, atol=3e-16)
    proposal = np.linalg.matrix_power(auxiliary, mc.surrogate_steps)
    exact = [SharedBaseLSA(2, ElementaryRows(), known_q=q[a, list(r)])
             for a, r in states]
    logpi = np.array([model.log_joint(x)+np.log(mc.alpha_weights[a])
                      for model, (a, _) in zip(exact, states)])
    pi = np.exp(logpi-logsumexp(logpi))
    transition = np.zeros_like(proposal)
    for s, (a, ranks) in enumerate(states):
        current = target.evaluate(a, ranks)
        for t, (b, destination) in enumerate(states):
            correction = logaux[s]-logaux[t]
            mc.surrogate_proposal = lambda target, state, rng, b=b, r=destination, c=correction: (
                b, np.array(r), c)
            acceptance = min(1., np.exp(logpi[t]-logpi[s]+correction))
            for u in (acceptance/2, (1+acceptance)/2):
                actual = mc._step(target, current, OuterRNG(0, [], [], u),
                                  np.zeros(4, int), np.zeros(4, int))
                expected = (b, destination) if u < acceptance else (a, ranks)
                assert (actual.alpha_index, tuple(actual.ranks)) == expected
            transition[s, t] += proposal[s, t]*acceptance
            transition[s, s] += proposal[s, t]*(1-acceptance)
    np.testing.assert_allclose(pi @ transition, pi, atol=3e-16)
    np.testing.assert_allclose(pi[:, None]*transition, (pi[:, None]*transition).T, atol=3e-16)

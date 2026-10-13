"""Independent finite-prior checks of the family-mixture Bayes interface.

These enumerate ordered binary sequences and integrate finite Bernoulli or
Markov priors with exact fractions. No production source or integrator is
used, and no shared-base evidence approximation is tested or implied.
"""
from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from mixture_bayes import (EVIDENCE_KIND, SequenceEvidence,
                          cumulative_mixture_losses, mixture_prediction,
                          posterior_family_weights)


def evidence(values, n):
    logs = [math.log(float(value)) if value else -np.inf for value in values]
    return SequenceEvidence(prefix_length=n, log_marginals=logs,
                            evidence_kind=EVIDENCE_KIND)


def bernoulli_marginals(xs):
    priors = ((F(1, 4), F(3, 4)), (F(1, 10), F(9, 10)), (F(1, 2),))
    return [sum(math.prod(p if x else 1-p for x in xs) for p in ps) / len(ps)
            for ps in priors]


def markov_marginals(xs):
    # Each pair contains P(next=1 | previous=0), P(next=1 | previous=1).
    # Every chain has the same uniform initial law, and each family has a
    # genuinely different finite prior over transition matrices.
    priors = (((F(1, 4), F(3, 4)), (F(3, 4), F(1, 4))),
              ((F(1, 8), F(1, 8)), (F(7, 8), F(7, 8))),
              ((F(1, 3), F(2, 3)),))
    if not xs:
        return [F(1)] * len(priors)
    return [sum(F(1, 2) * math.prod(row[a] if b else 1-row[a]
                    for a, b in zip(xs[:-1], xs[1:])) for row in rows) / len(rows)
            for rows in priors]


@pytest.mark.parametrize("marginal", [bernoulli_marginals, markov_marginals])
def test_prediction_matches_independently_enumerated_joint_ratios(marginal):
    prior = [F(1, 5), F(3, 10), F(1, 2)]
    for n in range(6):
        for xs in product((0, 1), repeat=n):
            parents = marginal(xs)
            children = [marginal(xs + (a,)) for a in (0, 1)]
            total = sum(w*p for w, p in zip(prior, parents))
            expected_weights = [float(w*p / total) for w, p in zip(prior, parents)]
            expected_prediction = [float(sum(w*p for w, p in zip(prior, child)) / total)
                                   for child in children]
            component_predictions = np.array([[float(children[a][f] / parents[f])
                                               for a in (0, 1)] for f in range(3)])
            child_logs = np.log(np.array(children, dtype=float).T)
            current = evidence(parents, n)
            np.testing.assert_allclose(posterior_family_weights(current, prior),
                                       expected_weights, rtol=0, atol=1e-14)
            actual = mixture_prediction(current, prior, component_predictions,
                                        successor_log_marginals=child_logs)
            np.testing.assert_allclose(actual, expected_prediction, rtol=0, atol=1e-14)
        # The independently constructed priors really are normalized over
        # ordered sequences, not over family labels or sufficient statistics.
        totals = [sum(marginal(xs)[f] for xs in product((0, 1), repeat=n))
                  for f in range(3)]
        assert totals == [F(1)] * 3


@pytest.mark.parametrize("marginal", [bernoulli_marginals, markov_marginals])
def test_cumulative_loss_matches_sequential_scoring_and_family_bound(marginal):
    prior = [F(1, 3)] * 3
    for xs in product((0, 1), repeat=5):
        full = cumulative_mixture_losses(evidence(marginal(xs), len(xs)), prior)
        sequential = 0.0
        for t, token in enumerate(xs):
            parents = marginal(xs[:t])
            children = [marginal(xs[:t] + (a,)) for a in (0, 1)]
            distributions = [[float(children[a][f] / parents[f]) for a in (0, 1)]
                             for f in range(3)]
            prediction = mixture_prediction(evidence(parents, t), prior, distributions)
            sequential -= math.log2(prediction[token])
        assert full["mixture_loss_bits"] == pytest.approx(sequential, abs=1e-13)
        np.testing.assert_allclose(full["upper_bound_bits"], math.log2(3), atol=1e-14)
        assert np.all(full["mixture_minus_component_bits"] <= math.log2(3) + 1e-13)
        omitted = cumulative_mixture_losses(evidence(marginal(xs), len(xs)), prior,
                           first_token_evidence=evidence(marginal(xs[:1]), 1))
        assert omitted["scored_tokens"] == 4
        assert omitted["mixture_loss_bits"] == pytest.approx(sequential-1, abs=1e-13)
        np.testing.assert_allclose(omitted["component_loss_bits"],
                                   full["component_loss_bits"]-1, atol=1e-14)
        np.testing.assert_array_equal(omitted["mixture_minus_component_bits"],
                                      full["mixture_minus_component_bits"])


def test_prior_average_cost_equals_reduction_in_family_entropy():
    prior = np.array([.2, .3, .5])
    n = 6
    mean_cost = 0.0
    mean_posterior_entropy = 0.0
    for xs in product((0, 1), repeat=n):
        marginals = np.array(markov_marginals(xs), dtype=float)
        current = evidence(marginals, n)
        result = cumulative_mixture_losses(current, prior)
        mean_cost += np.sum(prior * marginals * result["mixture_minus_component_bits"])
        posterior = posterior_family_weights(current, prior)
        mean_posterior_entropy += np.sum(prior * marginals) * -np.sum(
            posterior * np.log2(posterior))
    initial_entropy = -np.sum(prior * np.log2(prior))
    assert mean_cost == pytest.approx(initial_entropy - mean_posterior_entropy, abs=1e-13)
    assert 0 < mean_cost < initial_entropy


def test_zero_prior_and_impossible_families_have_no_posterior_weight():
    current = evidence([F(1, 2), F(0), F(1)], 2)
    prior = [.25, .75, 0]
    np.testing.assert_array_equal(posterior_family_weights(current, prior), [1, 0, 0])
    prediction = mixture_prediction(current, prior, [[.1, .9], [np.nan, np.nan],
                                                      [np.nan, np.nan]])
    np.testing.assert_allclose(prediction, [.1, .9], atol=1e-15)
    losses = cumulative_mixture_losses(current, prior)
    assert losses["mixture_loss_bits"] == pytest.approx(3)
    assert losses["mixture_minus_component_bits"][0] == pytest.approx(2)
    assert losses["component_loss_bits"][1] == np.inf
    assert losses["mixture_minus_component_bits"][1] == -np.inf
    assert losses["upper_bound_bits"][2] == np.inf


def test_a_sequence_can_favor_the_mixture_over_its_generating_family():
    result = cumulative_mixture_losses(evidence([F(1, 100), F(1, 2)], 3), [.5, .5])
    assert result["mixture_minus_component_bits"][0] < 0


def test_impossible_mixture_raises_instead_of_inventing_a_posterior():
    current = evidence([0, F(1, 2)], 1)
    for function in (posterior_family_weights, cumulative_mixture_losses):
        with pytest.raises(ValueError, match="impossible under every"):
            function(current, [1, 0])
    with pytest.raises(ValueError, match="impossible under every"):
        mixture_prediction(current, [1, 0], [[np.nan, np.nan], [.5, .5]])


def test_extremely_small_evidence_remains_stable():
    current = SequenceEvidence(prefix_length=100000, log_marginals=[-100000., -100001.],
                               evidence_kind=EVIDENCE_KIND)
    expected = np.array([1, math.exp(-1)]) / (1 + math.exp(-1))
    np.testing.assert_allclose(posterior_family_weights(current, [.5, .5]), expected,
                               rtol=0, atol=1e-14)
    assert np.isfinite(cumulative_mixture_losses(current, [.5, .5])["mixture_loss_bits"])


@pytest.mark.parametrize("kind", ["posterior", "unnormalized_likelihood", "log_density", None])
def test_posterior_or_unnormalized_evidence_declarations_are_rejected(kind):
    with pytest.raises(ValueError, match="absolute normalized sequence"):
        SequenceEvidence(prefix_length=3, log_marginals=[-.1, -2.], evidence_kind=kind)


def test_empty_prefix_anchor_rejects_prior_or_posterior_weights_as_evidence():
    with pytest.raises(ValueError, match="empty-prefix"):
        evidence([F(1, 2), F(1, 2)], 0)
    current = evidence([1, 1], 0)
    np.testing.assert_allclose(posterior_family_weights(current, [.2, .8]), [.2, .8])
    assert cumulative_mixture_losses(current, [.2, .8])["mixture_loss_bits"] == pytest.approx(0)
    with pytest.raises(TypeError, match="explicit SequenceEvidence"):
        posterior_family_weights(np.array([.2, .8]), [.2, .8])


@pytest.mark.parametrize("logs", [[.01, -1], [np.nan, -1], [np.inf, -1], []])
def test_invalid_log_marginals_are_rejected(logs):
    with pytest.raises(ValueError):
        SequenceEvidence(prefix_length=2, log_marginals=logs, evidence_kind=EVIDENCE_KIND)


@pytest.mark.parametrize("prior", [[2, 1], [.1, .2], [-.1, 1.1], [.5], [np.nan, .5]])
def test_invalid_priors_are_not_renormalized(prior):
    with pytest.raises(ValueError, match="prior weights"):
        posterior_family_weights(evidence([F(1, 2), F(1, 4)], 2), prior)


def test_invalid_component_predictions_are_not_renormalized():
    current = evidence([F(1, 2), F(1, 4)], 2)
    for probabilities in ([[.2, .2], [.5, .5]], [[-.1, 1.1], [.5, .5]],
                           [[np.nan, np.nan], [.5, .5]], [[.5, .5], [np.nan, .5]]):
        with pytest.raises(ValueError):
            mixture_prediction(current, [.5, .5], probabilities)


def test_children_reject_inconsistent_normalization_or_prediction():
    current = evidence([F(1, 2), F(1, 4)], 2)
    probabilities = [[.2, .8], [.5, .5]]
    children = np.log([[.1, .4], [.125, .125]])
    mixture_prediction(current, [.5, .5], probabilities, successor_log_marginals=children)
    with pytest.raises(ValueError, match="not normalized to its parent"):
        mixture_prediction(current, [.5, .5], probabilities,
                           successor_log_marginals=children + math.log(.5))
    # Normalizing evidence over families erases the absolute sequence scale.
    posterior_children = np.log([[.1/.225, .4/.525], [.125/.225, .125/.525]])
    with pytest.raises(ValueError, match="not normalized to its parent"):
        mixture_prediction(current, [.5, .5], probabilities,
                           successor_log_marginals=posterior_children)
    with pytest.raises(ValueError, match="disagree with evidence ratios"):
        mixture_prediction(current, [.5, .5], [[.3, .7], [.5, .5]],
                           successor_log_marginals=children)


def test_impossible_parent_requires_impossible_children():
    current = evidence([F(1, 2), 0], 2)
    probabilities = [[.5, .5], [np.nan, np.nan]]
    valid = [[math.log(.25), math.log(.25)], [-np.inf, -np.inf]]
    np.testing.assert_allclose(mixture_prediction(current, [.5, .5], probabilities,
                               successor_log_marginals=valid), [.5, .5])
    with pytest.raises(ValueError, match="impossible prefix"):
        mixture_prediction(current, [.5, .5], probabilities,
                           successor_log_marginals=np.log([[.25, .25], [.1, .1]]))


def test_first_token_omission_requires_common_matching_marginals():
    current = evidence([F(1, 8), F(1, 16)], 3)
    with pytest.raises(ValueError, match="common finite marginal"):
        cumulative_mixture_losses(current, [.5, .5],
                                   first_token_evidence=evidence([F(1, 2), F(1, 3)], 1))
    with pytest.raises(ValueError, match="exceed its first-token"):
        cumulative_mixture_losses(current, [.5, .5],
                                   first_token_evidence=evidence([F(1, 32)]*2, 1))
    first = evidence([F(1, 2)]*2, 1)
    zero = cumulative_mixture_losses(first, [.5, .5], first_token_evidence=first)
    assert zero["scored_tokens"] == 0
    assert zero["mixture_loss_bits"] == pytest.approx(0)
    np.testing.assert_allclose(zero["component_loss_bits"], 0, atol=1e-15)
    with pytest.raises(ValueError, match="must agree"):
        cumulative_mixture_losses(evidence([F(1, 4)]*2, 1), [.5, .5],
                                   first_token_evidence=first)

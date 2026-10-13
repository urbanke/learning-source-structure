"""Independent counting and probability identities for natural copying curves."""
from collections import Counter
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from copier_time_metrics import CONDITIONS, natural_metrics


def constant_predictions(n, probabilities):
    return {name: np.tile(probabilities, (n, 1)) for name in CONDITIONS}


def test_all_binary_histories_use_only_completed_transitions():
    for path in product(range(2), repeat=6):
        tokens = np.asarray(path)
        metrics, checks = natural_metrics(tokens, constant_predictions(5, [.5, .5]), np.full((2, 2), .5))
        for t in range(1, len(tokens)):
            previous_edges = Counter(zip(tokens[:t-1], tokens[1:t]))
            expected = [previous_edges[tokens[t-1], b] for b in range(2)]
            np.testing.assert_array_equal(metrics["row_counts"][t-1], expected)
            np.testing.assert_array_equal(metrics["successor_mask"][t-1], np.asarray(expected) > 0)
            assert metrics["row_totals"][t-1] == sum(expected)
        assert checks["finite_metrics"]
        for name in CONDITIONS:
            np.testing.assert_array_equal(metrics["excess_" + name], 0)
            np.testing.assert_array_equal(metrics["oracle_regret_" + name], 0)
        for key in ("copier_causal_mass", "control_causal_mass", "benefit_copier_bits",
                    "benefit_control_bits", "benefit_bayes_bits", "intact_bayes_kl_bits"):
            np.testing.assert_array_equal(metrics[key], 0)


def test_unobserved_target_cannot_change_support_or_oracle_metrics():
    tokens = np.array([0, 1, 0, 0, 1, 0])
    other = tokens.copy()
    other[-1] = 1
    transition = np.array([[.8, .2], [.3, .7]])
    predictions = constant_predictions(5, [.6, .4])
    first, _ = natural_metrics(tokens, predictions, transition)
    second, _ = natural_metrics(other, predictions, transition)
    for key in first:
        if "observed_loss" not in key:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        else:
            np.testing.assert_array_equal(first[key][:-1], second[key][:-1])
            assert first[key][-1] != second[key][-1] or key == "observed_loss_uniform"


def test_analytic_oracle_benefits_kl_directions_and_cumulative_sum():
    tokens = np.array([0, 1, 0, 0, 1])
    source = np.array([[.75, .25], [.25, .75]])
    actual = source[tokens[:-1]]
    predictions = {
        "intact": actual.copy(),
        "copier_deleted": np.full((4, 2), .5),
        "control_deleted": np.tile([.6, .4], (4, 1)),
        "bayes": np.tile([.8, .2], (4, 1)),
    }
    out, _ = natural_metrics(tokens, predictions, source)
    entropy = -.75*np.log2(.75) - .25*np.log2(.25)
    np.testing.assert_allclose(out["source_entropy_bits"], entropy, atol=1e-15)
    np.testing.assert_allclose(out["oracle_regret_intact"], 0, atol=1e-15)
    np.testing.assert_allclose(out["benefit_copier_bits"], 1-entropy, atol=1e-15)
    np.testing.assert_allclose(out["cumulative_benefit_copier_bits"], np.arange(1, 5)*(1-entropy), atol=1e-15)
    for t, p in enumerate(actual):
        for name, qrows in predictions.items():
            q = qrows[t]
            direct_kl = sum(float(p[b])*np.log2(p[b]/q[b]) for b in range(2))
            assert out["oracle_regret_" + name][t] == pytest.approx(direct_kl, abs=2e-15)
        bayes_ce = -sum(float(p[b])*np.log2(predictions["bayes"][t, b]) for b in range(2))
        assert out["benefit_bayes_bits"][t] == pytest.approx(1-bayes_ce, abs=2e-15)
        assert out["intact_bayes_kl_bits"][t] == pytest.approx(bayes_ce-entropy, abs=2e-15)
    forward = .6*np.log2(.6/.5) + .4*np.log2(.4/.5)
    reverse = .5*np.log2(.5/.6) + .5*np.log2(.5/.4)
    np.testing.assert_allclose(out["control_deleted_to_uniform_kl_bits"], forward, atol=1e-15)
    np.testing.assert_allclose(out["uniform_to_control_deleted_kl_bits"], reverse, atol=1e-15)
    # At the final current token (0), both outcomes have previously followed it.
    assert out["support_sizes"][-1] == 2
    np.testing.assert_allclose(out["copier_causal_mass"][-1], 0, atol=1e-15)


def test_raw_bayes_error_is_retained_while_tighter_checks_reject_it_elsewhere():
    tokens = np.array([0, 1, 0, 0, 1])
    source = np.full((2, 2), .5)
    predictions = constant_predictions(4, [.5, .5])
    predictions["bayes"][3, 0] += 5e-8
    original = {key: value.copy() for key, value in predictions.items()}
    out, checks = natural_metrics(tokens, predictions, source)
    assert out["mass_bayes"][3] == pytest.approx(1+5e-8, abs=1e-15)
    assert out["excess_bayes"][3] == pytest.approx(5e-8, abs=1e-15)
    assert checks["normalization_error"]["bayes"] == pytest.approx(5e-8, abs=1e-15)
    for key in predictions:
        np.testing.assert_array_equal(predictions[key], original[key])
    for name in ("intact", "copier_deleted", "control_deleted"):
        corrupt = {key: value.copy() for key, value in original.items()}
        corrupt[name][0, 0] += 5e-8
        with pytest.raises(ValueError, match="normalization"):
            natural_metrics(tokens, corrupt, source)
    predictions["bayes"][3, 0] += 2e-7
    with pytest.raises(ValueError, match="normalization"):
        natural_metrics(tokens, predictions, source)
    corrupt_source = source.copy()
    corrupt_source[0, 0] += 5e-8
    with pytest.raises(ValueError, match="normalization"):
        natural_metrics(tokens, original, corrupt_source)


def test_zero_source_entries_have_exact_entropy_without_a_probability_floor():
    tokens = np.array([0, 1, 0, 1])
    source = np.array([[0., 1.], [1., 0.]])
    out, _ = natural_metrics(tokens, constant_predictions(3, [.75, .25]), source)
    np.testing.assert_array_equal(out["source_entropy_bits"], 0)
    np.testing.assert_array_equal(out["source_observed_loss_bits"], 0)
    np.testing.assert_allclose(out["oracle_ce_intact"], [2, -np.log2(.75), 2])
    with pytest.raises(ValueError, match="zero source"):
        natural_metrics(np.array([0, 0, 1, 0]), constant_predictions(3, [.75, .25]), source)


def test_invalid_probability_entries_and_fractional_tokens_are_not_repaired():
    tokens = np.array([0, 1, 0])
    source = np.full((2, 2), .5)
    for bad in ([-1e-16, 1.0], [np.nan, .5], [0., 1.]):
        predictions = constant_predictions(2, [.5, .5])
        predictions["bayes"][0] = bad
        with pytest.raises(ValueError):
            natural_metrics(tokens, predictions, source)
    with pytest.raises(ValueError, match="integer"):
        natural_metrics(np.array([0., 1.5, 0.]), constant_predictions(2, [.5, .5]), source)

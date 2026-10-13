"""Independent checks of functional uncertainty and paired source accounting."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shared_base_full_context_summary import (LN2, NUMERICAL_LIMIT_BITS, RUNS,
    aggregate_sources, comparison_metrics, covariance_of_mean, metric_refinement, probability_refinement, stable_seed)


def test_frozen_source_seed_protocol_known_values():
    assert [stable_seed("shared-base-full-context-v1", 20260922, index) for index in range(4)] == [
        1527047719, 1527047028, 1527046337, 1527045646]


def test_covariances_match_independent_scalar_batch_and_chain_calculations():
    rng = np.random.default_rng(20260924)
    samples = rng.dirichlet([2, 4, 3], size=(4, 40))
    result = covariance_of_mean(samples, batch_length=5)
    gradient = np.array([1.0, -2.0, 0.4])
    scalar = samples @ gradient
    batches = scalar.reshape(4, 8, 5).mean(axis=2).ravel()
    expected_batch = batches.var(ddof=1) / len(batches)
    expected_chain = scalar.mean(axis=1).var(ddof=1) / 4
    assert gradient @ result["covariance_batch"] @ gradient == pytest.approx(expected_batch, rel=1e-13)
    assert gradient @ result["covariance_between"] @ gradient == pytest.approx(expected_chain, rel=1e-13)
    np.testing.assert_allclose(result["covariance_batch"].sum(axis=1), 0, atol=1e-17)
    np.testing.assert_allclose(result["mean"], samples.mean(axis=(0, 1)), atol=1e-15)


def test_metric_of_mean_is_not_average_state_kl():
    samples = np.tile(np.array([[0.9, 0.1], [0.1, 0.9]]), (4, 8, 1))
    moments = covariance_of_mean(samples)
    metrics = comparison_metrics(moments["mean"], [0.5, 0.5], [0.5, 0.5],
                                 moments["covariance_batch"], moments["covariance_between"])
    assert metrics["kl_bits"] == pytest.approx(0, abs=1e-15)
    state_kl = np.mean(np.sum(samples * np.log2(samples / 0.5), axis=2))
    assert state_kl > 0.5


def test_centered_gradients_match_simplex_finite_differences():
    b, t, r = np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.1, 0.5]), np.array([0.1, 0.7, 0.2])
    covariance = np.zeros((3, 3))
    metrics = comparison_metrics(b, t, r, covariance, covariance)
    direction = np.array([0.2, -0.5, 0.3])
    step = 1e-6
    plus = comparison_metrics(b + step * direction, t, r, covariance, covariance)
    minus = comparison_metrics(b - step * direction, t, r, covariance, covariance)
    for key, gradient in (("kl_bits", "gradient_kl"), ("true_expected_loss_gap_bits", "gradient_gap")):
        derivative = (plus[key] - minus[key]) / (2 * step)
        assert derivative == pytest.approx(np.asarray(metrics[gradient]) @ direction, abs=1e-9)
    ce_t, ce_b = -r @ np.log2(t), -r @ np.log2(b)
    assert metrics["true_expected_loss_gap_bits"] == pytest.approx(ce_t - ce_b)


def test_functional_mcse_maximizes_projections_not_covariance_entries():
    c1 = 0.01 * np.array([[1, -1], [-1, 1]])
    c2 = c1 / 2
    metrics = comparison_metrics([0.6, 0.4], [0.3, 0.7], [0.5, 0.5], c1, c2)
    gradient = np.array(metrics["gradient_kl"])
    expected = np.sqrt(gradient @ c1 @ gradient)
    wrong = np.sqrt(gradient @ np.maximum(c1, c2) @ gradient)
    assert metrics["kl_uncertainty"]["mcse_bits"] == pytest.approx(expected)
    assert expected > wrong


def test_noise_scale_uses_conservative_diagonal_envelope():
    v1, v2 = np.array([1, -1, 0.0]), np.array([0, 1, -1.0])
    c1, c2 = 1e-5 * np.outer(v1, v1), 1e-5 * np.outer(v2, v2)
    b = np.ones(3) / 3
    metrics = comparison_metrics(b, b, b, c1, c2)
    expected = 0.5 / LN2 * np.sum(np.maximum(np.diag(c1), np.diag(c2)) / b)
    assert metrics["kl_second_order"]["noise_scale_bits"] == pytest.approx(expected)
    assert expected > max(metrics["kl_second_order"][name]["mean_scale_bits"] for name in ("batch", "between"))
    assert metrics["kl_uncertainty"]["mcse_bits"] < 1e-15
    assert metrics["gap_uncertainty"]["mcse_bits"] < 1e-15
    assert metrics["kl_second_order"]["noise_scale_bits"] > 0


def test_rare_label_gap_curvature_can_fail_when_first_order_cancels():
    # One poorly mixed chain assigns the rare label 8%; the other three
    # assign almost zero. The mean is 2%, with an empirically realizable
    # between-chain covariance rather than an arbitrary oversized matrix.
    samples = np.empty((4, 16, 2))
    samples[:, :, 1] = np.array([0.08, 1e-12, 1e-12, 1e-12])[:, None]
    samples[:, :, 0] = 1 - samples[:, :, 1]
    moments = covariance_of_mean(samples)
    b = moments["mean"]
    metrics = comparison_metrics(b, b, b, moments["covariance_batch"], moments["covariance_between"])
    assert metrics["gap_uncertainty"]["mcse_bits"] == pytest.approx(0, abs=1e-15)
    assert metrics["gap_second_order"]["noise_scale_bits"] > NUMERICAL_LIMIT_BITS
    assert metrics["max_relative_probability_mcse"] == pytest.approx(1)
    refinement = metric_refinement(metrics, metrics)
    assert refinement["checks"]["gap_numerical_precision"] is False
    assert not refinement["passed"]


def test_paired_aggregation_uses_four_sources_and_never_drops_unresolved():
    rows = []
    for index in range(4):
        for run in RUNS:
            value = index + (10 if run == RUNS[1] else 0)
            rows.append(dict(sequence_index=index, run=run, resolved=True,
                             refined=dict(total_variation=value / 100, kl_bits=float(value),
                                          true_expected_loss_gap_bits=-float(value))))
    result = aggregate_sources(rows)
    first = result["by_model"][RUNS[0]]["kl_bits"]
    assert first["mean"] == pytest.approx(1.5)
    assert first["source_se"] == pytest.approx(np.std([0, 1, 2, 3], ddof=1) / 2)
    paired = result["paired_seed1_minus_seed0"]["kl_bits"]
    assert paired["mean"] == 10 and paired["source_se"] == 0
    rows[2]["resolved"] = False
    withheld = aggregate_sources(rows)
    assert withheld["reported"] is False and withheld["unresolved_source_indices"] == [1]
    with pytest.raises(ValueError, match="all four"):
        aggregate_sources(rows[:-1])


def test_equal_scalar_metrics_do_not_hide_failed_vector_refinement():
    first, second = np.array([0.4, 0.3, 0.3]), np.array([0.3, 0.4, 0.3])
    uniform, covariance = np.ones(3) / 3, np.zeros((3, 3))
    coarse = comparison_metrics(first, uniform, uniform, covariance, covariance)
    refined = comparison_metrics(second, uniform, uniform, covariance, covariance)
    assert metric_refinement(coarse, refined)["passed"]
    vector = probability_refinement(first, second, np.zeros(3), np.zeros(3))
    assert not vector["passed"]
    assert not vector["checks"]["max_absolute_difference_below_0_01"]
    # Even a difference below the absolute cap must pass the MCSE check.
    small = probability_refinement([0.5, 0.5], [0.5003, 0.4997], [0, 0], [0, 0])
    assert small["checks"]["max_absolute_difference_below_0_01"]
    assert not small["passed"]

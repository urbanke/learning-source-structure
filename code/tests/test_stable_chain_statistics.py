"""Regression against the long-chain arithmetic failure and independent sums."""
from pathlib import Path
import math
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from full_context_statistics import predictive_moments
from shared_base_mcmc import chain_summary
from stable_chain_statistics import (
    SUMMARY_SPEC, stable_chain_summary, stable_predictive_moments,
)


def test_long_repeated_prediction_matches_fsum_and_both_report_representations():
    # Fixed predictions are common when a rank assignment does not move.
    # The former strided reduction incurs >2e-14 error for this example.
    values = np.broadcast_to([.3, .2, .5], (4, 65536, 3)).copy()
    expected = np.array([math.fsum(values[:, :, j].ravel()) / (4*65536) for j in range(3)])
    old = chain_summary(values)
    assert np.max(np.abs(old["mean"] - expected)) > 2e-14
    summary = stable_chain_summary(values)
    moments = stable_predictive_moments(values)
    np.testing.assert_array_equal(summary["mean"], expected)
    np.testing.assert_array_equal(moments["mean"], expected)
    np.testing.assert_array_equal(summary["chain_means"].mean(axis=0), expected)
    np.testing.assert_array_equal(summary["mcse"], np.zeros(3))
    np.testing.assert_array_equal(summary["rhat"], np.ones(3))
    np.testing.assert_array_equal(summary["ess"], np.full(3, 4*65536))
    np.testing.assert_array_equal(moments["cov_batch"], np.zeros((3, 3)))
    np.testing.assert_array_equal(moments["cov_chain"], np.zeros((3, 3)))


def _reference_coordinate(values):
    chains, draws = values.shape
    means = [math.fsum(chain)/draws for chain in values]
    mean = math.fsum(means)/chains
    length = max(2, math.isqrt(draws))
    batches = [math.fsum(chain[j:j+length])/length
               for chain in values for j in range(0, draws-length+1, length)]
    mcse = math.sqrt(max(statistics.variance(batches)/len(batches),
                         statistics.variance(means)/chains))
    half = draws//2
    split = list(values[:, :half]) + list(values[:, -half:])
    within = math.fsum(statistics.variance(part) for part in split)/(2*chains)
    between = half*statistics.variance([math.fsum(part)/half for part in split])
    rhat = math.sqrt(((half-1)/half*within + between/half)/within)
    ess = min(chains*draws, statistics.variance(values.ravel())/(mcse*mcse))
    return mean, mcse, rhat, ess


def test_odd_draw_count_statistics_match_independent_scalar_definitions():
    rng = np.random.default_rng(319)
    values = rng.dirichlet([2, 3, 7], size=(4, 67))
    summary = stable_chain_summary(values)
    for coordinate in range(values.shape[2]):
        expected = _reference_coordinate(values[:, :, coordinate])
        actual = [summary[key][coordinate] for key in ("mean", "mcse", "rhat", "ess")]
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-15)


def test_covariance_projections_equal_independent_scalar_batch_and_chain_variances():
    rng = np.random.default_rng(741)
    values = rng.dirichlet(np.arange(1, 6), size=(4, 129))
    weights = np.array([1., -2., 3., -4., 2.])
    moments = stable_predictive_moments(values)
    summary = stable_chain_summary(values)
    scalar = values @ weights
    length = math.isqrt(129)
    batch_means = [math.fsum(chain[j:j+length])/length
                   for chain in scalar for j in range(0, 129-length+1, length)]
    chain_means = [math.fsum(chain)/129 for chain in scalar]
    expected_batch = statistics.variance(batch_means)/len(batch_means)
    expected_chain = statistics.variance(chain_means)/len(chain_means)
    assert weights @ moments["cov_batch"] @ weights == pytest.approx(expected_batch, rel=2e-14)
    assert weights @ moments["cov_chain"] @ weights == pytest.approx(expected_chain, rel=2e-14)
    np.testing.assert_array_equal(moments["mean"], summary["mean"])
    np.testing.assert_allclose(moments["mcse_probs"], summary["mcse"], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(moments["cov_batch"].sum(axis=1), 0, atol=2e-18)
    np.testing.assert_allclose(moments["cov_chain"].sum(axis=1), 0, atol=2e-18)


def test_new_arithmetic_preserves_well_behaved_original_statistics():
    values = np.random.default_rng(17).dirichlet([1, 2, 3], size=(4, 256))
    expected = chain_summary(values)
    actual = stable_chain_summary(values)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-14, rtol=3e-13)
    expected_moments = predictive_moments(values)
    actual_moments = stable_predictive_moments(values)
    for key in expected_moments:
        np.testing.assert_allclose(actual_moments[key], expected_moments[key], atol=2e-14, rtol=3e-13)


def test_constant_but_inconsistent_chains_still_fail_rhat():
    values = np.empty((4, 32, 2))
    values[:2] = [.25, .75]
    values[2:] = [.75, .25]
    assert np.isinf(stable_chain_summary(values)["rhat"]).all()


@pytest.mark.parametrize("values", [
    np.ones((1, 16, 2))/2,
    np.ones((2, 15, 2))/2,
    np.ones((2, 16, 0)),
    np.full((2, 16, 2), np.nan),
    np.zeros((2, 16, 2)),
    np.ones((2, 16, 2)),
])
def test_predictive_validation_is_not_relaxed(values):
    with pytest.raises(ValueError):
        stable_predictive_moments(values)


def test_summary_metadata_identifies_the_new_reduction():
    assert SUMMARY_SPEC["id"] == "shared-base-stable-chain-statistics-v1"
    assert SUMMARY_SPEC["accumulator_mantissa_bits"] >= 53

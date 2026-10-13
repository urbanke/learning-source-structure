"""Independent arithmetic checks for the saved two-prior coin audit."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coin_bayes_audit import (LN2, entropy, exact_prior_regret, posterior_prediction,
                              reconstruct_binary, source_summary)


def test_binary_reconstruction_uses_target_label_and_natural_log():
    p = np.array([[0.01, 0.3, 0.8], [0.2, 0.9, 0.4]])
    targets = np.array([[0, 1, 0], [1, 0, 1]])
    lp = np.log(np.where(targets == 1, p, 1 - p))
    np.testing.assert_allclose(reconstruct_binary(lp, targets), p, atol=1e-15)


def test_two_point_prediction_matches_independent_posterior_weights():
    s, f = np.array([0, 2, 4, 23]), np.array([1, 2, 3, 18])
    first, second = 0.2 ** s * 0.8 ** f, 0.8 ** s * 0.2 ** f
    expected = (0.2 * first + 0.8 * second) / (first + second)
    np.testing.assert_allclose(posterior_prediction(s, s + f, "twopoint"), expected, atol=1e-15)


def test_free_first_token_changes_two_point_limit():
    result = exact_prior_regret("twopoint")
    assert result["finite_prior_limit_bits"] == pytest.approx(float(entropy(0.2)), abs=1e-15)
    assert result["final_regret_bits"] == pytest.approx(float(entropy(0.2)), abs=2e-12)
    assert result["cumulative_regret_bits"][0] == pytest.approx(float(entropy(0.68) - entropy(0.2)), abs=1e-15)


def test_uniform_prior_regret_matches_independent_ordered_sequence_evidence():
    # Under Beta(1,1), the count is uniform and every ordered sequence
    # with s ones has mass 1 / ((N+1) choose(N,s)).
    n = 257
    s = np.arange(n + 1)
    log_choose = gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1)
    joint_entropy = np.log2(n + 1) + np.mean(log_choose) / LN2
    expected = joint_entropy - (n - 1) / (2 * LN2) - 1
    assert exact_prior_regret("uniform", n - 1)["final_regret_bits"] == pytest.approx(expected, abs=2e-12)


def test_source_se_has_one_unit_per_complete_sequence():
    values = np.array([1, 2, 4, 5.0])
    result = source_summary(values)
    assert result["source_se"] == pytest.approx(values.std(ddof=1) / 2)
    assert result["n_sources"] == 4
    with pytest.raises(ValueError, match="per independent sequence"):
        source_summary(np.tile(values[:, None], (1, 256)))

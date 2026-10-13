"""Covariance projections, metric definitions and log-loss signs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transformer'))
from full_context_statistics import compare_prediction, predictive_moments, projected_mcse
from shared_base_mcmc import chain_summary


def test_covariance_projection_matches_direct_scalar_batch_estimator():
    rng = np.random.default_rng(53)
    x = rng.dirichlet([3., 5., 2.], size=(4, 256))
    # Introduce within-chain persistence and different chain means.
    x = .8*x + .2*x[:, :1]
    m = predictive_moments(x)
    for g in ([1., -2., 4.], [100., 100., 100.]):
        wanted = chain_summary((x @ g)[:, :, None])['mcse'][0]
        assert projected_mcse(g, m) == pytest.approx(wanted, abs=1e-14)
    np.testing.assert_allclose(m['mcse_probs'], chain_summary(x)['mcse'], atol=1e-14)
    np.testing.assert_allclose(m['mean'], m['chain_means'].mean(axis=0), atol=1e-15)


def test_metrics_are_computed_from_posterior_mean_and_can_have_negative_gap():
    x = np.tile(np.array([[[.2, .8], [.8, .2]]]), (4, 128, 1))
    m = predictive_moments(x)
    result = compare_prediction(m['mean'], [.5, .5], [.2, .8], m)
    assert result['kl_bits'] == pytest.approx(0, abs=1e-14)
    assert np.mean(np.sum(x*np.log2(x/.5), axis=2)) > .2
    result = compare_prediction([.5, .5], [.2, .8], [.2, .8], m)
    assert result['true_loss_gap_bits'] < 0
    assert result['true_loss_gap_bits'] == pytest.approx(
        result['transformer_true_cross_entropy_bits']-result['bayes_true_cross_entropy_bits'])


def test_smooth_metric_gradients_on_simplex():
    b = np.array([.2, .5, .3]); t = np.array([.4, .4, .2]); r = np.array([.7, .1, .2])
    direction = np.array([.4, -.6, .2]); h = 1e-6
    for fn, gradient in [(lambda p: p@np.log2(p/t), np.log2(b/t)+1/np.log(2)),
                         (lambda p: r@np.log2(p/t), r/(b*np.log(2)))]:
        numeric = (fn(b+h*direction)-fn(b-h*direction))/(2*h)
        assert numeric == pytest.approx(gradient@direction, abs=2e-10)


def test_kl_noise_scale_is_positive_when_first_order_mcse_vanishes():
    x = np.random.default_rng(18).dirichlet([10, 10], size=(4, 256))
    m = predictive_moments(x)
    r = compare_prediction(m['mean'], m['mean'], [.5, .5], m)
    assert r['kl_mcse_bits'] < 1e-15
    assert r['kl_noise_scale_bits'] > 0


def test_invalid_probabilities_do_not_get_renormalized():
    x = np.full((4, 16, 2), .51)
    with pytest.raises(ValueError):
        predictive_moments(x)

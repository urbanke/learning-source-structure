"""Paired predictive discrepancies and numerical sampling uncertainty, in bits."""
from __future__ import annotations

import numpy as np

LN2 = np.log(2.)
RESOLUTION_BITS = 0.01
NUMERICAL_TOLERANCE_BITS = RESOLUTION_BITS / 3


def predictive_moments(samples):
    """Covariance estimates for the posterior mean, from batches and chains."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 16:
        raise ValueError('need at least two chains and sixteen predictive draws')
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('predictive samples must be finite and positive')
    if np.max(np.abs(values.sum(axis=2)-1)) > 1e-7:
        raise ValueError('predictive samples are not normalized')
    chains, draws, d = values.shape
    length = max(2, int(np.sqrt(draws)))
    n_batches = draws // length
    batches = values[:, :n_batches*length].reshape(chains, n_batches, length, d).mean(axis=2)
    chain_means = values.mean(axis=1)
    batch_means = batches.reshape(chains*n_batches, d)
    cov_batch = np.cov(batch_means, rowvar=False, ddof=1)/(chains*n_batches)
    cov_chain = np.cov(chain_means, rowvar=False, ddof=1)/chains
    diagonal = np.maximum(np.diag(cov_batch), np.diag(cov_chain))
    return dict(mean=values.mean(axis=(0, 1)), mcse_probs=np.sqrt(np.maximum(diagonal, 0)),
                cov_batch=cov_batch, cov_chain=cov_chain, chain_means=chain_means,
                batch_length=length, batches_per_chain=n_batches, chains=chains, draws=draws)


def _probabilities(values, size=None):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or (size is not None and len(p) != size):
        raise ValueError('probability vector shape mismatch')
    if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 1e-7:
        raise ValueError('invalid raw probability vector')
    return p


def projected_mcse(gradient, moments):
    # Constants disappear along the probability simplex. Centering reduces
    # cancellation when covariance entries are nearly equal.
    g = np.asarray(gradient, dtype=float)
    g = g-g.mean()
    return float(np.sqrt(max(0., g @ moments['cov_batch'] @ g,
                             g @ moments['cov_chain'] @ g)))


def compare_prediction(bayes, transformer, oracle, moments):
    b = _probabilities(bayes)
    t = _probabilities(transformer, len(b))
    r = _probabilities(oracle, len(b))
    if (b <= 0).any() or (t <= 0).any():
        raise ValueError('Bayes and transformer probabilities must be positive')
    ratio = np.log2(b/t)
    kl = float(b @ ratio)
    gap = float(r @ ratio)
    kl_se = projected_mcse(ratio + 1/LN2, moments)
    gap_se = projected_mcse(r/(b*LN2), moments)
    # A conservative second-order noise scale (not a rigorous error bound).
    # This remains informative when KL's first derivative vanishes at b=t.
    variance = np.maximum(np.diag(moments['cov_batch']), np.diag(moments['cov_chain']))
    noise = float(np.sum(np.maximum(variance, 0)/b)/(2*LN2))
    entropy = float(-r[r > 0] @ np.log2(r[r > 0]))
    return dict(tv=float(np.abs(b-t).sum()/2), kl_bits=kl,
                true_loss_gap_bits=gap, kl_mcse_bits=kl_se, gap_mcse_bits=gap_se,
                kl_noise_scale_bits=noise,
                bayes_true_cross_entropy_bits=float(-r @ np.log2(b)),
                transformer_true_cross_entropy_bits=float(-r @ np.log2(t)),
                oracle_entropy_bits=entropy,
                numerical_tolerance_bits=NUMERICAL_TOLERANCE_BITS,
                metric_precision_passed=max(kl_se, gap_se, noise) <= NUMERICAL_TOLERANCE_BITS)


def diagnostic_gates(report):
    d = report['diagnostics']
    rhats = d['rhat_probs']+d['rhat_depth']+d['rhat_alpha_and_loglik']
    valid = all(v is not None and np.isfinite(v) for v in rhats)
    ess = np.asarray(d['ess_probs'], dtype=float)
    return dict(split_rhat_below_1_05=valid and min(rhats) >= 0 and max(rhats) < 1.05,
                probability_ess_at_least_200=bool(np.isfinite(ess).all() and min(ess) >= 200),
                maximum_probability_mcse_at_most_0_005=bool(np.max(report['mcse_probs']) <= .005),
                raw_normalization=bool(d['max_conditional_probability_sum_error'] <= 1e-7
                                       and abs(np.sum(report['predictive_probs'])-1) <= 1e-7))

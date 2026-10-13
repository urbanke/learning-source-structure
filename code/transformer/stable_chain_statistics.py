"""Stable reductions for new shared-base runs; frozen reports are unchanged.

These are the same batch-means, independent-chain MCSE, classic split-Rhat
and ESS definitions used by shared_base_mcmc/full_context_statistics. Only
the reduction arithmetic changes. The overall mean is reduced from the
chain means, so both report representations agree. No sample is adjusted,
renormalized, thinned, dropped or assigned a different statistical weight.

Explicit pairwise sums are needed on platforms where numpy.longdouble has
the same precision as float64. Extended precision is used when available.
"""
from __future__ import annotations

import numpy as np


SUMMARY_SPEC = {
    "id": "shared-base-stable-chain-statistics-v1",
    "reduction": "explicit pairwise sums with numpy.longdouble accumulator",
    "accumulator_mantissa_bits": int(np.finfo(np.longdouble).nmant + 1),
    "output_dtype": "float64",
    "overall_mean": "equal-weight mean of independently reduced chain means",
    "diagnostics": "unchanged classic split Rhat and max(batch, between-chain) MCSE",
}


def _pairwise_sum(values, axis):
    """Sum a nonempty axis with a binary tree, avoiding strided linear sums."""
    work = np.moveaxis(np.asarray(values, dtype=np.longdouble), axis, -1)
    if work.shape[-1] == 0:
        raise ValueError("cannot reduce an empty axis")
    while work.shape[-1] > 1:
        pairs = work.shape[-1] // 2
        reduced = work[..., :2*pairs:2] + work[..., 1:2*pairs:2]
        if work.shape[-1] % 2:
            reduced = np.concatenate((reduced, work[..., -1:]), axis=-1)
        work = reduced
    return work[..., 0]


def _mean(values, axis):
    return _pairwise_sum(values, axis) / values.shape[axis]


def _variance(values, axis):
    center = np.expand_dims(_mean(values, axis), axis)
    difference = np.asarray(values, dtype=np.longdouble) - center
    return _pairwise_sum(difference*difference, axis) / (values.shape[axis] - 1)


def _samples(samples):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 16 or not values.shape[2]:
        raise ValueError("need at least two chains and sixteen draws with nonempty coordinates")
    if not np.isfinite(values).all():
        raise ValueError("samples must be finite")
    return values


def _batch_means(values):
    chains, draws, coordinates = values.shape
    length = max(2, int(np.sqrt(draws)))
    count = draws // length
    grouped = values[:, :count*length].reshape(chains, count, length, coordinates)
    return _mean(grouped, 2).reshape(chains*count, coordinates), length, count


def _covariance_of_mean(values):
    """Sample covariance divided by the number of independent groups."""
    size, coordinates = values.shape
    centered = values - _mean(values, 0)
    covariance = np.empty((coordinates, coordinates), dtype=np.longdouble)
    for i in range(coordinates):
        products = centered[:, i, None] * centered[:, i:]
        covariance[i, i:] = _pairwise_sum(products, 0) / ((size - 1)*size)
        covariance[i:, i] = covariance[i, i:]
    return covariance


def stable_chain_summary(samples):
    """Drop-in replacement for chain_summary with stable reductions."""
    values = _samples(samples)
    chains, draws, coordinates = values.shape
    chain_means = _mean(values, 1)
    mean = _mean(chain_means, 0)
    between_variance = _variance(chain_means, 0) / chains
    batches, batch_length, _ = _batch_means(values)
    batch_variance = _variance(batches, 0) / len(batches)
    mcse = np.sqrt(np.maximum(batch_variance, between_variance))
    half = draws // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = _mean(_variance(split, 1), 0)
    between = half * _variance(_mean(split, 1), 0)
    variance = (half - 1)/half * within + between/half
    rhat = np.ones(coordinates, dtype=np.longdouble)
    np.divide(variance, within, out=rhat, where=within > 1e-30)
    rhat = np.sqrt(np.maximum(rhat, 0))
    rhat[(within <= 1e-30) & (between > 1e-30)] = np.inf
    flat = values.reshape(chains*draws, coordinates)
    marginal_variance = _variance(flat, 0)
    ess = np.full(coordinates, chains*draws, dtype=np.longdouble)
    np.divide(marginal_variance, mcse*mcse, out=ess, where=mcse > 1e-30)
    ess = np.minimum(ess, chains*draws)
    return dict(mean=np.asarray(mean, dtype=float), mcse=np.asarray(mcse, dtype=float),
                rhat=np.asarray(rhat, dtype=float), ess=np.asarray(ess, dtype=float),
                chain_means=np.asarray(chain_means, dtype=float), batch_length=batch_length)


def stable_predictive_moments(samples):
    """Drop-in predictive_moments replacement, retaining its validation gates."""
    values = _samples(samples)
    if (values <= 0).any():
        raise ValueError("predictive samples must be finite and positive")
    if np.max(np.abs(values.sum(axis=2) - 1)) > 1e-7:
        raise ValueError("predictive samples are not normalized")
    chains, draws, _ = values.shape
    chain_means = _mean(values, 1)
    batches, length, count = _batch_means(values)
    cov_batch = _covariance_of_mean(batches)
    cov_chain = _covariance_of_mean(chain_means)
    diagonal = np.maximum(np.diag(cov_batch), np.diag(cov_chain))
    return dict(mean=np.asarray(_mean(chain_means, 0), dtype=float),
                mcse_probs=np.asarray(np.sqrt(np.maximum(diagonal, 0)), dtype=float),
                cov_batch=np.asarray(cov_batch, dtype=float),
                cov_chain=np.asarray(cov_chain, dtype=float),
                chain_means=np.asarray(chain_means, dtype=float), batch_length=length,
                batches_per_chain=count, chains=chains, draws=draws)

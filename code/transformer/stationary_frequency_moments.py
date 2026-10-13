"""Nonnegative rank-assignment moments for the stationary-frequency surrogate.

Equal tail counts share polynomial variables. The coefficient of each target
count vector sums distinct-rank assignments with indistinguishable equal-count
labels; factorial multiplicities restore the labelled sum. No signed moment
subtractions, clipped probabilities or renormalization are used.
"""
from collections import Counter
from functools import lru_cache
import math

import numpy as np


def frequency_type(tokens, d=64):
    values = np.asarray(tokens)
    if (values.ndim != 1 or not 1 <= len(values) <= 17 or values.dtype.kind not in 'iu'
            or np.any(values < 0) or np.any(values >= d)):
        raise ValueError('Need one through seventeen integer tokens in the alphabet')
    tail = Counter(map(int, values[1:]))
    first_count = tail.pop(int(values[0]), 0)
    return first_count, tuple(sorted(tail.values()))


@lru_cache(maxsize=4096)
def assignment_states(kind, d):
    first_count, other_counts = kind
    if (not isinstance(first_count, int) or first_count < 0
            or tuple(sorted(other_counts)) != other_counts
            or any(not isinstance(x, int) or x <= 0 for x in other_counts)
            or 1 + first_count + sum(other_counts) > 17 or len(other_counts) + 1 > d):
        raise ValueError('Invalid frequency type')
    groups = sorted(Counter(other_counts).items())
    powers = ((1, first_count),) + tuple((0, degree) for degree, _ in groups)
    sizes = (2,) + tuple(multiplicity + 1 for _, multiplicity in groups)
    coordinates = np.indices(sizes).reshape(len(sizes), -1)
    strides = [math.prod(sizes[i+1:]) for i in range(len(sizes))]
    updates = []
    for axis, stride in enumerate(strides):
        target = np.flatnonzero(coordinates[axis] > 0)
        updates.append((target, target-stride))
    factor = math.prod(math.factorial(m) for _, m in groups)
    divisor = math.prod(range(d-len(other_counts), d+1))
    return powers, math.prod(sizes), tuple(updates), factor/divisor


def stationary_joint_type(kind, q, nu, *, block_size=256):
    """Return one permutation-averaged joint per latent (q,nu) pair.

    Evidence must subsequently be integrated over the latent prior *before*
    predictive ratios. Blocks bound working memory for large source banks.
    """
    q, nu = np.asarray(q, dtype=float), np.asarray(nu, dtype=float)
    if (q.ndim != 2 or q.shape != nu.shape or q.shape[1] < 2
            or not np.isfinite(q).all() or not np.isfinite(nu).all()
            or np.any(q < 0) or np.any(nu < 0)
            or np.any(abs(q.sum(1)-1) > 1e-10) or np.any(abs(nu.sum(1)-1) > 1e-10)
            or not isinstance(block_size, int) or block_size < 1):
        raise ValueError('Invalid normalized latent laws or block size')
    powers, size, updates, factor = assignment_states(kind, q.shape[1])
    result = np.empty(len(q))
    for start in range(0, len(q), block_size):
        stop = min(start+block_size, len(q))
        state = np.zeros((stop-start, size)); state[:, 0] = 1
        factors = [q[start:stop]**a * nu[start:stop]**b for a, b in powers]
        for rank in range(q.shape[1]):
            previous = state.copy()
            for values, (target, source) in zip(factors, updates):
                state[:, target] += previous[:, source] * values[:, rank, None]
        result[start:stop] = state[:, -1] * factor
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Invalid joint probabilities')
    return result

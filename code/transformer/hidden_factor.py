"""Hidden product source and its finite, exact Bayesian reference.

Two independent chains have iid Dirichlet(1,...,1) transition rows. A fresh
uniform bijection maps the r*s coordinate pairs to observed symbols in each
sequence. The initial pair is uniform. Encodings that differ only by names
of coordinate states (or exchange of equal-size coordinates) have identical
integrated likelihoods. The exact reference sums ONE representative of each
such orbit, with equal prior weight; this is the full uniform permutation
prior, not an approximation.

All returned log_evidence_bits entries are base-2 log probabilities, including
the uniform first token. Entry t scores xs[:t], so entry 0 is zero. Predictive
time t means t observed tokens, and predicts the next token. References use
float64 arithmetic and log-sum-exp, with no likelihood pruning.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations, product
from math import factorial
import numpy as np


def _dimensions(dimensions):
    values = tuple(dimensions)
    if len(values) != 2 or any(not isinstance(x, (int, np.integer)) or x < 2 for x in values):
        raise ValueError("dimensions must be two integers at least 2")
    return tuple(int(x) for x in values)


def encoding_count(dimensions=(3, 3)):
    """Number of distinct coordinate decompositions under the specified prior."""
    r, s = _dimensions(dimensions)
    return factorial(r*s) // (factorial(r)*factorial(s)*(2 if r == s else 1))


def draw_batch(rng, batch, length, dimensions=(3, 3)):
    """Draw independent sources and sequences, keeping their true parameters."""
    r, s = _dimensions(dimensions)
    if not isinstance(batch, (int, np.integer)) or batch < 1 or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("batch and length must be positive integers")
    matrices = tuple(rng.dirichlet(np.ones(k), size=(batch, k)) for k in (r, s))
    encoding = rng.permuted(np.broadcast_to(np.arange(r*s), (batch, r*s)), axis=1)
    coordinates = np.column_stack((rng.integers(r, size=batch), rng.integers(s, size=batch)))
    xs = np.empty((batch, length), dtype=np.int64)
    ids = np.arange(batch)
    xs[:, 0] = encoding[ids, coordinates[:, 0]*s + coordinates[:, 1]]
    cumulative = tuple(np.cumsum(matrix, axis=-1) for matrix in matrices)
    for t in range(1, length):
        for coordinate, k in enumerate((r, s)):
            row = cumulative[coordinate][ids, coordinates[:, coordinate]]
            coordinates[:, coordinate] = np.minimum((rng.random(batch)[:, None] > row).sum(axis=1), k-1)
        xs[:, t] = encoding[ids, coordinates[:, 0]*s + coordinates[:, 1]]
    return {"xs": xs, "matrices": matrices, "encoding": encoding}


def draw_hidden_factor_batch(rng, batch_size, length, r, s):
    """Convenience interface for training; sampling is identical to draw_batch."""
    return draw_batch(rng, batch_size, length, dimensions=(r, s))["xs"]


def _partitions(labels, width):
    """Unordered fixed-size groups, canonically ordered by their smallest label."""
    if not labels:
        yield ()
        return
    first, rest = labels[0], labels[1:]
    for selected in combinations(rest, width-1):
        group = (first,) + selected
        selected_set = set(group)
        remainder = tuple(x for x in labels if x not in selected_set)
        for tail in _partitions(remainder, width):
            yield (group,) + tail


def _canonical_encoding(matrix):
    """Canonicalize coordinate names without exchanging the two coordinates."""
    row_zero = int(np.flatnonzero(np.any(matrix == 0, axis=1))[0])
    order = np.argsort(matrix[row_zero])
    matrix = matrix[:, order]
    other_rows = sorted((i for i in range(len(matrix)) if i != row_zero), key=lambda i: int(matrix[i].min()))
    return tuple(matrix[[row_zero] + other_rows].ravel())


def iter_encodings(dimensions=(3, 3), chunk_size=65536):
    """Canonical pair->token permutations, generated without visiting all d!.

    The row containing symbol 0 is first and its entries are sorted (fixing
    column names). Other row groups are ordered by their smallest symbol.
    Each remaining row can have arbitrary column order. For square products,
    lexicographic comparison with the canonical transpose removes coordinate
    exchange. The group action is free, hence the representatives have equal
    prior mass.
    """
    r, s = _dimensions(dimensions)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    d = r*s
    if d > 255:
        raise ValueError("exact encoding enumeration supports at most 255 symbols")
    result = np.empty((min(chunk_size, encoding_count((r, s))), d), dtype=np.uint8)
    row = 0
    total = 0
    for groups in _partitions(tuple(range(d)), s):
        first, *remaining = groups
        for tails in product(*(permutations(group) for group in remaining)):
            candidate = first + sum(tails, ())
            if r == s and candidate > _canonical_encoding(np.asarray(candidate).reshape(r, s).T):
                continue
            result[row] = candidate
            row += 1
            total += 1
            if row == len(result):
                yield result
                result = np.empty_like(result)
                row = 0
    if row:
        yield result[:row]
    if total != encoding_count((r, s)):
        raise AssertionError(f"enumerated {total} encodings, expected {encoding_count((r, s))}")


@lru_cache(maxsize=4)
def encodings(dimensions=(3, 3)):
    """Cached canonical encodings for moderate problems (use iterator for large)."""
    dimensions = _dimensions(dimensions)
    if encoding_count(dimensions) > 1_000_000:
        raise ValueError("use iter_encodings for more than one million hypotheses")
    result = np.concatenate(list(iter_encodings(dimensions)), axis=0)
    result.flags.writeable = False
    return result


encoding_chunks = iter_encodings


def _arguments(xs, times, dimensions):
    dimensions = _dimensions(dimensions)
    xs = np.asarray(xs)
    d = int(np.prod(dimensions))
    if xs.ndim != 1 or xs.dtype.kind not in "iu" or np.any(xs < 0) or np.any(xs >= d):
        raise ValueError("xs must be a one-dimensional integer token array")
    if times is None:
        times = np.arange(len(xs)+1)
    times = np.asarray(times)
    if times.size == 0:
        times = times.astype(np.int64)
    if times.ndim != 1 or times.dtype.kind not in "iu" or np.any(times < 0) or np.any(times > len(xs)):
        raise ValueError("times must be integer observed lengths between 0 and len(xs)")
    if len(np.unique(times)) != len(times):
        raise ValueError("times must be unique")
    return xs.astype(np.int64, copy=False), times.astype(np.int64, copy=False), dimensions


def _logsumexp(values, axis=None):
    largest = np.max(values, axis=axis, keepdims=True)
    answer = largest + np.log(np.exp(values-largest).sum(axis=axis, keepdims=True))
    return answer.squeeze() if axis is None else np.squeeze(answer, axis=axis)


def _encoding_curve(xs, times, dimensions, candidates, chunk_size, candidate_count=None):
    r, s = dimensions
    d = r*s
    n = len(xs)
    mixture_evidence = np.full(n+1, -np.inf)
    mixture_joints = np.full((len(times), d), -np.inf)
    wanted = {int(t): i for i, t in enumerate(times)}
    if candidate_count is None:
        candidate_count = len(candidates)
        array = candidates
        candidates = (array[start:start+chunk_size] for start in range(0, candidate_count, chunk_size))
    for encoding in candidates:
        inverse = np.argsort(encoding, axis=1)
        coordinates = (inverse // s, inverse % s)
        m = len(encoding)
        ids = np.arange(m)
        counts = tuple(np.zeros((m, k, k), dtype=np.int32) for k in dimensions)
        totals = tuple(np.zeros((m, k), dtype=np.int32) for k in dimensions)
        log_likelihoods = np.zeros(m)
        for t in range(n+1):
            if t == 1:
                log_likelihoods[:] = -np.log(d)
            elif t >= 2:
                previous, current = xs[t-2:t]
                for coordinate, k in enumerate(dimensions):
                    a = coordinates[coordinate][:, previous]
                    b = coordinates[coordinate][:, current]
                    log_likelihoods += np.log((counts[coordinate][ids, a, b]+1) / (totals[coordinate][ids, a]+k))
                    counts[coordinate][ids, a, b] += 1
                    totals[coordinate][ids, a] += 1
            chunk_evidence = _logsumexp(log_likelihoods)
            mixture_evidence[t] = np.logaddexp(mixture_evidence[t], chunk_evidence)
            if t in wanted:
                if t == 0:
                    log_joint = np.full(d, chunk_evidence-np.log(d))
                else:
                    probabilities = np.ones((m, d))
                    for coordinate, k in enumerate(dimensions):
                        a = coordinates[coordinate][:, xs[t-1]]
                        probabilities *= (counts[coordinate][ids[:, None], a[:, None], coordinates[coordinate]]+1) / (totals[coordinate][ids, a, None]+k)
                    log_joint = _logsumexp(log_likelihoods[:, None] + np.log(probabilities), axis=0)
                mixture_joints[wanted[t]] = np.logaddexp(mixture_joints[wanted[t]], log_joint)
    predictive = np.exp(mixture_joints-mixture_evidence[times, None])
    log_evidence = (mixture_evidence-np.log(candidate_count))/np.log(2)
    log_evidence[0] = 0.0
    return {"times": times, "predictive": predictive, "log_evidence_bits": log_evidence,
            "encoding_count": candidate_count, "dimensions": dimensions}


def exact_curve(xs, times=None, dimensions=(3, 3), chunk_size=65536):
    """Exact hidden-encoding Bayes; selected predictions, evidence at all times.

    chunk_size bounds count-table memory but does not change the reference.
    Calling with times=[] requests only all-time sequence evidence.
    """
    xs, times, dimensions = _arguments(xs, times, dimensions)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    count = encoding_count(dimensions)
    if count <= 1_000_000:
        return _encoding_curve(xs, times, dimensions, encodings(dimensions), chunk_size)
    return _encoding_curve(xs, times, dimensions, iter_encodings(dimensions, chunk_size),
                           chunk_size, candidate_count=count)


def known_encoding_curve(xs, encoding, times=None, dimensions=(3, 3)):
    """Bayes given the pair->token encoding, but not either transition matrix."""
    xs, times, dimensions = _arguments(xs, times, dimensions)
    encoding = np.asarray(encoding)
    d = int(np.prod(dimensions))
    if encoding.shape != (d,) or encoding.dtype.kind not in "iu" or not np.array_equal(np.sort(encoding), np.arange(d)):
        raise ValueError("encoding must be a pair->token permutation")
    return _encoding_curve(xs, times, dimensions, encoding[None, :], 1)


def flat_curve(xs, times=None, dimensions=(3, 3), concentration=1.0):
    """Unstructured d-state chain: independent Dirichlet(a,...,a) rows.

    a=concentration is the PER-CELL concentration, not the row total.
    This is an explicit structural baseline, not a prior matched to the
    hidden product's induced distribution on full transition matrices.
    """
    xs, times, dimensions = _arguments(xs, times, dimensions)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    d = int(np.prod(dimensions))
    counts = np.zeros((d, d), dtype=np.int64)
    totals = np.zeros(d, dtype=np.int64)
    log_evidence = np.zeros(len(xs)+1)
    prediction = np.empty((len(times), d))
    wanted = {int(t): i for i, t in enumerate(times)}
    for t in range(len(xs)+1):
        if t == 1:
            log_evidence[t] = -np.log2(d)
        elif t >= 2:
            a, b = xs[t-2:t]
            log_evidence[t] = log_evidence[t-1] + np.log2((counts[a, b]+concentration)/(totals[a]+d*concentration))
            counts[a, b] += 1
            totals[a] += 1
        if t in wanted:
            prediction[wanted[t]] = np.full(d, 1/d) if t == 0 else (counts[xs[t-1]]+concentration)/(totals[xs[t-1]]+d*concentration)
    return {"times": times, "predictive": prediction, "log_evidence_bits": log_evidence, "dimensions": dimensions,
            "concentration_per_cell": concentration}


def true_curve(xs, matrices, encoding, times=None, dimensions=(3, 3)):
    """Source next-token probabilities and realized prefix log probabilities."""
    xs, times, dimensions = _arguments(xs, times, dimensions)
    r, s = dimensions
    d = r*s
    encoding = np.asarray(encoding)
    if encoding.shape != (d,) or encoding.dtype.kind not in "iu" or not np.array_equal(np.sort(encoding), np.arange(d)):
        raise ValueError("encoding must be a pair->token permutation")
    if len(matrices) != 2:
        raise ValueError("matrices must contain two transition matrices")
    for matrix, k in zip(matrices, dimensions):
        if np.asarray(matrix).shape != (k, k) or not np.isfinite(matrix).all() or np.any(np.asarray(matrix) < 0) or not np.allclose(np.sum(matrix, axis=-1), 1):
            raise ValueError("invalid source transition matrix")
    inverse = np.argsort(encoding)
    coordinates = (inverse // s, inverse % s)
    transition = np.asarray(matrices[0])[coordinates[0][:, None], coordinates[0][None, :]] * np.asarray(matrices[1])[coordinates[1][:, None], coordinates[1][None, :]]
    predictive = np.array([np.full(d, 1/d) if t == 0 else transition[xs[t-1]] for t in times]).reshape(len(times), d)
    log_evidence = np.zeros(len(xs)+1)
    if len(xs):
        log_evidence[1] = -np.log2(d)
        with np.errstate(divide="ignore"):
            log_evidence[2:] = log_evidence[1] + np.cumsum(np.log2(transition[xs[:-1], xs[1:]]))
    return {"times": times, "predictive": predictive, "log_evidence_bits": log_evidence, "dimensions": dimensions}

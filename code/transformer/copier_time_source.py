"""Independent natural draws from the uniform-base transformer's exact prior.

Each call draws a fresh shared depth, transition matrix, full path, and label
permutation. The entire n_ctx+1-token path is sampled before relabeling, in
the same random-number order as Source.batch(1). Prefixes must subsequently
be sliced from that fixed path rather than regenerated at shorter lengths.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from source import Source


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def validate_natural(record, *, d, n_ctx, prob_tol=1e-12):
    """Check raw source arrays without clipping or renormalizing them."""
    d, n_ctx = _integer(d, "d", 2), _integer(n_ctx, "n_ctx", 1)
    if not np.isfinite(prob_tol) or prob_tol <= 0:
        raise ValueError("prob_tol must be finite and positive")
    required = {"tokens", "initial_probs", "transition_probs", "permutation", "source_depth"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("Natural source record fields differ from the declared schema")
    x, permutation = np.asarray(record["tokens"]), np.asarray(record["permutation"])
    if x.shape != (n_ctx+1,) or x.dtype.kind not in "iu" or np.any(x < 0) or np.any(x >= d):
        raise ValueError("tokens must be an integer n_ctx+1 path within the alphabet")
    if permutation.shape != (d,) or permutation.dtype.kind not in "iu" or not np.array_equal(np.sort(permutation), np.arange(d)):
        raise ValueError("permutation must relabel every source symbol exactly once")
    raw_depth = record["source_depth"]
    if isinstance(raw_depth, np.ndarray) and raw_depth.ndim == 0 and raw_depth.dtype.kind in "iu":
        raw_depth = raw_depth.item()
    depth = _integer(raw_depth, "source_depth", 0)
    if depth > 8:
        raise ValueError("The source prior has depths zero through eight")
    initial, transition = np.asarray(record["initial_probs"]), np.asarray(record["transition_probs"])
    errors = {}
    for name, value, shape in (("initial", initial, (d,)), ("transition", transition, (d, d))):
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"Invalid {name} probabilities")
        error = float(np.max(np.abs(value.sum(axis=-1)-1)))
        if error > prob_tol:
            raise ValueError(f"Unnormalized {name} probabilities: {error}")
        errors[name+"_normalization_error"] = error
    uniform_error = float(np.max(np.abs(initial-1/d)))
    if uniform_error > prob_tol:
        raise ValueError("The initial source distribution must be uniform")
    zero_error = float(np.max(np.abs(transition-1/d))) if depth == 0 else None
    if zero_error is not None and zero_error > prob_tol:
        raise ValueError("Depth zero must be the uniform iid source")
    return dict(**errors, initial_uniform_error=uniform_error,
                depth_zero_uniform_error=zero_error,
                minimum_transition_probability=float(transition.min()),
                probability_tolerance=float(prob_tol))


def draw_natural(d, n_ctx, sequence_seed):
    """Return one exactly replayable natural source/path with an oracle matrix.

    Parameters follow Source(…,alpha=(0.,0.),ls=(0,8),permute=True). The
    point interval for alpha intentionally consumes the same random draw as
    the checkpoint's source configuration. The scalar source_depth is
    metadata for evaluation and must never be passed to the family Bayes
    predictor or transformer.
    """
    d, n_ctx = _integer(d, "d", 2), _integer(n_ctx, "n_ctx", 1)
    sequence_seed = _integer(sequence_seed, "sequence_seed", 0)
    source = Source(d, n_ctx+1, alpha=(0., 0.), ls=(0, 8), permute=True)
    rng = np.random.default_rng(sequence_seed)
    alpha, depths = source.draw_params(1, rng)
    initial, transition = source.matrices(alpha, depths, rng)
    original = source.sequences(initial, transition, rng)[0]
    permutation = rng.permutation(d)
    inverse = np.argsort(permutation)
    record = dict(tokens=np.asarray(permutation[original], dtype="<i8"),
                  initial_probs=np.asarray(initial[0, inverse], dtype="<f8"),
                  transition_probs=np.asarray(transition[0][inverse][:, inverse], dtype="<f8"),
                  permutation=np.asarray(permutation, dtype="<i8"),
                  source_depth=int(depths[0]))
    validate_natural(record, d=d, n_ctx=n_ctx)
    return record

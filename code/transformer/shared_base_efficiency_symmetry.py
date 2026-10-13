"""Exact label-symmetry reuse for the full uniform-rank shared-base prior.

Bayes depends on a prefix's equality pattern, not the numeric token labels.
A canonical prediction may be reused for every prefix with that pattern.
Each reused reference must still identify its full numerical/prior settings.
This module changes only labels; it does not condition on source parameters.
"""
import numpy as np

from shared_depth import _tokens


def canonical_prefix(tokens, d):
    """Return first-appearance labels and a complete canonical-to-input map."""
    tokens = _tokens(tokens, d)
    seen = list(dict.fromkeys(map(int, tokens)))
    unseen = [i for i in range(d) if i not in seen]
    canonical_to_input = np.array(seen + unseen, dtype=np.int64)
    input_to_canonical = np.argsort(canonical_to_input)
    return input_to_canonical[tokens], canonical_to_input


def restore_prediction(canonical_probs, canonical_to_input):
    """Map a full canonical next-token vector back to the input labels."""
    probs = np.asarray(canonical_probs, dtype=float)
    labels = np.asarray(canonical_to_input)
    if probs.ndim != 1 or labels.shape != probs.shape:
        raise ValueError('probabilities and label map must have equal vector shape')
    if labels.dtype.kind not in 'iu' or not np.array_equal(np.sort(labels), np.arange(len(labels))):
        raise ValueError('label map must be a complete permutation')
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError('probabilities must be finite and nonnegative')
    result = np.empty_like(probs)
    result[labels] = probs
    return result

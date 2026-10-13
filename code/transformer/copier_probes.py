"""Controlled, token-count-matched probes of a transformer's copying circuit.

These are designed diagnostic sequences, not draws from a Markov source.
Two earlier motifs A B and D C share a final query A. The counterfactual
swaps only B and C; an unrelated-query control replaces final A by unseen E.
Every filler token is distinct and excludes A, D, B, C and E. Thus no token
before the query repeats, and the neutral-query sequence is entirely unique.
"""
from __future__ import annotations

import operator

import numpy as np

DISCOVERY_SEED = 20260930
HELDOUT_SEED = 20261001
ARRAY_KEYS = (
    "clean", "counterfactual", "neutral_clean", "neutral_counterfactual",
    "target_b", "target_c", "source_b_pos", "source_c_pos", "query_pos",
    "context_a", "context_d", "control_e",
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def validate_probes(probes, *, vocab):
    """Check the intervention, query controls, labels and absence of repeats."""
    if any(key not in probes for key in ARRAY_KEYS):
        raise ValueError("Missing a required probe array")
    clean = np.asarray(probes["clean"])
    if clean.ndim != 2 or clean.shape[0] < 1:
        raise ValueError("Expected a nonempty probe-by-token array")
    n, length = clean.shape
    if not 8 <= length <= vocab:
        raise ValueError("Probe length must lie between 8 and the vocabulary size")
    for key in ARRAY_KEYS:
        values = np.asarray(probes[key])
        shape = (n, length) if key in ARRAY_KEYS[:4] else (n,)
        if values.shape != shape or values.dtype.kind not in "iu":
            raise ValueError(f"Incorrect integer array shape: {key}")
    token_keys = ARRAY_KEYS[:4] + ("target_b", "target_c", "context_a", "context_d", "control_e")
    if any(np.any(probes[key] < 0) or np.any(probes[key] >= vocab) for key in token_keys):
        raise ValueError("A probe token is outside the vocabulary")
    rows = np.arange(n)
    bpos, cpos, query = (probes[key] for key in ("source_b_pos", "source_c_pos", "query_pos"))
    if (np.any(bpos < 2) or np.any(cpos < 2) or np.any(bpos > length-3)
            or np.any(cpos > length-3) or np.any(np.abs(bpos.astype(int)-cpos.astype(int)) < 3)
            or np.any(query != length-1)):
        raise ValueError("Motifs overlap, touch the edges, or use the wrong query position")
    a, d, b, c, e = (probes[key] for key in ("context_a", "context_d", "target_b", "target_c", "control_e"))
    labels = np.stack((a, d, b, c, e), axis=1)
    if np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0):
        raise ValueError("A, D, B, C and E must be distinct")
    for positions, expected in ((bpos-1, a), (bpos, b), (cpos-1, d), (cpos, c), (query, a)):
        if not np.array_equal(clean[rows, positions], expected):
            raise ValueError("Clean sequence does not contain the declared motifs and query")
    if np.any(np.diff(np.sort(clean[:, :-1], axis=1), axis=1) == 0):
        raise ValueError("Every token before the query must be distinct")
    if np.any(clean == e[:, None]):
        raise ValueError("The unrelated query E must be unseen in the clean sequence")
    expected_counterfactual = clean.copy()
    expected_counterfactual[rows, bpos] = c
    expected_counterfactual[rows, cpos] = b
    if not np.array_equal(probes["counterfactual"], expected_counterfactual):
        raise ValueError("Counterfactual must change only the two successor tokens")
    for key, original in (("neutral_clean", clean), ("neutral_counterfactual", expected_counterfactual)):
        expected = original.copy()
        expected[rows, query] = e
        if not np.array_equal(probes[key], expected):
            raise ValueError("Neutral control must change only final A to unseen E")
        if np.any(np.diff(np.sort(probes[key], axis=1), axis=1) == 0):
            raise ValueError("Every neutral-query token must be distinct")
    return True


def make_probes(n, length=64, vocab=64, seed=HELDOUT_SEED):
    """Return independent controlled probe pairs, with zero-based positions.

    Successor positions are sampled uniformly from ordered pairs in
    {2,...,length-3}, separated by at least three positions. Both temporal
    orders occur: B is not systematically more recent than C. Motifs have
    at least one filler between them and between the later successor and
    the final query. Fillers are sampled without replacement.
    """
    n, length, vocab, seed = (_integer(value, name) for value, name in
                              ((n, "n"), (length, "length"), (vocab, "vocab"), (seed, "seed")))
    if n < 1 or not 8 <= length <= vocab or seed < 0:
        raise ValueError("Require n>=1, 8<=length<=vocab, and a nonnegative seed")
    rng = np.random.default_rng(seed)
    possible = np.arange(2, length-2, dtype=np.int64)
    pairs = np.array([(left, right) for left in possible for right in possible
                      if abs(left-right) >= 3], dtype=np.int64)
    clean = np.empty((n, length), dtype=np.int64)
    labels = np.empty((n, 5), dtype=np.int64)
    positions = np.empty((n, 2), dtype=np.int64)
    alphabet = np.arange(vocab, dtype=np.int64)
    for index in range(n):
        a, d, b, c, e = rng.choice(alphabet, size=5, replace=False)
        labels[index] = (a, d, b, c, e)
        bpos, cpos = pairs[rng.integers(len(pairs))]
        positions[index] = (bpos, cpos)
        free = np.ones(length, dtype=bool)
        free[[bpos-1, bpos, cpos-1, cpos, length-1]] = False
        available = np.ones(vocab, dtype=bool)
        available[[a, d, b, c, e]] = False
        clean[index, free] = rng.choice(alphabet[available], size=length-5, replace=False)
        clean[index, [bpos-1, bpos, cpos-1, cpos, length-1]] = (a, b, d, c, a)
    a, d, b, c, e = labels.T
    bpos, cpos = positions.T
    rows = np.arange(n)
    counterfactual = clean.copy()
    counterfactual[rows, bpos] = c
    counterfactual[rows, cpos] = b
    neutral_clean, neutral_counterfactual = clean.copy(), counterfactual.copy()
    neutral_clean[:, -1], neutral_counterfactual[:, -1] = e, e
    result = dict(clean=clean, counterfactual=counterfactual,
                  neutral_clean=neutral_clean, neutral_counterfactual=neutral_counterfactual,
                  target_b=b.copy(), target_c=c.copy(), source_b_pos=bpos.copy(), source_c_pos=cpos.copy(),
                  query_pos=np.full(n, length-1, dtype=np.int64),
                  context_a=a.copy(), context_d=d.copy(), control_e=e.copy())
    validate_probes(result, vocab=vocab)
    return result

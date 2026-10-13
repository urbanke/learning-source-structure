"""Read a fixed copying candidate from existing instrumented attention.

This is a hypothesis about how attention becomes token probabilities. It
is not an isolated neural expert: value projections, MLPs, and the final
readout are deliberately not decoded here. The candidate is independent
of the intact output probabilities used to evaluate its reconstruction.
"""
from __future__ import annotations

import numpy as np
import torch

from circuit_probe import run_probe


def copy_candidates(tokens, attention, vocab):
    """Pool last-layer heads, condition on matching predecessors, scatter.

    tokens has only the observed input (no final target). attention has
    shape [heads,time,time]. Returns undefined C as uniform placeholders
    with a separate `defined` flag. They must not receive reported weights.
    """
    tokens = np.asarray(tokens)
    attention = np.asarray(attention)
    if (tokens.ndim != 1 or tokens.dtype.kind not in "iu" or len(tokens) < 1
            or not isinstance(vocab, int) or vocab < 2
            or np.any(tokens < 0) or np.any(tokens >= vocab)):
        raise ValueError("Expected a nonempty integer input and valid vocabulary")
    n = len(tokens)
    if (attention.ndim != 3 or attention.shape[0] < 1
            or attention.shape[1:] != (n, n) or not np.isfinite(attention).all()
            or np.any(attention < 0)
            or np.max(np.abs(attention.sum(-1)-1)) > 1e-10
            or np.max(np.triu(attention, k=1)) > 1e-12):
        raise ValueError("Expected normalized finite causal attention")
    mean_attention = attention.mean(axis=0)
    matched = np.zeros((n, n), dtype=bool)
    if n > 1:
        matched[:, 1:] = tokens[:, None] == tokens[None, :-1]
    matched &= np.tri(n, dtype=bool)
    counts = np.zeros((n, vocab), dtype=np.int64)
    transported = np.zeros((n, vocab), dtype=np.float64)
    # Sum over sites, not unique labels: repeated continuations contribute
    # their full frequency/attention mass.
    for site in range(1, n):
        counts[:, tokens[site]] += matched[:, site]
        transported[:, tokens[site]] += matched[:, site] * mean_attention[:, site]
    totals = counts.sum(-1)
    defined = totals > 0
    mass = transported.sum(-1)
    if np.any(mass[defined] <= 0):
        raise ValueError("Matching sites have zero attention mass")
    support = counts > 0
    result = {}
    for name, values, denominator in (
            ("attention", transported, mass),
            ("empirical", counts, totals),
            ("support_uniform", support, support.sum(-1))):
        candidate = np.full((n, vocab), 1/vocab, dtype=np.float64)
        candidate[defined] = values[defined]/denominator[defined, None]
        if np.max(np.abs(candidate.sum(-1)-1)) > 1e-12:
            raise ValueError("Candidate does not normalize")
        result[name] = candidate
    return dict(candidates=result, defined=defined, support=support,
                counts=counts, matching_attention=mass)


@torch.no_grad()
def read_copy_candidates(model, sequence):
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or sequence.dtype.kind not in "iu" or len(sequence) < 2:
        raise ValueError("Expected a complete integer sequence with its final target")
    tokens = torch.from_numpy(np.array(sequence[:-1], dtype=np.int64, copy=True))[None]
    result = run_probe(model, tokens, return_cache=True)
    copied = copy_candidates(sequence[:-1], result["attention"][-1][0].numpy(), model.cfg["vocab"])
    copied["intact_probs"] = result["logits"][0].softmax(-1).numpy().copy()
    return copied

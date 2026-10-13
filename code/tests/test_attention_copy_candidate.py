"""Independent site enumeration and causal replay for the proposed copier."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from attention_copy_candidate import copy_candidates, read_copy_candidates
from model import GPT


def test_independent_enumeration_counts_duplicate_successors_and_self_repeats():
    tokens = np.array([0, 1, 0, 2, 0, 1, 0, 0])
    n = len(tokens)
    rng = np.random.default_rng(743)
    attention = np.tril(rng.uniform(.1, 1, (3, n, n)))
    attention /= attention.sum(-1, keepdims=True)
    result = copy_candidates(tokens, attention, 4)
    for query in range(n):
        sites = [s for s in range(1, query+1) if tokens[s-1] == tokens[query]]
        assert result["defined"][query] == bool(sites)
        expected_count = np.array([sum(tokens[s] == b for s in sites) for b in range(4)])
        np.testing.assert_array_equal(result["counts"][query], expected_count)
        if sites:
            expected_mass = np.array([sum(float(attention[h, query, s]) for h in range(3)
                                         for s in sites if tokens[s] == b) for b in range(4)])
            np.testing.assert_allclose(result["candidates"]["attention"][query], expected_mass/expected_mass.sum())
            np.testing.assert_allclose(result["candidates"]["empirical"][query], expected_count/len(sites))
            np.testing.assert_array_equal(result["support"][query], expected_count > 0)
    assert result["counts"][-1, 0] == 1  # Completed self-repeat at last input site.
    assert result["counts"][-1, 1] == 2  # Both earlier 0->1 transitions count.


def test_uniform_attention_equals_empirical_copier():
    tokens = np.array([0, 1, 0, 2, 0, 1, 0])
    attention = np.tril(np.ones((2, 7, 7)))
    attention /= attention.sum(-1, keepdims=True)
    result = copy_candidates(tokens, attention, 4)
    np.testing.assert_allclose(result["candidates"]["attention"], result["candidates"]["empirical"])
    assert not np.allclose(result["candidates"]["attention"][-1], result["candidates"]["support_uniform"][-1])


def test_future_target_excluded_and_every_shorter_prefix_replays():
    with torch.random.fork_rng():
        torch.manual_seed(79)
        model = GPT(vocab=4, n_ctx=12, n_layer=2, n_head=2, d_model=8).double().eval()
    sequence = np.array([0, 1, 0, 2, 0, 1, 3, 0, 2])
    full = read_copy_candidates(model, sequence)
    changed = sequence.copy()
    changed[-1] = 3
    other = read_copy_candidates(model, changed)
    np.testing.assert_array_equal(full["intact_probs"], other["intact_probs"])
    for name in full["candidates"]:
        np.testing.assert_array_equal(full["candidates"][name], other["candidates"][name])
    for length in range(1, len(sequence)):
        truncated = read_copy_candidates(model, sequence[:length+1])
        for name in full["candidates"]:
            np.testing.assert_allclose(truncated["candidates"][name][-1], full["candidates"][name][length-1], atol=2e-14)
        np.testing.assert_allclose(truncated["intact_probs"][-1], full["intact_probs"][length-1], atol=2e-14)


def test_reject_noncausal_attention():
    with pytest.raises(ValueError, match="causal"):
        copy_candidates(np.array([0, 1]), np.full((1, 2, 2), .5), 2)

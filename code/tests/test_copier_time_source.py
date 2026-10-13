"""Replay and independent relabeling checks for the natural copier cohort."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"transformer"))
from copier_time_source import draw_natural, validate_natural
from source import Source


@pytest.mark.parametrize("d,n_ctx,seed", [(2, 1, 13), (7, 32, 721), (64, 1024, 20261004)])
def test_exact_frozen_source_batch_replay(d, n_ctx, seed):
    record = draw_natural(d, n_ctx, seed)
    tokens, metadata = Source(d, n_ctx+1, alpha=(0., 0.), ls=(0, 8), permute=True).batch(1, np.random.default_rng(seed))
    np.testing.assert_array_equal(record["tokens"], tokens[0])
    np.testing.assert_array_equal(record["permutation"], metadata["perm"][0])
    assert record["source_depth"] == int(metadata["ls"][0])
    assert record["tokens"].dtype == np.int64
    assert record["transition_probs"].dtype == np.float64


def test_oracle_relabeling_preserves_every_edge_and_path_probability():
    d, n_ctx, seed = 7, 32, 721
    actual = draw_natural(d, n_ctx, seed)
    source = Source(d, n_ctx+1, alpha=(0., 0.), ls=(0, 8), permute=True)
    rng = np.random.default_rng(seed)
    alpha, depth = source.draw_params(1, rng)
    q, t = source.matrices(alpha, depth, rng)
    old = source.sequences(q, t, rng)[0]
    permutation = rng.permutation(d)
    # Check the forward meaning of the relabeling directly, rather than
    # reproducing the helper's inverse-index expression.
    for i in range(d):
        assert actual["initial_probs"][permutation[i]] == q[0, i]
        for j in range(d):
            assert actual["transition_probs"][permutation[i], permutation[j]] == t[0, i, j]
    x = actual["tokens"]
    old_logp = np.log(q[0, old[0]])+np.log(t[0, old[:-1], old[1:]]).sum()
    new_logp = np.log(actual["initial_probs"][x[0]])+np.log(actual["transition_probs"][x[:-1], x[1:]]).sum()
    assert old_logp == new_logp


def test_full_prior_including_uniform_depth_zero_is_retained():
    seen = set()
    for seed in range(64):
        record = draw_natural(3, 5, seed)
        seen.add(record["source_depth"])
        if record["source_depth"] == 0:
            np.testing.assert_array_equal(record["transition_probs"], np.full((3, 3), 1/3))
    assert seen == set(range(9))


def test_numpy_scalar_depth_roundtrip_is_valid():
    record = draw_natural(3, 7, 39)
    record["source_depth"] = np.asarray(record["source_depth"], dtype=np.int64)
    assert validate_natural(record, d=3, n_ctx=7)["transition_normalization_error"] < 1e-12


@pytest.mark.parametrize("field,changed", [
    ("tokens", np.zeros(8, dtype=float)),
    ("permutation", np.array([0, 0, 2])),
    ("initial_probs", np.array([.2, .3, .5])),
    ("transition_probs", np.zeros((3, 3))),
    ("transition_probs", np.full((3, 3), np.nan)),
    ("transition_probs", np.full((3, 3), .5)),
    ("source_depth", 9),
    ("source_depth", True),
])
def test_invalid_cached_source_records_rejected(field, changed):
    record = copy.deepcopy(draw_natural(3, 7, 39))
    record[field] = changed
    with pytest.raises(ValueError):
        validate_natural(record, d=3, n_ctx=7)


@pytest.mark.parametrize("d,n_ctx,seed", [(1, 3, 1), (3, 0, 1), (3, 2, -1), (True, 2, 1), (3, 2., 1)])
def test_invalid_draw_arguments_rejected(d, n_ctx, seed):
    with pytest.raises(ValueError):
        draw_natural(d, n_ctx, seed)

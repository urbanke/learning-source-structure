"""Independent checks of exact prefix sharing and probability/oracle mappings."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_copying_payoff_references as inventory


def test_canonical_mapping_preserves_order_and_last_context():
    tokens = [47, 9, 47, 12, 9]
    pattern, seen = inventory.canonicalize(tokens)
    assert pattern == (0, 1, 0, 2, 1)
    assert seen == (47, 9, 12)
    mapping = inventory.alphabet_map(seen)
    assert len(set(mapping)) == 64
    assert mapping[np.asarray(pattern)].tolist() == tokens
    assert mapping[pattern[-1]] == tokens[-1]


def test_count_equal_order_different_prefixes_are_not_shared():
    left, _ = inventory.canonicalize([5, 8, 5, 9])
    right, _ = inventory.canonicalize([5, 5, 8, 9])
    assert sorted(left) == sorted(right)
    assert inventory.reference_key(left) != inventory.reference_key(right)


def test_alphabet_permutations_share_but_keep_every_use():
    data = {'L0': np.asarray([[47, 9, 47, 12, 9], [2, 7, 2, 8, 7]]),
            'L2': np.asarray([[16, 31, 16, 53, 31]])}
    refs, uses = inventory.inventory_tokens(data, times=[5])
    assert len(refs) == 1 and len(uses) == 3
    ref = next(iter(refs.values()))
    assert ref['use_count'] == 3
    assert ref['uses_by_condition'] == {'L0': 2, 'L2': 1}
    for use in uses:
        tokens = data[use['condition']][use['index']]
        mapping = inventory.alphabet_map(use['canonical_to_original_seen'])
        assert mapping[ref['tokens']].tolist() == tokens.tolist()


def test_full64_prediction_and_expected_loss_mapping_agree():
    tokens = [47, 9, 47, 12, 9]
    pattern, seen = inventory.canonicalize(tokens)
    canonical = np.arange(1, 65, dtype=float)
    canonical /= canonical.sum()
    rows = np.asarray([np.roll(np.arange(1, 65, dtype=float), i) for i in range(64)])
    rows /= rows.sum(axis=1, keepdims=True)
    original = inventory.map_prediction(canonical, seen)
    canonical_oracle = inventory.canonical_oracle(rows, tokens, seen)
    assert np.allclose(original[inventory.alphabet_map(seen)], canonical, rtol=0, atol=0)
    original_loss = -rows[tokens[-1]] @ np.log2(original)
    canonical_loss = -canonical_oracle @ np.log2(canonical)
    assert abs(original_loss - canonical_loss) < 2e-14
    assert np.isclose(original.sum(), 1.)


def test_horizon_counts_require_both_orders_and_candidates_are_not_accepted():
    refs, uses = inventory.inventory_tokens({'L0': np.asarray([[1, 4, 1, 4, 1, 4, 1, 4]])}, times=[5, 8])
    refs[next(iter(refs))]['existing_reference_candidates'] = [{'order':16}, {'order':16}]
    counts = inventory.counts_by_horizon(refs, uses)
    assert counts[0]['reference_tasks_both_orders'] == 0
    assert counts[1]['reference_tasks_both_orders'] == 4
    assert counts[1]['candidate_matched_reference_tasks'] == 1
    assert counts[1]['accepted_reused_reference_tasks'] == 0
    assert counts[1]['tasks_requiring_new_sampling_or_independent_reuse_audit'] == 4


@pytest.mark.parametrize('tokens', [[], [64], [-1]])
def test_invalid_tokens_rejected(tokens):
    with pytest.raises(ValueError):
        inventory.canonicalize(tokens)


def test_invalid_mapping_rejected():
    with pytest.raises(ValueError):
        inventory.alphabet_map([1, 1])

"""Canonical reuse agrees with independent full-rank reference integration."""
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transformer'))
from shared_base import SharedBaseLSA
from shared_base_efficiency_symmetry import canonical_prefix, restore_prediction
from test_shared_base import ElementaryRows


@pytest.mark.parametrize('d,depths,x', [(2, [0, 1], [1, 1, 0, 1, 0]),
                                       (3, [0], [2, 0, 2, 2, 0]),
                                       (3, [0], [1, 0, 2, 1])])
def test_canonical_reuse_equals_full_rank_enumeration(d, depths, x):
    model = SharedBaseLSA(d, ElementaryRows(depths), alpha_order=6)
    canonical, labels = canonical_prefix(x, d)
    np.testing.assert_array_equal(labels[canonical], x)
    reference = model.run(list(canonical)+[0], full_probs=True,
                          positions=[len(x)])['probs'][0]
    for permutation in permutations(range(d)):
        observed = np.array(permutation)[x]
        changed, mapping = canonical_prefix(observed, d)
        np.testing.assert_array_equal(changed, canonical)
        exact = model.run(list(observed)+[0], full_probs=True,
                          positions=[len(x)])['probs'][0]
        np.testing.assert_allclose(restore_prediction(reference, mapping), exact, atol=3e-12)


def test_time_four_has_only_fifteen_canonical_patterns():
    patterns = {tuple(canonical_prefix(x, 4)[0]) for x in product(range(4), repeat=4)}
    assert len(patterns) == 15


def test_unseen_labels_remain_a_complete_mapping():
    canonical, mapping = canonical_prefix([28, 28, 52, 28], 64)
    np.testing.assert_array_equal(canonical, [0, 0, 1, 0])
    np.testing.assert_array_equal(np.sort(mapping), np.arange(64))
    probs = np.full(64, .5/62)
    probs[:2] = [.3, .2]
    restored = restore_prediction(probs, mapping)
    assert restored[28] == .3 and restored[52] == .2
    assert restored.sum() == pytest.approx(1.0)

"""Scientific identity checks for paired product mismatch/change sources."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from product_shift_source import generate_product_shift_cases


@pytest.fixture
def cases():
    return generate_product_shift_cases(3, n_tokens=41, switch_observed=12)


def test_relabeling_preserves_path_and_maps_both_context_and_output(cases):
    original, relabeled = cases["stationary"], cases["relabeled"]
    perm = relabeled.permutation
    np.testing.assert_array_equal(relabeled.tokens, perm[original.tokens])
    observed = np.arange(1, len(original.tokens))
    np.testing.assert_array_equal(
        relabeled.true_probs(observed)[:, perm], original.true_probs(observed))
    # Score of each realized target is unchanged, including repeated contexts.
    positions = np.arange(len(observed))
    np.testing.assert_array_equal(
        relabeled.true_probs(observed)[positions, relabeled.tokens[observed]],
        original.true_probs(observed)[positions, original.tokens[observed]])


def test_switch_boundary_and_unchanged_first_segment(cases):
    original = cases["stationary"]
    change = original.switch_observed
    for condition in ("switch_one", "switch_both"):
        case = cases[condition]
        np.testing.assert_array_equal(case.tokens[:change], original.tokens[:change])
        np.testing.assert_array_equal(case.true_probs(np.arange(1, change)),
                                      original.true_probs(np.arange(1, change)))
        for n, matrices in ((change - 1, case.pre_matrices),
                            (change, case.post_matrices),
                            (change + 1, case.post_matrices)):
            u, v = divmod(int(case.tokens[n - 1]), 8)
            expected = np.outer(matrices[0, u], matrices[1, v]).ravel()
            np.testing.assert_array_equal(case.true_probs(n), expected)


def test_one_factor_retains_matrix_and_entire_coordinate_path(cases):
    stationary, one, both = (cases[k] for k in
                             ("stationary", "switch_one", "switch_both"))
    np.testing.assert_array_equal(one.pre_matrices[1], one.post_matrices[1])
    np.testing.assert_array_equal(one.tokens % 8, stationary.tokens % 8)
    np.testing.assert_array_equal(one.post_matrices[0], both.post_matrices[0])
    np.testing.assert_array_equal(one.tokens // 8, both.tokens // 8)
    assert not np.array_equal(one.pre_matrices[0], one.post_matrices[0])
    assert not np.array_equal(both.pre_matrices[1], both.post_matrices[1])


@pytest.mark.parametrize("depths", [(1, 1), (4, 4)])
def test_all_true_rows_normalized_without_modifying_probabilities(depths):
    cases = generate_product_shift_cases(7, depths=depths, n_tokens=65,
                                         switch_observed=16)
    for case in cases.values():
        probs = case.true_probs(np.arange(1, 65))
        assert probs.shape == (64, 64)
        assert np.all(probs > 0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(case.pre_matrices.sum(axis=-1), 1,
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(case.post_matrices.sum(axis=-1), 1,
                                   rtol=0, atol=1e-15)


def test_index_seed_and_generation_length_do_not_change_shared_prefix(cases):
    long = generate_product_shift_cases(3, n_tokens=80, switch_observed=12)
    other = generate_product_shift_cases(4, n_tokens=41, switch_observed=12)
    for condition, case in cases.items():
        np.testing.assert_array_equal(case.tokens, long[condition].tokens[:41])
        np.testing.assert_array_equal(case.pre_matrices, long[condition].pre_matrices)
        np.testing.assert_array_equal(case.post_matrices, long[condition].post_matrices)
        assert not np.array_equal(case.pre_matrices, other[condition].pre_matrices)
        assert not case.tokens.flags.writeable


@pytest.mark.parametrize("observed", [0, 41, -1, 1.5, [1, 2.0], True, [[1, 2]]])
def test_invalid_prediction_indices_fail(cases, observed):
    with pytest.raises(ValueError):
        cases["stationary"].true_probs(observed)


def test_switch_at_first_transition_continues_initial_state():
    cases = generate_product_shift_cases(2, n_tokens=10, switch_observed=1)
    assert len({int(cases[k].tokens[0]) for k in
                ("stationary", "switch_one", "switch_both")}) == 1
    case = cases["switch_both"]
    u, v = divmod(int(case.tokens[0]), 8)
    np.testing.assert_array_equal(case.true_probs(1),
                                  np.outer(case.post_matrices[0, u],
                                           case.post_matrices[1, v]).ravel())

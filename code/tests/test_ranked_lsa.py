"""Rank-cache cross-checks against direct weighted-row summation."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from ranked_lsa import RankedLSARowEvidence
from weighted_lsa import WeightedLSARowEvidence


def test_matches_direct_rows_with_arbitrary_observed_symbol_ranks():
    kernel = RankedLSARowEvidence(5, [0.5, 1.0, 1.5], [0, 1])
    direct = WeightedLSARowEvidence([0, 1])
    # Observed symbol b has zero-based latent rank rank_of_symbol[b].
    rank_of_symbol = np.array([4, 0, 2, 1, 3])
    observed_counts = np.array([1, 3, 0, 2, 5])
    rank_counts = observed_counts[np.argsort(rank_of_symbol)]
    for a, q in enumerate(kernel.base_laws):
        for m in (np.zeros(5, dtype=int), rank_counts, np.array([0, 0, 1, 0, 0])):
            np.testing.assert_allclose(kernel.log_evidence(m, a),
                                       direct.log_evidence(m[rank_of_symbol], q[rank_of_symbol]),
                                       rtol=0, atol=2e-13)


def test_extension_consistency_and_nonuniform_first_observation():
    kernel = RankedLSARowEvidence(4, [1.5], [1, 0])
    for counts in (np.zeros(4, dtype=int), np.array([4, 1, 0, 3])):
        before = kernel.log_evidence(counts, 0)
        ratios = [np.exp(kernel.log_evidence(counts + e, 0) - before)
                  for e in np.eye(4, dtype=int)]
        np.testing.assert_allclose(np.sum(ratios, axis=0), 1, rtol=0, atol=3e-12)
    assert abs(np.exp(kernel.log_evidence([1, 0, 0, 0], 0)[0]) - kernel.base_laws[0, 0]) > 0.02


def test_exact_uniform_shortcuts_and_rank_cache_canonicalization():
    kernel = RankedLSARowEvidence(4, [0.0], [0, 1, 8])
    # The singleton requires no store even when a high depth is requested.
    np.testing.assert_array_equal(kernel.log_evidence([1, 0, 0, 0], 0), np.full(3, -np.log(4)))
    np.testing.assert_array_equal(kernel.log_evidence([0, 0, 0, 1], 0), np.full(3, -np.log(4)))
    assert kernel.cache_info()["row_cache_hits"] == 1
    uniform = RankedLSARowEvidence(4, [0.0], [1, 0])
    counts = np.array([3, 2, 0, 4])
    np.testing.assert_allclose(uniform.log_evidence(counts, 0),
                               [gammaln(4) - gammaln(13) + gammaln(counts + 1).sum(),
                                -9 * np.log(4)], rtol=0, atol=1e-14)
    assert uniform.cache_info()["phi_terms_computed"] == 0


def test_cache_bounds_clear_and_output_isolation():
    kernel = RankedLSARowEvidence(4, [0.5, 1.5], [0, 1], term_cache_size=2,
                                 evidence_cache_size=2)
    expected = kernel.log_evidence([1, 2, 0, 0], 0)
    mutation = kernel.log_evidence([1, 2, 0, 0], 0)
    mutation[:] = 99
    np.testing.assert_array_equal(kernel.log_evidence([1, 2, 0, 0], 0), expected)
    for a in range(2):
        kernel.log_evidence([2, 0, 1, 0], a)
    info = kernel.cache_info()
    assert info["stored_phi_terms"] <= 2
    assert info["stored_rows"] <= 2
    assert info["stored_baselines"] == 2
    assert info["phi_cache_bytes"] <= 2 * 2049 * 8
    with pytest.raises(ValueError):
        kernel.base_laws[0, 0] = 0.5
    kernel.clear_cache()
    assert kernel.cache_info()["stored_baselines"] == 0
    assert kernel.cache_info()["stored_phi_terms"] == 0
    np.testing.assert_array_equal(kernel.log_evidence([1, 2, 0, 0], 0), expected)


def test_matrix_evaluation_and_exact_depth_zero():
    kernel = RankedLSARowEvidence(64, [0.5, 1.5], [0])
    counts = np.zeros((3, 64), dtype=int)
    counts[1, 63] = 10
    counts[2, [0, 17, 63]] = [2, 3, 4]
    for a in range(2):
        np.testing.assert_array_equal(kernel.log_evidence_many(counts, a)[:, 0],
                                      counts @ kernel.log_base_laws[a])
    assert kernel.log_evidence_many(np.empty((0, 64), dtype=int), 0).shape == (0, 1)
    assert kernel.cache_info()["phi_terms_computed"] == 0


def test_bad_bounds_are_not_normalized():
    kernel = RankedLSARowEvidence(3, [1.0], [1], u_lo=-1, u_hi=1)
    probabilities = [np.exp(kernel.log_evidence(e, 0)[0]) for e in np.eye(3, dtype=int)]
    assert sum(probabilities) < 0.8


def test_predictive_cache_shape_rank_order_and_isolation():
    kernel = RankedLSARowEvidence(4, [0.0, 1.2], [0, 1], predictive_cache_size=1)
    for a in range(2):
        counts = np.array([3, 0, 1, 2])
        expected = np.stack([kernel.log_evidence(counts + e, a)
                             - kernel.log_evidence(counts, a)
                             for e in np.eye(4, dtype=int)], axis=1)
        actual = kernel.log_predictive(counts, a)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(np.exp(actual).sum(1), 1, rtol=0, atol=2e-12)
        actual[:] = 88
        np.testing.assert_array_equal(kernel.log_predictive(counts, a), expected)
    assert kernel.cache_info()["stored_predictives"] == 1
    assert kernel.cache_info()["predictive_cache_hits"] == 2
    kernel.clear_cache()
    assert kernel.cache_info()["stored_predictives"] == 0


def test_sparse_cache_keys_retain_ranks_and_omit_unobserved_entries():
    kernel = RankedLSARowEvidence(64, [1.0], [0])
    counts = np.zeros(64, dtype=int)
    counts[[0, 63]] = [2, 7]
    expected = kernel.log_evidence(counts, 0)
    assert list(kernel._evidence) == [(0, ((0, 2), (63, 7)))]
    prediction = kernel.log_predictive(counts, 0)
    assert list(kernel._predictive) == [(0, ((0, 2), (63, 7)))]
    assert all(len(key[1]) <= 3 for key in kernel._evidence)
    swapped = counts.copy()
    swapped[[0, 63]] = swapped[[63, 0]]
    assert not np.array_equal(kernel.log_evidence(swapped, 0), expected)
    np.testing.assert_array_equal(kernel.log_predictive(counts, 0), prediction)


def test_sparse_uniform_evidence_keys_preserve_count_permutation_symmetry():
    kernel = RankedLSARowEvidence(64, [0.0], [0, 1])
    counts = np.zeros(64, dtype=int)
    counts[[0, 63]] = [2, 7]
    expected = kernel.log_evidence(counts, 0)
    assert list(kernel._evidence) == [(0, ((62, 2), (63, 7)))]
    permuted = np.roll(counts, 17)
    np.testing.assert_array_equal(kernel.log_evidence(permuted, 0), expected)
    assert kernel.cache_info()["row_cache_hits"] == 1
    # Predictive output still retains the supplied rank order.
    np.testing.assert_allclose(np.roll(kernel.log_predictive(counts, 0), 17, axis=1),
                               kernel.log_predictive(permuted, 0), rtol=0, atol=1e-13)


@pytest.mark.skipif(not (os.environ.get("LSA_STORE") and os.environ.get("PMM_SRC")),
                    reason="higher-depth cross-check needs the external read-only anchor store")
def test_all_depths_sparse_d64_against_direct_weighted_sums():
    kernel = RankedLSARowEvidence(64, [0.5, 1.5])
    direct = WeightedLSARowEvidence()
    for a in range(2):
        counts = np.zeros(64, dtype=int)
        counts[[0, 3, 21, 63]] = [3, 2, 1, 4]
        np.testing.assert_allclose(kernel.log_evidence(counts, a),
                                   direct.log_evidence(counts, kernel.base_laws[a]),
                                   rtol=0, atol=2e-12)


def test_backend_refuses_unsealed_store_before_import_or_creation(tmp_path):
    before = list(tmp_path.iterdir())
    kernel = RankedLSARowEvidence(3, [1.0], [2], store=str(tmp_path), src=str(tmp_path))
    with pytest.raises(RuntimeError, match="sealed"):
        kernel.log_evidence([1, 0, 0], 0)
    assert list(tmp_path.iterdir()) == before


def test_backend_refuses_conflicting_already_imported_mode(tmp_path, monkeypatch):
    (tmp_path / "anchors.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    old_mode = SimpleNamespace(_LADDER_EVERY=0, _LADDER_DEGREE=7, _SADDLE_MIN_L=0)
    monkeypatch.setitem(sys.modules, "product_model_with_memory.universal_tables", old_mode)
    kernel = RankedLSARowEvidence(3, [1.0], [2], store=str(tmp_path), src=str(tmp_path))
    with pytest.raises(RuntimeError, match="already imported"):
        kernel.log_evidence([1, 0, 0], 0)


@pytest.mark.skipif(not (os.environ.get("LSA_STORE") and os.environ.get("PMM_SRC")),
                    reason="production configuration check needs the external read-only store")
def test_existing_production_guard_rejects_conflicting_environment(monkeypatch):
    monkeypatch.setenv("PMM_PHI_LADDER_DEGREE", "7")
    kernel = RankedLSARowEvidence(3, [1.0], [2])
    with pytest.raises(RuntimeError, match="production requires PMM_PHI_LADDER_DEGREE"):
        kernel.log_evidence([1, 0, 0], 0)


@pytest.mark.parametrize("kwargs", [
    {"d": 1}, {"d": True}, {"d": 3.0}, {"alpha_nodes": []},
    {"alpha_nodes": [1, 1]}, {"alpha_nodes": [-1]}, {"alpha_nodes": [np.nan]},
    {"alpha_nodes": [[1]]}, {"term_cache_size": 1.5},
    {"evidence_cache_size": False}, {"alpha_nodes": [1e5]},
])
def test_invalid_constructor(kwargs):
    options = dict(d=3, alpha_nodes=[1.0], depths=[0, 1])
    options.update(kwargs)
    with pytest.raises(ValueError):
        RankedLSARowEvidence(**options)


@pytest.mark.parametrize("counts,index", [
    ([1.0, 0, 0], 0), ([-1, 0, 0], 0), ([1, 0], 0),
    ([1, 0, 0], -1), ([1, 0, 0], 1), ([1, 0, 0], 0.0),
    ([1, 0, 0], True), ([np.iinfo(np.int64).max, 1, 0], 0),
    (np.array([2**63, 0, 0], dtype=np.uint64), 0),
])
def test_invalid_row_input(counts, index):
    with pytest.raises(ValueError):
        RankedLSARowEvidence(3, [1.0], [0, 1]).log_evidence(counts, index)

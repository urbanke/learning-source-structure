"""Independent analytic checks for weighted rows; no anchor store needed."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from weighted_lsa import WeightedLSARowEvidence


def test_nonuniform_first_transition_is_not_the_base_law():
    q = np.array([0.8, 0.2])
    kernel = WeightedLSARowEvidence([0, 1])
    p = np.exp(kernel.log_evidence([1, 0], q))
    expected = q[0] / (q[0] - q[1]) - q[0] * q[1] / (q[0] - q[1])**2 * np.log(q[0] / q[1])
    np.testing.assert_allclose(p, [q[0], expected], rtol=0, atol=3e-13)
    assert abs(p[1] - q[0]) > 0.08


def test_depth_one_matches_independent_uniform_simplex_integration():
    kernel = WeightedLSARowEvidence([1])
    q = np.array([0.7, 0.3])
    for m in ([0, 0], [1, 0], [2, 3], [9, 2]):
        # A uniform Dirichlet(1,1) row is (u,1-u). Weight and normalize
        # this row directly, without the Laplace-transform representation.
        def integrand(u):
            p = q[0] * u / (q[0] * u + q[1] * (1 - u))
            return p**m[0] * (1 - p)**m[1]
        independent, _ = quad(integrand, 0, 1, epsabs=1e-14, epsrel=1e-12)
        assert np.exp(kernel.log_evidence(m, q)[0]) == pytest.approx(independent, rel=2e-11, abs=2e-14)


def test_ordered_observations_and_extension_consistency():
    kernel = WeightedLSARowEvidence([0, 1])
    q = np.array([0.6, 0.3, 0.1])
    np.testing.assert_array_equal(kernel.log_evidence([0, 0, 0], q), [0, 0])
    assert np.exp(kernel.log_evidence([1, 1, 0], q)[0]) == pytest.approx(0.18)
    counts = np.array([3, 1, 2])
    evidence = np.exp(kernel.log_evidence(counts, q))
    extensions = np.stack([np.exp(kernel.log_evidence(counts + delta, q)) for delta in np.eye(3, dtype=int)])
    np.testing.assert_allclose(extensions.sum(0), evidence, rtol=2e-11, atol=1e-14)


def test_permutation_scale_invariance_and_cache_isolation():
    kernel = WeightedLSARowEvidence([0, 1])
    m = np.array([2, 1, 4])
    q = np.array([0.6, 0.3, 0.1])
    expected = kernel.log_evidence(m, q)
    perm = np.array([2, 0, 1])
    np.testing.assert_array_equal(kernel.log_evidence(m[perm], q[perm]), expected)
    np.testing.assert_allclose(kernel.log_evidence(m, q * 7), expected, atol=1e-13)
    copy = kernel.log_evidence(m, q)
    copy[:] = 99
    np.testing.assert_array_equal(kernel.log_evidence(m, q), expected)
    assert kernel.cache_info()["row_cache_hits"] >= 3


def test_bad_integration_bounds_are_not_hidden_by_renormalization():
    kernel = WeightedLSARowEvidence([1], u_lo=-1, u_hi=1)
    q = np.array([0.7, 0.3])
    p = [np.exp(kernel.log_evidence(m, q)[0]) for m in ([1, 0], [0, 1])]
    assert sum(p) < 0.8


@pytest.mark.parametrize("counts,q", [
    ([1.0, 0.0], [0.5, 0.5]), ([-1, 2], [0.5, 0.5]),
    ([1, 0], [0, 1]), ([1, 0], [np.nan, 1]),
    ([1, 0], [1]), ([1], [1]),
])
def test_invalid_row_input(counts, q):
    with pytest.raises(ValueError):
        WeightedLSARowEvidence([0, 1]).log_evidence(counts, q)

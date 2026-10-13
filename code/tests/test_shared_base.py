"""Outer hierarchical Bayes checks using independent elementary row laws."""
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from shared_base import SharedBaseLSA
from shared_depth import SharedDepthLSA
from test_shared_depth import AnalyticKernel


class ElementaryRows:
    """Depth zero is deterministic; depth one integrates a uniform simplex.

    Nonuniform tests use d=2, so the independent integral is just over
    U~Uniform[0,1], rather than the implementation's Laplace representation.
    """

    def __init__(self, depths=(0, 1)):
        self.depths = list(depths)
        self.cache = {}

    def log_evidence(self, counts, q):
        key = (tuple(counts), tuple(q))
        if key in self.cache:
            return self.cache[key]
        counts = np.asarray(counts)
        q = np.asarray(q)
        total = int(counts.sum())
        values = []
        for depth in self.depths:
            if total == 0:
                value = 0.0
            elif depth == 0:
                value = float(counts @ np.log(q))
            elif np.allclose(q, 1 / len(q), atol=1e-15, rtol=0):
                value = float(gammaln(len(q)) - gammaln(len(q) + total) + gammaln(counts + 1).sum())
            else:
                assert depth == 1 and len(q) == 2
                def integrand(u):
                    denominator = q[0] * u + q[1] * (1 - u)
                    return (q[0] * u / denominator) ** counts[0] * (q[1] * (1 - u) / denominator) ** counts[1]
                integral, _ = quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13)
                value = np.log(integral)
            values.append(value)
        result = np.array(values)
        self.cache[key] = result
        return result


def independent_iid_evidence(tokens, d, bounds=(0.5, 1.5)):
    """Continuous exponent integral, independent of Gauss--Legendre nodes."""
    counts = np.bincount(tokens, minlength=d)
    assignments = list(permutations(range(1, d + 1)))
    def integrand(alpha):
        total = 0.0
        for ranks in assignments:
            q = np.array(ranks, dtype=float) ** -alpha
            q /= q.sum()
            total += float(np.prod(q ** counts))
        return total / len(assignments)
    value, _ = quad(integrand, *bounds, epsabs=1e-13, epsrel=1e-13)
    return value / (bounds[1] - bounds[0])


def test_depth_zero_matches_independent_continuous_iid_integral():
    reference = SharedBaseLSA(3, ElementaryRows([0]), alpha_order=12)
    for n in range(1, 4):
        total = 0.0
        for x in product(range(3), repeat=n):
            expected = independent_iid_evidence(x, 3)
            result = reference.run(x, full_probs=True)
            assert reference.joint(x) == pytest.approx(expected, abs=3e-14)
            assert np.exp(result["log_evidence"][-1]) == pytest.approx(expected, abs=3e-14)
            total += reference.joint(x)
        assert total == pytest.approx(1, abs=3e-14)


def test_joint_extension_and_sequential_ratios_with_shared_base_and_depth():
    reference = SharedBaseLSA(2, ElementaryRows(), alpha_order=6, depth_prior=[2, 3])
    for n in range(1, 5):
        total = 0.0
        for x in product(range(2), repeat=n):
            result = reference.run(x, full_probs=True, posteriors=True)
            total += reference.joint(x)
            for t in range(n):
                joint = reference.joint(x[:t + 1])
                previous = reference.joint(x[:t])
                assert np.exp(result["log_evidence"][t]) == pytest.approx(joint, abs=3e-14)
                assert result["loss"][t] == pytest.approx(-np.log2(joint / previous), abs=3e-13)
                for b in range(2):
                    expected = reference.joint(x[:t] + (b,)) / previous
                    assert result["probs"][t, b] == pytest.approx(expected, abs=3e-13)
            np.testing.assert_allclose(result["posterior"].sum(axis=(1, 2)), 1, atol=3e-14)
            np.testing.assert_allclose(result["prob_sum_error"], 0, atol=3e-13)
        assert total == pytest.approx(1, abs=3e-14)


def test_first_token_updates_base_posterior_but_not_depth_prior():
    reference = SharedBaseLSA(2, ElementaryRows(), alpha_order=3, depth_prior=[1, 4])
    result = reference.run([0, 1], full_probs=True, posteriors=True)
    np.testing.assert_allclose(result["probs"][0], [0.5, 0.5], atol=3e-15)
    expected = reference.hyper_prior * reference.base_laws[:, 0]
    expected /= expected.sum()
    np.testing.assert_allclose(result["base_posterior"][1], expected, atol=3e-15)
    assert not np.allclose(expected, reference.hyper_prior)
    np.testing.assert_allclose(result["depth_posterior"][1], reference.depth_prior, atol=3e-15)
    assert result["loss"][1] == pytest.approx(-(reference.log_joint([0, 1]) - reference.log_joint([0])) / np.log(2))


def test_known_q_initial_draw_differs_from_weighted_row_singleton():
    q = np.array([0.8, 0.2])
    rows = ElementaryRows([1])
    reference = SharedBaseLSA(2, rows, known_q=q)
    result = reference.run([0, 0, 1], full_probs=True, posteriors=True)
    np.testing.assert_allclose(result["probs"][0], q)
    expected = np.exp(rows.log_evidence([1, 0], q))[0]
    assert result["probs"][1, 0] == pytest.approx(expected, abs=1e-13)
    assert abs(expected - q[0]) > 0.05
    np.testing.assert_allclose(result["base_posterior"], 1)


def test_uniform_base_reduces_to_shared_depth_predictor():
    x = [0, 0, 1, 1, 1, 0, 1, 0]
    base = SharedBaseLSA(2, ElementaryRows(), fixed_alpha=0, depth_prior=[1, 3])
    depth = SharedDepthLSA(AnalyticKernel(), prior=[1, 3])
    actual = base.run(x, full_probs=True, posteriors=True)
    expected = depth.run(x, full_probs=True, depth_posteriors=True)
    np.testing.assert_allclose(actual["loss"], expected["loss"], atol=2e-14)
    np.testing.assert_allclose(actual["probs"], expected["probs"], atol=2e-14)
    np.testing.assert_allclose(actual["depth_posterior"], expected["depth_posterior"], atol=2e-14)


def test_unknown_ranks_are_equivariant_and_selected_positions_match():
    reference = SharedBaseLSA(3, ElementaryRows([0]), alpha_order=5)
    x = np.array([0, 2, 1, 0, 2])
    sigma = np.array([2, 0, 1])
    original = reference.run(x, full_probs=True)
    relabeled = reference.run(sigma[x], full_probs=True)
    np.testing.assert_allclose(relabeled["probs"][:, sigma], original["probs"], atol=3e-14)
    np.testing.assert_allclose(relabeled["loss"], original["loss"], atol=3e-14)
    selected = reference.run(x, full_probs=True, positions=[4, 0, 2])
    np.testing.assert_array_equal(selected["positions"], [0, 2, 4])
    np.testing.assert_allclose(selected["probs"], original["probs"][[0, 2, 4]])
    np.testing.assert_allclose(reference.run(x)["loss"], original["loss"])


def test_known_order_and_fixed_alpha_have_the_expected_iid_law():
    reference = SharedBaseLSA(3, ElementaryRows([0]), known_order=True, fixed_alpha=1)
    q = np.array([1, 1 / 2, 1 / 3])
    q /= q.sum()
    x = [2, 1, 0, 1]
    result = reference.run(x, full_probs=True)
    np.testing.assert_allclose(result["probs"], np.tile(q, (len(x), 1)))
    assert reference.joint(x) == pytest.approx(float(np.prod(q[x])))
    assert reference.n_hypotheses == 1
    np.testing.assert_array_equal(reference.perms, [[1, 2, 3]])


def test_empty_sequence_and_invalid_arguments():
    reference = SharedBaseLSA(2, ElementaryRows(), alpha_order=2)
    result = reference.run([], full_probs=True, posteriors=True)
    assert result["probs"].shape == (0, 2)
    assert result["posterior"].shape == (0, 4, 2)
    assert reference.joint([]) == 1
    assert reference.log_joint([]) == 0
    for kwargs in [{"alpha_order": 0}, {"alpha_bounds": (1, 1)}, {"fixed_alpha": -1},
                   {"known_q": [0.2, 0.2]}, {"known_q": [1, 0]}, {"depth_prior": [0, 1]},
                   {"known_q": [0.5, 0.5], "fixed_alpha": 0}]:
        with pytest.raises(ValueError):
            SharedBaseLSA(2, ElementaryRows(), **kwargs)
    with pytest.raises(ValueError):
        reference.run([0, 2])
    with pytest.raises(ValueError):
        reference.run([0, 1], full_probs=True, positions=[1, 1])

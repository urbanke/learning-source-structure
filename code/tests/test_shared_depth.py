"""Independent finite-state checks; no numerical LSA tables are required."""
from fractions import Fraction
from itertools import product
from math import factorial, log
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from shared_depth import ProductSharedDepthLSA, SharedDepthLSA


def row_evidence(counts, depth):
    d = len(counts)
    k = sum(counts)
    if depth == 0:
        return Fraction(1, d ** k)
    assert depth == 1
    numerator = factorial(d - 1)
    for count in counts:
        numerator *= factorial(count)
    return Fraction(numerator, factorial(d + k - 1))


def batch_evidence(x, d=2, prior=(Fraction(1, 2), Fraction(1, 2))):
    if not len(x):
        return Fraction(1)
    counts = [[0] * d for _ in range(d)]
    for a, b in zip(x[:-1], x[1:]):
        counts[a][b] += 1
    evidence = Fraction(0)
    for depth, weight in enumerate(prior):
        conditional = Fraction(1, d)
        for row in counts:
            conditional *= row_evidence(row, depth)
        evidence += weight * conditional
    return evidence


class AnalyticKernel:
    """SeqLSA-compatible row interface with independent rational evidence.

    Deliberately contaminate numerical evidence for single-observation
    rows to test the required exact anchor, including its effect later.
    """

    def __init__(self, d=2, depths=(0, 1)):
        self.d = d
        self.depths = list(depths)

    def new_table(self):
        return AnalyticTable(self)


class AnalyticTable:
    def __init__(self, kernel):
        self.kernel = kernel
        self.m = np.zeros(kernel.d, dtype=int)
        self.k = 0
        self.logQ = None

    def _logQ_after(self, count):
        counts = self.m.copy()
        counts[np.flatnonzero(counts == count)[0]] += 1
        evidence = np.array([log(float(row_evidence(counts.tolist(), depth))) for depth in self.kernel.depths])
        if self.k == 0:
            evidence += np.arange(len(evidence)) * 0.2 - 0.1
        return evidence

    def observe(self, symbol):
        self.logQ = self._logQ_after(self.m[symbol])
        self.m[symbol] += 1
        self.k += 1
        return 0.0  # legacy row-local loss is intentionally not used


def test_all_binary_sequences_match_rational_batch_and_predictive():
    reference = SharedDepthLSA(AnalyticKernel())
    for n in range(1, 7):
        total = 0.0
        for x in product(range(2), repeat=n):
            result = reference.run(x, full_probs=True, depth_posteriors=True)
            expected_joint = batch_evidence(x)
            total += np.exp(result["log_evidence"][-1])
            assert np.exp(result["log_evidence"][-1]) == pytest.approx(float(expected_joint), abs=2e-15)
            assert result["loss"].sum() == pytest.approx(-log(float(expected_joint)) / log(2), abs=2e-13)
            for t in range(n):
                prefix = x[:t]
                for b in range(2):
                    expected = batch_evidence(prefix + (b,)) / batch_evidence(prefix)
                    assert result["probs"][t, b] == pytest.approx(float(expected), abs=2e-14)
                assert result["loss"][t] == pytest.approx(-np.log2(result["probs"][t, x[t]]), abs=2e-13)
            np.testing.assert_allclose(result["probs"].sum(1), 1.0, atol=3e-14)
            np.testing.assert_allclose(result["depth_posterior"].sum(1), 1.0, atol=3e-14)
        assert total == pytest.approx(1.0, abs=2e-14)


def test_shared_depth_differs_from_local_row_mixtures():
    reference = SharedDepthLSA(AnalyticKernel())
    # Store a prediction after prefix 0011 by appending its next token.
    result = reference.run([0, 0, 1, 1, 1], full_probs=True, depth_posteriors=True)
    assert result["probs"][4, 1] == pytest.approx(17 / 30)
    assert result["probs"][4, 1] != pytest.approx(7 / 12)  # row-local mixture
    np.testing.assert_allclose(result["depth_posterior"][4], [3 / 5, 2 / 5])
    # First token and each row's first observation have equal evidence at
    # every depth; quadrature noise must not update the depth posterior.
    np.testing.assert_allclose(result["depth_posterior"][:3], np.full((3, 2), 0.5))


def test_arbitrary_prior_selected_positions_and_fresh_run():
    reference = SharedDepthLSA(AnalyticKernel(), prior=[1, 3])
    x = [0, 0, 1, 0, 1, 1]
    result = reference.run(x, full_probs=True, positions=[5, 0, 3])
    np.testing.assert_array_equal(result["positions"], [0, 3, 5])
    exact = batch_evidence(x, prior=(Fraction(1, 4), Fraction(3, 4)))
    assert np.exp(result["log_evidence"][-1]) == pytest.approx(float(exact))
    np.testing.assert_allclose(result["loss"], reference.run(x)["loss"])
    np.testing.assert_allclose(result["probs"], reference.run(x, full_probs=True)["probs"][[0, 3, 5]])


def test_product_factorization_against_independent_joint_formula():
    first = SharedDepthLSA(AnalyticKernel(2), prior=[1, 3])
    second = SharedDepthLSA(AnalyticKernel(3), prior=[2, 1])
    reference = ProductSharedDepthLSA(first, second)
    x = [0, 1, 4, 5, 2, 3, 0]
    result = reference.run(x, full_probs=True, depth_posteriors=True)
    u = tuple(token // 3 for token in x)
    v = tuple(token % 3 for token in x)
    prior1 = (Fraction(1, 4), Fraction(3, 4))
    prior2 = (Fraction(2, 3), Fraction(1, 3))
    for t in range(len(x)):
        joint = batch_evidence(u[:t + 1], 2, prior1) * batch_evidence(v[:t + 1], 3, prior2)
        assert np.exp(result["log_evidence"][t]) == pytest.approx(float(joint), abs=2e-15)
        for token in range(6):
            p1 = batch_evidence(u[:t] + (token // 3,), 2, prior1) / batch_evidence(u[:t], 2, prior1)
            p2 = batch_evidence(v[:t] + (token % 3,), 3, prior2) / batch_evidence(v[:t], 3, prior2)
            assert result["probs"][t, token] == pytest.approx(float(p1 * p2), abs=2e-14)
    np.testing.assert_allclose(result["depth_posterior1"], first.run(u, depth_posteriors=True)["depth_posterior"])
    np.testing.assert_allclose(result["depth_posterior2"], second.run(v, depth_posteriors=True)["depth_posterior"])


def test_single_depth_zero_and_one_exactness():
    x = [1, 1, 1, 0, 1, 0, 0]
    zero = SharedDepthLSA(AnalyticKernel(depths=[0])).run(x, full_probs=True)
    np.testing.assert_allclose(zero["loss"], 1.0, atol=1e-14)
    one = SharedDepthLSA(AnalyticKernel(depths=[1])).run(x)
    expected = batch_evidence(x, prior=(Fraction(0), Fraction(1)))
    assert np.exp(one["log_evidence"][-1]) == pytest.approx(float(expected))


@pytest.mark.parametrize("prior", [[0, 1], [-1, 2], [1], [1, np.inf], [np.nan, 1]])
def test_invalid_prior(prior):
    with pytest.raises(ValueError):
        SharedDepthLSA(AnalyticKernel(), prior=prior)


@pytest.mark.parametrize("x", [[0, 2], [-1, 0], [0.0, 1.0], [[0, 1]], [True, False]])
def test_invalid_tokens(x):
    with pytest.raises(ValueError):
        SharedDepthLSA(AnalyticKernel()).run(x)


@pytest.mark.parametrize("positions", [[-1], [2], [1, 1], [0.5]])
def test_invalid_positions(positions):
    with pytest.raises(ValueError):
        SharedDepthLSA(AnalyticKernel()).run([0, 1], full_probs=True, positions=positions)


def test_empty_sequence_and_stable_large_prior_weights():
    reference = SharedDepthLSA(AnalyticKernel(), prior=[1e308, 1e308])
    result = reference.run([], full_probs=True, depth_posteriors=True)
    assert result["loss"].shape == (0,)
    assert result["probs"].shape == (0, 2)
    assert result["depth_posterior"].shape == (0, 2)
    np.testing.assert_allclose(reference.prior, [0.5, 0.5])


def test_insufficient_kernel_range_reports_row_total():
    kernel = AnalyticKernel()
    kernel.rmax = 1
    with pytest.raises(ValueError, match="total number of transitions"):
        SharedDepthLSA(kernel).run([0, 1, 0, 0])

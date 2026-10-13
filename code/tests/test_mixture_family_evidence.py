"""Independent finite-state checks for endpoint family evidence and alignment."""
from fractions import Fraction
from itertools import product
from math import factorial
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from mixture_family_evidence import chain_log_marginal, product_log_marginal, family_evidence


class ExactRows:
    def __init__(self, depths=(0, 1)):
        self.depths = depths

    def row(self, counts, depth):
        k, d = sum(counts), len(counts)
        if not k:
            return 0.
        if depth == 0:
            return -k * np.log(d)
        assert depth == 1
        numerator = factorial(d - 1)
        for count in counts:
            numerator *= factorial(int(count))
        return np.log(float(Fraction(numerator, factorial(d + int(k) - 1))))


def sequential_rational(tokens, d, depth):
    probability = Fraction(1, d) if tokens else Fraction(1)
    counts = [[0 for _ in range(d)] for _ in range(d)]
    for a, b in zip(tokens[:-1], tokens[1:]):
        probability *= Fraction(1, d) if depth == 0 else Fraction(counts[a][b] + 1, sum(counts[a]) + d)
        counts[a][b] += 1
    return probability


def test_chain_endpoint_matches_independent_sequential_fractions_and_normalization():
    kernel = ExactRows()
    for n in range(7):
        total = 0.
        for tokens in product(range(2), repeat=n):
            x = np.asarray(tokens, dtype=np.int64)
            observed = chain_log_marginal(x, 2, kernel)
            expected = (sequential_rational(tokens, 2, 0) + sequential_rational(tokens, 2, 1)) / 2
            assert observed == pytest.approx(np.log(float(expected)), abs=3e-14)
            total += np.exp(observed)
        assert total == pytest.approx(1., abs=2e-14)


def test_product_separate_global_depths_and_first_token():
    kernel = ExactRows()
    tokens = np.array([0, 1, 4, 5, 2, 2, 4, 1], dtype=np.int64)
    first, second = (tokens // 3).tolist(), (tokens % 3).tolist()
    for n in range(len(tokens) + 1):
        q1 = sum(sequential_rational(first[:n], 2, depth) for depth in (0, 1)) / 2
        q2 = sum(sequential_rational(second[:n], 3, depth) for depth in (0, 1)) / 2
        assert product_log_marginal(tokens[:n], 2, 3, kernel) == pytest.approx(np.log(float(q1 * q2)), abs=2e-14)
    assert product_log_marginal(tokens[:1], 2, 3, kernel) == pytest.approx(-np.log(6))


def test_global_depth_is_not_row_local_mixture():
    kernel = ExactRows()
    tokens = np.array([0, 0, 1, 1, 1])
    global_evidence = chain_log_marginal(tokens, 2, kernel)
    rows = ([1, 1], [0, 2])
    local_evidence = -np.log(2) + sum(np.log((np.exp(kernel.row(row, 0)) + np.exp(kernel.row(row, 1))) / 2) for row in rows)
    assert abs(global_evidence - local_evidence) > 1e-3


@pytest.mark.parametrize("tokens", [np.array([0, -1]), np.array([0, 2]), np.array([0., 1.]), np.array([[0, 1]])])
def test_invalid_chain_tokens(tokens):
    with pytest.raises(ValueError):
        chain_log_marginal(tokens, 2, ExactRows())


def test_family_evidence_requires_matching_training_depths():
    with pytest.raises(ValueError, match="training prior"):
        family_evidence(np.array([0, 1, 1]), np.array([1, 2]), "uniform_base", ExactRows())


def test_prediction_times_score_exactly_t_tokens():
    class IdenticalUniformRows:
        depths = tuple(range(9))

        def row(self, counts, depth):
            return -sum(counts) * np.log(len(counts))

    times = np.array([1, 4, 16])
    marginal, losses = family_evidence(np.arange(17), times, "uniform_base", IdenticalUniformRows())
    np.testing.assert_allclose(marginal, -(times + 1) * np.log(64), atol=5e-14)
    np.testing.assert_allclose(losses, times * 6, atol=1e-13)

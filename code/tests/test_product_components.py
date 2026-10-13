"""Independent checks of the product-chain predictive backoff decomposition.

Exact rational row laws isolate the counting, posterior and mixture logic
from the production moment tables. Depths zero and one are the actual LSA
anchors. The explicitly marked synthetic depth-two law is Dirichlet(1/2),
used only to exercise the unanchored virtual-floor interface.
"""
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from shared_depth import SharedDepthLSA, ProductSharedDepthLSA
from product_components import row_components, coordinate_components, product_components


def rising(value, n):
    result = Fraction(1)
    for j in range(n):
        result *= value + j
    return result


def evidence(counts, depth):
    d, n = len(counts), sum(counts)
    if depth == 0:
        return Fraction(1, d ** n)
    shape = Fraction(1) if depth == 1 else Fraction(1, 2)
    return math.prod(rising(shape, count) for count in counts) / rising(d * shape, n)


class ExactKernel:
    def __init__(self, d=2, depths=(0, 1)):
        self.d = d
        self.depths = list(depths)
        self.rmax = 100

    def new_table(self):
        return ExactTable(self)


class ExactTable:
    def __init__(self, kernel):
        self.S = kernel
        self.k = 0
        self.m = np.zeros(kernel.d, dtype=int)
        self.logQ = None
        self.virtual_counts_requested = []

    def _logQ_after(self, count):
        # This remains meaningful for count=0 even when every actual label
        # has positive count. It then represents the virtual moment ratio.
        self.virtual_counts_requested.append(int(count))
        result = []
        for depth in self.S.depths:
            if depth == 0:
                probability = Fraction(1, self.S.d)
            else:
                shape = Fraction(1) if depth == 1 else Fraction(1, 2)
                probability = (count + shape) / (self.k + self.S.d * shape)
            result.append(math.log(float(evidence(self.m.tolist(), depth) * probability)))
        return np.array(result)

    def observe(self, symbol):
        self.logQ = self._logQ_after(int(self.m[symbol]))
        self.m[symbol] += 1
        self.k += 1
        return 0.0  # SharedDepthLSA uses per-depth evidence, not this value.


def exact_state(prefix, d, depths, prior=None):
    """Compute the entire-prefix posterior directly, without sequential code."""
    counts = [[0] * d for _ in range(d)]
    for a, b in zip(prefix[:-1], prefix[1:]):
        counts[a][b] += 1
    weights = [Fraction(1)] * len(depths) if prior is None else list(map(Fraction, prior))
    weights = [w * math.prod(evidence(row, depth) for row in counts)
               for w, depth in zip(weights, depths)]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    row = counts[prefix[-1]] if prefix else [0] * d
    n = sum(row)
    probability, floor = [Fraction(0)] * d, Fraction(0)
    for weight, depth in zip(weights, depths):
        if depth == 0 or n == 0:
            per_depth = [Fraction(1, d)] * d
            per_floor = Fraction(1, d)
        else:
            shape = Fraction(1) if depth == 1 else Fraction(1, 2)
            per_depth = [(count + shape) / (n + d * shape) for count in row]
            per_floor = shape / (n + d * shape)
        floor += weight * per_floor
        probability = [p + weight * q for p, q in zip(probability, per_depth)]
    lam = 1 - d * floor
    context = [(p - floor) / lam for p in probability] if lam else [Fraction(1, d)] * d
    return dict(counts=np.array(row), posterior=np.array(list(map(float, weights))),
                probs=np.array(list(map(float, probability))), floor=float(floor),
                weight=float(lam), context=np.array(list(map(float, context))))


def table_with_counts(reference, counts):
    table = reference.kernel.new_table()
    for symbol, count in enumerate(counts):
        for _ in range(count):
            table.observe(symbol)
    return table


@pytest.mark.parametrize("counts", [(0, 0, 0), (4, 0, 2), (2, 2, 2), (1, 2, 5)])
def test_depth_one_exact_floor_and_empirical_continuation(counts):
    reference = SharedDepthLSA(ExactKernel(3, (1,)))
    table = table_with_counts(reference, counts)
    result = row_components(reference, table, np.zeros(1))
    n = sum(counts)
    np.testing.assert_allclose(result["prediction_probs"], (np.array(counts) + 1) / (n + 3), atol=2e-14)
    assert result["uniform_floor"] == pytest.approx(1 / (n + 3), abs=2e-14)
    assert result["context_weight"] == pytest.approx(n / (n + 3), abs=2e-14)
    assert bool(result["context_defined"]) == (n > 0)
    expected_context = np.array(counts) / n if n else np.full(3, 1 / 3)
    np.testing.assert_allclose(result["context_probs"], expected_context, atol=2e-14)
    np.testing.assert_allclose(result["uniform_probs"], np.full(3, 1 / 3), atol=2e-14)


def test_uniform_prediction_does_not_identify_a_zero_copy_weight():
    reference = SharedDepthLSA(ExactKernel(3, (1,)))
    result = row_components(reference, table_with_counts(reference, [2, 2, 2]), np.zeros(1))
    assert result["context_weight"] == pytest.approx(2 / 3)
    assert result["context_defined"]
    np.testing.assert_allclose(result["prediction_probs"], result["uniform_probs"], atol=2e-14)
    np.testing.assert_allclose(result["context_probs"], result["uniform_probs"], atol=2e-14)


@pytest.mark.parametrize("counts", [(0, 0, 0), (3, 0, 1), (1, 2, 3)])
def test_depth_zero_has_zero_context_weight_even_after_repetitions(counts):
    reference = SharedDepthLSA(ExactKernel(3, (0,)))
    result = row_components(reference, table_with_counts(reference, counts), np.zeros(1))
    assert result["uniform_floor"] == pytest.approx(1 / 3)
    assert result["context_weight"] == 0
    assert not result["context_defined"]
    np.testing.assert_allclose(result["prediction_probs"], np.full(3, 1 / 3), atol=2e-14)
    assert np.isfinite(result["context_probs"]).all()


def test_all_seen_virtual_floor_is_below_every_actual_probability():
    # Synthetic Gamma(1/2) weights exercise the higher-depth interface only;
    # no identity between that law and the actual depth-two LSA is assumed.
    reference = SharedDepthLSA(ExactKernel(3, (2,)))
    table = table_with_counts(reference, [1, 2, 3])
    table.virtual_counts_requested.clear()
    result = row_components(reference, table, np.zeros(1))
    assert 0 in table.virtual_counts_requested
    assert result["uniform_floor"] == pytest.approx(0.5 / 7.5, abs=2e-14)
    assert result["uniform_floor"] < min(result["prediction_probs"])
    np.testing.assert_allclose(result["context_probs"], np.array([1, 2, 3]) / 6, atol=2e-14)
    assert result["context_weight"] == pytest.approx(6 / 7.5, abs=2e-14)


def test_global_depth_mixture_uses_weighted_floors_and_copy_excesses():
    reference = SharedDepthLSA(ExactKernel(3, (0, 1, 2)))
    weights = np.array([0.2, 0.3, 0.5])
    counts = np.array([4, 0, 2])
    n = counts.sum()
    result = row_components(reference, table_with_counts(reference, counts), np.log(weights))
    floors = np.array([1 / 3, 1 / (n + 3), 0.5 / (n + 1.5)])
    probabilities = np.array([np.full(3, 1 / 3), (counts + 1) / (n + 3), (counts + 0.5) / (n + 1.5)])
    np.testing.assert_allclose(result["depth_floors"], floors, atol=2e-14)
    np.testing.assert_allclose(result["prediction_probs"], weights @ probabilities, atol=2e-14)
    assert result["uniform_floor"] == pytest.approx(weights @ floors, abs=2e-14)
    assert result["context_weight"] == pytest.approx(weights @ (1 - 3 * floors), abs=2e-14)
    assert result["context_probs"][1] == pytest.approx(0, abs=2e-14)
    np.testing.assert_allclose(result["context_probs"], counts / n, atol=2e-14)


def test_depth_averaged_context_uses_copy_mass_not_plain_depth_weights():
    # A second synthetic positive iid weight law makes its context vector
    # differ from the Dirichlet one, exposing an incorrect plain average.
    def finite_evidence(counts):
        terms = []
        for weights in product((1, 4), repeat=len(counts)):
            denominator = sum(weights)
            terms.append(math.prod(Fraction(w, denominator) ** count
                                   for w, count in zip(weights, counts)))
        return sum(terms) / len(terms)

    class FiniteTable(ExactTable):
        def _logQ_after(self, count):
            candidate = self.m.copy()
            candidate[np.flatnonzero(candidate == count)[0]] += 1
            return np.array([math.log(float(evidence(candidate.tolist(), 1))),
                             math.log(float(finite_evidence(candidate.tolist())))])

    kernel = ExactKernel(3, (1, 2))
    kernel.new_table = lambda: FiniteTable(kernel)
    reference = SharedDepthLSA(kernel)
    counts = np.array([3, 0, 1])
    weights = np.array([0.4, 0.6])
    result = row_components(reference, table_with_counts(reference, counts), np.log(weights))
    per_depth = [(counts + 1) / (counts.sum() + 3)]
    finite_before = finite_evidence(counts)
    finite_prediction = []
    for b in range(3):
        after = counts.copy()
        after[b] += 1
        finite_prediction.append(float(finite_evidence(after) / finite_before))
    per_depth = np.array(per_depth + [finite_prediction])
    floors = per_depth[:, 1]
    lambdas = 1 - 3 * floors
    contexts = (per_depth - floors[:, None]) / lambdas[:, None]
    expected = (weights * lambdas) @ contexts / (weights @ lambdas)
    assert np.max(np.abs(expected - weights @ contexts)) > 1e-4
    np.testing.assert_allclose(result["context_probs"], expected, atol=3e-14)
    np.testing.assert_allclose(result["prediction_probs"], weights @ per_depth, atol=3e-14)


def test_every_small_binary_prefix_matches_direct_joint_posterior():
    reference = SharedDepthLSA(ExactKernel(2, (0, 1)), prior=[1, 3])
    for path in product(range(2), repeat=7):
        result = coordinate_components(reference, np.array(path))
        for j, t in enumerate(range(1, len(path))):
            expected = exact_state(path[:t], 2, (0, 1), [1, 3])
            np.testing.assert_array_equal(result["row_counts"][j], expected["counts"])
            np.testing.assert_allclose(result["depth_posterior"][j], expected["posterior"], atol=3e-14)
            np.testing.assert_allclose(result["prediction_probs"][j], expected["probs"], atol=3e-14)
            np.testing.assert_allclose(result["context_probs"][j], expected["context"], atol=3e-14)
            assert result["context_weight"][j] == pytest.approx(expected["weight"], abs=3e-14)


def test_other_rows_update_shared_depth_posterior_but_empty_current_row_stays_uniform():
    reference = SharedDepthLSA(ExactKernel(2, (0, 1)))
    prefix = (0,) * 9 + (1,)
    result = coordinate_components(reference, np.array(prefix + (1,)), positions=[len(prefix)])
    expected = exact_state(prefix, 2, (0, 1))
    assert expected["posterior"][1] > 0.7
    np.testing.assert_allclose(result["depth_posterior"][0], expected["posterior"], atol=2e-14)
    np.testing.assert_array_equal(result["row_counts"][0], [0, 0])
    assert result["context_weight"][0] == 0
    np.testing.assert_allclose(result["prediction_probs"][0], [0.5, 0.5], atol=2e-14)


def test_posterior_is_global_not_refitted_only_to_the_current_row():
    reference = SharedDepthLSA(ExactKernel(2, (0, 1)))
    prefix = (0,) * 9 + (1, 1)
    result = coordinate_components(reference, np.array(prefix + (0,)), positions=[len(prefix)])
    expected = exact_state(prefix, 2, (0, 1))
    # A row with just one observation leaves a row-local prior at (1/2,1/2).
    assert abs(expected["posterior"][1] - 0.5) > 0.1
    np.testing.assert_allclose(result["depth_posterior"][0], expected["posterior"], atol=2e-14)
    np.testing.assert_allclose(result["prediction_probs"][0], expected["probs"], atol=2e-14)


def test_selected_prediction_excludes_its_target_and_all_later_tokens():
    reference = SharedDepthLSA(ExactKernel(2, (0, 1)))
    prefix = (0, 0, 1, 0, 1, 1)
    first = coordinate_components(reference, np.array(prefix + (0, 0, 0)), positions=[len(prefix)])
    second = coordinate_components(reference, np.array(prefix + (1, 0, 1, 1)), positions=[len(prefix)])
    for key in ("prediction_probs", "context_probs", "context_weight", "uniform_floor", "depth_posterior", "row_counts"):
        np.testing.assert_array_equal(first[key], second[key])


def test_product_four_weights_and_joint_law_match_independent_coordinate_evidence():
    first = SharedDepthLSA(ExactKernel(2, (0, 1)), prior=[1, 3])
    second = SharedDepthLSA(ExactKernel(3, (0, 1)), prior=[2, 1])
    u = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0])
    v = np.array([2, 1, 2, 0, 2, 2, 1, 0, 1])
    x = u * 3 + v
    result = product_components(first, second, x)
    assert tuple(result["component_names"]) == ("UU", "CU", "UC", "CC")
    direct = ProductSharedDepthLSA(first, second).run(x, full_probs=True, positions=np.arange(1, len(x)))
    np.testing.assert_allclose(result["joint_probs"], direct["probs"], atol=3e-14)
    np.testing.assert_allclose(result["reconstructed_probs"], direct["probs"], atol=3e-14)
    for j, t in enumerate(range(1, len(x))):
        e1 = exact_state(tuple(u[:t]), 2, (0, 1), [1, 3])
        e2 = exact_state(tuple(v[:t]), 3, (0, 1), [2, 1])
        a, b = e1["weight"], e2["weight"]
        np.testing.assert_allclose(result["weights"][j], [(1-a)*(1-b), a*(1-b), (1-a)*b, a*b], atol=3e-14)
        np.testing.assert_allclose(result["joint_probs"][j], np.outer(e1["probs"], e2["probs"]).ravel(), atol=3e-14)
    assert result["weights"].min() >= -2e-14
    np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=2e-14)
    np.testing.assert_allclose(result["joint_probs"].sum(axis=1), 1, atol=2e-14)
    np.testing.assert_array_equal(result["weights"][0], [1, 0, 0, 0])


@pytest.mark.parametrize("tokens", [[], [2]])
def test_no_scored_predictions_preserves_empty_product_shapes(tokens):
    first = SharedDepthLSA(ExactKernel(2, (0, 1)))
    second = SharedDepthLSA(ExactKernel(3, (0, 1)))
    result = product_components(first, second, np.asarray(tokens, dtype=int))
    assert result["weights"].shape == (0, 4)
    assert result["component_probs"].shape == (0, 4, 6)
    assert result["joint_probs"].shape == (0, 6)
    assert result["reconstructed_probs"].shape == (0, 6)
    assert result["reconstruction_error"] == 0


def test_material_probability_error_is_rejected_without_normalization():
    reference = SharedDepthLSA(ExactKernel(3, (2,)))
    table = table_with_counts(reference, [2, 0, 1])
    original = table._logQ_after
    table._logQ_after = lambda count: original(count) + math.log(1.01)
    with pytest.raises(FloatingPointError, match="not normalized"):
        row_components(reference, table, np.zeros(1))

"""Independent labelled-sequence checks for the short family forecast."""
import itertools
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from section4_family_forecast import canonical, enumerate_curves, orbits, summarize


class ToyFamily:
    """Three normalized invariant sources, with genuinely different laws."""
    def __init__(self, kind):
        self.kind = kind

    def log_joint(self, tokens):
        x = np.asarray(tokens, dtype=int)
        if self.kind == "iid":
            return -len(x) * math.log(4)
        if self.kind == "product":
            result = -math.log(4)
            for a, b in zip(x[:-1], x[1:]):
                result += math.log(.8 if a // 2 == b // 2 else .2)
                result += math.log(.6 if a % 2 == b % 2 else .4)
            return result
        counts = np.zeros((4, 4), dtype=int)
        for a, b in zip(x[:-1], x[1:]):
            counts[a, b] += 1
        return -math.log(4) + sum(gammaln(4) - gammaln(4 + sum(row))
                                 + sum(gammaln(row + 1)) for row in counts)


@pytest.mark.parametrize("length", range(1, 6))
def test_orbits_partition_every_labelled_sequence(length):
    representatives, weights = orbits(length, 2, 2)
    actual = Counter()
    for tokens in itertools.product(range(4), repeat=length):
        actual[(canonical(np.asarray(tokens) // 2), canonical(np.asarray(tokens) % 2))] += 1
    expected = {(tuple(x // 2), tuple(x % 2)): int(w) for x, w in zip(representatives, weights)}
    assert dict(actual) == expected
    assert sum(weights) == 4 ** length


def test_population_metrics_equal_direct_labelled_enumeration():
    references = [ToyFamily(kind) for kind in ("product", "dirichlet", "iid")]
    compressed, _ = enumerate_curves(references, 2, 2, maximum_predictions=4)
    for length, result in enumerate(compressed, 1):
        all_tokens = list(itertools.product(range(4), repeat=length))
        probabilities = np.array([[math.exp(r.log_joint(x)) for r in references] for x in all_tokens])
        mixture = probabilities.mean(axis=1)
        costs = np.sum(probabilities * np.log2(probabilities / mixture[:, None]), axis=0)
        posteriors = probabilities / probabilities.sum(axis=1, keepdims=True)
        entropy = -np.sum(mixture[:, None] * posteriors * np.log2(posteriors))
        accuracy = np.sum(np.max(probabilities, axis=1)) / 3
        np.testing.assert_allclose(probabilities.sum(axis=0), 1, atol=3e-14, rtol=0)
        np.testing.assert_allclose(result["family_cost_bits"], costs, atol=3e-14, rtol=0)
        assert abs(result["posterior_entropy_bits"] - entropy) < 3e-14
        assert abs(result["optimal_family_classification_accuracy"] - accuracy) < 3e-14


def test_coordinate_orbits_do_not_collapse_distinct_joint_patterns():
    reference = ToyFamily("product")
    # Both have two distinct joint symbols but evidence differs by coordinate.
    assert canonical([0, 1]) == canonical([0, 2])
    assert abs(reference.log_joint([0, 1]) - reference.log_joint([0, 2])) > .1
    tokens, _ = orbits(2, 2, 2)
    assert [0, 1] in tokens.tolist() and [0, 2] in tokens.tolist()


def test_bad_probability_mass_is_rejected_and_small_residual_is_retained():
    logs = np.log(np.array([[.4, .3, .2], [.6, .7, .8]]))
    with pytest.raises(ValueError, match="probability mass"):
        summarize(logs + math.log(2), np.ones(2))
    residual = 1e-10
    result = summarize(logs + math.log1p(residual), np.ones(2))
    np.testing.assert_allclose(result["family_probability_mass"], 1 + residual, atol=1e-15, rtol=0)
    assert result["total_mixture_probability_mass"] > 1


"""Independent finite-alphabet checks of the equality-pattern calculation."""
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from conditional_crossing_exact import horizon_summary, multiplicity, restricted_growth_patterns


def row_probability(counts, depth):
    d = len(counts)
    if depth == 0:
        return Fraction(1, d ** sum(counts))
    assert depth == 1
    return Fraction(math.factorial(d - 1) * math.prod(math.factorial(c) for c in counts),
                    math.factorial(d + sum(counts) - 1))


@pytest.mark.parametrize("d", [2, 3, 7])
def test_orbit_multiplicities_partition_every_labelled_prefix(d):
    for n in range(1, 8):
        patterns = list(restricted_growth_patterns(n, d))
        assert len(set(patterns)) == len(patterns)
        assert sum(multiplicity(d, len(set(x))) for x in patterns) == d ** (n - 1)


@pytest.mark.parametrize("d", [2, 3])
def test_orbit_sums_match_independent_full_label_enumeration(d):
    def evidence(profile):
        counts = list(profile) + [0] * (d - len(profile))
        return np.array([math.log(row_probability(counts, depth)) for depth in (0, 1)])

    for n in range(1, 7):
        terms = []
        masses = [Fraction(0)] * 3
        for rest in product(range(d), repeat=n - 1):
            x = (0,) + rest
            counts = [x.count(b) for b in range(d)]
            rows = [[0] * d for _ in range(d)]
            for a, b in zip(x[:-1], x[1:]):
                rows[a][b] += 1
            p = math.prod(row_probability(row, 1) for row in rows)
            b = (math.prod(row_probability(row, 0) for row in rows) + p) / 2
            u = d * row_probability(counts, 1)
            for j, value in enumerate((p, b, u)):
                masses[j] += value
            terms.append((float(p) * math.log2(p / u), float(p) * math.log2(p / b)))
        assert masses == [1, 1, 1]
        result = horizon_summary(n, d, evidence, source_depth=1,
                                 bayes_depths=(0, 1), unigram_depths=(1,))
        a = math.fsum(t[0] for t in terms)
        divergence = math.fsum(t[1] for t in terms)
        assert result["A_marginal_divergence_bits"] == pytest.approx(a, abs=2e-14)
        assert result["D_marginal_divergence_bits"] == pytest.approx(divergence, abs=2e-14)
        assert result["cumulative_gap_bits"] == pytest.approx(a - divergence, abs=2e-14)
        for key in ("source_probability_mass", "bayes_probability_mass", "unigram_probability_mass"):
            assert result[key] == pytest.approx(1, abs=2e-14)
        if n == 2:
            repeat = 2 / (d + 1)
            analytic = -(math.log2(d * repeat) + (d - 1) * math.log2(d * (1 - repeat) / (d - 1))) / d
            assert result["D_marginal_divergence_bits"] == pytest.approx(0, abs=2e-14)
            assert result["cumulative_gap_bits"] == pytest.approx(analytic, abs=2e-14)

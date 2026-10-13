"""Independent exact integration and symmetry tests for hidden-factor Bayes."""
from itertools import permutations, product
from math import lgamma
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from hidden_factor import (draw_batch, draw_hidden_factor_batch, encoding_count,
                           encodings, exact_curve, flat_curve, iter_encodings,
                           known_encoding_curve, true_curve)


def direct_marginal(xs, candidates, dimensions):
    """Integrate count monomials by Dirichlet gamma functions, independently."""
    r, s = dimensions
    d = r*s
    if len(xs) == 0:
        return 1.0
    inverse = np.argsort(candidates, axis=1)
    mapped = inverse[:, xs]
    log_prob = np.full(len(candidates), -np.log(d))
    ids = np.arange(len(candidates))
    for states, k in ((mapped // s, r), (mapped % s, s)):
        counts = np.zeros((len(candidates), k, k), dtype=int)
        for t in range(1, len(xs)):
            np.add.at(counts, (ids, states[:, t-1], states[:, t]), 1)
        gamma = np.array([lgamma(v+1) for v in range(len(xs)+k+1)])
        log_prob += (lgamma(k)-gamma[counts.sum(axis=-1)+k-1] + gamma[counts].sum(axis=-1)).sum(axis=-1)
    return np.exp(log_prob).mean()


@pytest.mark.parametrize("dimensions, expected", [((2, 2), 3), ((3, 3), 5040),
                                                   ((2, 5), 15120), ((2, 6), 332640),
                                                   ((3, 4), 3326400), ((2, 7), 8648640),
                                                   ((2, 8), 259459200)])
def test_encoding_count(dimensions, expected):
    assert encoding_count(dimensions) == expected


@pytest.mark.parametrize("dimensions", [(2, 2), (2, 3), (3, 3)])
def test_quotient_matches_all_permutations_and_gamma_integral(dimensions):
    d = int(np.prod(dimensions))
    candidates = np.array(list(permutations(range(d))), dtype=np.uint8)
    xs = np.array([0, d-1, 0, 1, d-1], dtype=int)
    times = [0, 1, len(xs)]
    curve = exact_curve(xs, times=times, dimensions=dimensions, chunk_size=997)
    for t in range(len(xs)+1):
        expected = direct_marginal(xs[:t], candidates, dimensions)
        assert curve["log_evidence_bits"][t] == pytest.approx(np.log2(expected), abs=2e-13)
    parent = direct_marginal(xs, candidates, dimensions)
    for next_token in (0, d-1):
        child = direct_marginal(np.append(xs, next_token), candidates, dimensions)
        assert curve["predictive"][-1, next_token] == pytest.approx(child/parent, abs=2e-14)
    np.testing.assert_allclose(curve["predictive"].sum(axis=1), 1, atol=2e-14)
    np.testing.assert_allclose(curve["predictive"][0], 1/d, atol=2e-14)


def test_small_prior_normalizes_all_ordered_sequences():
    total = 0.0
    for xs in product(range(4), repeat=4):
        curve = exact_curve(np.array(xs), times=[], dimensions=(2, 2))
        assert curve["predictive"].shape == (0, 4)
        total += 2**curve["log_evidence_bits"][-1]
    assert total == pytest.approx(1, abs=1e-13)


def test_chunking_iterator_and_label_equivariance():
    dimensions = (2, 3)
    xs = np.array([0, 1, 0, 3, 4, 3, 0, 5, 5, 0])
    label_map = np.array([3, 0, 5, 2, 1, 4])
    a = exact_curve(xs, dimensions=dimensions, chunk_size=7)
    b = exact_curve(label_map[xs], dimensions=dimensions, chunk_size=50)
    np.testing.assert_allclose(a["log_evidence_bits"], b["log_evidence_bits"], atol=1e-13)
    np.testing.assert_allclose(a["predictive"], b["predictive"][:, label_map], atol=1e-14)
    chunks = list(iter_encodings(dimensions, 7))
    np.testing.assert_array_equal(np.concatenate(chunks), encodings(dimensions))
    assert all(len(chunk) <= 7 for chunk in chunks)
    assert len(np.unique(np.concatenate(chunks), axis=0)) == encoding_count(dimensions)


def test_source_and_known_encoding_reference():
    dimensions = (2, 3)
    rng = np.random.default_rng(139)
    batch = draw_batch(rng, 3, 25, dimensions)
    repeat = draw_batch(np.random.default_rng(139), 3, 25, dimensions)
    np.testing.assert_array_equal(batch["xs"], repeat["xs"])
    np.testing.assert_array_equal(batch["xs"], draw_hidden_factor_batch(np.random.default_rng(139), 3, 25, *dimensions))
    for i in range(3):
        xs, encoding = batch["xs"][i], batch["encoding"][i]
        matrices = tuple(matrix[i] for matrix in batch["matrices"])
        source = true_curve(xs, matrices, encoding, dimensions=dimensions)
        known = known_encoding_curve(xs, encoding, dimensions=dimensions)
        hidden = exact_curve(xs, dimensions=dimensions)
        direct = direct_marginal(xs, encoding[None, :], dimensions)
        assert known["log_evidence_bits"][-1] == pytest.approx(np.log2(direct), abs=2e-13)
        inverse = np.argsort(encoding)
        transition = np.kron(*matrices)[inverse[:, None], inverse[None, :]]
        np.testing.assert_allclose(source["predictive"][1:], transition[xs], atol=2e-15)
        np.testing.assert_allclose(source["log_evidence_bits"][1:], np.cumsum(np.log2(source["predictive"][np.arange(len(xs)), xs])), atol=1e-13)
        for curve in (known, hidden):
            realized = np.cumsum(np.log2(curve["predictive"][np.arange(len(xs)), xs]))
            np.testing.assert_allclose(curve["log_evidence_bits"][1:], realized, atol=1e-13)
        # Finite-mixture pointwise bound versus any one encoding, including true.
        assert np.all(known["log_evidence_bits"]-hidden["log_evidence_bits"] <= np.log2(encoding_count(dimensions))+1e-12)


def test_flat_prior_has_explicit_per_cell_concentration():
    xs = np.array([0, 0, 1, 0, 1, 1, 0])
    curve = flat_curve(xs, dimensions=(2, 2), concentration=.25)
    counts = np.zeros((4, 4))
    evidence = 1/4
    for t in range(1, len(xs)+1):
        if t > 1:
            a, b = xs[t-2:t]
            evidence *= (counts[a, b]+.25)/(counts[a].sum()+1)
            counts[a, b] += 1
        np.testing.assert_allclose(curve["predictive"][t], (counts[xs[t-1]]+.25)/(counts[xs[t-1]].sum()+1), atol=1e-15)
        assert curve["log_evidence_bits"][t] == pytest.approx(np.log2(evidence), abs=1e-13)


@pytest.mark.parametrize("kwargs", [{"times": [1, 1]}, {"times": [-1]},
                                  {"times": [4]}, {"dimensions": (1, 3)},
                                  {"chunk_size": 0}])
def test_invalid_reference_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        exact_curve(np.array([0, 1, 0]), **kwargs)

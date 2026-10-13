"""Independent checks of the controlled experiment's contrasts and grouping."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"transformer"))
from copier_probes import make_probes
from identify_copier import contrasts, query_inputs, probability_summaries


def test_query_group_order_and_histogram_control():
    p = make_probes(5, seed=902)
    x = query_inputs(p, 1, 4).reshape(6, 3, 64)
    assert np.array_equal(x[0], p["clean"][1:4])
    assert np.array_equal(x[1], p["counterfactual"][1:4])
    assert np.array_equal(x[2, :, -1], p["context_d"][1:4])
    assert np.array_equal(x[4, :, -1], p["control_e"][1:4])
    for g in (0, 2, 4):
        assert np.array_equal(np.sort(x[g], axis=1), np.sort(x[g+1], axis=1))


def test_independent_odds_ratio_calculation_and_query_reversal():
    p = make_probes(5, seed=901)
    rng = np.random.default_rng(83)
    logits = rng.normal(size=(5, 6, 64))
    actual = contrasts(logits, p)
    prob = np.exp(logits); prob /= prob.sum(axis=-1, keepdims=True)
    for i in range(5):
        b, c = p["target_b"][i], p["target_c"][i]
        responses = []
        for g in (0, 2, 4):
            ratio = (prob[i, g, b]/prob[i, g, c])/(prob[i, g+1, b]/prob[i, g+1, c])
            responses.append(np.log2(ratio))
        for k, value in zip(("A", "D", "E"), responses):
            np.testing.assert_allclose(actual[k][i], value, atol=1e-14)
        np.testing.assert_allclose(actual["query_specific"][i], responses[0]-responses[2], atol=1e-14)
    swapped = logits[:, [1, 0, 3, 2, 5, 4]]
    for k in actual:
        np.testing.assert_allclose(contrasts(swapped, p)[k], -actual[k], atol=1e-14)


def test_successor_probabilities_use_opposite_targets_in_paired_conditions():
    p = make_probes(3, seed=73)
    logits = np.zeros((3, 6, 64))
    for i in range(3):
        logits[i, 0, p["target_b"][i]] = np.log(3)
        logits[i, 1, p["target_c"][i]] = np.log(3)
    stats = probability_summaries(logits, p)
    np.testing.assert_allclose(stats["demonstrated_successor"]["mean"], 3/66)
    np.testing.assert_allclose(stats["other_successor"]["mean"], 1/66)

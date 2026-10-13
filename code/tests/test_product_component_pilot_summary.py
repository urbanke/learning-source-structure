"""Pure-array checks of paired estimation and observable transition semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import product_component_pilot_summary as summary


def test_paired_statistics_use_identical_draws_and_paired_uncertainty():
    bayes = np.array([[.1, .2, .3], [.7, .4, .6], [.9, .8, .1]])
    transformer = bayes + np.array([[.1, .05, .2], [.1, .3, .1], [-.7, .1, .2]])
    eligible = np.array([[True, True, False], [True, False, False], [False, False, False]])
    out = summary.paired_statistics(bayes, transformer, eligible)
    assert out["bayes"]["count"] == [2, 1, 0]
    assert out["transformer"]["count"] == [2, 1, 0]
    assert out["bayes"]["mean"] == pytest.approx([.4, .2, None])
    assert out["transformer"]["mean"] == pytest.approx([.5, .25, None])
    assert out["transformer_minus_bayes"]["mean"] == pytest.approx([.1, .05, None])
    assert out["bayes"]["se"][0] == pytest.approx(.3)
    # Perfectly paired offsets have zero sampling error even when each mean
    # is uncertain. One or zero eligible draws cannot estimate an SE.
    assert out["transformer_minus_bayes"]["se"][0] == pytest.approx(0, abs=1e-15)
    assert out["transformer_minus_bayes"]["se"][1:] == [None, None]


def test_first_observed_positive_is_not_automatically_a_transition():
    out = summary.crossing_record(np.array([0, 1, 1, 0]), np.arange(1, 5))
    assert out["first_positive"] == 2
    assert out["first_positive_after_negative"] is None
    assert out["first_switch_bracket"] is None
    assert out["status"] == "above_observed_without_prior_below"
    assert out["unresolved_before_first_positive"] == 1
    assert out["observed_resolved_state_upcrossings"] == 0


def test_unknown_gaps_and_later_recrossings_remain_explicit():
    out = summary.crossing_record(np.array([-1, 0, 1, 0, -1, 0, 1]), np.arange(1, 8))
    assert out["status"] == "observed_below_to_above"
    assert out["first_positive_after_negative"] == 3
    assert out["first_switch_bracket"] == dict(direction="up", left=1, right=3, unresolved_between=1)
    assert out["observed_resolved_state_upcrossings"] == 2
    assert out["observed_resolved_state_downcrossings"] == 1
    assert out["unresolved_count"] == 3
    assert out["transitions"][1] == dict(direction="down", left=3, right=5, unresolved_between=1)
    assert summary.crossing_record(np.zeros(3, dtype=int), np.arange(1, 4))["status"] == "unresolved_throughout"


def test_tolerance_radius_and_degenerate_components_do_not_force_a_status():
    radius = summary.robustness_radius(np.array([0, .001, 1.]), np.array([1., .5, 0]), .002)
    np.testing.assert_allclose(radius, [np.sqrt(3)*.001, np.sqrt(3)*.004, 1.])
    np.testing.assert_array_equal(summary.robust_states(np.array([.51, .49, .99]),
                                                       np.array([.02, .02, 1.]), np.ones(3, dtype=bool)), [0, 0, 0])
    np.testing.assert_array_equal(summary.robust_states(np.array([.9, .1, .9]),
                                                       np.full(3, .1), np.array([True, True, False])), [1, -1, 0])


def test_full_copy_dominance_is_largest_component_not_majority_mass():
    weights = np.array([[.2, .2, .2, .4], [.6, .1, .1, .2], [.25, .25, .25, .25],
                        [.2, .2, .2, .4], [.6, .1, .1, .2], [.2, .2, .2, .4]])
    residual = np.array([0, 0, 0, 0, .03, 0])
    sigma = np.array([.1, .1, .1, 0, .1, .1])
    eligible = np.array([True, True, True, True, True, False])
    states = summary.full_copy_states(weights, residual, sigma, eligible, .001)
    # CC can dominate below one half. Ties, degeneracy, a large residual,
    # or missing component identification all prevent a robust decision.
    np.testing.assert_array_equal(states, [1, -1, 0, 0, 0, 0])

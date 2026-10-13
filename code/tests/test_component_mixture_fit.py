"""Independent algebraic checks for scalar component fits and held-out shape."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"transformer"))
from component_mixture_fit import fit_component_mixture, fit_support_mass, token_half_cross_validation


def test_exact_mixture_recovery_and_complementary_raw_token_halves():
    u = np.tile([.25, .25, .25, .25], (4, 1))
    c = np.tile([.4, .3, .2, .1], (4, 1))
    w = np.array([0., .2, .7, 1.])
    t = (1-w[:, None])*u+w[:, None]*c
    original = [v.copy() for v in (t, u, c)]
    result = fit_component_mixture(t, u, c)
    np.testing.assert_allclose(result["weights"], w, atol=1e-15)
    np.testing.assert_allclose(result["fitted_probs"], t, atol=1e-15)
    np.testing.assert_allclose(result["squared_error"], 0, atol=1e-30)
    np.testing.assert_allclose(result["cross_validation"]["weights"], np.repeat(w[:, None], 2, axis=1), atol=1e-15)
    np.testing.assert_allclose(result["cross_validation"]["heldout_squared_error"], 0, atol=1e-30)
    assert result["identifiable"].all() and result["cross_validation"]["both_halves_identifiable"].all()
    for actual, before in zip((t, u, c), original):
        np.testing.assert_array_equal(actual, before)


def test_outside_hull_coefficients_are_clipped_to_correct_endpoints():
    u = np.tile([.3, .3, .2, .2], (2, 1))
    c = np.tile([.4, .2, .2, .2], (2, 1))
    t = u+np.array([1.5, -.5])[:, None]*(c-u)
    result = fit_component_mixture(t, u, c)
    np.testing.assert_array_equal(result["weights"], [1., 0.])
    np.testing.assert_allclose(result["fitted_probs"], np.stack([c[0], u[1]]))
    np.testing.assert_allclose(result["squared_error"], [.005, .005], atol=1e-15)


def test_orthogonal_residual_remains_and_error_matches_geometry():
    u = np.array([[.25]*4]); c = np.array([[.4, .3, .2, .1]])
    residual = np.array([[.01, -.01, -.01, .01]])
    assert abs(np.sum(residual*(c-u))) < 1e-18
    t = u+.4*(c-u)+residual
    result = fit_component_mixture(t, u, c)
    np.testing.assert_allclose(result["weights"], [.4], atol=1e-15)
    np.testing.assert_allclose(result["squared_error"], [.0004], atol=1e-15)
    np.testing.assert_allclose(result["tv_error"], [.02], atol=1e-15)
    np.testing.assert_allclose(result["baseline_squared_error"], [.4**2*np.sum((c-u)**2)+.0004], atol=1e-15)


def test_identical_components_have_no_meaningful_weight_but_defined_error():
    u = np.array([[.25]*4]); t = np.array([[.1, .2, .3, .4]])
    result = fit_component_mixture(t, u, u)
    assert not result["identifiable"][0] and np.isnan(result["weights"][0])
    np.testing.assert_array_equal(result["fitted_probs"], u)
    np.testing.assert_array_equal(result["squared_error"], result["baseline_squared_error"])
    cv = result["cross_validation"]
    assert not cv["identifiable"].any() and np.isnan(cv["weights"]).all()
    assert np.isnan(cv["heldout_squared_error"]).all()


def test_nonzero_weak_direction_is_flagged_without_repairing_component():
    u = np.array([[.25]*4]); c = u+np.array([[1e-11, -1e-11, 0, 0]])
    t = .6*u+.4*c
    result = fit_component_mixture(t, u, c)
    assert 0 < result["direction_squared_norm"][0] <= 1e-20
    assert not result["identifiable"][0] and np.isnan(result["weights"][0])
    np.testing.assert_allclose(result["fitted_probs"], t, atol=1e-16)


def test_token_half_holdout_detects_incompatible_shapes_without_renormalization():
    u = np.array([[.25]*4]); c = np.array([[.35, .35, .15, .15]])
    t = np.array([[.27, .33, .23, .17]])
    result = fit_component_mixture(t, u, c)
    np.testing.assert_allclose(result["weights"], [.5], atol=1e-15)
    np.testing.assert_allclose(result["squared_error"], [.0036], atol=1e-15)
    cv = token_half_cross_validation(t, u, c)
    np.testing.assert_allclose(cv["weights"], [[.2, .8]], atol=1e-15)
    np.testing.assert_allclose(cv["heldout_squared_error_by_direction"], [[.0072, .0072]], atol=1e-15)
    np.testing.assert_allclose(cv["heldout_squared_error"], [.0144], atol=1e-15)
    assert cv["heldout_squared_error"][0] > cv["baseline_squared_error"][0]


def test_half_identification_failure_is_explicit_even_when_full_fit_exists():
    u = np.array([[.25]*4]); c = np.array([[.35, .25, .15, .25]])
    t = .7*u+.3*c
    result = fit_component_mixture(t, u, c)
    assert result["identifiable"].all()
    np.testing.assert_array_equal(result["cross_validation"]["identifiable"], [[True, False]])
    assert np.isnan(result["cross_validation"]["heldout_squared_error"]).all()


def test_common_column_permutation_preserves_full_fit_and_respected_half_split():
    u = np.array([[.25]*4]); c = np.array([[.35, .35, .15, .15]])
    t = np.array([[.27, .33, .23, .17]])
    original = fit_component_mixture(t, u, c)
    for permutation in ([3, 0, 2, 1], [2, 3, 0, 1], [1, 0, 3, 2]):
        result = fit_component_mixture(t[:, permutation], u[:, permutation], c[:, permutation])
        for key in ("weights", "direction_squared_norm", "squared_error", "baseline_squared_error", "tv_error"):
            np.testing.assert_allclose(result[key], original[key], atol=1e-15)
        np.testing.assert_allclose(result["fitted_probs"], original["fitted_probs"][:, permutation], atol=1e-15)
    for permutation in ([2, 3, 0, 1], [1, 0, 3, 2]):
        result = token_half_cross_validation(t[:, permutation], u[:, permutation], c[:, permutation])
        np.testing.assert_allclose(result["heldout_squared_error"], original["cross_validation"]["heldout_squared_error"], atol=1e-15)


def test_odd_vocabulary_halves_score_each_coordinate_once():
    u = np.array([[.2]*5]); c = np.array([[.3, .1, .25, .15, .2]])
    t = .65*u+.35*c
    cv = token_half_cross_validation(t, u, c)
    np.testing.assert_allclose(cv["weights"], [[.35, .35]], atol=1e-15)
    np.testing.assert_allclose(cv["heldout_squared_error"], 0, atol=1e-30)


def test_support_mass_recovers_exact_mixture_without_candidate_shape():
    u = np.tile([.25]*4, (3, 1)); c = np.tile([.6, .4, 0, 0], (3, 1))
    w = np.array([0., .3, .8]); t = (1-w[:, None])*u+w[:, None]*c
    result = fit_support_mass(t, u, c > 0)
    np.testing.assert_allclose(result["weights_raw"], w, atol=1e-15)
    assert result["identified"].all() and result["convex"].all() and result["strictly_convex"].all()


def test_support_mass_negative_weight_is_retained_as_a_failure():
    u = np.array([[.25]*4]); t = np.array([[.1, .1, .4, .4]])
    result = fit_support_mass(t, u, np.array([[True, True, False, False]]))
    np.testing.assert_allclose(result["weights_raw"], [-.6], atol=1e-15)
    assert result["identified"][0] and not result["convex"][0]
    assert not result["strictly_convex"][0]


def test_support_mass_full_support_is_unidentified():
    u = np.array([[.25]*4]); t = np.array([[.1, .2, .3, .4]])
    result = fit_support_mass(t, u, np.ones_like(t, dtype=bool))
    assert not result["identified"].any() and not result["convex"].any()
    assert np.isnan(result["weights_raw"]).all()


def test_support_mass_boundary_tolerance_does_not_clip_the_raw_weight():
    u = np.array([[.25]*4]); epsilon = 2e-11
    t = u+np.array([[-epsilon, 0, epsilon, 0]])
    result = fit_support_mass(t, u, np.array([[True, True, False, False]]))
    assert -1e-10 < result["weights_raw"][0] < 0
    assert result["convex"][0] and not result["strictly_convex"][0]


@pytest.mark.parametrize("bad", [np.array([[.2, .2, .2, .2]]), np.array([[.1, .2, .3, np.nan]]), np.array([[0., .2, .3, .5]])])
def test_invalid_target_is_not_silently_repaired(bad):
    u = np.array([[.25]*4])
    with pytest.raises(ValueError):
        fit_component_mixture(bad, u, u)


def test_empty_row_batch_is_supported_with_consistent_shapes():
    x = np.empty((0, 4))
    result = fit_component_mixture(x, x, x)
    assert result["weights"].shape == (0,)
    assert result["fitted_probs"].shape == (0, 4)
    assert result["cross_validation"]["weights"].shape == (0, 2)

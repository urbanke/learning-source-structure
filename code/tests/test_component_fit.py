"""Independent numerical checks of the predictive-component KL fit."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import rel_entr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from component_fit import fit_components


def product_dictionary():
    u1, u2 = np.full(3, 1 / 3), np.full(4, 1 / 4)
    c1, c2 = np.array([0.05, 0.15, 0.8]), np.array([0.05, 0.05, 0.2, 0.7])
    return np.array([np.outer(a, b).ravel() for a, b in
                     [(u1, u2), (c1, u2), (u1, c2), (c1, c2)]])


def check_result(target, components, result, tolerance=1e-7):
    assert result.converged
    assert np.min(result.weights) >= 0
    assert abs(result.weights.sum() - 1) < 2e-14
    np.testing.assert_allclose(result.fitted_probs, result.weights @ components, atol=0, rtol=0)
    positive = target > 0
    gradient = -(components[:, positive] @ (target[positive] / result.fitted_probs[positive]))
    gap = float(gradient @ result.weights - gradient.min())
    assert result.optimality_gap == pytest.approx(max(0.0, gap), abs=2e-14)
    assert gap <= tolerance
    expected_kl = float(rel_entr(target, result.fitted_probs).sum() / np.log(2))
    assert result.kl_bits == pytest.approx(expected_kl, abs=2e-14)
    assert result.tv == pytest.approx(0.5 * np.abs(target - result.fitted_probs).sum(), abs=2e-14)


@pytest.mark.parametrize("weights", [[0.1, 0.2, 0.3, 0.4], [0.9, 0.01, 0.08, 0.01],
                                    [0, 0.2, 0.3, 0.5], [0, 0, 0, 1], [1, 0, 0, 0]])
def test_exact_product_mixtures_interior_and_boundary(weights):
    components = product_dictionary()
    target = np.array(weights) @ components
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.identifiable
    assert result.affine_rank == 3
    np.testing.assert_allclose(result.weights, weights, atol=2e-6)
    assert abs(result.kl_bits) < 2e-12


def test_fit_objective_has_requested_kl_direction():
    # The two projections have visibly different optima. Use the requested
    # derivative's scalar root as an independent oracle, not the fit optimizer.
    components = np.array([[0.85, 0.1, 0.05], [0.05, 0.4, 0.55]])
    target = np.array([0.6, 0.35, 0.05])
    delta = components[0] - components[1]
    derivative = lambda weight: -np.sum(target * delta / (components[1] + weight * delta))
    weight = brentq(derivative, 0, 1, xtol=1e-14)
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.weights[0] == pytest.approx(weight, abs=2e-7)
    reverse_derivative = lambda a: np.sum(delta * (np.log((components[1] + a * delta) / target) + 1))
    reverse_weight = brentq(reverse_derivative, 0, 1)
    assert abs(weight - reverse_weight) > 0.02
    assert result.kl_bits > 0.1


@pytest.mark.parametrize("target", [[0.999, 0.001, 0], [0, 0.001, 0.999], [0.1, 0.8, 0.1]])
def test_two_component_scalar_oracle_and_boundary(target):
    components = np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
    target = np.array(target)
    delta = components[0] - components[1]
    derivative = lambda a: -np.sum(target * delta / (components[1] + a * delta))
    optimum = 0.0 if derivative(0) >= 0 else 1.0 if derivative(1) <= 0 else brentq(derivative, 0, 1)
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.weights[0] == pytest.approx(optimum, abs=3e-7)


def test_target_outside_four_component_span():
    components = product_dictionary()
    target = np.array([0.30, 0.01, 0.02, 0.05, 0.03, 0.25, 0.01, 0.08, 0.04, 0.02, 0.04, 0.15])
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.kl_bits > 0.05
    rng = np.random.default_rng(9172)
    trial_weights = rng.dirichlet(np.ones(4), size=1000)
    trial_kl = rel_entr(target, trial_weights @ components).sum(axis=1) / np.log(2)
    assert result.kl_bits <= trial_kl.min() + 1e-12


def test_small_background_mass_requires_gradient_certified_polishing():
    # A well-separated dictionary with a very small active background weight
    # has highly unequal curvature. Objective-only convergence can leave a
    # KKT violation even though the probability error already looks tiny.
    a = np.array([1.32216621e-7, 0.041671838, 0.0250014456, 6.43601493e-7,
                  6.41232524e-5, 0.853399574, 0.0798592191, 3.02428617e-6])
    b = np.array([0.000100721577, 6.38514620e-5, 4.83679104e-6, 0.0572350802,
                  0.000356973699, 1.12546723e-5, 0.941281252, 0.000946029914])
    a, b = a / a.sum(), b / b.sum()
    u = np.full(8, 1 / 8)
    components = np.array([np.outer(x, y).ravel() for x, y in [(u, u), (a, u), (u, b), (a, b)]])
    weights = np.array([1.08220278e-5, 0.904305815, 0.0925720705, 0.00311129289])
    weights /= weights.sum()
    target = weights @ components
    result = fit_components(target, components)
    check_result(target, components, result)
    np.testing.assert_allclose(result.weights, weights, atol=2e-6)


def test_duplicate_components_use_first_representative_without_claiming_identification():
    a, b = np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.4, 0.1])
    components = np.array([a, a, b, b])
    target = 0.3 * a + 0.7 * b
    result = fit_components(target, components, initial=[0, 0.3, 0, 0.7])
    check_result(target, components, result)
    assert not result.identifiable
    assert result.duplicate_groups == ((0, 1), (2, 3))
    assert result.min_pairwise_tv == 0
    np.testing.assert_allclose(result.weights, [0.3, 0, 0.7, 0], atol=2e-14)


def test_empty_context_identical_uniform_components():
    components = np.full((4, 64), 1 / 64)
    target = np.random.default_rng(5).dirichlet(np.ones(64))
    result = fit_components(target, components)
    check_result(target, components, result)
    np.testing.assert_array_equal(result.weights, [1, 0, 0, 0])
    assert not result.identifiable
    assert result.affine_rank == 0
    assert result.iterations == 0


def test_distinct_affinely_dependent_components_are_not_identifiable():
    components = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.45, 0.4, 0.15]])
    target = np.array([0.6, 0.2, 0.2])
    result = fit_components(target, components)
    check_result(target, components, result)
    assert not result.identifiable
    assert result.affine_rank == 1
    assert result.duplicate_groups == ()
    assert result.min_pairwise_tv > 0.1


def test_near_duplicates_remain_distinct_with_small_singular_value():
    components = product_dictionary()
    components[1] = components[0] + 1e-7 * (components[1] - components[0])
    target = np.array([0.1, 0.2, 0.3, 0.4]) @ components
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.identifiable  # Algebraic rank only, not statistical stability.
    assert result.duplicate_groups == ()
    assert result.affine_singular_values[-1] < 1e-7
    assert result.min_pairwise_tv < 1e-7
    assert result.tv < 1e-6


def test_zero_probabilities_and_initial_support_restart():
    components = np.eye(3)
    target = np.array([0, 0.4, 0.6])
    result = fit_components(target, components, initial=[1, 0, 0])
    check_result(target, components, result)
    np.testing.assert_allclose(result.weights, target, atol=5e-7)


def test_single_component_needs_no_optimization():
    components, target = np.array([[0.2, 0.8]]), np.array([0.6, 0.4])
    result = fit_components(target, components)
    check_result(target, components, result)
    assert result.identifiable
    assert result.iterations == 0


@pytest.mark.parametrize("target,components", [
    ([0.3, 0.6], [[0.5, 0.5]]),
    ([0.3, 0.7], [[0.5, 0.4]]),
    ([-1e-16, 1], [[0.5, 0.5]]),
    ([0.3, 0.7], [[-1e-16, 1]]),
    ([np.nan, 1], [[0.5, 0.5]]),
    ([0.3, 0.7], [[np.inf, 0]]),
    ([0.3, 0.7], [[0.2, 0.3, 0.5]]),
    ([0.3, 0.7], [0.5, 0.5]),
    ([], [[]]),
    ([0.3, 0.7], [[1, 0]]),
])
def test_invalid_or_unsupported_distributions_are_rejected(target, components):
    with pytest.raises(ValueError):
        fit_components(target, components)


def test_accepted_small_probability_sum_error_is_not_repaired():
    components = np.array([[0.2, 0.8 + 1e-8]])
    target = np.array([0.2, 0.8])
    result = fit_components(target, components)
    np.testing.assert_array_equal(result.fitted_probs, components[0])
    assert result.probability_sum_error == pytest.approx(1e-8, abs=2e-16)


def test_gap_failure_is_returned_even_if_solver_stops():
    components = product_dictionary()
    target = np.array([0.01, 0.89, 0.04, 0.06]) @ components
    result = fit_components(target, components, max_iterations=1)
    assert not result.converged
    assert result.optimality_gap > 1e-7


def test_validated_inputs_are_unchanged():
    components = product_dictionary()
    target = np.array([0.1, 0.2, 0.3, 0.4]) @ components
    initial = np.array([0.25] * 4)
    originals = [x.copy() for x in [target, components, initial]]
    fit_components(target, components, initial)
    for value, original in zip([target, components, initial], originals):
        np.testing.assert_array_equal(value, original)

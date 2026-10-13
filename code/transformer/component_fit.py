"""Fit predictive component weights by KL(target || fitted prediction).

The input distributions are checked, never clipped or renormalized. Convex
optimization is certified separately by the Frank--Wolfe/KKT gap in nats,
which upper-bounds the objective suboptimality. A fitted weight describes
prediction, not an internal transformer mechanism or model posterior.

Exact duplicate components are combined before optimization and their mass
is assigned to the first original index. This deterministic convention does
not identify the original weights; ``identifiable`` is false in that case.
Near dependence is exposed by singular values, not hidden by merging rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import rel_entr


@dataclass(frozen=True)
class ComponentFit:
    weights: np.ndarray
    fitted_probs: np.ndarray
    kl_bits: float
    tv: float
    optimality_gap: float
    converged: bool
    iterations: int
    singular_values: np.ndarray
    affine_singular_values: np.ndarray
    identifiable: bool
    affine_rank: int
    min_pairwise_tv: float
    duplicate_groups: tuple[tuple[int, ...], ...]
    probability_sum_error: float


def _distribution_array(value, ndim, name, tolerance):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    error = float(np.max(np.abs(array.sum(axis=-1) - 1)))
    if error > tolerance:
        raise ValueError(f"{name} is not normalized: raw sum error {error:.3g}")
    return array, error


def _simplex_projection(values):
    """An optimizer step: project candidate weights, never probabilities."""
    ordered = np.sort(values)[::-1]
    levels = (np.cumsum(ordered) - 1) / np.arange(1, len(ordered) + 1)
    last = np.flatnonzero(ordered > levels)[-1]
    return np.maximum(values - levels[last], 0)


def fit_components(target, components, initial=None, *, gap_tolerance=1e-7,
                   probability_tolerance=1e-7, max_iterations=250):
    """Minimize KL(target || weights @ components) on the weight simplex.

    ``target`` has shape (d,), ``components`` has shape (k,d), and optional
    ``initial`` has shape (k,). ``optimality_gap`` and ``gap_tolerance`` are
    in nats; ``kl_bits`` is in bits. Only a passed numerical gap certificate
    gives ``converged=True``, regardless of the optimizer's success flag.

    ``identifiable`` means that the *given components* have full affine rank
    at the machine-precision rank threshold. It is not a statistical claim
    that fitted weights are reliable: inspect affine singular values and
    component separation too. Nonidentifiability can also occur for distinct
    but affinely dependent components. Tiny negative KL values caused by
    floating point or accepted input sum error are reported without clipping.
    """
    if (not np.isfinite(gap_tolerance) or gap_tolerance <= 0
            or not np.isfinite(probability_tolerance) or probability_tolerance <= 0):
        raise ValueError("numerical tolerances must be finite and positive")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    p, p_error = _distribution_array(target, 1, "target", probability_tolerance)
    all_components, c_error = _distribution_array(
        components, 2, "components", probability_tolerance)
    count, alphabet = all_components.shape
    if p.shape != (alphabet,):
        raise ValueError("target and components have different alphabet sizes")
    if initial is not None:
        initial, _ = _distribution_array(initial, 1, "initial weights", probability_tolerance)
        if initial.shape != (count,):
            raise ValueError("initial weights must have one entry per component")

    # Rank and separation always use the original components, before deduplication.
    singular_values = np.linalg.svd(all_components.T, compute_uv=False)
    contrasts = (all_components[1:] - all_components[0]).T
    affine_singular_values = np.linalg.svd(contrasts, compute_uv=False)
    rank_tolerance = (max(contrasts.shape) * np.finfo(float).eps
                      * (affine_singular_values[0] if affine_singular_values.size else 0))
    affine_rank = int(np.count_nonzero(affine_singular_values > rank_tolerance))
    identifiable = affine_rank == count - 1
    min_pairwise_tv = min(
        (0.5 * float(np.abs(all_components[i] - all_components[j]).sum())
         for i in range(count) for j in range(i)), default=float("nan"))
    groups = []
    for index, component in enumerate(all_components):
        for group in groups:
            if np.array_equal(component, all_components[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])
    representatives = np.array([group[0] for group in groups])
    c = all_components[representatives]
    positive = p > 0
    cp, pp = c[:, positive], p[positive]
    if np.any(np.all(cp == 0, axis=0)):
        raise ValueError("target has positive mass outside the union of component supports")
    reduced_count = len(c)
    w = (np.full(reduced_count, 1 / reduced_count) if initial is None else
         _simplex_projection(np.array([initial[group].sum() for group in groups])))
    # A boundary initial weight may omit part of target support. A uniform
    # restart is an explicit feasible optimizer initialization in that case.
    if np.any(w @ cp <= 0):
        w = np.full(reduced_count, 1 / reduced_count)

    def objective_gradient(weights):
        q = weights @ cp
        if np.any(q <= 0):
            return float("inf"), np.zeros(reduced_count)
        return float(-pp @ np.log(q)), -(cp @ (pp / q))

    def certificate(weights):
        _, gradient = objective_gradient(weights)
        if not np.isfinite(weights @ cp).all() or np.any(weights @ cp <= 0):
            return float("inf")
        return max(0.0, float(gradient @ weights - gradient.min()))

    iterations = 0
    if reduced_count > 1 and certificate(w) > gap_tolerance:
        result = minimize(
            objective_gradient, w, jac=True, method="SLSQP",
            bounds=[(0.0, 1.0)] * reduced_count,
            constraints={"type": "eq", "fun": lambda x: x.sum() - 1,
                         "jac": lambda x: np.ones_like(x)},
            options={"ftol": 1e-13, "maxiter": max_iterations, "disp": False})
        iterations += int(result.nit)
        candidate = _simplex_projection(result.x)
        if objective_gradient(candidate)[0] <= objective_gradient(w)[0]:
            w = candidate

    # SLSQP's objective-based stopping can precede a stringent gradient-gap
    # certificate. An active-face Newton step or a pairwise line search
    # resolves remaining violations without relying on that stopping flag.
    while reduced_count > 1 and certificate(w) > gap_tolerance and iterations < max_iterations:
        _, gradient = objective_gradient(w)
        entering = int(np.argmin(gradient))
        support = np.flatnonzero(w > 0)
        leaving = int(support[np.argmax(gradient[support])])
        if entering == leaving:
            break
        direction = np.zeros(reduced_count)
        direction[entering], direction[leaving] = 1, -1
        bound = w[leaving]
        if entering in support and len(support) > 1:
            pivot = int(support[np.argmax(w[support])])
            other = support[support != pivot]
            differences = cp[other] - cp[pivot]
            q = w @ cp
            curvature = (differences * (pp / q ** 2)) @ differences.T
            coordinate_step = np.linalg.lstsq(
                curvature, -(gradient[other] - gradient[pivot]), rcond=1e-14)[0]
            newton = np.zeros(reduced_count)
            newton[other], newton[pivot] = coordinate_step, -coordinate_step.sum()
            negative = newton < 0
            if np.any(negative) and float(gradient @ newton) < 0:
                feasible_step = min(1.0, float(np.min(w[negative] / -newton[negative])))
                if feasible_step > 1e-12:
                    direction, bound = newton, feasible_step
        q, delta = w @ cp, direction @ cp

        def line_derivative(step):
            point = q + step * delta
            # A disappearing positive-target outcome has positive infinite
            # derivative at this endpoint; interior steps retain support.
            if np.any(point <= 0):
                return float("inf")
            return float(-pp @ (delta / point))

        # Root the derivative rather than minimize almost equal objective
        # values: the latter loses the precision needed by the gap check.
        step = bound if line_derivative(bound) <= 0 else brentq(
            line_derivative, 0.0, bound, xtol=5e-16,
            rtol=4 * np.finfo(float).eps, maxiter=100)
        iterations += 1
        if step == 0:
            break
        w = w + step * direction
        # A Newton step that hits a simplex face can leave a negative value
        # at roundoff scale. This is a weight projection, with the final
        # predictive probabilities and gap always recomputed afterwards.
        if np.any(w < 0):
            w = _simplex_projection(w)
        # Correct only accumulated simplex arithmetic, keeping exact zeros.
        w[np.argmax(w)] += 1.0 - w.sum()

    weights = np.zeros(count)
    weights[representatives] = w
    fitted = weights @ all_components
    gap = certificate(w)
    return ComponentFit(
        weights=weights, fitted_probs=fitted,
        kl_bits=float(rel_entr(p, fitted).sum() / np.log(2)),
        tv=float(0.5 * np.abs(p - fitted).sum()),
        optimality_gap=gap, converged=bool(gap <= gap_tolerance),
        iterations=iterations, singular_values=singular_values,
        affine_singular_values=affine_singular_values,
        identifiable=identifiable, affine_rank=affine_rank,
        min_pairwise_tv=min_pairwise_tv,
        duplicate_groups=tuple(tuple(group) for group in groups if len(group) > 1),
        probability_sum_error=max(p_error, c_error))

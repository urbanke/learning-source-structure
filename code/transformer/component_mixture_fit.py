"""Scalar convex mixture fits to full probability vectors, with shape checks.

No observed next token, source matrix, or outcome label enters these fits.
Every row is independent. The token-half check holds out output coordinates,
not independent sequences; neither half is renormalized. Undefined weights
are NaN and always accompanied by explicit identification flags.
"""
from __future__ import annotations

import numpy as np


def _tolerance(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} scalar")
    return float(value)


def _probability_array(value, name, tolerance, *, positive, shape=None):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real matrix [rows, vocabulary>=2]")
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have the same shape as target")
    array = np.asarray(raw, dtype=np.float64)
    if (not np.isfinite(array).all() or np.any(array < 0)
            or (positive and np.any(array == 0))):
        raise ValueError(f"{name} contains invalid probabilities")
    error = float(np.max(np.abs(array.sum(axis=1)-1), initial=0.0))
    if error > tolerance:
        raise ValueError(f"{name} is not normalized: raw error {error}")
    return array, error


def _inputs(target, baseline, component, probability_tolerance):
    tolerance = _tolerance(probability_tolerance, "probability_tolerance")
    t, et = _probability_array(target, "target", tolerance, positive=True)
    u, eu = _probability_array(baseline, "baseline", tolerance, positive=False, shape=t.shape)
    c, ec = _probability_array(component, "component", tolerance, positive=False, shape=t.shape)
    return t, u, c, dict(target=et, baseline=eu, component=ec)


def _coefficient(target, baseline, component, tolerance):
    direction = component-baseline
    denominator = np.einsum("ij,ij->i", direction, direction)
    numerator = np.einsum("ij,ij->i", target-baseline, direction)
    coefficient = np.zeros(len(target), dtype=np.float64)
    np.divide(numerator, denominator, out=coefficient, where=denominator > 0)
    coefficient = np.clip(coefficient, 0.0, 1.0)
    identified = denominator > tolerance
    reported = np.where(identified, coefficient, np.nan)
    return coefficient, reported, identified, denominator


def _token_halves(target, baseline, component, denominator_tolerance):
    weights, identified, denominators, scores = [], [], [], []
    for training, heldout in ((slice(0, None, 2), slice(1, None, 2)),
                              (slice(1, None, 2), slice(0, None, 2))):
        coefficient, reported, valid, denominator = _coefficient(
            target[:, training], baseline[:, training], component[:, training], denominator_tolerance)
        prediction = ((1-coefficient[:, None])*baseline[:, heldout]
                      + coefficient[:, None]*component[:, heldout])
        error = np.sum((target[:, heldout]-prediction)**2, axis=1)
        weights.append(reported)
        identified.append(valid)
        denominators.append(denominator)
        scores.append(np.where(valid, error, np.nan))
    identified = np.stack(identified, axis=1)
    scores = np.stack(scores, axis=1)
    return dict(weights=np.stack(weights, axis=1), identifiable=identified,
                direction_squared_norm=np.stack(denominators, axis=1),
                heldout_squared_error_by_direction=scores,
                heldout_squared_error=scores.sum(axis=1),
                both_halves_identifiable=identified.all(axis=1),
                baseline_squared_error=np.sum((target-baseline)**2, axis=1),
                training_halves=("even_token_ids", "odd_token_ids"),
                heldout_halves=("odd_token_ids", "even_token_ids"))


def token_half_cross_validation(target, baseline, component, *,
                                denominator_tolerance=1e-20, probability_tolerance=1e-10):
    """Fit on even token IDs and score odd IDs, then reverse, without rescaling.

    Columns are zero-based token IDs. Weights/identification/norms have shape
    [rows,2], in train-even/score-odd then train-odd/score-even order. Each
    coordinate is scored exactly once in heldout_squared_error. A direction
    with squared norm <= denominator_tolerance has undefined weight and
    held-out error; the combined error is NaN unless both directions identify
    a weight. Arbitrary column permutations change this fixed split.
    """
    tolerance = _tolerance(denominator_tolerance, "denominator_tolerance", allow_zero=True)
    t, u, c, _ = _inputs(target, baseline, component, probability_tolerance)
    return _token_halves(t, u, c, tolerance)


def fit_component_mixture(target, baseline, component, *,
                          denominator_tolerance=1e-20, probability_tolerance=1e-10):
    """Fit Q=(1-w)baseline+w*component by rowwise full-vector squared error.

    w is the scalar least-squares coefficient clipped to [0,1]. Returned
    weights are NaN when ||component-baseline||^2 <= denominator_tolerance.
    The fitted vector and its errors remain defined: the bounded minimizer
    is used for nonzero directions; exactly equal components use baseline.
    No meaningful coefficient is claimed for either kind of weak direction.

    Return weights/identifiable/direction_squared_norm [rows], fitted_probs
    [rows,vocabulary], squared_error, baseline_squared_error and tv_error
    [rows], plus cross_validation from the complementary token-half check.
    No explained-error ratio is formed here; aggregate sums before dividing.
    Inputs are validated as received and never renormalized or modified.
    """
    tolerance = _tolerance(denominator_tolerance, "denominator_tolerance", allow_zero=True)
    t, u, c, errors = _inputs(target, baseline, component, probability_tolerance)
    coefficient, weights, identified, denominator = _coefficient(t, u, c, tolerance)
    prediction = (1-coefficient[:, None])*u+coefficient[:, None]*c
    residual = t-prediction
    return dict(weights=weights, identifiable=identified, direction_squared_norm=denominator,
                fitted_probs=prediction, squared_error=np.sum(residual**2, axis=1),
                baseline_squared_error=np.sum((t-u)**2, axis=1),
                tv_error=np.sum(np.abs(residual), axis=1)/2,
                cross_validation=_token_halves(t, u, c, tolerance),
                normalization_errors=errors)


def fit_support_mass(target, baseline, support_mask, *,
                     outside_mass_tolerance=1e-12, convex_tolerance=1e-10,
                     probability_tolerance=1e-10):
    """Infer a weight using only probability mass outside the candidate support.

    If C is zero outside S, an exact mixture must have
    w=1-T(S-complement)/baseline(S-complement). No shape of C is used here.
    The raw weight is never clipped. Identified means baseline's outside
    mass exceeds outside_mass_tolerance. `convex` permits only the stated
    numerical boundary tolerance; `strictly_convex` checks the exact [0,1]
    interval. Both flags are false for unidentified rows, whose weight is NaN.
    A caller may evaluate candidate vectors with valid weights; this function
    cannot form predictions or their errors without a candidate component.
    """
    tolerance = _tolerance(probability_tolerance, "probability_tolerance")
    outside_tolerance = _tolerance(outside_mass_tolerance, "outside_mass_tolerance", allow_zero=True)
    interval_tolerance = _tolerance(convex_tolerance, "convex_tolerance", allow_zero=True)
    t, et = _probability_array(target, "target", tolerance, positive=True)
    u, eu = _probability_array(baseline, "baseline", tolerance, positive=False, shape=t.shape)
    support = np.asarray(support_mask)
    if support.shape != t.shape or support.dtype.kind != "b":
        raise ValueError("support_mask must be a boolean matrix matching target")
    outside = ~support
    target_mass = np.sum(t*outside, axis=1)
    baseline_mass = np.sum(u*outside, axis=1)
    identified = baseline_mass > outside_tolerance
    weight = np.full(len(t), np.nan, dtype=np.float64)
    np.divide(target_mass, baseline_mass, out=weight, where=identified)
    weight = 1-weight
    strict = identified & (weight >= 0) & (weight <= 1)
    convex = identified & (weight >= -interval_tolerance) & (weight <= 1+interval_tolerance)
    return dict(weights_raw=weight, identified=identified, convex=convex, strictly_convex=strict,
                outside_target_mass=target_mass, outside_baseline_mass=baseline_mass,
                support_size=support.sum(axis=1),
                normalization_errors=dict(target=et, baseline=eu))

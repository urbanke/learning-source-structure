"""Read-only scale diagnostics for one saved, marginalized depth weight.

No inference, acceptance decision, or modification of the original summaries
occurs here. Rescaling preserves classic split R-hat's mathematical definition.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from stable_chain_statistics import _mean, _variance


def _split_moments(values):
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = _mean(_variance(split, 1), 0)
    between = half * _variance(_mean(split, 1), 0)
    return within, between


def _scientific(value):
    return np.format_float_scientific(np.longdouble(value), unique=True, trim="k")


def _original_units(value, scale):
    # Decimal preserves the exponent when squaring a float64-scale value would
    # underflow, including platforms where longdouble has float64 precision.
    with localcontext() as context:
        context.prec = 50
        return format(Decimal(_scientific(value)) * Decimal.from_float(scale)**2, ".20E")


def _range(values):
    return dict(mean=float(_mean(values, 0)), min=float(values.min()),
                max=float(values.max()), zero_fraction=float(np.mean(values == 0)))


def coordinate_summary(samples):
    """Summarize float64 weights shaped (chains, draws); return strict JSON data.

    Odd draw counts omit the middle draw only from split R-hat, matching the
    original definition. Shared constants are explicitly uninformative; chains
    stuck at different constants remain infinite after rescaling.
    """
    values = np.asarray(samples)
    if (values.dtype != np.float64 or values.ndim != 2
            or values.shape[0] < 2 or values.shape[1] < 16
            or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
        raise ValueError("need finite float64 weights in [0,1], >=2 chains and >=16 draws")
    chains, draws = values.shape
    half = draws // 2
    scale = float(values.max())
    constant = bool(np.all(values == values[0, 0]))
    within, between = _split_moments(values)
    cutoff_trigger = bool(within <= 1e-30 and between > 1e-30)
    original_status = ("constant_uninformative" if constant else
                       "cutoff_infinity" if cutoff_trigger else
                       "cutoff_unit_fallback" if within <= 1e-30 else "finite")
    if cutoff_trigger:
        original_rhat = None
    elif within <= 1e-30:
        original_rhat = 1.0
    else:
        original_rhat = float(np.sqrt(((half-1)/half * within + between/half) / within))
    normalized = values.astype(np.longdouble)
    if scale:
        normalized /= np.longdouble(scale)
    scaled_within, scaled_between = _split_moments(normalized)
    split = np.concatenate((normalized[:, :half], normalized[:, -half:]), axis=0)
    split_constant = bool(np.all(split == split[0, 0]))
    # Exact constants must not gain spurious variance through an odd-length
    # sum/division. No tolerance or materiality threshold is used.
    if np.all(split == split[:, :1]):
        scaled_within = np.longdouble(0)
    if split_constant:
        scaled_between = np.longdouble(0)
        rescaled_rhat, rescaled_status = None, "constant_uninformative"
    elif scaled_within == 0:
        rescaled_rhat = None
        rescaled_status = ("different_constant_chains" if scaled_between > 0
                           else "numerically_unresolved")
    else:
        value = np.sqrt(((np.longdouble(half-1)/half) * scaled_within
                         + scaled_between/half) / scaled_within)
        rescaled_rhat = float(value) if np.isfinite(value) and value <= np.finfo(float).max else None
        rescaled_status = "finite" if rescaled_rhat is not None else "infinite"
    return dict(
        schema="depth-rhat-coordinate-diagnostic-v1", chains=chains, draws=draws,
        half_draws=half, omitted_middle_draws_per_chain=draws-2*half, scale=scale,
        mean=float(_mean(_mean(values, 1), 0)), min=float(values.min()), max=scale,
        zero_fraction=float(np.mean(values == 0)), old_cutoff_trigger=cutoff_trigger,
        chain_summaries=[dict(chain=i, **_range(values[i])) for i in range(chains)],
        half_chain_summaries=[dict(chain=i, half=label, **_range(values[i, section]))
                              for i in range(chains)
                              for label, section in (("first", slice(None, half)),
                                                     ("last", slice(-half, None)))],
        original=dict(rhat=original_rhat, status=original_status,
                      within_variance_scientific=_scientific(within),
                      between_variance_scientific=_scientific(between)),
        rescaled=dict(rhat=rescaled_rhat, status=rescaled_status,
                      within_variance=float(scaled_within), between_variance=float(scaled_between),
                      within_variance_scientific=_scientific(scaled_within),
                      between_variance_scientific=_scientific(scaled_between),
                      within_variance_original_units=_original_units(scaled_within, scale),
                      between_variance_original_units=_original_units(scaled_between, scale)),
        interpretation="Diagnostics of saved weights only; no acceptance gate or inference result is changed.")

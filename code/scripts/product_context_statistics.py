"""Paired whole-sequence uncertainty for a frozen context-length forecast.

Arrays have shape (independent sequences, tested times, 2), with informed
Bayes first and the transformer second. A bootstrap draw resamples whole
sequences: the two predictors and every time stay together. Conditions use
independent random streams. The maximum standardized deviation is taken over
all supplied conditions, predictors and times, giving one approximate
simultaneous confidence band. Band inversion uses first crossings on the
actual tested grid, with neither interpolation nor monotonic smoothing.

Unreached means no tested time met the target; it is not an infinite
population context requirement. No interval drops unreached outcomes.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

PREDICTORS = ('bayes', 'transformer')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _times(values):
    raw = np.asarray(values)
    _require(raw.ndim == 1 and raw.size > 0 and raw.dtype.kind in 'iu'
             and np.all(raw > 0) and np.all(raw[1:] > raw[:-1]),
             'Times must be nonempty, strictly increasing positive integers')
    return raw.astype(np.int64, copy=False)


def first_tested_context(risks, times, target):
    """Return the first tested crossing, or None when every point exceeds it."""
    times = _times(times)
    risks = np.asarray(risks, dtype=float)
    _require(risks.shape == times.shape and np.isfinite(risks).all()
             and math.isfinite(target) and target >= 0, 'Invalid risk curve or target')
    found = np.flatnonzero(risks <= target)
    return int(times[found[0]]) if len(found) else None


def invert_first_crossing_band(lower, upper, times, target):
    """All first crossings compatible with the rectangular mean-risk band.

    A crossing at j is possible if lower[j] <= target and no earlier upper
    bound forces a crossing already. This retains holes on a nonmonotone grid.
    Unreached is possible precisely when every upper bound exceeds the target.
    """
    times = _times(times)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    _require(lower.shape == upper.shape == times.shape
             and np.isfinite(lower).all() and np.isfinite(upper).all()
             and np.all(lower <= upper) and math.isfinite(target) and target >= 0,
             'Invalid confidence band or target')
    candidates = []
    forced = False
    for lo, hi, t in zip(lower, upper, times):
        if not forced and lo <= target:
            candidates.append(int(t))
        forced = forced or hi <= target
    unreached = not forced
    _require(candidates or unreached, 'Empty inverted confidence set')
    return dict(tested_contexts=candidates, unreached_possible=bool(unreached),
        minimum_tested_context=min(candidates) if candidates else None,
        maximum_tested_context=max(candidates) if candidates and not unreached else None,
        maximum_status='unreached_possible' if unreached else 'finite')


def compatible_comparison(bayes, transformer):
    """Conservative joint ranges induced by the same simultaneous risk band.

    If either confidence set permits nonattainment, no finite-only range is
    reported for differences or ratios. The possible attainment outcomes are
    kept explicitly. Ranges are in transformer-minus-Bayes / transformer:Bayes
    order. The rectangular band makes these conservative compatibility ranges.
    """
    b, t = bayes['tested_contexts'], transformer['tested_contexts']
    bu, tu = bayes['unreached_possible'], transformer['unreached_possible']
    outcomes = []
    if b and t:
        outcomes.append('both_reached')
    if bu and t:
        outcomes.append('bayes_unreached_only')
    if b and tu:
        outcomes.append('transformer_unreached_only')
    if bu and tu:
        outcomes.append('both_unreached')
    finite = bool(b and t and not bu and not tu)
    return dict(possible_attainment_outcomes=outcomes,
        difference_interval=[min(t)-max(b), max(t)-min(b)] if finite else None,
        ratio_interval=[min(t)/max(b), max(t)/min(b)] if finite else None,
        finite_intervals_available=finite)


def _mean_se(values):
    """Stable means and SEs; exact constant coordinates stay exact.

    Only exact lack of empirical variation is treated as zero SE. Scaling the
    residuals before squaring avoids a small absolute variance cutoff and
    avoids squaring tiny differences directly. Extreme unrepresentable SEs
    raise an error rather than silently becoming deterministic coordinates.
    """
    offset = values - values[0]
    mean_offset = np.mean(offset, axis=0)
    mean = values[0] + mean_offset
    residual = offset - mean_offset
    scale = np.max(np.abs(residual), axis=0)
    varying = np.any(values != values[0], axis=0)
    scaled = np.zeros_like(residual)
    np.divide(residual, scale, out=scaled, where=scale > 0)
    n = len(values)
    se = scale * np.sqrt(np.sum(scaled*scaled, axis=0)/(n*(n-1)))
    _require(np.isfinite(mean).all() and np.isfinite(se).all()
             and np.all(se[varying] > 0), 'Risk variation exceeds representable numerical range')
    standardized = np.zeros_like(residual)
    np.divide(residual, se, out=standardized, where=varying)
    return mean, se, standardized, ~varying


def _crossing_indices(curves, targets):
    """B x T x 2 -> B x targets x 2; index T denotes unreached."""
    found = curves[:, :, None, :] <= targets[None, None, :, None]
    first = np.argmax(found, axis=1)
    return np.where(np.any(found, axis=1), first, curves.shape[1]).astype(np.int32)


def _bootstrap_attainment(indices, times):
    n = len(indices)
    counts = np.bincount(indices, minlength=len(times)+1)
    return dict(counts_by_tested_context=[dict(context=int(t), count=int(c))
            for t, c in zip(times, counts[:-1])],
        unreached_count=int(counts[-1]), unreached_fraction=float(counts[-1]/n))


def _joint_attainment(indices, sentinel):
    b, t = indices[:, 0] == sentinel, indices[:, 1] == sentinel
    masks = dict(both_reached=~b & ~t, bayes_unreached_only=b & ~t,
                 transformer_unreached_only=~b & t, both_unreached=b & t)
    return {key:dict(count=int(np.sum(mask)), fraction=float(np.mean(mask))) for key,mask in masks.items()}


def analyze_context_requirements(risks_by_condition, times, targets=(.1, .2, .5), *,
        bootstrap_replicates=4096, seed=20260917, confidence=.95, batch_size=64):
    """Return a strict-JSON-compatible paired bootstrap analysis.

    ``times`` is either one shared grid or a mapping with exactly the same
    condition keys as ``risks_by_condition``. Conditions are sorted by key for
    reproducibility; each receives its own spawned PCG64 random stream. Small
    bootstrap batches bound memory without constructing B x N x T x2 arrays.
    The caller must freeze conditions, grids, targets and checkpoint identities
    before evaluating confirmation sequences. Numerical reference validation
    and variation between independently trained models remain separate.
    """
    _require(isinstance(risks_by_condition, Mapping) and risks_by_condition
             and all(isinstance(k,str) and k for k in risks_by_condition),
             'Provide nonempty named conditions')
    _require(type(bootstrap_replicates) is int and bootstrap_replicates >= 2
             and type(batch_size) is int and batch_size >= 1
             and type(seed) is int and seed >= 0 and 0 < confidence < 1,
             'Invalid bootstrap size, seed, confidence or batch size')
    targets = np.asarray(targets, dtype=float)
    _require(targets.ndim == 1 and targets.size > 0 and np.isfinite(targets).all()
             and np.all(targets >= 0) and len(np.unique(targets)) == len(targets),
             'Targets must be finite, nonnegative and unique')
    names = sorted(risks_by_condition)
    if isinstance(times, Mapping):
        _require(set(times) == set(names), 'Time grids do not match conditions')
        grids = {name:_times(times[name]) for name in names}
    else:
        grids = {name:_times(times) for name in names}
    prepared = {}
    for name in names:
        values = np.asarray(risks_by_condition[name], dtype=float)
        _require(values.ndim == 3 and values.shape[0] >= 2
                 and values.shape[1:] == (len(grids[name]), 2)
                 and np.isfinite(values).all() and np.min(values) >= -1e-12,
                 f'Invalid risk array for {name}; need sequences x times x2 finite risks')
        mean,se,standardized,zero = _mean_se(values)
        prepared[name] = dict(n=len(values), mean=mean, se=se, standardized=standardized,
                              zero=zero, minimum_input_risk=float(np.min(values)))
    maxima = np.zeros(bootstrap_replicates)
    crossings = {}
    streams = np.random.SeedSequence(seed).spawn(len(names))
    for name,stream in zip(names,streams):
        item = prepared[name]; n = item['n']; width = len(grids[name])
        rng = np.random.Generator(np.random.PCG64(stream))
        selected = np.empty((bootstrap_replicates,len(targets),2),dtype=np.int32)
        for start in range(0,bootstrap_replicates,batch_size):
            stop = min(start+batch_size,bootstrap_replicates)
            counts = rng.multinomial(n,np.full(n,1/n),size=stop-start)
            deviations = (counts @ item['standardized'].reshape(n,-1)/n).reshape(stop-start,width,2)
            maxima[start:stop] = np.maximum(maxima[start:stop],np.max(np.abs(deviations),axis=(1,2)))
            bootstrap_means = item['mean'][None,:,:] + deviations*item['se'][None,:,:]
            selected[start:stop] = _crossing_indices(bootstrap_means,targets)
        crossings[name] = selected
    critical = float(np.quantile(maxima,confidence,method='higher'))
    _require(math.isfinite(critical), 'Nonfinite simultaneous bootstrap critical value')
    conditions = {}
    for name in names:
        item = prepared[name]; grid = grids[name]
        lower,upper = item['mean']-critical*item['se'], item['mean']+critical*item['se']
        thresholds = []
        for j,target in enumerate(targets):
            predictions = {}
            for k,predictor in enumerate(PREDICTORS):
                point = first_tested_context(item['mean'][:,k],grid,float(target))
                predictions[predictor] = dict(point_context=point, point_unreached=point is None,
                    compatible=invert_first_crossing_band(lower[:,k],upper[:,k],grid,float(target)),
                    bootstrap=_bootstrap_attainment(crossings[name][:,j,k],grid))
            b,t = (predictions[p]['point_context'] for p in PREDICTORS)
            joint = compatible_comparison(*(predictions[p]['compatible'] for p in PREDICTORS))
            joint.update(point_difference=t-b if b is not None and t is not None else None,
                         point_ratio=t/b if b is not None and t is not None else None,
                         bootstrap_attainment=_joint_attainment(crossings[name][:,j,:],len(grid)))
            thresholds.append(dict(target_bits=float(target),**predictions,comparison=joint))
        conditions[name] = dict(n_sequences=item['n'], times=grid.tolist(),
            mean_risk_bits=item['mean'].tolist(), standard_error_bits=item['se'].tolist(),
            lower_band_bits=lower.tolist(), upper_band_bits=upper.tolist(),
            zero_standard_error=item['zero'].tolist(), minimum_input_risk=item['minimum_input_risk'],
            thresholds=thresholds)
    return dict(schema='product-context-statistics-v1', predictor_order=list(PREDICTORS),
        confidence=float(confidence), bootstrap_replicates=bootstrap_replicates, seed=seed,
        batch_size=batch_size, condition_order=names, simultaneous_critical_value=critical,
        bootstrap_quantile_method='higher', conditions=conditions,
        method='Fixed-original-SE standardized bootstrap mean deviations; a single maximum across all supplied conditions, predictors and times. Independent whole-sequence resampling across conditions, paired across predictors and times within each condition.',
        crossing_rule='First tested time with mean risk <= target, without interpolation or monotonicity enforcement; exact floating-point comparison with no added threshold tolerance.',
        uncertainty_scope='Approximate simultaneous confidence bands conditional on the frozen trained model(s). Reference numerical error and between-training-run variation are not included.',
        zero_se_rule='Exactly constant empirical coordinates have zero-width bands and zero bootstrap deviation. This is an empirical-bootstrap limitation, not proof of a deterministic population.',
        nonattainment='Unreached is categorical beyond the tested grid. Fractions are bootstrap diagnostics, not posterior probabilities. Difference/ratio intervals are absent whenever either compatible set includes nonattainment; no finite-only intervals are reported.',
        numerical_rule='Input risks down to -1e-12 are accepted as possible evaluation roundoff and preserved, not clipped. This tolerance is not used for crossing decisions.')

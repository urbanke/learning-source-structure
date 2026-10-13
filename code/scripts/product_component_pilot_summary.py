"""Independently check and summarize the predeclared product-component pilot.

All source draws and fit errors are retained. Weight comparisons use the
same component-identification mask for Bayes and transformer. Robust status
intervals are deterministic prediction-tolerance bounds, not confidence
intervals. A first observed switch requires an earlier observed background
state; unresolved prefixes and later reversals remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from eval_conditional_crossing import array_hash, fingerprint
from eval_product_components import checked_load, SCHEMA
from source import seed_of

CONTRASTS = np.array([[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=float)
CONTRASTS /= np.sqrt([2, 6, 12])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _close(actual, expected, name, tolerance=2e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Invalid shape or finite values: {name}")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if error > tolerance:
        raise ValueError(f"Recomputed {name} differs by {error:.6g}")
    return error


def _probabilities(values, name, tolerance):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid probabilities: {name}")
    error = float(np.max(np.abs(values.sum(axis=-1) - 1)))
    if error > tolerance:
        raise ValueError(f"Unnormalized {name}: {error:.6g}")
    return error


def kl_bits(p, q):
    """KL of the full predictive vectors, without clipping or normalization."""
    positive = p > 0
    if np.any(q[positive] <= 0):
        raise ValueError("Fitted/reference distribution omits positive target mass")
    terms = np.zeros_like(p, dtype=float)
    terms[positive] = p[positive] * np.log2(p[positive] / q[positive])
    return terms.sum(axis=-1)


def robustness_radius(residual_l2, sigma, epsilon):
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    radius = np.ones_like(sigma, dtype=float)
    np.divide(np.sqrt(3) / 2 * (2 * residual_l2 + epsilon), sigma,
              out=radius, where=sigma > 1e-12)
    return np.minimum(radius, 1.0)


def robust_states(weight, radius, eligible):
    lower, upper = np.maximum(0, weight - radius), np.minimum(1, weight + radius)
    return np.where(eligible & (lower > 0.5), 1,
                    np.where(eligible & (upper < 0.5), -1, 0)).astype(np.int8)


def full_copy_states(weights, residual, sigma, eligible, epsilon=0.001):
    """CC is uniquely largest, robustly non-largest, or unresolved.

    Pairwise weight differences have norm sqrt(2) on the affine simplex;
    CC majority (>1/2) is a different event and is not used here.
    """
    weights, residual, sigma, eligible = map(np.asarray, (weights, residual, sigma, eligible))
    if (weights.shape[-1] != 4 or weights.shape[:-1] != residual.shape or residual.shape != sigma.shape
            or eligible.shape != sigma.shape or not np.isfinite(weights).all()
            or not np.isfinite(residual).all() or not np.isfinite(sigma).all()
            or np.any(residual < 0) or np.any(sigma < 0) or epsilon < 0 or not np.isfinite(epsilon)):
        raise ValueError("Invalid inputs for full-copy robustness")
    bound = np.ones_like(sigma, dtype=float)
    np.divide(np.sqrt(2)*(2*residual+epsilon), sigma, out=bound, where=sigma > 1e-12)
    bound = np.minimum(bound, 1)
    differences = weights[..., 3, None] - weights[..., :3]
    return np.where(eligible & (differences.min(axis=-1) > bound), 1,
                    np.where(eligible & ((-differences).max(axis=-1) > bound), -1, 0)).astype(np.int8)


def crossing_record(states, positions):
    """Finite observed transitions, with unknown intervals never filled in."""
    states, positions = np.asarray(states), np.asarray(positions)
    if (states.ndim != 1 or states.shape != positions.shape or not len(states)
            or not np.isin(states, [-1, 0, 1]).all() or positions.dtype.kind not in "iu"
            or np.any(positions < 1) or np.any(np.diff(positions) <= 0)):
        raise ValueError("Expected aligned ordered prefix positions and -1/0/+1 states")
    positive, negative = np.flatnonzero(states == 1), np.flatnonzero(states == -1)
    resolved = np.flatnonzero(states != 0)
    transitions = []
    for left, right in zip(resolved[:-1], resolved[1:]):
        if states[left] != states[right]:
            transitions.append(dict(direction="up" if states[right] == 1 else "down",
                                    left=int(positions[left]), right=int(positions[right]),
                                    unresolved_between=int(np.sum(states[left + 1:right] == 0))))
    ups = [event for event in transitions if event["direction"] == "up"]
    first_positive = int(positions[positive[0]]) if len(positive) else None
    if ups:
        status = "observed_below_to_above"
    elif len(positive):
        status = "above_observed_without_prior_below"
    elif len(negative):
        status = "no_above_observed"
    else:
        status = "unresolved_throughout"
    return dict(status=status, first_positive=first_positive,
                first_negative=int(positions[negative[0]]) if len(negative) else None,
                first_positive_after_negative=ups[0]["right"] if ups else None,
                first_switch_bracket=ups[0] if ups else None,
                unresolved_before_first_positive=int(np.sum(states[:positive[0]] == 0)) if len(positive) else None,
                unresolved_count=int(np.sum(states == 0)), resolved_count=int(len(resolved)),
                observed_resolved_state_upcrossings=len(ups),
                observed_resolved_state_downcrossings=sum(event["direction"] == "down" for event in transitions),
                transitions=transitions,
                scope="Observed prefix window only; unresolved intervals can contain additional reversals.")


def statistics(values, mask=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite source-by-prefix values")
    mask = np.ones_like(values, dtype=bool) if mask is None else np.asarray(mask)
    if mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("Invalid source-by-prefix mask")
    means, errors, counts = [], [], []
    for column in range(values.shape[1]):
        selected = values[mask[:, column], column]
        counts.append(int(len(selected)))
        means.append(float(selected.mean()) if len(selected) else None)
        errors.append(float(selected.std(ddof=1) / np.sqrt(len(selected))) if len(selected) > 1 else None)
    return dict(mean=means, se=errors, count=counts)


def paired_statistics(bayes, transformer, eligible):
    if np.asarray(bayes).shape != np.asarray(transformer).shape:
        raise ValueError("Paired weight arrays must have identical shape")
    return dict(bayes=statistics(bayes, eligible), transformer=statistics(transformer, eligible),
                transformer_minus_bayes=statistics(np.asarray(transformer) - bayes, eligible),
                scope="Both predictors use exactly the same eligible source draws at each prefix; coverage can change.")


def inspect_arrays(arrays, *, sigma_threshold=0.02, epsilon=0.001, prob_tol=1e-7):
    """Recompute predictions, fit certificate, identification and sensitivity."""
    a = arrays
    p, b, c = (np.asarray(a[key]) for key in ("transformer_probs", "bayes_probs", "component_probs"))
    n = len(p)
    shapes = dict(tokens=(n + 1,), prefix_lengths=(n,), source_depths=(2,),
                  transformer_probs=(n, 64), bayes_probs=(n, 64), component_probs=(n, 4, 64),
                  bayes_weights=(n, 4), transformer_weights=(n, 4), coordinate_copy_weights=(n, 2),
                  coordinate_separation_tv=(n, 2), row_totals=(n, 2))
    vector_keys = ("fit_kl_bits", "fit_tv", "fit_l2", "bayes_kl_bits", "fit_optimality_gap",
                   "fit_converged", "affine_sigma_min", "eligible", "robustness_radius",
                   "copy_lower", "copy_upper", "both_lower", "both_upper")
    shapes.update({key: (n,) for key in vector_keys})
    if n < 1:
        raise ValueError("No evaluated prefixes")
    for key, shape in shapes.items():
        if key not in a or np.asarray(a[key]).shape != shape or not np.isfinite(a[key]).all():
            raise ValueError(f"Missing, malformed or nonfinite array: {key}")
    for key in ("tokens", "prefix_lengths", "source_depths", "row_totals"):
        if np.asarray(a[key]).dtype.kind not in "iu":
            raise ValueError(f"Expected integer array: {key}")
    for key in ("eligible", "fit_converged"):
        if np.asarray(a[key]).dtype.kind != "b":
            raise ValueError(f"Expected boolean array: {key}")
    if not np.array_equal(a["prefix_lengths"], np.arange(1, n + 1)):
        raise ValueError("Wrong observed-prefix/target alignment")
    if np.any(a["tokens"] < 0) or np.any(a["tokens"] >= 64) or np.any(a["row_totals"] < 0):
        raise ValueError("Invalid token labels or context counts")
    if np.any(a["source_depths"] < 1) or np.any(a["source_depths"] > 8):
        raise ValueError("Source draw is outside the declared depth prior")
    checks = {key + "_sum_error": _probabilities(a[key], key, prob_tol)
              for key in ("transformer_probs", "bayes_probs", "component_probs", "bayes_weights", "transformer_weights")}
    bw, tw = a["bayes_weights"], a["transformer_weights"]
    checks["bayes_reconstruction_error"] = _close(np.einsum("tk,tkd->td", bw, c), b, "Bayes reconstruction")
    fitted = np.einsum("tk,tkd->td", tw, c)
    checks["fitted_sum_error"] = _probabilities(fitted, "fitted probabilities", prob_tol)
    fit_kl = kl_bits(p, fitted)
    bayes_kl = kl_bits(p, b)
    fit_tv = np.abs(p - fitted).sum(axis=1) / 2
    residual = np.linalg.norm(p - fitted, axis=1)
    ratio = np.zeros_like(p)
    np.divide(p, fitted, out=ratio, where=p > 0)
    gradient = -np.einsum("tkd,td->tk", c, ratio)
    gap = np.maximum(0, np.einsum("tk,tk->t", tw, gradient) - gradient.min(axis=1))
    for key, value in (("fit_kl_bits", fit_kl), ("bayes_kl_bits", bayes_kl), ("fit_tv", fit_tv),
                       ("fit_l2", residual), ("fit_optimality_gap", gap)):
        checks[key + "_recomputation_error"] = _close(a[key], value, key)
    if not a["fit_converged"].all() or gap.max() > 1.01e-7:
        raise ValueError("A required convex fit did not pass its numerical optimality-gap check")
    if np.max(fit_kl - bayes_kl) > 2e-7:
        raise ValueError("Fit is worse than the feasible Bayesian component weights")
    checks["maximum_optimality_gap_nats"] = float(gap.max())
    checks["maximum_fit_minus_feasible_bayes_kl_bits"] = float(np.max(fit_kl - bayes_kl))
    sigma = np.linalg.svd(c.transpose(0, 2, 1) @ CONTRASTS, compute_uv=False)[:, -1]
    checks["affine_sigma_recomputation_error"] = _close(a["affine_sigma_min"], sigma, "affine singular value")
    eligible = sigma >= sigma_threshold
    if not np.array_equal(eligible, a["eligible"]):
        raise ValueError("Saved eligibility differs from the predeclared component-only criterion")
    # Recover coordinate components from joint marginals independently.
    reshaped = c.reshape(n, 4, 8, 8)
    c1 = reshaped[:, 1].sum(axis=2)
    c2 = reshaped[:, 2].sum(axis=1)
    uniform = np.full((n, 8), 1 / 8)
    predicted_components = np.stack([(x[:, :, None] * y[:, None, :]).reshape(n, 64)
                                     for x, y in ((uniform, uniform), (c1, uniform), (uniform, c2), (c1, c2))], axis=1)
    checks["component_factorization_error"] = _close(c, predicted_components, "component factorization")
    separation = np.stack((np.abs(c1 - uniform).sum(axis=1), np.abs(c2 - uniform).sum(axis=1)), axis=1) / 2
    checks["coordinate_separation_error"] = _close(a["coordinate_separation_tv"], separation, "coordinate separation")
    l1, l2 = a["coordinate_copy_weights"].T
    if np.any(a["coordinate_copy_weights"] < 0) or np.any(a["coordinate_copy_weights"] > 1):
        raise ValueError("Nonconvex coordinate copy weights")
    expected_weights = np.stack(((1-l1)*(1-l2), l1*(1-l2), (1-l1)*l2, l1*l2), axis=1)
    checks["bayes_weight_factorization_error"] = _close(bw, expected_weights, "Bayes product weights")
    # Reconstruct all current-context totals from the observed prefix, before target.
    tokens = a["tokens"].astype(int)
    totals = np.zeros((n, 2), dtype=int)
    for coordinate, x in enumerate((tokens // 8, tokens % 8)):
        seen = np.zeros(8, dtype=int)
        for j in range(n):
            totals[j, coordinate] = seen[x[j]]
            seen[x[j]] += 1
    if not np.array_equal(totals, a["row_totals"]):
        raise ValueError("Current-row totals include a future transition or use the wrong context")
    if np.any(a["coordinate_copy_weights"][totals == 0] != 0):
        raise ValueError("Empty context must have zero canonical copying weight")
    radius = robustness_radius(residual, sigma, epsilon)
    checks["radius_recomputation_error"] = _close(a["robustness_radius"], radius, "robustness radius")
    for key, values in (("copy", 1-tw[:, 0]), ("both", tw[:, 3])):
        _close(a[key + "_lower"], np.maximum(0, values-radius), key + " lower bound")
        _close(a[key + "_upper"], np.minimum(1, values+radius), key + " upper bound")
    return dict(checks=checks, fit_kl=fit_kl, bayes_kl=bayes_kl, fit_tv=fit_tv, fit_l2=residual,
                sigma=sigma, eligible=eligible, radius=radius,
                bayes_weights=bw, transformer_weights=tw,
                bayes_copy=1-bw[:, 0], transformer_copy=1-tw[:, 0],
                bayes_both=bw[:, 3], transformer_both=tw[:, 3])


def _verify_spec(spec, *, allow_smoke=False):
    expected = dict(schema=SCHEMA, source_family="product:8x8", source_depths=list(range(1, 9)),
                    alphabet_sizes=[8, 8], master_seed=20260928, prior_unchanged=True,
                    identification_sigma_threshold=0.02, sensitivity_epsilon_l2=0.001,
                    optimality_gap_tolerance_nats=1e-7)
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Protocol mismatch: {key}")
    if (not allow_smoke and (spec["n_seq"], spec["n_ctx"]) != (32, 1024)):
        raise ValueError("The primary report requires every one of the 32 declared length1024 draws")
    if spec["n_seq"] < (1 if allow_smoke else 2) or spec["n_ctx"] < 1:
        raise ValueError("Need the declared source draws and at least one prefix")
    for name, value in spec["provenance"].items():
        if sha(value["path"]) != value["sha256"]:
            raise ValueError(f"Current provenance hash changed: {name}")
    configuration = json.loads(Path(spec["provenance"]["config"]["path"]).read_text())
    if (configuration["family"] != "product:8x8" or configuration["d"] != 64
            or configuration["ls"] != [1, 8] or configuration["seed"] != 0):
        raise ValueError("Checkpoint configuration does not match the predeclared family")


def aggregate(directory, *, allow_smoke=False):
    directory = Path(directory)
    protocol_path = directory / "protocol.json"
    spec = json.loads(protocol_path.read_text())
    _verify_spec(spec, allow_smoke=allow_smoke)
    spec_id = fingerprint(spec)
    records, inputs, sequence_info = [], [], []
    for index in range(spec["n_seq"]):
        path = directory / f"seq_{index:06d}.npz"
        arrays = checked_load(path, spec_id)
        if arrays is None:
            raise ValueError(f"Missing predeclared source draw {index}; no subset aggregate is permitted")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
        sequence_seed = seed_of(SCHEMA, spec["master_seed"], index)
        if (meta["spec"] != spec or fingerprint(meta["spec"]) != spec_id
                or meta["index"] != index or meta["sequence_seed"] != sequence_seed):
            raise ValueError(f"Source index/seed/spec mismatch in draw {index}")
        if len(arrays["prefix_lengths"]) != spec["n_ctx"]:
            raise ValueError("A source draw has the wrong declared context length")
        if (not np.isfinite(meta["causal_prefix_replay_error"]) or meta["causal_prefix_replay_error"] > 2e-6
                or not np.isfinite(meta["bayes_reconstruction_error"]) or meta["bayes_reconstruction_error"] > 1e-7):
            raise ValueError("Stored causal/alignment or Bayes reconstruction check failed")
        records.append(inspect_arrays(arrays, sigma_threshold=spec["identification_sigma_threshold"],
                                      epsilon=spec["sensitivity_epsilon_l2"]))
        inputs.append(dict(path=relative(path), sha256=sha(path), index=index,
                           tokens_sha256=array_hash(arrays["tokens"]), metadata_sha256=fingerprint(meta)))
        sequence_info.append(dict(index=index, source_depths=arrays["source_depths"].tolist(), sequence_seed=sequence_seed))
    matrices = {key: np.stack([record[key] for record in records]) for key in records[0] if key != "checks"}
    eligible = matrices["eligible"]
    paired = {key: paired_statistics(matrices["bayes_"+key], matrices["transformer_"+key], eligible)
              for key in ("copy", "both")}
    positions = np.arange(1, spec["n_ctx"] + 1)
    sensitivity, crossings = {}, []
    for epsilon in (0.0, 0.001, 0.005):
        radius = robustness_radius(matrices["fit_l2"], matrices["sigma"], epsilon)
        states = robust_states(matrices["transformer_copy"], radius, eligible)
        key = format(epsilon, ".3f")
        lower = np.maximum(0, matrices["transformer_copy"] - radius)
        upper = np.minimum(1, matrices["transformer_copy"] + radius)
        cc_states = full_copy_states(matrices["transformer_weights"], matrices["fit_l2"], matrices["sigma"], eligible, epsilon)
        sensitivity[key] = dict(epsilon_l2=epsilon, robust_positive_fraction=(states == 1).mean(axis=0).tolist(),
                                robust_negative_fraction=(states == -1).mean(axis=0).tolist(),
                                unresolved_fraction=(states == 0).mean(axis=0).tolist(),
                                lower_bound_mean=lower.mean(axis=0).tolist(), upper_bound_mean=upper.mean(axis=0).tolist(),
                                per_sequence=[crossing_record(row, positions) for row in states],
                                cc_robust_positive_fraction=(cc_states == 1).mean(axis=0).tolist(),
                                cc_robust_negative_fraction=(cc_states == -1).mean(axis=0).tolist(),
                                cc_unresolved_fraction=(cc_states == 0).mean(axis=0).tolist(),
                                cc_per_sequence=[crossing_record(row, positions) for row in cc_states])
    primary = sensitivity["0.001"]
    cc_primary_states = full_copy_states(matrices["transformer_weights"], matrices["fit_l2"], matrices["sigma"], eligible, 0.001)
    for index, info in enumerate(sequence_info):
        bw, tw, mask = matrices["bayes_copy"][index], matrices["transformer_copy"][index], eligible[index]
        bayes_states = np.where(bw > .5, 1, np.where(bw < .5, -1, 0))
        point_states = np.where(mask & (tw > .5), 1, np.where(mask & (tw < .5), -1, 0))
        paired_bayes_states = np.where(mask, bayes_states, 0)
        bweights, tweights = matrices["bayes_weights"][index], matrices["transformer_weights"][index]
        bmargin = bweights[:, 3] - bweights[:, :3].max(axis=1)
        tmargin = tweights[:, 3] - tweights[:, :3].max(axis=1)
        bcc = np.where(bmargin > 0, 1, np.where(bmargin < 0, -1, 0))
        tcc = np.where(mask & (tmargin > 0), 1, np.where(mask & (tmargin < 0), -1, 0))
        crossings.append(dict(**info, bayes_canonical=crossing_record(bayes_states, positions),
                              bayes_on_eligible=crossing_record(paired_bayes_states, positions),
                              transformer_eligible_point=crossing_record(point_states, positions),
                              transformer_robust=primary["per_sequence"][index],
                              cc_bayes_canonical=crossing_record(bcc, positions),
                              cc_bayes_on_eligible=crossing_record(np.where(mask, bcc, 0), positions),
                              cc_transformer_eligible_point=crossing_record(tcc, positions),
                              cc_transformer_robust=primary["cc_per_sequence"][index]))
    diagnostics = {key: statistics(matrices[key]) for key in ("fit_kl", "bayes_kl", "fit_tv", "fit_l2", "sigma")}
    checks = {key: max(record["checks"][key] for record in records) for key in records[0]["checks"]}
    status_counts = {name: sum(record["transformer_robust"]["status"] == name for record in crossings)
                     for name in ("observed_below_to_above", "above_observed_without_prior_below",
                                  "no_above_observed", "unresolved_throughout")}
    cc_status_counts = {name: sum(record["cc_transformer_robust"]["status"] == name for record in crossings)
                        for name in status_counts}
    timing_pairs = [dict(index=row["index"], bayes_first=row["cc_bayes_on_eligible"]["first_positive"],
                         transformer_first=row["cc_transformer_eligible_point"]["first_positive"],
                         transformer_robust_state_at_point_first=(int(cc_primary_states[row["index"], row["cc_transformer_eligible_point"]["first_positive"]-1])
                             if row["cc_transformer_eligible_point"]["first_positive"] is not None else None))
                    for row in crossings]
    timing_complete = all(row["bayes_first"] is not None and row["transformer_first"] is not None for row in timing_pairs)
    timing = dict(all_draws_have_both_first_positive=timing_complete, pairs=timing_pairs,
                  interpretation="First eligible point-estimate CC-largest observations; these differ from robust switch brackets. A point-estimate timing difference does not establish robust timing disagreement, especially when the fitted first event is unresolved. No permanence follows.")
    if timing_complete:
        differences = np.array([row["transformer_first"]-row["bayes_first"] for row in timing_pairs])
        timing.update(exact_matches=int(np.sum(differences == 0)), within_five=int(np.sum(np.abs(differences) <= 5)),
                      signed_mean=float(differences.mean()), mean_absolute=float(np.abs(differences).mean()),
                      median_absolute=float(np.median(np.abs(differences))), maximum_absolute=int(np.abs(differences).max()),
                      paired_source_statistics=statistics(differences[:, None]))
    window_summaries = []
    for left, right in ((1, 32), (33, 128), (129, 1024)):
        right = min(right, spec["n_ctx"])
        if left > right:
            continue
        sl = slice(left-1, right)
        window = dict(first_prefix=left, last_prefix=right, n_sequences=spec["n_seq"],
                      eligible_prefix_count_per_sequence=eligible[:, sl].sum(axis=1).tolist(),
                      eligible_fraction=float(eligible[:, sl].mean()),
                      fit_kl_bits=statistics(matrices["fit_kl"][:, sl].mean(axis=1)[:, None]),
                      bayes_kl_bits=statistics(matrices["bayes_kl"][:, sl].mean(axis=1)[:, None]),
                      fit_tv=statistics(matrices["fit_tv"][:, sl].mean(axis=1)[:, None]))
        errors = {name: matrices["transformer_"+name][:, sl]-matrices["bayes_"+name][:, sl] for name in ("copy", "both")}
        errors["four_weight_l1"] = np.abs(matrices["transformer_weights"][:, sl]-matrices["bayes_weights"][:, sl]).sum(axis=-1)
        for name, difference in errors.items():
            available = eligible[:, sl].any(axis=1)
            def reduce_on_shared_mask(values):
                means = np.array([row[mask].mean() if mask.any() else 0 for row, mask in zip(values, eligible[:, sl])])
                return statistics(means[:, None], available[:, None])
            window[name+"_paired_error"] = dict(signed=reduce_on_shared_mask(difference),
                                                 absolute=reduce_on_shared_mask(np.abs(difference)),
                                                 mean_squared=reduce_on_shared_mask(difference**2))
        window_summaries.append(window)
    return dict(schema="product-component-pilot-summary-v1", passed=True, units="bits",
                spec=spec, spec_id=spec_id, protocol=dict(path=relative(protocol_path), sha256=sha(protocol_path)),
                inputs=inputs, n_sequences=spec["n_seq"], prefix_lengths=positions.tolist(),
                paired_weights=paired, diagnostics=diagnostics,
                eligible_fraction=eligible.mean(axis=0).tolist(), sensitivity=sensitivity, crossings=crossings,
                crossing_status_counts=status_counts, cc_crossing_status_counts=cc_status_counts,
                cc_point_first_observation_comparison=timing,
                cc_robust_draws_with_later_return_below=sum(row["cc_transformer_robust"]["observed_resolved_state_downcrossings"] > 0 for row in crossings),
                any_copy_draws_with_robust_background=sum(row["transformer_robust"]["first_negative"] is not None for row in crossings),
                early_middle_late_windows=window_summaries, maximum_check_errors=checks,
                per_sequence_time_average_fit_kl_bits=statistics(matrices["fit_kl"].mean(axis=1)[:, None]),
                uncertainty="One standard error across independent source draws at each prefix; paired weight differences use the same eligibility mask. One fixed training checkpoint.",
                weight_interpretation="Weights describe the projection onto specified predictive components. They are not source-model posterior probabilities or evidence of an internal transformer mechanism.",
                robustness_interpretation="For any mixture with L2 residual to transformer at most fitted residual+epsilon, copy weight lies in the stated deterministic bound. These bounds are not confidence intervals.",
                crossing_interpretation="Observed below-to-above switches require an earlier observed below state. Unknown gaps and subsequent resolved-state reversals are retained. No all-future crossing or permanence claim.",
                full_copy_event="CC weight is strictly greater than each of UU, CU and UC; this is not the CC>1/2 majority event.",
                script=relative(__file__), script_sha256=sha(__file__))


def _line(ax, x, stats, *, label, color, style="-"):
    mean, se = np.array(stats["mean"], dtype=float), np.array(stats["se"], dtype=float)
    ax.plot(x, mean, style, color=color, linewidth=1.1, label=label)
    ax.fill_between(x, mean-se, mean+se, color=color, alpha=.12)


def plot(report, pdf, png):
    import os
    png.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(png.parent / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    t = np.array(report["prefix_lengths"])
    blue, orange, green = "#22688a", "#bf5d32", "#397447"
    for key, name, style in (("copy", "any copy", "-"), ("both", "both copy", "--")):
        _line(axes[0, 0], t, report["paired_weights"][key]["bayes"], label="Bayes: "+name, color=blue, style=style)
        _line(axes[0, 0], t, report["paired_weights"][key]["transformer"], label="Transformer: "+name, color=orange, style=style)
    axes[0, 0].set_ylim(-.025, 1.025)
    axes[0, 0].set_title("A  Paired weights on identifiable prefixes", loc="left")
    axes[0, 0].set_ylabel("Mean predictive weight")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")
    _line(axes[0, 1], t, report["diagnostics"]["fit_kl"], label="Four-component fit", color=orange)
    _line(axes[0, 1], t, report["diagnostics"]["bayes_kl"], label="Family Bayes", color=blue)
    axes[0, 1].set_title("B  Prediction discrepancy: all source draws", loc="left")
    axes[0, 1].set_ylabel("KL(transformer || prediction) [bits]")
    axes[0, 1].legend(frameon=False, fontsize=8)
    primary = report["sensitivity"]["0.001"]
    axes[1, 0].plot(t, report["eligible_fraction"], color=green, label="Component-identifiable")
    axes[1, 0].plot(t, 1-np.array(primary["cc_unresolved_fraction"]), color=orange, label="Robust CC / non-CC")
    axes[1, 0].plot(t, primary["cc_robust_positive_fraction"], color=blue, linestyle="--", label="Robust CC uniquely largest")
    axes[1, 0].set_ylim(-.025, 1.025)
    axes[1, 0].set_ylabel("Fraction of all declared source draws")
    axes[1, 0].set_title("C  Identification and robust status coverage", loc="left")
    axes[1, 0].legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[1, 1]
    events = [row[key]["first_switch_bracket"] for row in report["crossings"]
              for key in ("cc_bayes_canonical", "cc_transformer_robust")]
    first_max = max([event["right"] for event in events if event] or [1])
    last_shown = min(t[-1], max(8, first_max*1.2))
    missing_event = any(event is None for event in events)
    endpoint = last_shown * 1.24
    for row in report["crossings"]:
        y = row["index"] + 1
        for key, color, offset in (("cc_bayes_canonical", blue, -.16), ("cc_transformer_robust", orange, .16)):
            event = row[key]["first_switch_bracket"]
            if event:
                ax.plot([event["left"], event["right"]], [y+offset]*2, color=color, linewidth=.8)
                ax.scatter(event["right"], y+offset, s=13, color=color, marker="o", zorder=3)
            else:
                ax.scatter(endpoint, y+offset, s=13, color=color, marker="x", linewidth=.8)
        if row["cc_transformer_robust"]["observed_resolved_state_downcrossings"]:
            event = row["cc_transformer_robust"]["first_switch_bracket"]
            if event:
                ax.scatter(event["right"], y+.16, s=30, facecolors="none", edgecolors=orange, linewidth=.6)
    ax.set_ylim(report["n_sequences"]+.8, .2)
    ticks = [1] + list(range(8, report["n_sequences"]+1, 8))
    ax.set_yticks(ticks)
    ax.set_ylabel("Predeclared source draw")
    ax.set_title("D  First observed non-CC to CC-largest intervals", loc="left")
    ax.plot([], [], color=blue, marker="o", markersize=3, label="Bayes canonical")
    ax.plot([], [], color=orange, marker="o", markersize=3, label="Transformer robust")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for row in axes:
        for axis in row:
            axis.set_xscale("log", base=2)
            axis.set_xticks([x for x in (1, 4, 16, 64, 256, 1024) if x <= t[-1]])
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.set_xlim(1, t[-1])
            axis.grid(axis="y", color=".9", linewidth=.5)
            axis.set_xlabel("Observed prefix length")
    first_shown = min([event["left"] for event in events if event] or [1])
    ax.set_xlim(2 if first_shown >= 2 else 1, endpoint*1.08 if missing_event else last_shown)
    ax.set_xticks([x for x in (2, 4, 8, 16, 32, 64, 128, 256, 1024) if x <= last_shown])
    if missing_event:
        ax.text(endpoint, 0, "none", ha="center", va="bottom", fontsize=7,
                transform=ax.get_xaxis_transform())
    fig.suptitle("Product-source diagnostic: predictive weights and their limits", y=.995, fontsize=12)
    fig.text(.5, .025, f"{report['n_sequences']} independent 8 x 8 source draws; one fixed checkpoint. Shading: one source standard error. "
             "Identification: sigma >= 0.02; robustness epsilon = 0.001.", ha="center", fontsize=7.7)
    fig.text(.5, .009, "D: resolved intervals, not confidence intervals; Bayes and robust transformer intervals use different criteria. "
             "Ring: later return to non-CC; x: no switch.", ha="center", fontsize=7.7)
    fig.tight_layout(rect=(0, .055, 1, .965), h_pad=2, w_pad=2.5)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=160)
    plt.close(fig)


def save_tables(report, directory):
    with (directory / "summary.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["quantity", "prefix_length", "mean", "source_standard_error", "n_sources"])
        for quantity, statistics_by_prefix in report["diagnostics"].items():
            for t, mean, se, count in zip(report["prefix_lengths"], *[statistics_by_prefix[k] for k in ("mean", "se", "count")]):
                writer.writerow([quantity, t, mean, se, count])
        for quantity, paired in report["paired_weights"].items():
            for predictor in ("bayes", "transformer", "transformer_minus_bayes"):
                stats = paired[predictor]
                for t, mean, se, count in zip(report["prefix_lengths"], *[stats[k] for k in ("mean", "se", "count")]):
                    writer.writerow([quantity+":"+predictor, t, mean, se, count])
    with (directory / "crossings.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source_index", "depth1", "depth2", "prediction", "status", "first_positive",
                         "first_positive_after_negative", "bracket_left", "bracket_right", "unresolved_count", "resolved_downcrossings"])
        for row in report["crossings"]:
            for key in ("bayes_canonical", "bayes_on_eligible", "transformer_eligible_point", "transformer_robust",
                        "cc_bayes_canonical", "cc_bayes_on_eligible", "cc_transformer_eligible_point", "cc_transformer_robust"):
                value = row[key]
                event = value["first_switch_bracket"] or {}
                writer.writerow([row["index"], *row["source_depths"], key, value["status"], value["first_positive"],
                                 value["first_positive_after_negative"], event.get("left"), event.get("right"),
                                 value["unresolved_count"], value["observed_resolved_state_downcrossings"]])
    with (directory / "windows.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["first_prefix", "last_prefix", "quantity", "mean", "source_standard_error", "n_sources", "eligible_fraction"])
        for window in report["early_middle_late_windows"]:
            values = {key: window[key] for key in ("fit_kl_bits", "bayes_kl_bits", "fit_tv")}
            for key in ("copy", "both", "four_weight_l1"):
                values.update({key+"_error:"+name: value for name, value in window[key+"_paired_error"].items()})
            for key, value in values.items():
                writer.writerow([window["first_prefix"], window["last_prefix"], key, value["mean"][0], value["se"][0],
                                 value["count"][0], window["eligible_fraction"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/product_component_pilot/run_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/product_component_pilot")
    parser.add_argument("--png", type=Path, default=ROOT / "output/product_component_pilot/diagnostic.png")
    parser.add_argument("--smoke", action="store_true", help="Permit a separately declared small cohort; never subset the full cohort")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        report = aggregate(args.input, allow_smoke=args.smoke)
    except (ValueError, OSError, KeyError) as error:
        failure = dict(schema="product-component-pilot-summary-v1", passed=False, error=str(error),
                       input=relative(args.input), script_sha256=sha(__file__))
        (args.out / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure))
        return 1
    pdf = args.out / "diagnostic.pdf"
    plot(report, pdf, args.png)
    report["figure"] = dict(path=relative(pdf), sha256=sha(pdf), all_positions_plotted=True, smoothing=False)
    save_tables(report, args.out)
    report["tables"] = {name: dict(path=relative(args.out / name), sha256=sha(args.out / name))
                        for name in ("summary.tsv", "crossings.tsv", "windows.tsv")}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, n_sequences=report["n_sequences"],
                          crossing_status_counts=report["crossing_status_counts"],
                          cc_crossing_status_counts=report["cc_crossing_status_counts"],
                          time_average_fit_kl_bits=report["per_sequence_time_average_fit_kl_bits"],
                          output=relative(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

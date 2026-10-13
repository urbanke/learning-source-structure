"""Summarize frozen product-predictor mismatch and parameter-switch experiments.

No probability array is renormalized. Partial summaries use the same contiguous
prefix of sequence indices in every depth condition, and are marked preliminary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import xlogy


SCHEMA = "product-shift-summary-v1"
CONDITIONS = ["stationary", "relabeled", "switch_one", "switch_both"]
PRECHANGE_TIMES = [192, 224, 240, 248, 255]
LN2 = np.log(2.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    """Write complete JSON atomically, rejecting nonfinite numerical results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def scalar(record, key, path):
    require(key in record and np.asarray(record[key]).shape == (),
            f"{path}: missing or nonscalar {key}")
    return record[key].item()


def statistics(values):
    values = np.asarray(values, dtype=float)
    require(len(values) >= 2 and np.isfinite(values).all(), "Statistics need at least two finite records")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return dict(mean=mean.tolist(), se=se.tolist(),
                lower_pointwise_95=(mean - 1.96 * se).tolist(),
                upper_pointwise_95=(mean + 1.96 * se).tolist())


def read_record(path, plan, pair, index):
    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    require(str(scalar(record, "plan_id", path)) == plan["plan_id"], f"{path}: different plan")
    cache_id = str(scalar(record, "cache_id", path))
    expected = hashlib.sha256(json.dumps(dict(plan_id=plan["plan_id"], pair=pair, index=index),
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(cache_id == expected, f"{path}: wrong cache identity")
    for name, value in record.items():
        if np.issubdtype(value.dtype, np.number):
            require(np.isfinite(value).all(), f"{path}: nonfinite {name}")
    if "index" in record:
        require(int(scalar(record, "index", path)) == index, f"{path}: wrong sequence index")
    for key in ("pair", "depth_pair"):
        if key in record:
            require(np.array_equal(record[key], pair), f"{path}: wrong depth pair")
    times = np.asarray(plan["times"], dtype=int)
    if "times" in record:
        require(np.array_equal(record["times"], times), f"{path}: wrong prediction times")
    if "conditions" in record:
        require(np.array_equal(record["conditions"], CONDITIONS), f"{path}: wrong condition order")
    tokens = record.get("tokens")
    require(tokens is not None and tokens.shape == (4, 1025) and tokens.dtype.kind in "iu",
            f"{path}: wrong token shape or dtype")
    require(np.all((tokens >= 0) & (tokens < 64)), f"{path}: token outside vocabulary")
    require(float(scalar(record, "elapsed_seconds", path)) >= 0, f"{path}: negative elapsed time")
    normalization = {}
    for key in ("true_probs", "bayes_probs", "transformer_probs"):
        value = record.get(key)
        require(value is not None and value.shape == (4, len(times), 64), f"{path}: wrong {key} shape")
        require(np.isfinite(value).all() and np.min(value) >= 0, f"{path}: invalid {key}")
        if key != "true_probs":
            require(np.min(value) > 0, f"{path}: zero predictor support")
        error = float(np.max(np.abs(value.sum(axis=-1) - 1.0)))
        tolerance = 2e-12 if key == "true_probs" else 2e-6
        require(error <= tolerance, f"{path}: raw {key} normalization error {error}")
        normalization[key] = error
    record["normalization"] = normalization
    return record


def risks(record):
    true, bayes, model = (np.asarray(record[key], dtype=float) for key in
                          ("true_probs", "bayes_probs", "transformer_probs"))
    entropy = -np.sum(xlogy(true, true), axis=-1) / LN2
    rb = np.sum(xlogy(true, true) - xlogy(true, bayes), axis=-1) / LN2
    rt = np.sum(xlogy(true, true) - xlogy(true, model), axis=-1) / LN2
    gap = np.sum(true * (np.log(bayes) - np.log(model)), axis=-1) / LN2
    agreement = np.sum(xlogy(bayes, bayes) - xlogy(bayes, model), axis=-1) / LN2
    require(np.max(np.abs(gap - (rt - rb))) <= 2e-12, "Paired regret identity failed")
    require(min(rb.min(), rt.min(), agreement.min()) >= -4e-6, "Negative KL beyond normalization tolerance")
    result = dict(transformer_regret_bits=rt, bayes_regret_bits=rb,
                  transformer_minus_bayes_bits=gap, kl_bayes_to_transformer_bits=agreement,
                  true_entropy_bits=entropy)
    require(all(np.isfinite(value).all() for value in result.values()), "Nonfinite derived metric")
    return result


def recovery(mean_risk, times, switch):
    """Descriptive grid crossing, with no confidence interval claim."""
    mean_risk = np.asarray(mean_risk, dtype=float)
    positions = [int(np.flatnonzero(times == t)[0]) for t in PRECHANGE_TIMES]
    baseline = float(mean_risk[positions].mean())
    threshold = baseline + 0.05
    post = np.flatnonzero(times >= switch)
    within = mean_risk[post] <= threshold
    persistent = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(persistent)
    result = dict(prechange_mean_regret_bits=baseline, threshold_bits=threshold,
                  attained=bool(len(candidates)), observed_tokens=None, observations_since_switch=None,
                  bracketing_tested_delays=None, uncertainty_interval=None,
                  uncertainty="Descriptive crossing of sample means; no crossing confidence interval computed.")
    if len(candidates):
        j = int(candidates[0])
        current = int(times[post[j]] - switch)
        preceding = int(times[post[j - 1]] - switch) if j else 0
        result.update(status="attained_on_tested_grid", observed_tokens=int(times[post[j]]),
                      observations_since_switch=current, bracketing_tested_delays=[preceding, current],
                      tested_points_within_threshold_through_horizon=int(len(post) - j))
    else:
        result.update(status="not_attained_within_observed_horizon",
                      censoring_delay=int(times[post[-1]] - switch))
    return result


def figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    titles = {"stationary": "Visible product labels", "relabeled": "Random joint labels",
              "switch_one": "One new matrix after t=256", "switch_both": "Two new matrices after t=256"}
    times = np.asarray(summary["times"])
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.3), squeeze=False)
    for row, pair in enumerate(summary["depth_pairs"]):
        for col, condition in enumerate(summary["conditions"]):
            axis = axes[row, col]
            group = next(g for g in summary["groups"] if g["depth_pair"] == pair and g["condition"] == condition)
            for key, color, style in (("transformer_regret_bits", "#1565a8", "-"),
                                      ("bayes_regret_bits", "#d77719", "--")):
                stat = group["metrics"][key]
                mean = np.asarray(stat["mean"])
                axis.plot(times, mean, color=color, linestyle=style, linewidth=1.6)
                axis.fill_between(times, stat["lower_pointwise_95"], stat["upper_pointwise_95"],
                                  color=color, alpha=.15, linewidth=0)
            axis.set_title(titles[condition] if row == 0 else "")
            axis.set_ylim(bottom=0)
            axis.grid(True, alpha=.15)
            if condition.startswith("switch"):
                axis.axvline(summary["switch_observed"], color=".35", linestyle=":", linewidth=1.1)
                axis.set_xticks([0, 256, 512, 768, 1024])
                axis.set_xlim(0, 1024)
                axis.text(summary["switch_observed"] + 18, .96, "change", fontsize=8,
                          transform=axis.get_xaxis_transform(), va="top", color=".35")
            else:
                axis.set_xscale("log", base=2)
                axis.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
                axis.set_xlim(1, 1024)
            if col == 0:
                axis.set_ylabel(rf"$L_1=L_2={pair[0]}$" + "\nNext-token regret (bits)")
            if row == 1:
                axis.set_xlabel("Observed tokens")
    fig.legend([Line2D([], [], color="#1565a8", lw=1.7),
                Line2D([], [], color="#d77719", lw=1.7, ls="--")],
               ["Frozen product transformer", "Bayes with the original product prior"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    label = "Complete cohort" if summary["complete"] else "Preliminary partial cohort"
    fig.text(.5, .01, f"{label}: {summary['evaluated_n_sequences_per_depth']} independent sources per depth; "
             "shading: mean ± 1.96 SE (pointwise).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .945))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"overview.{extension}", dpi=180, metadata={"Creator": SCHEMA})
    plt.close(fig)


def summarize(out, allow_partial=False):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    require(plan["conditions"] == CONDITIONS, "Unexpected condition order")
    require(plan["pairs"] == [[1, 1], [4, 4]], "Unexpected source depths")
    require(isinstance(plan["plan_id"], str) and plan["plan_id"], "Missing plan identity")
    plan_fingerprint = hashlib.sha256(json.dumps({key: value for key, value in plan.items() if key != "plan_id"},
                                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(plan["plan_id"] == plan_fingerprint, "Plan fingerprint differs from its contents")
    n = int(plan["n_seq"])
    require(n >= 2, "At least two sequences required")
    times = np.asarray(plan["times"])
    require(times.ndim == 1 and times.dtype.kind in "iu" and len(times) > 0 and
            np.all(np.diff(times) > 0) and times[0] >= 1 and times[-1] <= 1024,
            "Invalid prediction grid")
    require(int(plan["switch_observed"]) == 256 and set(PRECHANGE_TIMES).issubset(times.tolist()) and
            256 in times and 1024 in times, "Required switch/recovery times missing")
    prefixes, present = [], []
    for pair in plan["pairs"]:
        directory = out / "records" / f"L{pair[0]}_{pair[1]}"
        available = [i for i in range(n) if (directory / f"seq_{i:06d}.npz").is_file()]
        present.append(len(available))
        prefix = 0
        while prefix in available:
            prefix += 1
        prefixes.append(prefix)
    count = min(prefixes)
    complete = count == n
    require(complete or allow_partial, f"Incomplete cohort: contiguous counts {prefixes}, expected {n}")
    require(count >= 2, "Need at least two common contiguous sequence indices")
    groups, arrays, elapsed = [], {}, []
    raw_hash = hashlib.sha256()
    cache_ids = set()
    normalization = {key: 0.0 for key in ("true_probs", "bayes_probs", "transformer_probs")}
    for pair in plan["pairs"]:
        metrics, pair_elapsed = {}, []
        for i in range(count):
            path = out / "records" / f"L{pair[0]}_{pair[1]}" / f"seq_{i:06d}.npz"
            record = read_record(path, plan, pair, i)
            cache_id = str(record["cache_id"].item())
            require(cache_id not in cache_ids, f"{path}: repeated cache identity")
            cache_ids.add(cache_id)
            raw_hash.update(str(path.relative_to(out)).encode() + b":" + hashlib.sha256(path.read_bytes()).digest())
            for key, value in record["normalization"].items():
                normalization[key] = max(normalization[key], value)
            for key, value in risks(record).items():
                metrics.setdefault(key, []).append(value)
            pair_elapsed.append(float(record["elapsed_seconds"].item()))
        metrics = {key: np.asarray(value) for key, value in metrics.items()}
        for key, value in metrics.items():
            arrays.setdefault(key, []).append(value)
        elapsed.append(pair_elapsed)
        for j, condition in enumerate(CONDITIONS):
            group = dict(depth_pair=pair, condition=condition, n_sequences=count,
                         metrics={key: statistics(value[:, j]) for key, value in metrics.items()},
                         paired_condition_minus_stationary={key: statistics(value[:, j] - value[:, 0])
                                                            for key, value in metrics.items()})
            if condition.startswith("switch"):
                group["recovery"] = {key: recovery(metrics[key][:, j].mean(axis=0), times, 256)
                                     for key in ("transformer_regret_bits", "bayes_regret_bits")}
            groups.append(group)
    summary = dict(schema=SCHEMA, plan_id=plan["plan_id"], complete=complete,
                   status="complete" if complete else "preliminary_partial",
                   expected_n_sequences_per_depth=n, evaluated_n_sequences_per_depth=count,
                   available_records_per_depth=present, contiguous_records_per_depth=prefixes,
                   included_indices=list(range(count)), depth_pairs=plan["pairs"], conditions=CONDITIONS,
                   times=times.tolist(), switch_observed=256, groups=groups,
                   raw_records_sha256=raw_hash.hexdigest(), raw_normalization_max_error=normalization,
                   probabilities_renormalized=False,
                   uncertainty="Mean ± 1.96 standard errors across independent source/sequence draws; pointwise, not simultaneous. Condition contrasts use paired draws.",
                   model_scope="One frozen training seed; no training-seed uncertainty estimate.",
                   recovery_rule=dict(primary="Full regret curve", secondary="Return to own prechange mean plus 0.05 bits",
                       prechange_observed_tokens=PRECHANGE_TIMES, allowance_bits=0.05,
                       prechange_average="Unweighted average of the mean regrets at the five specified times",
                       persistence="All later tested times through 1024 must remain within threshold",
                       grid_interpretation="Observed-grid estimate only; no assertion between tested points",
                       uncertainty="No recovery-time confidence intervals computed"))
    risk_path = out / "per_sequence_risks.npz"
    temporary = risk_path.with_name(risk_path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, plan_id=plan["plan_id"], depth_pairs=plan["pairs"],
                                conditions=CONDITIONS, times=times, indices=np.arange(count),
                                elapsed_seconds=np.asarray(elapsed),
                                **{key: np.asarray(value) for key, value in arrays.items()})
        temporary.replace(risk_path)
    finally:
        temporary.unlink(missing_ok=True)
    figures(summary, out)
    atomic_json(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in
                      ("status", "complete", "evaluated_n_sequences_per_depth", "expected_n_sequences_per_depth")}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.out, args.allow_partial)


if __name__ == "__main__":
    main()

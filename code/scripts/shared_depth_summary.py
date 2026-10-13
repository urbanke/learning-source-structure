"""Independently verify and export the shared-depth reevaluation summaries.

Reads the evaluation driver's raw per-sequence caches, without importing its
aggregation code. Checks provenance, token/prefix alignment, evidence/loss
consistency, normalization, and independently recomputed paired means/SE and
predictive KL. Only verified small JSON/TSV summaries are exported.

    python code/scripts/shared_depth_summary.py
    python code/scripts/shared_depth_summary.py --input /private/tmp/smoke \
        --out /private/tmp/smoke-summary

All cumulative statistics use every scored token. Predictive KL is available
only at the selected prefixes and is never interpolated into cumulative KL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ("uniform_d64", "product_8x8")
LN2 = math.log(2.0)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def token_digest(tokens):
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype="<i2").tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def moments(values):
    """Independent sum/deviation implementation; rows are source draws."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        raise ValueError("At least two independent sequences are required for standard errors")
    mean = np.sum(values, axis=0) / n
    deviation = values - mean
    se = np.sqrt(np.sum(deviation * deviation, axis=0) / (n * (n - 1)))
    return {"mean": mean, "se": se}


def check_array(actual, expected, name, tolerance=2e-8):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Nonfinite or mismatched array: {name}")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if error > tolerance:
        raise ValueError(f"Independent check failed for {name}: absolute error {error:.6g} > {tolerance:.6g}")
    return error


def check_probabilities(values, name, tolerance):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid probability array: {name}")
    error = float(np.max(np.abs(np.sum(values, axis=1) - 1.0)))
    if error > tolerance:
        raise ValueError(f"Unnormalized probabilities: {name} ({error:.6g})")
    return error


def predictive_kl(p, q):
    """Direct probability-ratio computation, distinct from driver log subtraction."""
    terms = np.zeros_like(p, dtype=np.float64)
    positive = p > 0
    if (q[positive] <= 0).any():
        raise ValueError("Positive Bayes probability has zero comparison probability")
    terms[positive] = p[positive] * np.log(p[positive] / q[positive])
    return np.sum(terms, axis=1) / LN2


def audit_case(base):
    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text())
    n = int(summary["n_sequences"])
    horizon = int(summary["source"]["n_ctx"])
    alphabet = int(summary["source"]["d"])
    prefixes = np.asarray(summary["prefixes"], dtype=int)
    if n < 2 or horizon < 1 or not np.array_equal(prefixes, np.unique(prefixes)):
        raise ValueError("Invalid sample count, horizon or prefix ordering")
    if len(prefixes) == 0 or prefixes.min() < 1 or prefixes.max() != horizon:
        raise ValueError("Selected prefixes must be positive and include the complete horizon")
    if summary["first_token_scored"] or summary["units"] != "bits":
        raise ValueError("Unexpected scoring convention")
    tolerance = float(summary["reference"]["probability_tolerance"])
    losses = {name: np.empty((n, horizon), dtype=np.float64) for name in ("bayes", "local", "transformer")}
    divergence = {name: np.empty((n, len(prefixes))) for name in ("kl_bayes_to_transformer", "kl_bayes_to_local")}
    probability_errors = np.empty((n, 3))
    tokens = np.empty((n, horizon + 1), dtype=np.int16)
    depth_counts = {}
    raw_hash = hashlib.sha256()
    alignment_error = 0.0
    for i in range(n):
        name = f"seq_{i:06d}.npz"
        paths = [base / folder / name for folder in ("data", "references", "model")]
        source, ref, net = [read_npz(path) for path in paths]
        for path in paths:
            raw_hash.update(path.relative_to(base).as_posix().encode())
            raw_hash.update(bytes.fromhex(file_digest(path)))
        source_meta, ref_meta, net_meta = [json.loads(str(cache["metadata_json"])) for cache in (source, ref, net)]
        source_identity = {key: value for key, value in source_meta.items() if key != "runtime"}
        if str(source["cache_id"]) != digest(source_identity):
            raise ValueError(f"Source cache identity failed at sequence {i}")
        if source_meta["source"] != summary["source"] or source_meta["index"] != i:
            raise ValueError(f"Source specification/index mismatch at sequence {i}")
        if ref_meta["reference"] != summary["reference"] or net_meta["model"] != summary["model"]:
            raise ValueError(f"Predictor specification mismatch at sequence {i}")
        for cache, key in ((ref, "reference"), (net, "model")):
            expected_id = digest({"data_id": str(source["cache_id"]), key: summary[key]})
            if str(cache["cache_id"]) != expected_id:
                raise ValueError(f"{key} cache identity failed at sequence {i}")
        x = source["tokens"]
        if x.shape != (horizon + 1,) or not np.issubdtype(x.dtype, np.integer) or x.min() < 0 or x.max() >= alphabet:
            raise ValueError(f"Invalid token sequence {i}")
        token_hash = token_digest(x)
        if any(token_hash != str(value) for value in (source["tokens_sha256"], ref_meta["tokens_sha256"], net_meta["tokens_sha256"])):
            raise ValueError(f"Unpaired token hashes at sequence {i}")
        tokens[i] = x
        if not np.array_equal(ref["prefixes"], prefixes) or not np.array_equal(net["prefixes"], prefixes):
            raise ValueError(f"Prefix index mismatch at sequence {i}")
        if net["logp"].dtype != np.float32:
            raise ValueError("Transformer log probabilities are not stored in float32")
        for key, value in (("bayes", ref["bayes_loss"]), ("local", ref["local_loss"]), ("transformer", net["loss"])):
            if value.shape != (horizon,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} loss array at sequence {i}")
            losses[key][i] = value
        evidence = np.asarray(ref["log_evidence"], dtype=np.float64)
        if evidence.shape != (horizon + 1,):
            raise ValueError("Unexpected evidence length")
        check_array(-evidence[0] / LN2, math.log2(alphabet), "first-token cost")
        check_array(-np.diff(evidence) / LN2, losses["bayes"][i], "evidence increments", 2e-7)
        p = np.asarray(ref["bayes_probs"], dtype=np.float64)
        old = np.asarray(ref["local_probs"], dtype=np.float64)
        tf = np.exp(net["logp"].astype(np.float64))
        if p.shape != (len(prefixes), alphabet) or old.shape != p.shape or tf.shape != p.shape:
            raise ValueError("Unexpected probability array shape")
        for j, (key, distribution) in enumerate((("bayes", p), ("local", old), ("transformer", tf))):
            probability_errors[i, j] = check_probabilities(distribution, key, tolerance)
            target_probability = distribution[np.arange(len(prefixes)), x[prefixes]]
            alignment_error = max(alignment_error, check_array(
                -np.log2(target_probability), losses[key][i, prefixes - 1], f"{key} prefix/target alignment", 2e-6))
        divergence["kl_bayes_to_transformer"][i] = predictive_kl(p, tf)
        divergence["kl_bayes_to_local"][i] = predictive_kl(p, old)
        params = json.loads(str(source["parameters_json"]))
        depth_key = str(params["ls"]) if summary["source"]["family"] == "lsa" else f"{params['ls1']},{params['ls2']}"
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
    if token_digest(tokens) != summary["tokens_sha256"] or depth_counts != summary["prior_draw_depth_counts"]:
        raise ValueError("Aggregate token/depth identity mismatch")
    statistics = {}
    terminal = {}
    pairs = {"tf_minus_bayes": ("transformer", "bayes"),
             "local_minus_bayes": ("local", "bayes"),
             "tf_minus_local": ("transformer", "local")}
    for key, (first, second) in pairs.items():
        per_sequence = np.add.accumulate(losses[first] - losses[second], axis=1)
        full_curve = moments(per_sequence)
        statistics[key] = {stat: value[prefixes - 1] for stat, value in full_curve.items()}
        terminal[key] = {stat: value[-1] for stat, value in full_curve.items()}
    for key, values in divergence.items():
        if not np.isfinite(values).all() or values.min() < -5 * tolerance:
            raise ValueError(f"Invalid independent KL values: {key}")
        statistics[key] = moments(values)
    for key, values in losses.items():
        terminal[f"{key}_total_loss"] = moments(np.sum(values, axis=1))
    maximum_error = 0.0
    for calculated, saved, scope in ((statistics, summary["by_prefix"], "by_prefix"),
                                      (terminal, summary["terminal"], "terminal")):
        if set(calculated) != set(saved):
            raise ValueError(f"Unexpected summary metrics: {scope}")
        for key, metric in calculated.items():
            for stat, values in metric.items():
                maximum_error = max(maximum_error, check_array(values, saved[key][stat], f"{scope}.{key}.{stat}"))
    check_array(np.max(probability_errors, axis=0), summary["maximum_probability_sum_error"], "normalization audit")
    validated = dict(summary)
    validated["case"] = base.name
    validated["independent_validation"] = dict(
        passed=True, sequences_checked=n, scored_positions_per_sequence=horizon,
        selected_predictive_distributions_per_sequence=len(prefixes),
        raw_cache_files_checked=3 * n, raw_cache_digest_sha256=raw_hash.hexdigest(),
        driver_summary_sha256=file_digest(summary_path),
        exporter_sha256=file_digest(Path(__file__).resolve()),
        maximum_aggregate_discrepancy=maximum_error,
        maximum_prefix_target_loss_discrepancy=alignment_error,
        comparison_absolute_tolerance=2e-8,
        checks=["immutable source and predictor cache identities", "paired token hashes and depth counts",
                "all-token evidence/loss increments", "prefix/target probability alignment",
                "raw probability normalization", "independent paired means and standard errors",
                "independent selected-prefix predictive KL"])
    return validated


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "output/shared_depth/full_prior_final_2026-09-14")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/shared_depth")
    args = parser.parse_args()
    cases = [audit_case(args.input / name) for name in CASES]
    notes = ["Independent fresh source draws from each full training prior; one existing checkpoint per family.",
             "All means are in bits. Standard errors describe paired sequence/source sampling; training-seed uncertainty is not measured.",
             "The first token is supplied free. Cumulative losses use every subsequent token through the stated horizon.",
             "Predictive KL is measured only at the listed prefixes. No cumulative KL is inferred from sparse measurements."]
    report = dict(schema="shared_depth_curated_v1", notes=notes, cases=cases)
    atomic_text(args.out / "summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    metric_names = list(cases[0]["by_prefix"])
    fields = [f"{metric}_{stat}" for metric in metric_names for stat in ("mean", "se")]
    rows = ["\t".join(["case", "n_sequences", "prefix"] + fields)]
    for case in cases:
        for j, prefix in enumerate(case["prefixes"]):
            values = [f"{case['by_prefix'][metric][stat][j]:.12g}" for metric in metric_names for stat in ("mean", "se")]
            rows.append("\t".join([case["case"], str(case["n_sequences"]), str(prefix)] + values))
        terminal = case["terminal"]
        print(f"{case['case']}: independently validated {case['n_sequences']} sequences, "
              f"TF-Bayes {terminal['tf_minus_bayes']['mean']:.6g} +/- {terminal['tf_minus_bayes']['se']:.6g} SE bits; "
              f"local-Bayes {terminal['local_minus_bayes']['mean']:.6g} +/- {terminal['local_minus_bayes']['se']:.6g} SE bits")
    atomic_text(args.out / "summary.tsv", "\n".join(rows) + "\n")
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'summary.tsv'}")


if __name__ == "__main__":
    main()

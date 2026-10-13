"""Summarize all conditional-source draws and plot B/U regret and gap curves.

Consumes immutable per-sequence records produced by eval_conditional_crossing.
Uncertainty is across independent source/path draws, paired before averaging.
Reported positive suffixes concern the evaluated window only. They do not
establish the all-future crossing definitions in manuscript Section 2.3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from eval_conditional_crossing import load_record, make_identity, validate_record


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def cumul(values):
    return np.concatenate((np.zeros((len(values), 1)), np.cumsum(values, axis=1)), axis=1)


def statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two independent, finite path curves")
    return dict(mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist())


def finite_window_signs(mean, se, coordinates):
    mean, se, coordinates = np.asarray(mean), np.asarray(se), np.asarray(coordinates)
    if (mean.ndim != 1 or len(mean) == 0 or mean.shape != se.shape or mean.shape != coordinates.shape
            or not np.isfinite(mean).all() or not np.isfinite(se).all() or np.any(se < 0)
            or coordinates.dtype.kind not in "iu" or np.any(coordinates < 1) or np.any(np.diff(coordinates) <= 0)):
        raise ValueError("Unaligned means, errors or coordinates")
    suffix = np.logical_and.accumulate((mean > 0)[::-1])[::-1]
    first = np.flatnonzero(suffix)
    return dict(first_positive_mean_suffix=int(coordinates[first[0]]) if len(first) else None,
                negative_mean_positions=coordinates[mean < 0].tolist(),
                zero_mean_positions=coordinates[mean == 0].tolist(),
                pointwise_95_normal_lower_positive_count=int(np.sum(mean - 1.96 * se > 0)),
                minimum_mean=float(mean.min()), coordinate_at_minimum=int(coordinates[mean.argmin()]),
                scope="Evaluated positions only; pointwise normal intervals are not a simultaneous band.")


def aggregate(directory, n_seq):
    if n_seq < 2:
        raise ValueError("At least two independent draws required")
    records, inputs, identities, first_checks = [], [], [], []
    for index in range(n_seq):
        path = directory / f"seq_{index:06d}.npz"
        arrays, metadata = load_record(path)
        identity = metadata["identity"]
        source, prediction, backend = identity["source"], identity["prediction"], identity["backend"]
        expected = make_identity(source["d"], source["n_tokens"] - 1, source["source_depth"],
                                 identity["master_seed"], index, backend["store"], backend["src"],
                                 prediction["u_lo"], prediction["u_hi"], prediction["prob_tol"], prediction["loss_tol"])
        validate_record(arrays, metadata, expected)
        if identity["sequence_index"] != index:
            raise ValueError("Sequence index mismatch")
        common = {k: v for k, v in identity.items() if k not in ("sequence_index", "sequence_seed")}
        if identities and common != identities[0]:
            raise ValueError("Mixed source/protocol/code identities")
        identities.append(common)
        inputs.append(dict(path=relative(path), sha256=sha(path), cache_id=metadata["cache_id"]))
        first_checks.append(metadata["checks"]["source_averaged_first_gap_bits"])
        # Independent recomputation, including paired cumulative uncertainty.
        b, u, p = (arrays[k] for k in ("bayes_probs", "unigram_probs", "oracle_probs"))
        log_p = np.zeros_like(p)
        np.log2(p, out=log_p, where=p > 0)
        rb = np.sum(p * (log_p - np.log2(b)), axis=-1)
        ru = np.sum(p * (log_p - np.log2(u)), axis=-1)
        gap = np.sum(p * np.log2(b / u), axis=-1)
        if not np.allclose(gap, ru - rb, rtol=0, atol=2e-13):
            raise ValueError("Oracle cancellation failed")
        if not np.allclose(gap, arrays["instantaneous_gap_bits"], rtol=0, atol=2e-13):
            raise ValueError("Stored gap disagrees with independently recomputed gap")
        observed = np.log2(b[np.arange(len(b)), arrays["targets"]]
                           / u[np.arange(len(u)), arrays["targets"]])
        curves = dict(bayes_regret=rb, unigram_regret=ru, gap=gap, observed_gap=observed)
        # Optional depth-known diagnostic integrates T as well as the next token.
        if "conditioned_bayes_probs" in arrays:
            p2 = arrays["conditioned_bayes_probs"]
            curves["conditional_to_unigram_kl"] = np.sum(p2 * np.log2(p2 / u), axis=-1)
            curves["conditional_to_bayes_kl"] = np.sum(p2 * np.log2(p2 / b), axis=-1)
            curves["conditional_gap"] = np.sum(p2 * np.log2(b / u), axis=-1)
        records.append(curves)
    if np.ptp(first_checks) > 1e-10:
        raise ValueError("First-step symmetry check differs across label permutations")
    matrices = {key: np.stack([record[key] for record in records]) for key in records[0]}
    curves = {key: statistics(value) for key, value in matrices.items()}
    cumulative = {key: statistics(cumul(value)) for key, value in matrices.items()}
    t = list(range(1, len(matrices["gap"][0]) + 1))
    n = list(range(1, len(t) + 2))
    for key in curves:
        if not np.allclose(np.diff(cumulative[key]["mean"]), curves[key]["mean"], atol=2e-12, rtol=0):
            raise ValueError(f"Cumulative increment identity failed for {key}")
    return dict(schema="conditional-crossing-summary-v1", units="bits", n_sequences=n_seq,
                identity=identities[0], inputs=inputs, prefix_lengths=t, cumulative_prefix_lengths=n,
                first_source_averaged_gap_bits=float(np.mean(first_checks)),
                instantaneous=curves, cumulative=cumulative,
                signs=dict(instantaneous=finite_window_signs(**curves["gap"], coordinates=t),
                           cumulative=finite_window_signs(**cumulative["gap"], coordinates=n)),
                uncertainty="One standard error across all independent source/path draws; paired path sums for cumulative quantities.",
                primary_estimator="Average over the actual current transition row, then over source/path draws.",
                script=relative(__file__), script_sha256=sha(__file__))


def plot(report, pdf, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4))
    t, n = np.asarray(report["prefix_lengths"]), np.asarray(report["cumulative_prefix_lengths"])
    colors = dict(bayes_regret="#22688a", unigram_regret="#bf5d32", gap="#397447")
    for col, (group, coordinates) in enumerate((("instantaneous", t), ("cumulative", n))):
        for key, label in (("bayes_regret", "Family Bayes"), ("unigram_regret", "Original unigram")):
            values = report[group][key]
            mean, se = np.asarray(values["mean"]), np.asarray(values["se"])
            axes[0, col].plot(coordinates, mean, color=colors[key], label=label, linewidth=1.1)
            axes[0, col].fill_between(coordinates, mean - se, mean + se, color=colors[key], alpha=.16)
        values = report[group]["gap"]
        mean, se = np.asarray(values["mean"]), np.asarray(values["se"])
        axes[1, col].plot(coordinates, mean, color=colors["gap"], linewidth=1.1)
        axes[1, col].fill_between(coordinates, mean - se, mean + se, color=colors["gap"], alpha=.18)
        axes[1, col].axhline(0, color=".5", linewidth=.7)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64, 256, 1024])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(axis="y", color=".9", linewidth=.6)
            ax.set_xlim(1, coordinates[-1])
        axes[1, col].set_xlabel("Observed prefix length " + ("$t$" if col == 0 else "$n$"))
    axes[0, 0].set_title("Instantaneous regret")
    axes[0, 1].set_title("Cumulative regret")
    axes[0, 0].set_ylabel("Bits per prediction")
    axes[0, 1].set_ylabel("Bits")
    axes[1, 0].set_ylabel("$g(t)=r_U(t)-r_B(t)$ [bits]")
    axes[1, 1].set_ylabel("$G(n)=R_U(n)-R_B(n)$ [bits]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    source = report["identity"]["source"]
    fig.suptitle(f"Uniform-base LSA source: $d={source['d']}$, source depth $L_s={source['source_depth']}$", y=.99)
    fig.text(.5, .015, f"{report['n_sequences']} independent source/path draws; shading: one standard error. "
             "$X_1$ is observed free.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .965))
    for path in (pdf, png):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/conditional_crossing/uniform_d64_L2")
    parser.add_argument("--n-seq", type=int, default=128)
    parser.add_argument("--json", type=Path, default=ROOT / "results/transformer/conditional_crossing/uniform_d64_L2.json")
    parser.add_argument("--tsv", type=Path, default=ROOT / "results/transformer/conditional_crossing/uniform_d64_L2.tsv")
    parser.add_argument("--pdf", type=Path, default=ROOT / "figures/conditional_crossing_uniform_d64_L2.pdf")
    parser.add_argument("--png", type=Path, default=ROOT / "output/conditional_crossing/uniform_d64_L2.png")
    parser.add_argument("--numerical-validation", type=Path,
                        default=ROOT / "results/transformer/conditional_crossing/numerical_validation.json")
    parser.add_argument("--exact", type=Path,
                        default=ROOT / "results/transformer/conditional_crossing/exact_prefix.json")
    args = parser.parse_args()
    report = aggregate(args.input, args.n_seq)
    source, prediction = report["identity"]["source"], report["identity"]["prediction"]
    validation = json.loads(args.numerical_validation.read_text())
    if (validation.get("passed") is not True or validation["driver_sha256"]
            != report["identity"]["code_sha256"]["eval_conditional_crossing.py"]
            or validation["validator_sha256"] != sha(ROOT / "code/scripts/validate_conditional_crossing.py")
            or validation["d"] != source["d"] or validation["source_depth"] != source["source_depth"]
            or validation["n_ctx"] < source["n_tokens"] - 1
            or any(validation[key] != prediction[key] for key in ("bayes_depths", "unigram_depths"))):
        raise ValueError("Numerical validation is missing or refers to a different driver")
    exact = json.loads(args.exact.read_text())
    if (exact.get("passed") is not True or any(exact["parameters"][key] != source[key]
            for key in ("d", "alpha", "source_depth"))
            or any(exact["parameters"][key] != prediction[key] for key in ("bayes_depths", "unigram_depths"))):
        raise ValueError("Short-prefix calculation is missing or uses different source/predictor settings")
    for path, digest in exact["provenance"]["code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Short-prefix calculation code hash changed")
    backend = report["identity"]["backend"]
    if (exact["provenance"]["store_manifest_sha256"] != backend["manifest_sha256"]
            or exact["provenance"]["store_anchors_sha256"] != backend["anchors_sha256"]):
        raise ValueError("Short-prefix calculation uses a different locked store")
    report["numerical_validation"] = dict(path=relative(args.numerical_validation), sha256=sha(args.numerical_validation),
                                           passed=True, scope="One long path, wider bounds and half-cell grid shift.")
    report["deterministic_short_prefix"] = dict(path=relative(args.exact), sha256=sha(args.exact), passed=True)
    plot(report, args.pdf, args.png)
    report["figure"] = dict(path=relative(args.pdf), sha256=sha(args.pdf),
                           all_positions_plotted=True, smoothing=False, uncertainty_standard_errors=1)
    for path in (args.json, args.tsv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.tsv.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["quantity", "prefix_length", "mean_bits", "standard_error_bits"])
        for group, coordinate_key in (("instantaneous", "prefix_lengths"), ("cumulative", "cumulative_prefix_lengths")):
            for key, values in report[group].items():
                for coordinate, mean, se in zip(report[coordinate_key], values["mean"], values["se"]):
                    writer.writerow([group + ":" + key, coordinate, repr(mean), repr(se)])
    print(json.dumps(dict(n_sequences=args.n_seq, first_source_averaged_gap_bits=report["first_source_averaged_gap_bits"],
                          signs=report["signs"], final_cumulative_gap_bits=report["cumulative"]["gap"]["mean"][-1],
                          final_cumulative_gap_se=report["cumulative"]["gap"]["se"][-1])))


if __name__ == "__main__":
    main()

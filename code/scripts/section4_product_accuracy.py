"""Explore required context using the saved conditional product comparisons.

No inference, sampling, interpolation in context length, or retraining occurs.
Every displayed requirement inverts a mean over the same 256 saved sequences.
The targets are exploratory: this is not a prospective threshold validation.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/section4_product_accuracy.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/transformer/conditional_accuracy/product"
DEFAULT_OUT = ROOT / "results/transformer/section4_product_accuracy"
PAIRS = [[1, 1], [1, 8], [4, 4], [8, 8]]
COLORS = ["#2468A0", "#BF611C", "#288167", "#8A4E98"]
TARGETS = [.05, .1, .2, .5, 1., 2.]
KEYS = ("informed_bayes", "transformer")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def first_tested_context(times, means, target):
    """First qualifying tested time, not an interpolated or permanent crossing."""
    times, means = np.asarray(times), np.asarray(means)
    require(times.ndim == means.ndim == 1 and len(times) == len(means), "Invalid curve")
    require(np.all(np.diff(times) > 0) and np.isfinite(means).all(), "Invalid times/risks")
    eligible = np.flatnonzero(means <= target)
    return int(times[eligible[0]]) if len(eligible) else None


def audit_source():
    summary = json.loads((SOURCE / "summary.json").read_text())
    receipt = json.loads((SOURCE / "receipt.json").read_text())
    require(receipt["passed"] and receipt["validation"]["passed"], "Saved reference audit failed")
    for name, digest in receipt["files_sha256"].items():
        require(sha(ROOT / name) == digest, f"Changed source artifact: {name}")
    protocol = summary["protocol"]
    require(protocol["depth_pairs"] == PAIRS and protocol["n_sequences_per_pair"] == 256
            and protocol["n_tokens"] == 1025 and protocol["times"] == [2**j for j in range(11)]
            and protocol["reference_depths_revealed"] is False, "Unexpected source protocol")
    require([case["depth_pair"] for case in summary["cases"]] == PAIRS, "Changed case coverage")
    cases, worst_error = [], 0.
    for case, color in zip(summary["cases"], COLORS):
        directory = Path(case["raw_directory"])
        digest = hashlib.sha256()
        risks = {key: [] for key in KEYS}
        for part in ("data", "references", "model"):
            require(sorted(p.name for p in (directory / part).glob("seq_*.npz")) ==
                    [f"seq_{j:06d}.npz" for j in range(256)], "Incomplete or enlarged cohort")
        for index in range(256):
            records = {}
            for part in ("data", "references", "model"):
                path = directory / part / f"seq_{index:06d}.npz"
                digest.update(f"{part}/{path.name}:{sha(path)}".encode())
                with np.load(path, allow_pickle=False) as saved:
                    records[part] = {key: saved[key] for key in saved.files}
            data, reference, model = (records[key] for key in ("data", "references", "model"))
            for record in records.values():
                require(np.array_equal(record["prefixes"], protocol["times"]), "Time alignment changed")
            for record in (reference, model):
                require(json.loads(str(record["metadata_json"]))["tokens_sha256"] ==
                        str(data["tokens_sha256"]), "Mismatched sequence identities")
            true = data["true_probs"]
            for key, prediction in (("informed_bayes", reference["bayes_probs"]),
                                    ("transformer", np.exp(model["logp"].astype(float)))):
                require(true.shape == prediction.shape == (11, 64), "Changed probability shape")
                require(np.isfinite(prediction).all() and np.min(prediction) > 0
                        and np.max(np.abs(prediction.sum(-1) - 1)) < 2e-6,
                        "Invalid saved prediction")
                risks[key].append(np.sum(xlogy(true, true) - xlogy(true, prediction), axis=-1) / np.log(2))
        require(digest.hexdigest() == case["raw_records_sha256"] ==
                receipt["raw_cohort_digests"][str(case["depth_pair"])] , "Changed raw cohort")
        row = dict(id="L%d_%d" % tuple(case["depth_pair"]), depth_pair=case["depth_pair"], color=color,
                   times=case["times"], n_sequences=256, n_tokens=1025,
                   raw_directory=relative(directory), raw_records_sha256=digest.hexdigest())
        for key in KEYS:
            raw = np.asarray(risks[key])
            stats = dict(mean=raw.mean(0).tolist(), se=(raw.std(0, ddof=1) / 16).tolist())
            for moment in ("mean", "se"):
                error = float(np.max(np.abs(np.asarray(stats[moment]) - case[key][moment])))
                require(error < 1e-12, "Raw risk summary does not reproduce saved result")
                worst_error = max(worst_error, error)
            row[key] = stats
        row["targets"] = [dict(target_bits=target, **{
            key: first_tested_context(row["times"], row[key]["mean"], target) for key in KEYS})
            for target in TARGETS]
        cases.append(row)
    return dict(schema="section4-product-accuracy-v1", exploratory=True,
                definition="min tested t with mean next-token KL regret <= target; null means not reached on saved grid",
                threshold_confidence_intervals=None,
                uncertainty="Per-time SE over independent source-matrix/sequence draws; no threshold confidence intervals, interpolation, or training-seed uncertainty.",
                scope="Four fixed-depth subsets; Bayes retains the full training depth prior; one frozen transformer checkpoint.",
                targets_bits=TARGETS, compact_case_ids=["L1_1", "L8_8"], cases=cases,
                source_protocol=protocol, source_receipt_sha256=sha(SOURCE / "receipt.json"),
                source_summary_sha256=sha(SOURCE / "summary.json"),
                verification=dict(raw_sequences=1024, raw_records=3072, raw_summary_max_abs_error=worst_error,
                                  original_validation=receipt["validation"]),
                provenance=dict(script_sha256=sha(__file__), checkpoint=relative(summary["checkpoint"]),
                                checkpoint_sha256=protocol["checkpoint_sha256"]))


def draw_required_context(axis, summary, pair_ids=None, legend=True):
    """Reusable one-panel renderer. The top band is categorical, not extrapolation."""
    from matplotlib.lines import Line2D
    selected = [row for row in summary["cases"] if pair_ids is None or row["id"] in pair_ids]
    require(bool(selected), "No selected product cases")
    axis.axhspan(11, 12.8, color=".96", zorder=-2)
    axis.axhline(11, color=".75", lw=.6, ls=":")
    for row in selected:
        for key, ls, marker, size in (("transformer", "-", ".", 5),
                                      ("informed_bayes", "--", "o", 4)):
            means = row[key]["mean"]
            targets = sorted({.04, 4., *[value for value in means if .04 < value < 4.]})
            contexts = [first_tested_context(row["times"], means, target) for target in targets]
            # A step in epsilon is the exact inverse of the discrete-time estimator.
            reached = np.asarray([np.nan if n is None else np.log2(n) for n in contexts])
            axis.step(targets, reached, where="post", color=row["color"], ls=ls, lw=1.45)
            # Extend the nonattainment line exactly to the smallest observed regret.
            if contexts[0] is None:
                axis.plot([.04, min(means)], [12, 12], color=row["color"], ls=ls, lw=1.45)
            points = [first_tested_context(row["times"], means, e) for e in TARGETS]
            axis.plot(TARGETS, [12 if n is None else np.log2(n) for n in points],
                      ls="none", marker=marker, ms=size, color=row["color"],
                      markerfacecolor="white" if key == "informed_bayes" else row["color"],
                      markeredgewidth=.9)
    axis.set_xscale("log", base=2)
    axis.set_xlim(.04, 4)
    axis.set_ylim(-.4, 12.6)
    axis.set_xticks([.05, .1, .2, .5, 1, 2], [".05", ".1", ".2", ".5", "1", "2"])
    axis.set_yticks([0, 2, 4, 6, 8, 10, 12], ["1", "4", "16", "64", "256", "1024", "Not reached"])
    axis.set_xlabel("Target next-token regret (bits)")
    axis.set_ylabel("First tested context length")
    axis.grid(alpha=.13)
    axis.spines[["top", "right"]].set_visible(False)
    if legend:
        handles = [Line2D([], [], color=row["color"], lw=2,
                          label="Depths (%d,%d)" % tuple(row["depth_pair"])) for row in selected]
        handles += [Line2D([], [], color=".25", lw=1.5, ls="-", marker=".", label="Transformer"),
                    Line2D([], [], color=".25", lw=1.5, ls="--", marker="o", mfc="white", ms=4, label="Bayes")]
        axis.legend(handles=handles, fontsize=8, frameon=False, loc="lower left")
    return axis


def save_outputs(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["L1\tL2\ttarget_bits\tbayes_first_tested_context\ttransformer_first_tested_context"]
    for row in summary["cases"]:
        for target in row["targets"]:
            values = [*row["depth_pair"], target["target_bits"],
                      *[target[key] if target[key] is not None else "not_reached" for key in KEYS]]
            lines.append("\t".join(map(str, values)))
    (out / "requirements.tsv").write_text("\n".join(lines) + "\n")
    fig, axis = plt.subplots(figsize=(5.5, 3.6))
    draw_required_context(axis, summary, summary["compact_case_ids"])
    axis.set_title("Product source: context needed for target accuracy")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"compact.{ext}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.6), sharex=True, sharey=True)
    for axis, row in zip(axes.flat, summary["cases"]):
        draw_required_context(axis, summary, [row["id"]], legend=False)
        axis.set_title("Depths (%d,%d)" % tuple(row["depth_pair"]))
    fig.legend(handles=[Line2D([], [], color=".25", ls="-", marker=".", label="Transformer"),
                        Line2D([], [], color=".25", ls="--", marker="o", mfc="white", ms=4, label="Bayes")],
               loc="upper center", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"all_four.{ext}", dpi=220)
    plt.close(fig)
    receipt = dict(schema="section4-product-accuracy-receipt-v1", verified=True,
                   meaning="Verifies reused artifacts and derived discrete estimates; not prospective scientific confirmation.",
                   script_sha256=sha(__file__), source_receipt_sha256=summary["source_receipt_sha256"],
                   outputs_sha256={name: sha(out / name) for name in
                                   ("summary.json", "requirements.tsv", "compact.pdf", "compact.png", "all_four.pdf", "all_four.png")})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary = audit_source()
    if args.check_only:
        receipt = json.loads((args.out / "receipt.json").read_text())
        require(receipt["script_sha256"] == sha(__file__), "Changed analysis script")
        require(json.loads((args.out / "summary.json").read_text()) == summary, "Derived summary changed")
        for name, digest in receipt["outputs_sha256"].items():
            require(sha(args.out / name) == digest, f"Changed derived artifact: {name}")
    else:
        save_outputs(args.out, summary)
    print(json.dumps(dict(verified=True, sequences=1024, thresholds_exploratory=True,
                          output=relative(args.out), max_raw_summary_error=summary["verification"]["raw_summary_max_abs_error"])))


if __name__ == "__main__":
    main()

"""Plot all four product components from the complete validated pilot.

No model evaluation or numerical source calculation is performed. Every raw
record must match the existing validated summary, and the common component
identification mask is independently reconstructed. Standard errors are
computed from source-level raw weights, never by adding component errors.

    python code/scripts/product_all_components_figure.py
    python code/scripts/product_all_components_figure.py --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("UU", "CU", "UC", "CC")
TITLES = ("Uniform in both", "Copy first coordinate", "Copy second coordinate", "Copy both coordinates")
H = np.array([[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=float) / np.sqrt([2, 6, 12])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def sequence_seed(schema, master_seed, index):
    value = 1469598103934665603
    for part in (schema, master_seed, index):
        for byte in str(part).encode():
            value = ((value ^ byte)*1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value % (2**32-1))


def stats(values, included):
    """Equal source weights at each prefix, with the provided common mask."""
    values, included = np.asarray(values), np.asarray(included)
    if values.ndim != 2 or values.shape != included.shape or included.dtype.kind != "b" or not np.isfinite(values).all():
        raise ValueError("Expected finite source-by-prefix values and boolean inclusion mask")
    result = dict(mean=[], se=[], count=[])
    for t in range(values.shape[1]):
        selected = values[included[:, t], t]
        result["count"].append(int(len(selected)))
        result["mean"].append(float(selected.mean()) if len(selected) else None)
        result["se"].append(float(selected.std(ddof=1)/np.sqrt(len(selected))) if len(selected) > 1 else None)
    return result


def compare_stats(computed, saved, label):
    if computed["count"] != saved["count"]:
        raise ValueError(f"Included counts differ from validated summary: {label}")
    maximum = 0.0
    for key in ("mean", "se"):
        if len(computed[key]) != len(saved[key]):
            raise ValueError(f"Wrong number of prefixes: {label}")
        for actual, expected in zip(computed[key], saved[key]):
            if actual is None or expected is None:
                if actual is not None or expected is not None:
                    raise ValueError(f"Changed undefined statistic: {label}/{key}")
            else:
                if not np.isfinite(expected):
                    raise ValueError(f"Nonfinite saved statistic: {label}/{key}")
                maximum = max(maximum, abs(actual-expected))
    if maximum > 2e-12:
        raise ValueError(f"Raw statistics disagree with validated summary: {label}, {maximum}")
    return maximum


def check_probability_rows(value, name, tolerance=1e-7):
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Nonfinite or negative values: {name}")
    error = float(np.max(np.abs(value.sum(axis=-1)-1)))
    if error > tolerance:
        raise ValueError(f"Raw normalization error: {name}, {error}")
    return error


def build(summary_path, expected_n_sequences=None):
    summary_path = Path(summary_path)
    old = json.loads(summary_path.read_text())
    spec = old["spec"]
    n_seq = spec["n_seq"]
    if (isinstance(n_seq, bool) or not isinstance(n_seq, int) or n_seq < 2
            or expected_n_sequences is not None and n_seq != expected_n_sequences):
        raise ValueError("Unexpected declared source count")
    if (old.get("passed") is not True or old["n_sequences"] != n_seq
            or spec["n_ctx"] != 1024 or spec["source_family"] != "product:8x8"
            or spec["source_depths"] != list(range(1, 9)) or spec["alphabet_sizes"] != [8, 8]
            or spec["identification_sigma_threshold"] != .02
            or spec["prior_unchanged"] is not True or fingerprint(spec) != old["spec_id"]):
        raise ValueError("Expected a complete validated product-source cohort")
    if sha(ROOT / old["script"]) != old["script_sha256"]:
        raise ValueError("The existing summary's validating script has changed")
    if "reused_consumer" in old:
        reused = old["reused_consumer"]
        if sha(ROOT / reused["path"]) != reused["sha256"]:
            raise ValueError("The evidence summary's reused validating consumer has changed")
    for name, provenance in spec["provenance"].items():
        if sha(provenance["path"]) != provenance["sha256"]:
            raise ValueError(f"Current producer/checkpoint/store provenance changed: {name}")
    if len(old["inputs"]) != n_seq or [item["index"] for item in old["inputs"]] != list(range(n_seq)):
        raise ValueError("The validated source cohort is missing, reordered or duplicated")
    positions = np.arange(1, 1025)
    if old["prefix_lengths"] != positions.tolist():
        raise ValueError("Changed prefix convention in validated summary")
    bayes, transformer, masks, input_records = [], [], [], []
    maximum_sum_error, maximum_sigma_error = 0.0, 0.0
    for index, item in enumerate(old["inputs"]):
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Raw input differs from independently validated record: {path}")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
            arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        if (meta["spec"] != spec or meta["spec_id"] != old["spec_id"] or meta["index"] != index
                or meta["sequence_seed"] != sequence_seed(spec["schema"], spec["master_seed"], index)
                or fingerprint(meta) != item["metadata_sha256"]):
            raise ValueError(f"Wrong source/protocol metadata in draw {index}")
        if set(arrays) != set(meta["array_hashes"]):
            raise ValueError(f"Unexpected array set in draw {index}")
        for name, value in arrays.items():
            if array_hash(value) != meta["array_hashes"][name]:
                raise ValueError(f"Invalid array checksum in draw {index}: {name}")
        if array_hash(arrays["tokens"]) != item["tokens_sha256"] or not np.array_equal(arrays["prefix_lengths"], positions):
            raise ValueError(f"Wrong tokens or prefix indices in draw {index}")
        b, t, components, eligible = (arrays[name] for name in
                                    ("bayes_weights", "transformer_weights", "component_probs", "eligible"))
        if b.shape != (1024, 4) or t.shape != b.shape or components.shape != (1024, 4, 64):
            raise ValueError("Wrong predictive weight/component shapes")
        if eligible.shape != (1024,) or eligible.dtype.kind != "b":
            raise ValueError("Wrong eligibility array")
        for name, value in (("Bayes weights", b), ("transformer weights", t), ("component distributions", components)):
            maximum_sum_error = max(maximum_sum_error, check_probability_rows(value, name))
        sigma = np.linalg.svd(components.transpose(0, 2, 1) @ H, compute_uv=False)[:, -1]
        if not np.array_equal(eligible, sigma >= .02):
            raise ValueError("Mask differs from the predeclared component-separation criterion")
        sigma_error = float(np.max(np.abs(sigma-arrays["affine_sigma_min"])))
        maximum_sigma_error = max(maximum_sigma_error, sigma_error)
        if sigma_error > 2e-12 or not arrays["fit_converged"].all():
            raise ValueError("A saved identification or fit check failed")
        bayes.append(b)
        transformer.append(t)
        masks.append(eligible)
        input_records.append(dict(index=index, path=relative(path), sha256=item["sha256"],
                                  metadata_sha256=item["metadata_sha256"], sequence_seed=meta["sequence_seed"],
                                  source_depths=arrays["source_depths"].tolist()))
    bayes, transformer, included = np.stack(bayes), np.stack(transformer), np.stack(masks)
    full = {name: {"bayes": stats(bayes[:, :, j], included), "transformer": stats(transformer[:, :, j], included),
                   "transformer_minus_bayes": stats(transformer[:, :, j]-bayes[:, :, j], included)}
            for j, name in enumerate(NAMES)}
    checks = dict(raw_probability_maximum_sum_error=maximum_sum_error, maximum_sigma_recomputation_error=maximum_sigma_error)
    for predictor, weights in (("bayes", bayes), ("transformer", transformer)):
        checks[predictor+"_CC_summary_error"] = compare_stats(full["CC"][predictor], old["paired_weights"]["both"][predictor], predictor+" CC")
        checks[predictor+"_any_copy_summary_error"] = compare_stats(stats(1-weights[:, :, 0], included),
            old["paired_weights"]["copy"][predictor], predictor+" 1-UU")
    checks["CC_paired_difference_summary_error"] = compare_stats(full["CC"]["transformer_minus_bayes"],
        old["paired_weights"]["both"]["transformer_minus_bayes"], "CC paired difference")
    if not np.allclose(included.sum(axis=0), n_seq*np.asarray(old["eligible_fraction"]), atol=1e-10, rtol=0):
        raise ValueError("Included source counts differ from validated summary")
    for predictor in ("bayes", "transformer"):
        means = np.array([full[name][predictor]["mean"] for name in NAMES], dtype=float)
        if np.max(np.abs(means[:, included.any(axis=0)].sum(axis=0)-1)) > 2e-12:
            raise ValueError("Component means do not sum to one on the common source mask")
    early_mae, early_signed, per_sequence = [], [], []
    for index in range(n_seq):
        mask = included[index, :32]
        if not mask.any():
            per_sequence.append(dict(index=index, included_prefix_count=0,
                                     mean_absolute_error=dict.fromkeys(NAMES), mean_signed_error=dict.fromkeys(NAMES)))
            continue
        differences = transformer[index, :32][mask]-bayes[index, :32][mask]
        mae, signed = np.abs(differences).mean(axis=0), differences.mean(axis=0)
        early_mae.append(mae)
        early_signed.append(signed)
        per_sequence.append(dict(index=index, included_prefix_count=int(mask.sum()),
                                 mean_absolute_error=dict(zip(NAMES, mae.tolist())),
                                 mean_signed_error=dict(zip(NAMES, signed.tolist()))))
    n_early = len(early_mae)
    early_mae, early_signed = np.asarray(early_mae).reshape(n_early, 4), np.asarray(early_signed).reshape(n_early, 4)
    early = dict(first_prefix=1, last_prefix=32, included_pairs=int(included[:, :32].sum()),
                 total_pairs=n_seq*32, included_fraction=float(included[:, :32].mean()), per_sequence=per_sequence,
                 n_sequences_with_early_inclusion=n_early,
                 components={}, averaging=f"Within each sequence, average over its included prefixes1..32; then weight the {n_early} defined sequence averages equally. Sources without an included early prefix remain explicitly listed with undefined errors. Standard errors use the independent sequence averages.")
    def mean_se(values):
        return (float(values.mean()) if len(values) else None,
                float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None)
    for j, name in enumerate(NAMES):
        mae_mean, mae_se = mean_se(early_mae[:, j])
        signed_mean, signed_se = mean_se(early_signed[:, j])
        early["components"][name] = dict(mean_absolute_error=mae_mean, absolute_error_source_se=mae_se,
                                        mean_signed_error=signed_mean, signed_error_source_se=signed_se)
    four_average = early_mae.mean(axis=1)
    early["all_four_mean_absolute_error"], early["all_four_absolute_error_source_se"] = mean_se(four_average)
    if old["early_middle_late_windows"][0]["both_paired_error"]["absolute"]["count"] != [n_early]:
        raise ValueError("Early source count differs from validated summary")
    for value, saved in ((early["components"]["CC"]["mean_absolute_error"], old["early_middle_late_windows"][0]["both_paired_error"]["absolute"]["mean"][0]),
                          (early["components"]["CC"]["absolute_error_source_se"], old["early_middle_late_windows"][0]["both_paired_error"]["absolute"]["se"][0])):
        if value is None or saved is None:
            if value is not saved:
                raise ValueError("Early CC error differs from validated summary")
        elif abs(value-saved) > 2e-12:
            raise ValueError("Early CC error differs from validated summary")
    return dict(schema="product-all-components-figure-v1", passed=True, n_sequences=n_seq, component_names=list(NAMES),
                component_titles=dict(zip(NAMES, TITLES)), prefix_lengths=positions[:128].tolist(),
                included_source_counts=included.sum(axis=0)[:128].tolist(),
                components={name: {predictor: {key: values[:128] for key, values in stats_value.items()}
                                   for predictor, stats_value in component.items()} for name, component in full.items()},
                early_errors=early, validation=checks, inputs=input_records,
                source=dict(family=spec["source_family"], coordinate_sizes=[8, 8], depth_prior=spec["source_depths"],
                            depth_scope=spec["depth_scope"], master_seed=spec["master_seed"], scored_prefixes_per_source=1024),
                identification=dict(criterion="sigma_min(AH)>=0.02; columns of A are the four component distributions; H has orthonormal columns perpendicular to the all-ones vector", threshold=.02,
                                    common_mask=True, based_on_fit_quality=False),
                uncertainty="One standard error across included independent source draws at each prefix. Raw component weights are used separately; component standard errors are not added.",
                interpretation="Means of specified predictive coefficients, not fractions of sequences with component dominance. Changing early inclusion prevents an unconditional transition-time conclusion.",
                provenance=dict(validated_summary=dict(path=relative(summary_path), sha256=sha(summary_path)),
                                spec_id=old["spec_id"], producer_checkpoint_store=spec["provenance"],
                                validating_script=dict(path=old["script"], sha256=old["script_sha256"]),
                                reused_consumer=old.get("reused_consumer"),
                                plotting_script=dict(path=relative(__file__), sha256=sha(__file__))),
                command=f"python code/scripts/product_all_components_figure.py --summary {relative(summary_path)} --expected-n-sequences {n_seq}")


def plot(report, pdf, png):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/product_component_pilot/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(9, 7.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 3, .8], hspace=.33, wspace=.15)
    t = np.array(report["prefix_lengths"])
    axes = []
    for j, (name, title) in enumerate(zip(NAMES, TITLES)):
        ax = fig.add_subplot(grid[j//2, j%2], sharey=axes[0] if axes else None)
        axes.append(ax)
        for predictor, color, label, style in (("bayes", "#166B94", "Bayesian weight", "-"),
                                               ("transformer", "#D05A2B", "Fitted transformer weight", "--")):
            values = report["components"][name][predictor]
            mean, se = np.array(values["mean"], dtype=float), np.array(values["se"], dtype=float)
            ax.plot(t, mean, color=color, ls=style, lw=1.8, label=label)
            ax.fill_between(t, mean-se, mean+se, color=color, alpha=.12, linewidth=0)
        ax.set_title(f"{title} ({name})", fontsize=11.5, pad=8)
        ax.set_ylim(0, 1)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 128)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(labelsize=9, labelleft=j%2 == 0)
        ax.grid(axis="y", color="#e3e7ea", linewidth=.7)
        if j%2 == 0:
            ax.set_ylabel("Mean predictive weight")
    coverage = fig.add_subplot(grid[2, :])
    counts = np.array(report["included_source_counts"])
    coverage.fill_between(t, 0, counts, color="#A8B5BF", alpha=.5, step="mid")
    coverage.plot(t, counts, color="#667681", lw=1)
    coverage.set_xscale("log", base=2)
    coverage.set_xlim(1, 128)
    coverage.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    coverage.xaxis.set_major_formatter(ScalarFormatter())
    coverage.set_ylim(0, report["n_sequences"]*1.03)
    coverage.set_yticks([0, report["n_sequences"]//2, report["n_sequences"]])
    coverage.tick_params(labelsize=9)
    coverage.set_ylabel("Sequences\nincluded", fontsize=9)
    coverage.set_xlabel("Tokens observed in the new sequence", fontsize=10)
    fig.suptitle("All four components of the product prediction", fontsize=14, y=.991)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.52, .956),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .026, "All curves use the same included sequences at each prefix. Shading: one standard error across sequences.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.09, right=.985, top=.865, bottom=.13)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/transformer/product_component_pilot/summary.json")
    parser.add_argument("--out-dir", "--out", dest="out", type=Path, default=ROOT / "results/transformer/product_component_pilot")
    parser.add_argument("--expected-n-sequences", type=int, help="Require this complete declared source count; otherwise infer it from the validated summary")
    parser.add_argument("--check-only", action="store_true", help="Recheck every raw record and reference statistic without changing files")
    args = parser.parse_args(argv)
    report = build(args.summary, expected_n_sequences=args.expected_n_sequences)
    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        pdf, png = args.out / "all_components.pdf", args.out / "all_components.png"
        plot(report, pdf, png)
        report["artifacts"] = {kind: dict(path=relative(path), sha256=sha(path)) for kind, path in (("pdf", pdf), ("png", png))}
        (args.out / "all_components.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True, all_source_records=report["n_sequences"], prefix_count=128,
                          early_component_errors=report["early_errors"]["components"],
                          mean_four_component_absolute_error=report["early_errors"]["all_four_mean_absolute_error"],
                          four_component_source_se=report["early_errors"]["all_four_absolute_error_source_se"],
                          validation=report["validation"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

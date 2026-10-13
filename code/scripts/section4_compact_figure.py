"""Combine the saved Section 4 evidence into eight paper-sized vector panels.

Only saved scores and summaries are read; the explicitly provisional 128
endpoint remains separate from the accepted references. No inference, source
sampling, probability clipping, smoothing, or new uncertainty calculation is
performed. Recompute displayed means and standard errors from saved records
where available, and retain each original uncertainty convention.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/layered-section4-matplotlib")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from product_shift_additivity import load as load_shift
from copying_precision_figure_input import load as load_copying_precision
from copying_endpoint32_figure_input import load as load_copying_endpoint32
from copying_endpoint64_figure_input import load as load_copying_endpoint64
from copying_endpoint128_provisional_figure_input import load as load_copying_endpoint128

ROOT = Path(__file__).resolve().parents[2]
SHIFT = ROOT / "output/product_shift_2026-09-18"
UNSEEN = ROOT / "results/transformer/section4_unseen_context_refined_2026-09-20"
UNIFORM = ROOT / "results/transformer/section4_unseen_uniform_control_2026-09-17"
COPY = ROOT / "results/transformer/shared_base_shuffle_2026-09-18"
COPY_RAW = ROOT / "output/shared_base_shuffle_download_2026-09-18/eval"
COPY_PRECISION = ROOT / "results/transformer/copying_bayes_precision_2026-09-24"
COPY_ENDPOINT32 = ROOT / "results/transformer/copying_bayes_endpoint32_2026-09-24"
COPY_ENDPOINT64 = ROOT / "results/transformer/copying_bayes_endpoint64_2026-09-24"
COPY_ENDPOINT128 = ROOT / "results/transformer/copying_bayes_endpoint128_provisional_2026-09-25"
OUT = ROOT / "results/transformer/section4_compact"
MISMATCH_LARGER = ROOT / "results/transformer/product_mismatch_larger_2026-09-23"
COPY_EXTENSION = ROOT / "output/shared_base_shuffle_extension_recovery_2026-09-22"
COPY_PLAN = ROOT / "code/transformer/configs/shared_base_shuffle_extension_2026-09-22.json"
# Match Figure 1: color denotes condition; line versus marker denotes predictor.
BLUE, ORANGE, GRAY, DARK = "#2468A0", "#BF611C", "#777777", "#333333"
COPYING_BAYES_DISPLAY_TIMES = np.array([1, 2, 4, 8, 16])


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_stat(values, stat, se_name="se"):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all() and len(values) > 1, "Invalid saved scores")
    error = max(float(np.max(np.abs(values.mean(0) - stat["mean"]))),
                float(np.max(np.abs(values.std(0, ddof=1) / np.sqrt(len(values)) - stat[se_name]))))
    require(error < 2e-12, "Saved scores disagree with displayed summary")
    return error


def load_mismatch_larger(shift):
    paths = [MISMATCH_LARGER / name for name in ("curves.npz", "provenance.json.gz")]
    with gzip.open(paths[1], "rt") as stream:
        receipt = json.load(stream)
    validation = receipt["independent_validation"]
    require(receipt["schema"] == "product-mismatch-larger-release-v1" and receipt["passed"]
            and receipt["curves_sha256"] == sha(paths[0]) and validation["passed"]
            and validation["source_plan_id"] == shift["plan_id"]
            and {row["seed"] for row in validation["models"]} == {0, 1},
            "Invalid larger-model mismatch release")
    names = ("mean", "se", "lower_pointwise_95", "upper_pointwise_95")
    with np.load(paths[0], allow_pickle=False) as curves:
        require(np.array_equal(curves["times"], shift["times"]), "Wrong mismatch grid")
        groups = {seed: {name: curves[f"seed{seed}_L4_larger_regret_bits_{name}"].copy()
                         for name in names} for seed in (0, 1)}
    require(all(value.shape == np.shape(shift["times"]) and np.isfinite(value).all()
        for group in groups.values() for value in group.values()), "Invalid mismatch curves")
    return groups, paths


def release_copying_extension(source, out):
    """Replay the frozen score aggregates before exporting the small figure input."""
    summary, manifest, status = (read(source / name) for name in
                                 ("summary.json", "manifest.json", "status.json"))
    plan = read(COPY_PLAN)
    require(status["complete"] and status["summary_sha256"] == sha(source / "summary.json")
            and summary["complete"] and summary["scientific_reference_validated"]
            and summary["no_retraining"] and summary["no_new_bayesian_references"],
            "Copying extension is not validated")
    require(summary["manifest_sha256"] == sha(source / "manifest.json")
            and manifest["extension_plan_sha256"] == sha(COPY_PLAN)
            and (plan["n_sequences"], plan["retained_sequences"], plan["bootstrap"]) == (1024, 256, 1000),
            "Copying extension declaration changed")
    require(all(sha(ROOT / name) == digest for name, digest in
                plan["dependencies_sha256"].items()), "Frozen copying dependency changed")
    sys.path.insert(0, str(ROOT / "code/transformer"))
    from source import seed_of
    arrays = {"times": np.arange(1, 1025)}
    inputs = [COPY_PLAN, source / "summary.json", source / "manifest.json", source / "status.json"]
    checks = {}
    for depth in (2, 3):
        condition = f"a1_L{depth}"
        declared = next(c for c in manifest["conditions"] if c["id"] == condition)
        require(declared["indices"] == list(range(1024)), "Copying sources were selected")
        scores = {}
        for role in ("ordered", "shuffled"):
            key = f"{role}_seed0/{condition}"
            path = source / "eval" / (key + ".npz")
            old_path = COPY_RAW / (key + ".npz")
            protocol_path = path.parent / "protocol.json"
            protocol = read(protocol_path)
            require(sha(path) == summary["input_sha256"][key]
                    and protocol["manifest_sha256"] == summary["manifest_sha256"]
                    and protocol["checkpoint_sha256"] == plan["checkpoint_sha256"][f"{role}_seed0"]
                    == read(old_path.parent / "protocol.json")["checkpoint_sha256"],
                    "Copying score/checkpoint provenance changed")
            with np.load(path, allow_pickle=False) as raw, np.load(old_path, allow_pickle=False) as old:
                require(str(raw["protocol_sha256"]) == sha(protocol_path)
                        and str(raw["source_tokens_sha256"]) == declared["arrays_sha256"]["tokens"]
                        and np.array_equal(raw["times"], arrays["times"]), "Copying score identity differs")
                for name in ("expected_loss_bits", "regret_bits", "realized_loss_bits"):
                    value = raw[name]
                    require(value.shape == (1024, 1024) and np.isfinite(value).all()
                            and np.array_equal(value[:256], old[name]), "Retained copying scores changed")
                scores[role] = raw["expected_loss_bits"].copy()
            inputs.extend((path, old_path, protocol_path, old_path.parent / "protocol.json"))
        gap = scores["shuffled"] - scores["ordered"]
        for cohort, first, last in (("all1024", 0, 1024), ("retained256", 0, 256), ("new768", 256, 1024)):
            group = next(g for g in summary["groups"] if
                         (g["seed"], g["depth"], g["cohort"]) == (0, depth, cohort))
            require(group["n_sequences"] == last-first, "Wrong copying cohort size")
            values = gap[first:last]
            rng = np.random.default_rng(seed_of("shuffle-extension-bootstrap-v1", 0, depth, cohort))
            counts = rng.multinomial(len(values), np.full(len(values), 1/len(values)), size=1000).astype(float)
            for metric, value in (("instantaneous", values), ("cumulative", values.cumsum(1))):
                error = matching_stat(value, group[metric])
                bounds = np.quantile(counts @ value / len(value), [.025, .975], axis=0)
                error = max(error, float(np.max(np.abs(bounds - np.asarray(
                    [group[metric]["lower95"], group[metric]["upper95"]])))))
                require(error < 2e-12, "Saved copying bootstrap bands differ on replay")
                checks[f"L{depth}_{cohort}_{metric}_max_error"] = error
                if cohort == "all1024":
                    for name in ("mean", "se", "lower95", "upper95"):
                        arrays[f"L{depth}_{metric}_{name}"] = np.asarray(group[metric][name])
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "full_copying_curves.npz"
    np.savez_compressed(artifact, **arrays)
    provenance = dict(schema="copying-extension-figure-v1", passed=True,
        seed=0, depths=[2, 3], n_sequences=1024, retained_sequences=256,
        bootstrap_replicates=1000, arrays_sha256=sha(artifact), checks=checks,
        retained_scores_exact=True, inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        uncertainty="Pointwise 95% paired whole-sequence bootstrap; no training-seed uncertainty.",
        scope="Full transformer curves: all 1024 histories. Bayesian points: validated paired 256-history cohort.")
    (out / "full_copying_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(released=str(artifact), checks=checks), indent=2))


def load_extended_copying():
    path, receipt_path = OUT / "full_copying_curves.npz", OUT / "full_copying_provenance.json"
    receipt = read(receipt_path)
    contract = (receipt["schema"], receipt["n_sequences"], receipt["retained_sequences"],
                receipt["seed"], receipt["depths"], receipt["bootstrap_replicates"])
    require(contract in (("copying-extension-figure-v1", 1024, 256, 0, [2, 3], 1000),
                         ("copying-extension-figure-v2", 16384, 1024, 0, [2, 3], 1000))
            and receipt["passed"] and receipt["retained_scores_exact"]
            and receipt["arrays_sha256"] == sha(path)
            and max(receipt["checks"].values()) < 2e-12, "Invalid copying extension release")
    paths = [path, receipt_path]
    if receipt["schema"].endswith("v2"):
        require(receipt["independent_validation_path"] == "independent_validation.json",
                "Wrong copying validation path")
        validation_path = OUT / receipt["independent_validation_path"]
        validation = read(validation_path)
        scope = tuple(validation[name] for name in
                      ("schema", "n_sequences", "retained_sequences", "depths", "seed", "bootstrap_replicates"))
        require(scope == ("copying-large-independent-validation-v1", 16384, 1024, [0, 2, 3], 0, 1000)
                and receipt["independent_validation_sha256"] == sha(validation_path)
                and validation["passed"] and validation["retained_scores_exact"]
                and validation["arrays_sha256"] == receipt["arrays_sha256"]
                and validation["checks"] == receipt["checks"], "Invalid independent copying validation")
        paths.append(validation_path)
    groups = {}
    with np.load(path, allow_pickle=False) as arrays:
        require(np.array_equal(arrays["times"], np.arange(1, 1025)), "Wrong extended copying grid")
        for depth in (2, 3):
            groups[depth] = {metric: {name: arrays[f"L{depth}_{metric}_{name}"].copy()
                for name in ("mean", "se", "lower95", "upper95")}
                for metric in ("instantaneous", "cumulative")}
            for values in groups[depth].values():
                require(all(a.shape == (1024,) and np.isfinite(a).all() for a in values.values())
                        and np.all(values["se"] >= 0) and np.all(values["lower95"] <= values["upper95"]),
                        "Invalid extended copying values")
    return groups, receipt["n_sequences"], receipt["checks"], paths


def load_inputs(unseen_folder=UNSEEN):
    paths = []
    shift, risks, checks = load_shift(SHIFT)
    paths += [SHIFT / name for name in ("summary.json", "per_sequence_risks.npz")]
    mismatch_larger, mismatch_paths = load_mismatch_larger(shift)
    paths.extend(mismatch_paths)
    projection_path = SHIFT / "mismatch_limit/projection_summary.json"
    projection = read(projection_path)
    require(projection["source_plan_id"] == shift["plan_id"] and not projection["smoke"],
            "Wrong mismatch-limit cohort")
    require(projection["depths"]["4"]["n_sources"] == 1024, "Incomplete projection cohort")
    paths.append(projection_path)
    add_path = SHIFT / "additivity/summary.json"
    add = read(add_path)
    require(add["plan_id"] == shift["plan_id"] and all(sha(SHIFT / name) == digest
            for name, digest in add["input_sha256"].items()), "Additivity inputs changed")
    paths.append(add_path)
    post = risks["times"] >= shift["switch_observed"]
    row = next(g for g in add["groups"] if g["depth"] == 4 and g["metric"] == "transformer_regret_bits")
    values = risks["transformer_regret_bits"][1]
    checks["additivity_stat_max_error"] = max(
        matching_stat((values[:, 3] - values[:, 0])[:, post], row["two_factor_excess"]),
        matching_stat((2 * (values[:, 2] - values[:, 0]))[:, post], row["twice_one_factor_excess"]))
    require(np.array_equal(risks["times"][post], add["postchange_times"]), "Wrong additivity grid")

    unseen, unseen_check = read(unseen_folder / "summary.json"), read(unseen_folder / "validation.json")
    uniform, uniform_check = read(UNIFORM / "summary.json"), read(UNIFORM / "validation.json")
    for folder, receipt, key in ((unseen_folder, unseen_check, "files"), (UNIFORM, uniform_check, "outputs")):
        require(receipt["passed"], "Saved-data validation did not pass")
        for name, digest in receipt[key].items():
            require(sha(folder / name) == digest, "Unseen-context input changed")
            paths.append(folder / name)
        paths.append(folder / "validation.json")
    with (unseen_folder / "records.csv").open() as handle:
        records = list(csv.DictReader(handle))
    largest = 0.0
    for group in unseen["groups"]:
        if group["depth"] not in (1, 8):
            continue
        require(group["bayes_released"] == group["subset_numerical_validation"]["passed"],
                "Unseen Bayesian release flag differs")
        saved = [r for r in records if int(r["depth"]) == group["depth"] and int(r["time"]) == group["time"]]
        require(len(saved) == group["n_selected"], "Selected-history count differs")
        for name in ("larger_seed0", "bayes", "kt_unigram"):
            if name == "bayes" and not group["bayes_released"]:
                continue
            values = [float(r["uniform_loss_bits"]) - float(r[name + "_loss_bits"]) for r in saved]
            largest = max(largest, matching_stat(values, group["predictors"][name]["gain_over_uniform_bits"],
                                                 "standard_error"))
    with (UNIFORM / "records.csv").open() as handle:
        records = list(csv.DictReader(handle))
    for group in uniform["groups"]:
        if group["depth"] == 1:
            saved = [r for r in records if int(r["depth"]) == 1 and int(r["time"]) == group["time"]]
            require(len(saved) == group["n"], "Uniform selected-history count differs")
            for name in ("transformer", "counts"):
                largest = max(largest, matching_stat([float(r[name + "_excess_bits"]) for r in saved], group[name]))
    checks["unseen_csv_stat_max_error"] = largest

    copying, copying_check = read(COPY / "summary.json"), read(COPY / "validation.json")
    require(copying_check["passed"] and sha(COPY / "summary.json") == copying_check["summary_sha256"],
            "Copying summary is not the validated version")
    paths += [COPY / "summary.json", COPY / "validation.json"]
    errors = []
    for group in copying["groups"]:
        if group["seed"] != 0 or group["depth"] not in (2, 3):
            continue
        arrays = {}
        for role in ("ordered", "shuffled"):
            path = COPY_RAW / f"{role}_seed0" / (group["condition"] + ".npz")
            key = f"{role}_seed0/{group['condition']}"
            require(sha(path) == copying["input_sha256"][key], "Copying scores changed")
            paths.append(path)
            with np.load(path, allow_pickle=False) as raw:
                require(np.array_equal(raw["times"], np.arange(1, 1025)), "Copying time grid changed")
                arrays[role] = raw["expected_loss_bits"].copy()
        difference = arrays["shuffled"] - arrays["ordered"]
        require(difference.shape == (256, 1024), "Copying cohort changed")
        for metric, values in (("instantaneous", difference), ("cumulative", difference.cumsum(1))):
            errors.append(matching_stat(values, group[metric]))
            require(np.all(np.asarray(group[metric]["lower95"]) <= group[metric]["upper95"]),
                    "Invalid saved bootstrap intervals")
    checks["copying_stat_max_error"] = max(errors)
    copying_full, copying_full_n, extension_checks, extension_paths = load_extended_copying()
    checks.update(extension_checks)
    paths.extend(extension_paths)
    bayes_curves, precision_checks, precision_paths = load_copying_precision(COPY_PRECISION)
    checks.update(precision_checks)
    paths.extend(precision_paths)
    copying_endpoint32, endpoint_checks, endpoint_paths = load_copying_endpoint32(COPY_ENDPOINT32)
    checks.update(endpoint_checks)
    paths.extend(endpoint_paths)
    copying_endpoint64, endpoint64_checks, endpoint64_paths = load_copying_endpoint64(COPY_ENDPOINT64)
    checks.update(endpoint64_checks)
    paths.extend(endpoint64_paths)
    copying_endpoint128, endpoint128_checks, endpoint128_paths = load_copying_endpoint128(COPY_ENDPOINT128)
    checks.update(endpoint128_checks)
    paths.extend(endpoint128_paths)
    return dict(shift=shift, mismatch_larger=mismatch_larger,
                projection=projection, additivity=add, unseen=unseen,
                uniform=uniform, copying=copying, copying_full=copying_full,
                copying_full_n=copying_full_n, copying_bayes=bayes_curves,
                copying_endpoint32=copying_endpoint32, copying_endpoint64=copying_endpoint64,
                copying_endpoint128=copying_endpoint128), checks, paths


def draw(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": None})
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 6.75))
    fig.subplots_adjust(left=.115, right=.97, bottom=.065, top=.925,
                        wspace=.40, hspace=.80)
    axes = list(axes.flat)
    handles = [Line2D([], [], color=DARK, lw=1.2, label="Two-layer transformer"),
               Line2D([], [], color=DARK, lw=1.2, ls="--", label="Larger transformers"),
               Line2D([], [], color=DARK, ls="none", marker="o", mfc="white",
                      ms=3.5, label="Informed Bayes"),
               Line2D([], [], color=GRAY, lw=1.05, ls=":", marker="^", ms=3.2,
                      label="Token counts")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .997),
               ncol=4, frameon=False, columnspacing=.8, handlelength=1.65,
               handletextpad=.4, fontsize=7.0)

    def title(ax, letter, text):
        ax.set_title(f"({letter}) {text}", loc="left", pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8DEE5", lw=.45, zorder=-5)
        ax.tick_params(length=2.5, pad=2, labelsize=7.5)

    def context_axis(ax, ticks, limits):
        ax.set_xscale("log", base=2)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Observed tokens, $t$")

    def ribbon(ax, times, stat, color, style="-", label=None,
               names=("lower_pointwise_95", "upper_pointwise_95"), alpha=.12):
        ax.fill_between(times, stat[names[0]], stat[names[1]], color=color,
                        alpha=alpha, lw=0)
        ax.plot(times, stat["mean"], color=color, ls=style, lw=1.15, label=label)

    def se_ribbon(ax, times, stats, color, style, label=None):
        mean = np.asarray([v["mean"] for v in stats])
        se = np.asarray([v["standard_error"] for v in stats])
        ax.fill_between(times, mean-se, mean+se, color=color, alpha=.12, lw=0)
        ax.plot(times, mean, color=color, ls=style, lw=1.15, label=label)

    def interval_points(ax, times, stat, color, names=("lower_pointwise_95", "upper_pointwise_95"),
                        indices=None):
        mean = np.asarray(stat["mean"])
        lower, upper = np.asarray(stat[names[0]]), np.asarray(stat[names[1]])
        times = np.asarray(times)
        if indices is not None:
            times, mean, lower, upper = (value[indices] for value in (times, mean, lower, upper))
        ax.errorbar(times, mean, yerr=np.vstack((mean-lower, upper-mean)),
                    color=color, ls="none", marker="o", ms=3.0, mfc="white",
                    mew=.7, elinewidth=.55, capsize=1.0, zorder=5)

    def se_points(ax, times, stats, color):
        ax.errorbar(times, [v["mean"] for v in stats],
                    yerr=[v["standard_error"] for v in stats], color=color,
                    ls="none", marker="o", ms=3.2, mfc="white", mew=.7,
                    elinewidth=.6, capsize=1.2, zorder=5)

    times = np.asarray(data["shift"]["times"])
    mismatch = next(g for g in data["shift"]["groups"]
                    if g["depth_pair"] == [4, 4] and g["condition"] == "relabeled")
    ribbon(axes[0], times, mismatch["metrics"]["transformer_regret_bits"], BLUE,
           label="2 layers")
    for seed, style in ((0, "--"), (1, ":")):
        ribbon(axes[0], times, data["mismatch_larger"][seed], BLUE,
               style, f"4 layers, run {seed + 1}", alpha=.055)
    interval_points(axes[0], times, mismatch["metrics"]["bayes_regret_bits"], BLUE,
                    indices=np.isin(times, [1,2,4,8,16,32,64,128,256,512,1024]))
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.55,
                   labelspacing=.12, borderaxespad=.15, handletextpad=.4, fontsize=7.0)

    time_varying = next(g for g in data["shift"]["groups"]
                        if g["depth_pair"] == [4, 4] and g["condition"] == "switch_both")
    ribbon(axes[1], times, time_varying["metrics"]["transformer_regret_bits"], BLUE)
    interval_points(axes[1], times, time_varying["metrics"]["bayes_regret_bits"], BLUE)
    for ax in axes[:2]:
        ax.set_ylabel("Next-token regret (bits)")
        context_axis(ax, [1, 16, 256, 1024], (.87, 1180))
    title(axes[0], "a", "Mismatch: predicted residual loss")
    axes[0].set_ylim(2.08, 3.25)
    limit = data["projection"]["depths"]["4"]["quantities"]["stationary_kl_bits"]["mean"]
    axes[0].axhline(limit, color=GRAY, ls=":", lw=1)
    axes[0].annotate(f"Predicted limit: {limit:.3f}", (.04, limit), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points", color=".3", fontsize=8, va="bottom")
    title(axes[1], "b", "Time-varying: adaptation")
    axes[1].axvline(256, color=".55", ls=":", lw=.8, zorder=-2)
    axes[1].set(ylim=(-.05, 7.2), yticks=[0, 3, 6])
    axes[1].annotate("new matrices", (256, 6.4), xytext=(-3, 0),
                     textcoords="offset points", ha="right", fontsize=7, color=".4")

    title(axes[2], "c", "Time-varying: additivity")
    row = next(g for g in data["additivity"]["groups"] if g["depth"] == 4 and g["metric"] == "transformer_regret_bits")
    postchange = np.asarray(data["additivity"]["postchange_times"])
    ribbon(axes[2], postchange, row["two_factor_excess"], BLUE,
           label="Two new matrices: measured")
    ribbon(axes[2], postchange, row["twice_one_factor_excess"], ORANGE, ":",
           r"$2\times$ one new matrix: predicted", alpha=.07)
    context_axis(axes[2], [256, 512, 1024], (242, 1080))
    axes[2].set(ylim=(-.05, 6.75), yticks=[0, 3, 6],
                ylabel="Additional regret (bits)")
    axes[2].legend(frameon=False, loc="upper right", handlelength=1.6, labelspacing=.15,
                   borderaxespad=.05, handletextpad=.4, fontsize=7.0)

    def unseen_style(ax):
        ax.axhline(0, color=".6", lw=.6, zorder=-3)
        context_axis(ax, [4, 16, 64], (3.3, 78))
        ax.set_ylabel("Bits saved over uniform")

    for ax, letter, depth in ((axes[3], "d", 1), (axes[4], "e", 8)):
        title(ax, letter, f"Unseen shared base, $L={depth}$")
        selected = sorted((g for g in data["unseen"]["groups"] if g["depth"] == depth), key=lambda g:g["time"])
        transformer = [g["predictors"]["larger_seed0"]["gain_over_uniform_bits"] for g in selected]
        bayes = [g["predictors"]["bayes"]["gain_over_uniform_bits"] for g in selected]
        counts = [g["predictors"]["kt_unigram"]["gain_over_uniform_bits"] for g in selected]
        accepted = [g["bayes_released"] for g in selected]
        se_ribbon(ax, [4,16,64], transformer, BLUE, "--")
        se_points(ax, [t for t, ok in zip((4,16,64), accepted) if ok],
                  [v for v, ok in zip(bayes, accepted) if ok], BLUE)
        ax.errorbar([4,16,64], [v["mean"] for v in counts],
                    yerr=[v["standard_error"] for v in counts], color=GRAY,
                    marker="^", ls=":", ms=3.2, lw=1.05, capsize=1.2,
                    elinewidth=.6)
        unseen_style(ax)
        if depth == 8:
            ax.set_ylim(-.15, .49)
            missing = [str(g['time']) for g in selected if not g['bayes_released']]
            if missing:
                ax.text(.04,.93, ', '.join(missing) + ": informed Bayes withheld",
                        transform=ax.transAxes, ha="left", va="top", fontsize=7, color=".35")
        else:
            ax.set_ylim(-.03, .98)
            ax.set_yticks([0,.4,.8])

    title(axes[5], "f", "Unseen uniform base, $L=1$")
    selected = sorted((g for g in data["uniform"]["groups"] if g["depth"] == 1), key=lambda g:g["time"])
    transformer_mean = -np.asarray([g["transformer"]["mean"] for g in selected])
    transformer_se = np.asarray([g["transformer"]["se"] for g in selected])
    axes[5].fill_between([4,16,64], transformer_mean-transformer_se,
                         transformer_mean+transformer_se, color=BLUE, alpha=.12, lw=0)
    axes[5].plot([4,16,64], transformer_mean, color=BLUE, lw=1.15)
    axes[5].errorbar([4,16,64], [-g["counts"]["mean"] for g in selected],
                     yerr=[g["counts"]["se"] for g in selected], color=GRAY,
                     marker="^", ls=":", ms=3.2, lw=1.05, capsize=1.2,
                     elinewidth=.6)
    axes[5].plot([4,16,64], [0,0,0], color=BLUE, marker="o", mfc="white",
                 mew=.7, ms=3.2, ls="none", zorder=5)
    unseen_style(axes[5])
    axes[5].set(ylim=(-.42,.045), yticks=[-.4,-.2,0])

    for ax, letter, metric, heading in ((axes[6],"g","instantaneous","Copying: gain per token"),
                                       (axes[7],"h","cumulative","Copying: summed gain")):
        title(ax, letter, heading)
        for depth, color, style, marker in ((2, BLUE, "--", "o"),
                                             (3, ORANGE, "--", "s")):
            group = data["copying_full"][depth]
            ribbon(ax, np.arange(1,1025), group[metric], color, style,
                   names=("lower95","upper95"), alpha=.11 if depth == 2 else .055)
            early = data["copying_bayes"][depth][metric]
            bayes_times = np.arange(1, len(early["bayes_mean"]) + 1)
            shown = np.isin(bayes_times, COPYING_BAYES_DISPLAY_TIMES)
            ax.errorbar(bayes_times[shown], np.asarray(early["bayes_mean"])[shown],
                        yerr=np.asarray(early["bayes_source_se"])[shown],
                        color=color, ls="none", marker=marker, mfc="white", ms=3.2,
                        mew=.75, elinewidth=.6, capsize=1.0, zorder=6)
            if metric == "instantaneous":
                for time in (32, 64, 128):
                    endpoint = data[f"copying_endpoint{time}"][depth]
                    ax.errorbar([time], [endpoint["bayes_mean"]],
                                yerr=[endpoint["bayes_source_se"]], color=color,
                                ls="none", marker=marker, mfc="white", ms=3.2,
                                mew=.75, elinewidth=.6, capsize=1.0, zorder=6)
        ax.axhline(0,color=".5",lw=.6,zorder=-3)
        context_axis(ax, [1,16,256,1024], (.87,1180))
        ax.set(ylabel="Gain per token (bits)" if metric=="instantaneous"
               else "Summed gain (bits)")
    depth_handles = [Line2D([], [], color=color, ls=style, lw=1.15, marker=marker,
                            mfc="white", mew=.75, ms=3.2,
                            label=f"$L={depth}$")
                     for depth, color, style, marker in
                     ((2, BLUE, "--", "o"), (3, ORANGE, "--", "s"))]
    axes[6].legend(handles=depth_handles, frameon=False, loc="upper left", ncol=2,
                   handlelength=1.5, columnspacing=.8, handletextpad=.35,
                   borderaxespad=.1, fontsize=7.0)
    axes[6].set_ylim(-.035,1.15)
    axes[7].set_ylim(-25,850)
    fig.canvas.draw()
    from matplotlib.text import Text
    require(all(value.get_fontsize() >= 7.0 for value in fig.findobj(Text)
                if value.get_text()), "Figure contains text below the legibility floor")
    for extension in ("pdf","png"):
        fig.savefig(out / f"overview.{extension}", dpi=240,
                    metadata={"CreationDate": None, "ModDate": None} if extension == "pdf"
                    else {"Software":"section4_compact_figure.py"})
    plt.close(fig)


def make_receipt(data, checks, paths):
    with np.load(COPY_ENDPOINT128 / "paired_scores.npz", allow_pickle=False) as scores:
        endpoint128_histories = {str(depth): len(scores[f"a1_L{depth}_stationary_loss"])
                                 for depth in (0, 2, 3)}
    selected = {str(depth): [g["n_selected"] for g in sorted(
        (row for row in data["unseen"]["groups"] if row["depth"] == depth),
        key=lambda row: row["time"])] for depth in (1, 8)}
    selected["uniform_1"] = [g["n"] for g in sorted(
        (row for row in data["uniform"]["groups"] if row["depth"] == 1),
        key=lambda row: row["time"])]
    return dict(schema="section4-compact-figure-v10", no_new_experiments=True,
        script_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, checks=checks,
        size_inches=[5.5, 6.75], minimum_font_points=7.0,
        display={
            "x_axis":"Every panel uses a base-2 logarithmic axis of absolute observed tokens t.",
            "predictor_styles":"Two-layer transformers are solid; larger transformers are dashed (panel a run 2 dotted). Open markers show informed Bayes; gray dotted triangles show token counts. Copying depths share dashed four-layer curves, with circles for L=2 and squares for L=3.",
            "condition_colors":"Matched predictors share colors: blue is primary; panel (c) uses blue for measurement and orange for prediction; copying uses blue for L=2 and orange for L=3.",
            "uncertainty":"Copying transformer curves retain saved 95% paired bootstrap intervals; displayed Bayesian points show one sequence SE.",
            "copying_bayes_display_times":COPYING_BAYES_DISPLAY_TIMES.tolist(),
            "copying_bayes_endpoint_times":{"instantaneous":[32,64,128],"cumulative":[]},
            "copying_histories":{"full_transformer_curves":data["copying_full_n"],"bayesian_points_through64":256,"bayesian_points_at128":endpoint128_histories},
            "copying128":{"provisional":True,"scientific_reference_validated":False,"frequency_bank":0,"latent_rows":288,"numerical_uncertainty_certified":False},
            "selected_histories_at_t_4_16_64":selected},
        panels={"a":"Mismatch at product depth (4,4): two-layer transformer, both four-layer runs, informed Bayes and predicted stationary projection limit",
                "b":"Time-varying product source at depth (4,4): two new matrices after observed t=256",
                "c":"Time-varying source additivity: measured two-new-matrix excess and predicted twice-one-new-matrix excess",
                "d":"Unseen context, shared base L=1: larger transformer run 1, informed Bayes and token counts",
                "e":"Unseen context, shared base L=8: larger transformer run 1, informed Bayes and token counts; Bayes t64 " +
                    ("released" if next(g for g in data['unseen']['groups']
                     if (g['depth'],g['time'])==(8,64))['bayes_released'] else "withheld"),
                "f":"Unseen context, uniform base L=1: two-layer transformer, exact informed Bayes and token counts",
                "g":"Gain from token order per token at L=2,3; accepted Bayes through64 and explicitly provisional fixed-cohort Bayes at128",
                "h":"Summed gain from token order at L=2,3; full transformer cohort and informed-Bayes t=1..16 cohort"},
        uncertainty={"a,b,c":"Saved pointwise mean +/- 1.96 sequence SE",
                     "d,e,f":"Saved one sequence SE; selected counts differ by time",
                     "g,h":"Transformer curves: saved pointwise 95% paired whole-sequence bootstrap intervals. Informed Bayes points: one paired source-sequence SE."},
        scope="One frozen training run per displayed model; intervals do not measure training-seed uncertainty or simultaneous coverage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--unseen", type=Path, default=UNSEEN)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--release-copying-extension", action="store_true",
                        help="Replay completed extension scores and export compact figure inputs, then exit")
    args = parser.parse_args()
    if args.release_copying_extension:
        release_copying_extension(COPY_EXTENSION, OUT)
        return
    data, checks, paths = load_inputs(args.unseen.resolve())
    receipt = make_receipt(data, checks, paths)
    names = ("overview.pdf", "overview.png")
    if args.check_only:
        saved = read(args.out / "receipt.json")
        outputs = saved.pop("outputs")
        # The recomputed check errors are rounding residuals (below 1e-12) and can
        # differ in the last bit between machines; compare them with a tolerance.
        saved_checks, new_checks = saved.pop("checks"), receipt.pop("checks")
        require(saved_checks.keys() == new_checks.keys() and all(
            abs(saved_checks[k] - new_checks[k]) < 1e-9 for k in new_checks),
                "Compact figure numerical checks differ from the saved receipt")
        require(saved == receipt, "Compact figure receipt differs from its inputs or plotting code")
        require(outputs == {name: sha(args.out / name) for name in names},
                "Compact figure output hashes changed")
        print("Verified exact inputs, aggregate checks, display metadata and artifact hashes.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    draw(data, args.out)
    receipt["outputs"] = {name: sha(args.out / name) for name in names}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(checks=checks, outputs=receipt["outputs"],
                              size_inches=receipt["size_inches"]), indent=2))


if __name__=="__main__":
    main()

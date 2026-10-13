"""Render appendix Figures 3, 4, 6 and 7 from audited, unchanged results.

This presentation-only consumer leaves the original numerical releases,
figures and audits untouched. No sampling, inference or smoothing is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/layered-appendix-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import bayes_evidence_figure as divergence
import hidden_factor_campaign_results as hidden
import shared_base_campaign_results as shared
from appendix_figure_style import apply_style, style_axis, PALETTE, GRAY, DARK

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "results/transformer/appendix_style_2026-09-24"
RECEIPT = DESTINATION / "source_receipt.json"
BLUE, ORANGE, GREEN, PURPLE = PALETTE
INK = DARK
TIMES = [1, 4, 16, 64, 256, 1024]
INPUTS: dict[str, str] = {}
OUTPUTS: dict[str, str] = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    INPUTS[str(path.relative_to(ROOT))] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def axes_style(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(TIMES, [str(t) for t in TIMES])
    ax.set_xlim(.85, 1250)
    ax.minorticks_off()
    style_axis(ax)


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = stem.with_suffix("." + suffix)
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {"dpi": 240}
        fig.savefig(path, **kwargs)
        OUTPUTS[str(path.relative_to(ROOT))] = sha(path)
    plt.close(fig)


def bayes(ax, times, mean, se):
    ax.errorbar(times, mean, yerr=se, color=INK, linestyle="none", marker="o",
        markerfacecolor="white", markersize=3.5, markeredgewidth=.8,
        elinewidth=.7, capsize=1.5, zorder=5)


MODEL_SPECS = [("baseline", BLUE, "-", "2 layers: 6.4M"),
               ("large", ORANGE, "--", "4 layers: 6.4M"),
               ("continue_large", GREEN, "--", "4 layers: 25.6M"),
               ("xlarge", PURPLE, "--", "8 layers: 25.6M")]


def model_legend(fig, top):
    handles = [Line2D([], [], color=color, linestyle=style, label=label)
               for _, color, style, label in MODEL_SPECS]
    handles.append(Line2D([], [], color=INK, marker="o", markerfacecolor="white",
        linestyle="none", markersize=3.5, label="Informed Bayes"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, top),
        ncol=3, frameon=False, columnspacing=1.4, handlelength=2., labelspacing=.4)


def product_gaps():
    folder = ROOT / "results/transformer/conditional_accuracy/product"
    data, audit, receipt = [read(folder / name) for name in ("summary.json", "validation.json", "receipt.json")]
    require(audit["passed"] is True and receipt["passed"] is True,
            "Product release did not pass validation")
    for name in ("summary.json", "validation.json"):
        path = folder / name
        bind(path, receipt["files_sha256"][str(path.relative_to(ROOT))])
    require(len(data["cases"]) == 4, "Expected four product source settings")
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.65), sharex=True, sharey=True)
    bounds = []
    for i, (ax, case, color) in enumerate(zip(axes.flat, data["cases"], (BLUE, ORANGE, GREEN, PURPLE))):
        t = np.asarray(case["times"])
        mean, se = [np.asarray(case["transformer_minus_bayes"][key]) for key in ("mean", "se")]
        require(np.isfinite(mean).all() and np.isfinite(se).all() and (se >= 0).all(), "Invalid product scores")
        ax.axhline(0, color=GRAY, linewidth=.75)
        ax.errorbar(t, mean, yerr=se, color=color, linewidth=1.15, capsize=1.5, elinewidth=.7)
        a, b = case["depth_pair"]
        ax.set_title(rf"({chr(97+i)}) $L_1={a},\ L_2={b}$", loc="left", pad=5)
        bounds.extend((float((mean-se).min()), float((mean+se).max())))
        axes_style(ax)
    span = max(bounds)-min(bounds)
    axes[0, 0].set_ylim(min(bounds)-.06*span, max(bounds)+.07*span)
    fig.supylabel("Transformer minus Bayes regret (bits)", x=.016, fontsize=8)
    fig.supxlabel(r"Observed tokens, $t$", y=.024, fontsize=8)
    fig.subplots_adjust(left=.12, right=.985, bottom=.14, top=.94, hspace=.38, wspace=.16)
    save(fig, DESTINATION / "figure3")
    return {"sequences_per_setting": 256, "uncertainty": "One paired sequence standard error", "metric": "transformer_minus_bayes"}


def hidden_scaling():
    paths, data, _ = hidden.load_campaign()
    for path in paths:
        bind(path)
    group = next(g for g in data["groups"] if g["id"] == "full_prior")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), sharey=True)
    for seed, ax in enumerate(axes):
        for arm, color, style, _ in MODEL_SPECS:
            val = group["predictors"][f"{arm}_seed{seed}"]["next_kl_bits"]
            mean, se = np.asarray(val["mean"]), np.asarray(val["se"])
            ax.plot(TIMES, mean, color=color, linestyle=style)
            ax.fill_between(TIMES, np.maximum(0, mean-se), mean+se, color=color, alpha=.13, linewidth=0)
        val = group["predictors"]["bayes_hidden"]["next_kl_bits"]
        bayes(ax, TIMES, val["mean"], val["se"])
        axes_style(ax)
        ax.set_ylim(0, .9)
        ax.set_yticks([0, .3, .6, .9])
        ax.set_title(f"({chr(97+seed)}) Training run {seed+1}", loc="left", pad=5)
    model_legend(fig, 1.01)
    fig.supylabel("Expected next-token regret (bits)", x=.016, fontsize=8)
    fig.supxlabel(r"Observed tokens, $t$", y=.03, fontsize=8)
    fig.subplots_adjust(left=.105, right=.985, bottom=.21, top=.72, wspace=.16)
    save(fig, DESTINATION / "figure4")
    return {"sequences_per_run": 3840, "uncertainty": "One sequence standard error", "training_runs_separate": True}


def predictive_divergence():
    source = bind(ROOT / "results/transformer/shared_depth/summary.json")
    panels = divergence.load_panels(source)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.30), sharey=True)
    maximum = max(max(np.asarray(p["mean_bits"])+np.asarray(p["standard_error_bits"])) for p in panels)
    upper = np.ceil(1.06*maximum/.01)*.01
    for ax, panel, title in zip(axes, panels, ("(a) Known factorization", "(b) Known base law")):
        ax.errorbar(panel["observed_prefix"], panel["mean_bits"], yerr=panel["standard_error_bits"],
            color=BLUE, linewidth=1.15, marker=".", markersize=4, capsize=1.5, elinewidth=.7)
        axes_style(ax)
        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper+.005, .02))
        ax.set_title(title, loc="left", pad=5)
    axes[0].set_ylabel("Predictive divergence (bits)")
    fig.supxlabel(r"Observed tokens, $t$", y=.03, fontsize=8)
    fig.subplots_adjust(left=.11, right=.985, bottom=.22, top=.86, wspace=.16)
    save(fig, DESTINATION / "figure6")
    return {"metric": "kl_bayes_to_transformer", "sequences_per_family": 128,
            "uncertainty": "One sequence standard error", "full_distribution_kl_preserved": True}


def shared_long_points(report):
    """Bind every long-context point to the very same 256 source histories."""
    folder = ROOT / "results/transformer/shared_base_full_cohort_2026-09-23"
    data, audit = [read(folder / name) for name in ("summary.json", "validation.json")]
    bind(folder / "summary.json", audit["summary_sha256"])
    plan = ROOT / "code/transformer/configs/shared_base_posterior_full_plan_2026-09-17.json"
    bind(plan, data["plan_sha256"])
    require(data["passed"] is True and audit["passed"] is True and data["complete_campaign"] is True
            and not data["withheld"] and audit["groups_published"] == 8,
            "Long-context full-cohort numerical release is incomplete")
    scores_path = bind(folder / "paired_scores.npz", audit["outputs"]["paired_scores.npz"])
    expected = {(d, t) for d in shared.DEPTHS for t in (256, 1024)}
    require(len(data["groups"]) == 8 and {(g["depth"], g["time"]) for g in data["groups"]} == expected,
            "Long-context depth/time coverage differs")
    checked = 0
    with np.load(scores_path, allow_pickle=False) as arrays:
        for g in data["groups"]:
            d, t = g["depth"], g["time"]
            require(g["n_sequences"] == 256 and g["indices"] == list(range(256))
                and g["all_cases_retained"] is True and all(c["passed"] for c in g["numerical_checks"]),
                "A long-context point omitted histories or failed numerical checks")
            prefix = f"L{d}_t{t}"
            ids = arrays[prefix+"_source_ids"].tolist()
            require(ids == g["source_ids"] and len(set(ids)) == 256, "Long-context source IDs differ")
            values = arrays[prefix+"_bayes_regret"]
            require(values.shape == (256,) and np.isfinite(values).all(), "Incomplete Bayesian scores")
            require(abs(float(values.mean())-g["bayes"]["mean"]) < 2e-12
                and abs(float(values.std(ddof=1)/16)-g["bayes"]["se"]) < 2e-12,
                "Long-context Bayesian moments differ")
            for seed in shared.SEEDS:
                base = next(r for r in report["groups"] if r["depth"] == d and r["seed"] == seed)
                require(base["source_ids"] == ids, "Campaign and long-context source cohorts differ")
                for comparison in (c for c in g["comparisons"] if c["seed"] == seed):
                    value = base["predictors"][comparison["model"]]
                    j = value["times"].index(t)
                    require(comparison["regret"]["n"] == 256 and
                        abs(value["mean"][j]-comparison["regret"]["mean"]) < 2e-12 and
                        abs(value["se"][j]-comparison["regret"]["se"]) < 2e-12,
                        "Long-context model moments do not match the campaign")
                    checked += 1
    return data, {"complete_long_context_groups": 8, "source_ids_checked": 2048,
        "matching_model_moments": checked, "long_contexts": [256, 1024],
        "all_numerical_checks_passed": True, "new_computation": False}


def shared_scaling():
    paths, data, _ = shared.load_campaign()
    for path in paths:
        bind(path)
    long, checks = shared_long_points(data)
    names = {"baseline": "original100k", "large": "large100k",
             "continue_large": "continue_large_400000", "xlarge": "xlarge_400000"}
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 6.10), sharex=True, sharey="row")
    for row, depth in enumerate(shared.DEPTHS):
        limits = [0.]
        for seed, ax in enumerate(axes[row]):
            group = next(g for g in data["groups"] if g["depth"] == depth and g["seed"] == seed)
            for arm, color, style, _ in MODEL_SPECS:
                val = group["predictors"][names[arm]]
                mean, se = np.asarray(val["mean"]), np.asarray(val["se"])
                ax.plot(TIMES, mean, color=color, linestyle=style)
                ax.fill_between(TIMES, mean-se, mean+se, color=color, alpha=.13, linewidth=0)
                limits.extend((mean-se).tolist()); limits.extend((mean+se).tolist())
            extra = sorted((g for g in long["groups"] if g["depth"] == depth), key=lambda g: g["time"])
            b = group["bayes"]
            times = b["times"]+[g["time"] for g in extra]
            mean = b["mean"]+[g["bayes"]["mean"] for g in extra]
            se = b["se"]+[g["bayes"]["se"] for g in extra]
            require(times == TIMES, "Combined Bayesian contexts differ")
            bayes(ax, times, mean, se)
            limits.extend((np.asarray(mean)-se).tolist()); limits.extend((np.asarray(mean)+se).tolist())
            axes_style(ax)
            ax.set_title(rf"({chr(97+2*row+seed)}) $L={depth}$, run {seed+1}", loc="left", pad=5)
        span = max(limits)-min(limits)
        axes[row, 0].set_ylim(min(limits)-.025*span, max(limits)+.06*span)
    model_legend(fig, 1.)
    fig.supylabel("Expected next-token regret (bits)", x=.016, fontsize=8)
    fig.supxlabel(r"Observed tokens, $t$", y=.017, fontsize=8)
    fig.subplots_adjust(left=.10, right=.985, bottom=.085, top=.89, hspace=.40, wspace=.16)
    save(fig, DESTINATION / "figure7")
    return dict(checks, sequences_per_depth=256, uncertainty="One sequence standard error; numerical allowance separately in table")


def main():
    for module in (Path(__file__), Path(__file__).with_name("appendix_figure_style.py"),
                   Path(hidden.__file__), Path(shared.__file__), Path(divergence.__file__)):
        bind(module)
    with apply_style():
        panels = {"figure3": product_gaps(), "figure4": hidden_scaling(),
                  "figure6": predictive_divergence(), "figure7": shared_scaling()}
    receipt = {"schema": "appendix-source-figure-presentation-v1", "passed": True,
        "script_sha256": sha(__file__), "inputs": INPUTS, "outputs": OUTPUTS, "panels": panels,
        "scope": "Presentation-only in separate output paths; original numerical releases, figures and audits unchanged.",
        "sampling_performed": False, "inference_performed": False, "smoothing_performed": False,
        "command": "python code/scripts/appendix_source_figures.py"}
    RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "figures": list(panels), "long_context_checks": panels["figure7"]}, indent=2))


if __name__ == "__main__":
    main()

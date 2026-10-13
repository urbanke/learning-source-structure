"""Make paper-sized figures from the completed summary and its validation receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["stationary", "relabeled", "switch_one", "switch_both"]
LABELS = {"stationary": "Visible product labels", "relabeled": "Random joint labels",
          "switch_one": "One new matrix after $t=256$",
          "switch_both": "Two new matrices after $t=256$"}
BLUE, ORANGE, GREEN, PURPLE = "#2468A0", "#BF611C", "#288167", "#8A4E98"
DEPTH_COLORS = {1: ORANGE, 4: GREEN}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_stat(group, key, n_times):
    stat = group["metrics"][key]
    mean, se, lo, hi = (np.asarray(stat[k], dtype=float) for k in
                        ("mean", "se", "lower_pointwise_95", "upper_pointwise_95"))
    require(all(value.shape == (n_times,) and np.isfinite(value).all() for value in (mean, se, lo, hi)),
            "Nonfinite or incorrectly shaped summary statistic")
    require(np.all(se >= 0), "Negative standard error")
    require(np.allclose(lo, mean - 1.96 * se, rtol=0, atol=1e-12) and
            np.allclose(hi, mean + 1.96 * se, rtol=0, atol=1e-12), "Uncertainty band differs from mean ±1.96 SE")
    return mean, lo, hi


def load_summary(path):
    summary = json.loads(path.read_text())
    require(summary["schema"] == "product-shift-summary-v1", "Unsupported summary schema")
    require(summary["complete"] is True and summary["status"] == "complete", "Paper figures require a completed cohort")
    n = summary["evaluated_n_sequences_per_depth"]
    require(n >= 2 and n == summary["expected_n_sequences_per_depth"], "Incomplete sequence count")
    require(summary["included_indices"] == list(range(n)), "Sequence coverage is not contiguous and complete")
    require(summary["conditions"] == CONDITIONS and summary["depth_pairs"] == [[1, 1], [4, 4]],
            "Unexpected source conditions")
    require(summary["probabilities_renormalized"] is False, "Summary silently normalizes saved predictions")
    times = np.asarray(summary["times"], dtype=int)
    require(len(times) > 1 and np.all(np.diff(times) > 0) and times[0] >= 1 and times[-1] == 1024,
            "Unexpected context grid")
    groups = {}
    for group in summary["groups"]:
        key = (tuple(group["depth_pair"]), group["condition"])
        require(key not in groups and group["n_sequences"] == n, "Duplicate or incomplete group")
        groups[key] = group
        for metric in ("transformer_regret_bits", "bayes_regret_bits", "kl_bayes_to_transformer_bits"):
            checked_stat(group, metric, len(times))
    expected = {(tuple(pair), condition) for pair in summary["depth_pairs"] for condition in CONDITIONS}
    require(set(groups) == expected, "Missing or unexpected result group")
    return summary, groups, times


def save_figure(figure, results, stem):
    for extension in ("pdf", "png"):
        figure.savefig(results / f"{stem}.{extension}", dpi=220,
                       metadata={"Creator": "product_shift_paper_figures.py"})


def main_figures(summary, groups, times, results, plt, Line2D, projection):
    for pair in summary["depth_pairs"]:
        color = DEPTH_COLORS[pair[0]]
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
        for panel, (axis, condition) in enumerate(zip(axes.flat, CONDITIONS)):
            group = groups[tuple(pair), condition]
            transformer, tf_lower, tf_upper = checked_stat(
                group, "transformer_regret_bits", len(times))
            bayes, bayes_lower, bayes_upper = checked_stat(
                group, "bayes_regret_bits", len(times))
            axis.fill_between(times, tf_lower, tf_upper, color=color,
                              alpha=.13, linewidth=0)
            axis.plot(times, transformer, color=color, linewidth=1.25)
            axis.errorbar(times, bayes,
                          yerr=np.vstack((bayes-bayes_lower, bayes_upper-bayes)),
                          color=color, linestyle="none", marker="o", markersize=3.2,
                          markerfacecolor="white", markeredgewidth=.75,
                          elinewidth=.55, capsize=1.1, zorder=5)
            axis.set_title(f"({chr(97 + panel)}) {LABELS[condition]}",
                           fontsize=8, loc="left", pad=5)
            axis.set_ylim(bottom=0)
            axis.set_xscale("log", base=2)
            axis.set_xlim(.87, 1180)
            axis.set_xticks([1, 4, 16, 64, 256, 1024],
                            ["1", "4", "16", "64", "256", "1024"])
            axis.minorticks_off()
            axis.grid(axis="y", color="#D8DEE5", linewidth=.45)
            axis.tick_params(length=2.5, pad=2)
            if condition.startswith("switch"):
                axis.axvline(summary["switch_observed"], color=".45",
                             linewidth=.8, linestyle=":")
                axis.annotate("new matrices", (summary["switch_observed"], .96),
                              xycoords=("data", "axes fraction"), xytext=(3, 0),
                              textcoords="offset points", color=".35", fontsize=7.5,
                              ha="left", va="top")
            if condition == "relabeled":
                limit = projection["depths"][str(pair[0])]["quantities"]["stationary_kl_bits"]["mean"]
                axis.axhline(limit, color=".35", linestyle=":", linewidth=1.3)
                axis.annotate(f"Predicted limit: {limit:.3f}", xy=(.04, limit),
                              xycoords=("axes fraction", "data"), xytext=(0, -6),
                              textcoords="offset points", va="top", color=".3", fontsize=7.5)
        for axis in axes[:, 0]:
            axis.set_ylabel("Next-token regret (bits)")
        for axis in axes[1, :]:
            axis.set_xlabel("Observed tokens, $t$")
        handles = [Line2D([], [], color=color, linewidth=1.25),
                   Line2D([], [], color=color, linestyle="none", marker="o",
                          markerfacecolor="white", markersize=4)]
        fig.legend(handles, ["Transformer", "Informed Bayes"],
                   ncol=2, loc="upper center", frameon=False,
                   bbox_to_anchor=(.5, .995), fontsize=8)
        fig.text(.5, .01, rf"Source depths $L_1=L_2={pair[0]}$; "
                 f"{summary['evaluated_n_sequences_per_depth']:,} independent sources; intervals: mean ± 1.96 SE.",
                 ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .035, 1, .93), h_pad=1.5, w_pad=1.3)
        save_figure(fig, results, f"depth{pair[0]}")
        plt.close(fig)


def agreement_figure(summary, groups, times, results, plt, Line2D):
    styles = {"stationary": ("#555555", "-"), "relabeled": ("#1565a8", "--"),
              "switch_one": ("#237f53", "-."), "switch_both": ("#b24c27", ":")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45))
    for axis, pair in zip(axes, summary["depth_pairs"]):
        for condition in CONDITIONS:
            color, style = styles[condition]
            mean, lower, upper = checked_stat(groups[tuple(pair), condition], "kl_bayes_to_transformer_bits", len(times))
            axis.plot(times, mean, color=color, linestyle=style, linewidth=1.6)
            axis.fill_between(times, lower, upper, color=color, alpha=.10, linewidth=0)
        axis.axvline(summary["switch_observed"], color=".4", linewidth=1, linestyle=":")
        axis.text(summary["switch_observed"] + 15, .96, "new matrices", color=".35", fontsize=8,
                  transform=axis.get_xaxis_transform(), va="top")
        axis.set_title(rf"Source depths $L_1=L_2={pair[0]}$")
        axis.set_xlabel("Observed tokens")
        axis.set_xlim(0, 1024)
        axis.set_xticks([0, 256, 512, 768, 1024])
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=.16)
    axes[0].set_ylabel("KL(Bayes ∥ transformer) (bits)")
    handles = [Line2D([], [], color=styles[c][0], linestyle=styles[c][1], linewidth=1.6) for c in CONDITIONS]
    fig.legend(handles, [LABELS[c] for c in CONDITIONS], ncol=2, loc="upper center", frameon=False,
               bbox_to_anchor=(.5, 1.02), fontsize=9)
    fig.text(.5, .01, f"{summary['evaluated_n_sequences_per_depth']:,} independent sources per depth; "
             "shading: mean ± 1.96 SE (pointwise).", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .84), w_pad=1.2)
    save_figure(fig, results, "agreement")
    plt.close(fig)


def make_figures(summary_path, results):
    summary_path, results = Path(summary_path), Path(results)
    summary, groups, times = load_summary(summary_path)
    plan_path, benchmark_path = summary_path.with_name("plan.json"), summary_path.with_name("benchmark.json")
    plan, benchmark = (json.loads(path.read_text()) for path in (plan_path, benchmark_path))
    fingerprint = hashlib.sha256(json.dumps({key: value for key, value in plan.items() if key != "plan_id"},
                                           sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(fingerprint == plan["plan_id"] == summary["plan_id"] == benchmark["plan_id"],
            "Plan, summary and benchmark identities differ")
    require(benchmark["passed"] is True and benchmark["validation"]["passed"] is True,
            "Numerical benchmark has not passed")
    require(plan["n_seq"] == summary["evaluated_n_sequences_per_depth"] and
            plan["times"] == summary["times"] and plan["conditions"] == summary["conditions"],
            "Summary scope differs from its frozen plan")
    projection_path = summary_path.parent / "mismatch_limit/projection_summary.json"
    projection = json.loads(projection_path.read_text())
    require(projection["source_plan_id"] == summary["plan_id"] and projection["smoke"] is False,
            "Projection must use the same complete source cohort")
    require(all(projection["depths"][str(d)]["n_sources"] == plan["n_seq"] for d in (1, 4)),
            "Incomplete stationary projection")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "axes.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    results.mkdir(parents=True, exist_ok=True)
    main_figures(summary, groups, times, results, plt, Line2D, projection)
    agreement_figure(summary, groups, times, results, plt, Line2D)
    filenames = [f"{stem}.{extension}" for stem in ("depth1", "depth4", "agreement") for extension in ("pdf", "png")]
    rows = []
    for pair in summary["depth_pairs"]:
        for condition in CONDITIONS:
            group = groups[tuple(pair), condition]
            values = [rf"${pair[0]},{pair[1]}$", LABELS[condition]]
            for metric in ("transformer_regret_bits", "bayes_regret_bits", "transformer_minus_bayes_bits"):
                stat = group["metrics"][metric]
                values.append(rf"${stat['mean'][-1]:.4f}\pm {stat['se'][-1]:.4f}$")
            rows.append(" & ".join(values) + r" \\")
    (results / "endpoint_rows.tex").write_text("\n".join(rows) + "\n")
    filenames.append("endpoint_rows.tex")
    table = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"$(L_1,L_2)$ & Condition & Transformer & Bayes & Paired difference \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (results / "endpoint_table.tex").write_text("\n".join(table) + "\n")
    filenames.append("endpoint_table.tex")
    provenance = dict(schema="product-shift-paper-figures-v1", plan_id=summary["plan_id"],
                      summary=str(summary_path.resolve()), summary_sha256=sha(summary_path),
                      script_sha256=sha(__file__), complete=True,
                      plan_sha256=sha(plan_path), benchmark_sha256=sha(benchmark_path),
                      stationary_projection_sha256=sha(projection_path),
                      checkpoint_sha256=plan["model"]["checkpoint_sha256"],
                      numerical_validation=benchmark["validation"],
                      n_sequences_per_depth=summary["evaluated_n_sequences_per_depth"],
                      raw_records_sha256=summary["raw_records_sha256"],
                      raw_normalization_max_error=summary["raw_normalization_max_error"],
                      figure_sha256={name: sha(results / name) for name in filenames})
    (results / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    (results / "README.md").write_text(
        "# Product predictors under mismatch and time variation\n\n"
        f"Completed fixed cohort: {summary['evaluated_n_sequences_per_depth']:,} independent source draws "
        "at each of the two depth pairs (1,1) and (4,4), with four paired conditions per draw.\n\n"
        f"Frozen plan: `{summary['plan_id']}`. Transformer checkpoint SHA-256: "
        f"`{plan['model']['checkpoint_sha256']}`.\n\n"
        "`depth1` and `depth4` show next-token KL from the actual current transition row to each predictor. "
        "`agreement` shows KL from the training-prior Bayesian prediction to the frozen transformer prediction. "
        "Bands are mean ± 1.96 standard errors across independent source draws, pointwise rather than simultaneous. "
        "Only one trained transformer seed is used. No probabilities were renormalized.\n\n"
        "The dotted gray level in the random-label panel is the independently calculated stationary "
        "long-context Bayesian limit from the same source matrices; it is not a fit to these curves "
        "or a fixed-window transformer limit.\n\n"
        "The transformer weights and Bayesian product prior remain fixed, and both predictors retain the full history. "
        "The random label mapping is fixed within each sequence. New matrices first affect the prediction after 256 "
        "observed tokens; predictors are neither reset nor given a boundary signal. "
        "Predictive agreement outside the training family is an empirical question, not a Bayes optimality guarantee.\n\n"
        "Recovery descriptions in the source summary use each predictor's own prechange risk plus 0.05 bits, "
        "so the two numerical thresholds can differ. They are descriptive tested-grid crossings, without "
        "recovery-time confidence intervals.\n\n"
        "The benchmark passed wider-integration and exact depth-one checks on fixed source indices 0 and 3 "
        "in every condition. All saved records passed raw normalization and common-prefix causality checks. "
        "The test-law construction preserves the exact random label mapping and continues the current state across "
        "the parameter boundary.\n\n"
        f"Source summary: `{summary_path.resolve()}`. `provenance.json` records its checksum and the generated figure checksums.\n")
    print(json.dumps(dict(complete=True, results=str(results), figures=filenames)))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "output/product_shift_2026-09-18/summary.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/transformer/product_shift_2026-09-18")
    args = parser.parse_args()
    make_figures(args.summary, args.results)


if __name__ == "__main__":
    main()

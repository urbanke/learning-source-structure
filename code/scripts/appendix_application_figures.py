"""Restyle appendix Figures 9--11 from immutable saved numerical summaries."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from appendix_figure_style import apply_style, style_axis, BLUE, ORANGE, GREEN
from product_shift_paper_figures import load_summary, checked_stat, CONDITIONS, LABELS

ROOT = Path(__file__).resolve().parents[2]
DEPTH_COLORS = {1: BLUE, 4: GREEN}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_figure(fig, results, stem):
    stem = {"depth1": "figure9", "additivity_combined": "figure10", "coin_shift_main": "figure11"}[stem]
    fig.savefig(results / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(results / f"{stem}.png", dpi=220)

def displayed_bayes_indices(times):
    """Show a readable subset of markers, leaving every saved estimate intact."""
    selected = (1, 4, 16, 64, 128, 256, 288, 384, 512, 768, 1024)
    return np.flatnonzero(np.isin(times, selected))


def main_figures(summary, groups, times, results, plt, Line2D, projection):
    for pair in summary["depth_pairs"]:
        color = DEPTH_COLORS[pair[0]]
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.3))
        for panel, (axis, condition) in enumerate(zip(axes.flat, CONDITIONS)):
            group = groups[tuple(pair), condition]
            transformer, tf_lower, tf_upper = checked_stat(
                group, "transformer_regret_bits", len(times))
            bayes, bayes_lower, bayes_upper = checked_stat(
                group, "bayes_regret_bits", len(times))
            axis.fill_between(times, tf_lower, tf_upper, color=color,
                              alpha=.13, linewidth=0)
            axis.plot(times, transformer, color=color, linewidth=1.15)
            indices = displayed_bayes_indices(times)
            axis.errorbar(times[indices], bayes[indices],
                          yerr=np.vstack((bayes-bayes_lower, bayes_upper-bayes))[:, indices],
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
            style_axis(axis)
            if condition.startswith("switch"):
                axis.axvline(summary["switch_observed"], color=".45",
                             linewidth=.8, linestyle=":")
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
        handles = [Line2D([], [], color=color, linewidth=1.15),
                   Line2D([], [], color=color, linestyle="none", marker="o",
                          markerfacecolor="white", markersize=4)]
        fig.legend(handles, ["Two-layer transformer", "Informed Bayes"],
                   ncol=2, loc="upper center", frameon=False,
                   bbox_to_anchor=(.5, .995))
        fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.5, w_pad=1.3)
        save_figure(fig, results, f"depth{pair[0]}")
        plt.close(fig)


def additivity_figure(report, results):
    """Plot saved pointwise and simultaneous intervals without re-estimation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from appendix_figure_style import apply_style, style_axis, BLUE, ORANGE, GRAY

    times = np.asarray(report["postchange_times"])
    with apply_style():
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True)
        residual_limits = []
        for column, depth in enumerate((1, 4)):
            upper, lower = axes[:, column]
            tf = next(r for r in report["groups"]
                      if r["depth"] == depth and r["metric"] == "transformer_regret_bits")
            upper_limits = []
            for name, color, linestyle in (("two_factor_excess", BLUE, "-"),
                                           ("twice_one_factor_excess", ORANGE, ":")):
                stat = tf[name]
                upper.fill_between(times, stat["lower_pointwise_95"], stat["upper_pointwise_95"],
                                   color=color, alpha=.12, linewidth=0)
                upper.plot(times, stat["mean"], color=color, linestyle=linestyle, linewidth=1.15)
                upper_limits.append(max(stat["upper_pointwise_95"]))
            upper.set_ylim(0, 1.05 * max(upper_limits))
            lower.axhline(0, color=GRAY, linewidth=.9, zorder=2)
            for metric, marker, face, zorder in (("transformer_regret_bits", "s", BLUE, 4),
                                               ("bayes_regret_bits", "o", "white", 5)):
                row = next(r for r in report["groups"] if r["depth"] == depth and r["metric"] == metric)
                stat = row["contrast"]
                mean = np.asarray(stat["mean"])
                lo, hi = (np.asarray(stat[k]) for k in ("lower_simultaneous_95", "upper_simultaneous_95"))
                # Keep every tested time and the original simultaneous interval.
                lower.errorbar(times, mean, yerr=np.vstack((mean-lo, hi-mean)),
                               color=BLUE, linestyle="none", marker=marker, markersize=3.2,
                               markerfacecolor=face, markeredgewidth=.75,
                               elinewidth=.55, capsize=1.1, zorder=zorder)
                residual_limits.extend((float(lo.min()), float(hi.max())))
            for row, axis in enumerate((upper, lower)):
                style_axis(axis)
                axis.set_xscale("log", base=2)
                axis.set_xlim(245, 1080)
                axis.set_xticks([256, 512, 1024], ["256", "512", "1024"])
                axis.minorticks_off()
                axis.set_title(rf"({chr(97 + 2*row + column)}) Source depth $L_1=L_2={depth}$",
                               loc="left", pad=5)
            lower.set_xlabel("Observed tokens, $t$")
        limit = 1.08 * max(abs(x) for x in residual_limits)
        for axis in axes[1]:
            axis.set_ylim(-limit, limit)
        axes[1, 1].tick_params(labelleft=False)
        axes[0, 0].set_ylabel("Additional regret (bits)")
        axes[1, 0].set_ylabel("Departure from doubling (bits)")
        fig.legend([Line2D([], [], color=BLUE, linewidth=1.15),
                    Line2D([], [], color=ORANGE, linestyle=":", linewidth=1.15)],
                   ["Two new matrices", "Twice the one-new-matrix value"],
                   loc="upper center", bbox_to_anchor=(.5, 1), ncol=2, frameon=False)
        fig.legend([Line2D([], [], color=BLUE, ls="none", marker="s", ms=3.5),
                    Line2D([], [], color=BLUE, ls="none", marker="o", mfc="white", ms=3.5)],
                   ["Two-layer transformer", "Informed Bayes"],
                   loc="upper center", bbox_to_anchor=(.5, .515), ncol=2, frameon=False)
        fig.subplots_adjust(left=.12, right=.985, bottom=.095, top=.88, hspace=.66, wspace=.30)
        save_figure(fig, results, "additivity_combined")
        plt.close(fig)


def coin_figure(rows, results):
    times = np.arange(1, 257)
    with apply_style():
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
        for panel, (axis, name, title) in enumerate(zip(axes,
                ("mismatch_p04", "switch128"),
                (r"Mismatch: test bias $\theta=0.4$", "Change after 128 tokens"))):
            for prior, color in (("twopoint", BLUE), ("uniform", ORANGE)):
                row = next(r for r in rows if r["prior"] == prior and r["condition"]["name"] == name)
                mean, se = (np.asarray(row["transformer"][key]) for key in ("mean", "se"))
                axis.plot(times, mean, color=color, linewidth=1.15)
                axis.fill_between(times, mean - 1.96*se, mean + 1.96*se,
                                  color=color, alpha=.12, linewidth=0)
                selected = {1, 2, 4, 8, 16, 32, 64, 128, 256}
                if name == "switch128":
                    selected.update((129, 132, 144))
                indices = np.asarray(sorted(selected)) - 1
                bayes = np.asarray(row["exact_bayes"]["mean"])
                axis.plot(times[indices], bayes[indices], color=color, linestyle="none",
                          marker="o", markersize=3.2, markerfacecolor="white", markeredgewidth=.75,
                          zorder=5)
            axis.set_title(f"({chr(97+panel)}) {title}", loc="left", pad=5)
            style_axis(axis)
            axis.set_xscale("log", base=2)
            axis.set_xlim(.9, 272)
            axis.set_xticks([1, 4, 16, 64, 128, 256], ["1", "4", "16", "64", "128", "256"])
            axis.minorticks_off()
            axis.set_xlabel("Observed tokens, $t$")
            axis.set_ylim(bottom=0)
            axis.set_yticks([0, .05, .10, .15] if panel == 0 else [0, .4, .8, 1.2])
            if name == "switch128":
                axis.axvline(128, color=".45", linewidth=.75, linestyle=":")
        axes[0].set_ylabel("Next-token regret (bits)")
        fig.legend([Line2D([], [], color=BLUE, lw=1.15),
                    Line2D([], [], color=ORANGE, lw=1.15),
                    Line2D([], [], color="#333333", lw=1.15),
                    Line2D([], [], color="#333333", ls="none", marker="o", mfc="white", ms=3.5)],
                   [r"Prior $\{0.2,0.8\}$", r"Prior uniform on $[0,1]$",
                    "Two-layer transformer", "Informed Bayes (exact)"],
                   loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1))
        fig.subplots_adjust(left=.11, right=.985, bottom=.17, top=.72, wspace=.3)
        save_figure(fig, results, "coin_shift_main")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/appendix_style_2026-09-24")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    product = ROOT / "results/transformer/product_shift_2026-09-18"
    paths = [product / "summary.json", product / "additivity/summary.json",
             product / "mismatch_limit/projection_summary.json",
             ROOT / "results/transformer/coin_shift_2026-09-18/summary.json"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    summary, groups, times = load_summary(paths[0])
    projection = json.loads(paths[2].read_text())
    # The appendix shows depth one only; retain the complete source summary.
    presentation = dict(summary, depth_pairs=[[1, 1]])
    with apply_style():
        main_figures(presentation, groups, times, args.out, plt, Line2D, projection)
    additivity = json.loads(paths[1].read_text())
    additivity_figure(additivity, args.out)
    coin = json.loads(paths[3].read_text())
    assert coin["complete"] and coin["n_sequences_per_condition"] == 4096
    coin_figure(coin["rows"], args.out)
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    receipt = dict(schema="appendix-application-style-v1", source_sha256=before,
                   source_summaries_unchanged=True, no_new_estimates=True,
                   product_bayes_displayed_contexts=times[displayed_bayes_indices(times)].tolist(),
                   product_bayes_all_contexts=times.tolist(),
                   residual_contexts=additivity["postchange_times"],
                   simultaneous_critical_value=additivity["bootstrap"]["critical_value"],
                   residual_intervals="Original saved simultaneous endpoints at every context",
                   figure_sha256={f"figure{n}.{ext}": sha(args.out / f"figure{n}.{ext}")
                                  for n in (9, 10, 11) for ext in ("pdf", "png")})
    (args.out / "applications_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()

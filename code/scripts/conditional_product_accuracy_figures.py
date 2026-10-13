"""Render the fixed-depth product results from their saved numerical summary.

This is the canonical presentation step after conditional_product_accuracy.py.
It sets a shared y range from all four panels, avoiding the plotting default
that freezes the range after the first panel. No predictions are recomputed.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/transformer/conditional_accuracy/product"


def main():
    summary = json.loads((RESULT / "summary.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for is_gap in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(8., 5.7), sharex=True, sharey=True)
        bounds = []
        for axis, case in zip(axes.flat, summary["cases"]):
            times = np.asarray(case["times"])
            if is_gap:
                mean, se = [np.asarray(case["transformer_minus_bayes"][key]) for key in ("mean", "se")]
                axis.axhline(0, color=".5", linewidth=.8)
                axis.errorbar(times, mean, yerr=se, color="#1565a8", marker="o", markersize=3, capsize=2, lw=1.3)
                bounds.extend((float((mean-se).min()), float((mean+se).max())))
            else:
                for key, color, style in (("transformer", "#1565a8", "-"), ("informed_bayes", "#d77719", "--")):
                    mean, se = [np.asarray(case[key][stat]) for stat in ("mean", "se")]
                    axis.plot(times, mean, color=color, linestyle=style, linewidth=1.7)
                    axis.fill_between(times, np.maximum(0, mean-se), mean+se, color=color, alpha=.14, linewidth=0)
                    bounds.append(float((mean+se).max()))
            a, b = case["depth_pair"]
            axis.set_title(rf"$L_1={a},\ L_2={b}$")
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
            axis.set_xlim(1, 1024)
            axis.grid(True, alpha=.16)
        if is_gap:
            span = max(bounds) - min(bounds)
            axes[0, 0].set_ylim(min(bounds)-.07*span, max(bounds)+.07*span)
            fig.supylabel("Transformer regret minus Bayesian regret (bits)", x=.018)
        else:
            axes[0, 0].set_ylim(0, 1.04*max(bounds))
            fig.supylabel("Expected next-token regret (bits)", x=.018)
            fig.legend([Line2D([], [], color="#1565a8", lw=1.7), Line2D([], [], color="#d77719", lw=1.7, ls="--")],
                       ["Transformer", "Bayes with the training prior"], loc="upper center", ncol=2, frameon=False)
        fig.supxlabel("Observed tokens, $t$", y=.035)
        fig.tight_layout(rect=(.035, .05, 1, 1 if is_gap else .94))
        name = "gaps" if is_gap else "curves"
        for extension in ("pdf", "png"):
            fig.savefig(RESULT / f"{name}.{extension}", dpi=180,
                        metadata={"Creator": "conditional_product_accuracy_figures.py"})
        plt.close(fig)
    print("Rendered product curves and paired differences from saved summary")


if __name__ == "__main__":
    main()

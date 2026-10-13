"""One readable view of the validated product pilot; no new measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "results/transformer/product_component_pilot"
    report = json.loads((directory / "summary.json").read_text())
    if not report["passed"] or report["n_sequences"] != 32:
        raise ValueError("Expected the complete validated32-sequence pilot")
    t = np.asarray(report["prefix_lengths"])
    keep = t <= 128
    t = t[keep]
    fig, (ax, coverage) = plt.subplots(2, 1, figsize=(8.2, 6.1), sharex=True,
                                     gridspec_kw={"height_ratios": [4, 1], "hspace": .15})
    for predictor, color, label, style in (
            ("bayes", "#166B94", "Bayesian weight", "-"),
            ("transformer", "#D05A2B", "Fitted transformer weight", "--")):
        stats = report["paired_weights"]["both"][predictor]
        mean = np.asarray(stats["mean"], dtype=float)[keep]
        se = np.asarray(stats["se"], dtype=float)[keep]
        ax.plot(t, mean, color=color, ls=style, lw=2.3, label=label)
        ax.fill_between(t, mean-se, mean+se, color=color, alpha=.12, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean weight on copying both coordinates", fontsize=11)
    ax.set_title("Copying both coordinates receives increasing predictive weight", fontsize=13, pad=15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", color="#e3e7ea", lw=.7)
    count = 32*np.asarray(report["eligible_fraction"])[keep]
    coverage.fill_between(t, 0, count, color="#A8B5BF", alpha=.55, step="mid")
    coverage.plot(t, count, color="#667681", lw=1)
    coverage.set_ylim(0, 33)
    coverage.set_yticks([0, 16, 32])
    coverage.set_ylabel("Usable\nsequences", fontsize=10)
    coverage.set_xlabel("Tokens observed in the new sequence", fontsize=11)
    coverage.set_xscale("log", base=2)
    coverage.set_xlim(1, 128)
    coverage.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    coverage.xaxis.set_major_formatter(ScalarFormatter())
    for axis in (ax, coverage):
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .038, "Both lines average the same sequences at each prefix. Early weights cannot always be distinguished.",
             ha="center", fontsize=9)
    fig.text(.5, .013, "32 fresh product-source sequences; one fixed trained model. Shading: one standard error across sequences.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.12, right=.97, top=.9, bottom=.16)
    fig.savefig(directory / "copying_explained.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / "copying_explained.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

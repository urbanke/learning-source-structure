"""Plot the actual-chain learning costs and tabulate paired prediction errors.

MPLCONFIGDIR=/tmp/la-mpl python code/scripts/context_finite_prediction_figure.py

Reads the completed, independently audited 24-setting experiment. The figure
shows all 72 means; the table compares the longest runs at d=256. No fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/mechanism/context_finite_prediction_check.json"
DEPTHS, ALPHAS, DS = (1, 3, 6), (.5, 1., 1.5, 2.), (64, 256)


def main():
    data = json.loads(SOURCE.read_text())
    assert data["complete"] and len(data["cells"]) == 24
    audit = json.loads((ROOT / "results/mechanism/context_finite_prediction_audit.json").read_text())
    assert audit["status"] == "passed"
    assert audit["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    keyed = {(c["configuration"]["L"], c["configuration"]["alpha"],
              c["configuration"]["d"]): c for c in data["cells"]}
    assert set(keyed) == {(L, a, d) for L in DEPTHS for a in ALPHAS for d in DS}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42})
    colors = ("#2369a0", "#ba5b23")
    fig, axs = plt.subplots(3, 4, figsize=(6.5, 5.4), sharex=True)
    for i, L in enumerate(DEPTHS):
        for j, alpha in enumerate(ALPHAS):
            ax = axs[i, j]
            for d, color in zip(DS, colors):
                c = keyed[L, alpha, d]
                x = np.asarray(c["transitions"])
                y, se = np.asarray(c["redundancy"]["mean"]), np.asarray(c["redundancy"]["SE"])
                p = c["comparisons"]["joint_stationary"]["prediction"]
                pm, ps = np.asarray(p["mean"]), np.asarray(p["SE"])
                ax.plot(x, pm, color=color, lw=1.25)
                ax.fill_between(x, pm-2*ps, pm+2*ps, color=color, alpha=.13, lw=0)
                ax.errorbar(x, y, yerr=2*se, fmt="o", ms=3, mfc="white",
                            mec=color, color=color, elinewidth=.7, capsize=1.5)
            ax.set_xscale("log", base=4)
            ax.set_yscale("log")
            ax.set_xlim(640, 420000)
            ax.set_xticks([1024, 16384, 262144], ["1024", "16384", "262144"])
            ax.yaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000, 10000, 30000, 100000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.16, which="major", lw=.5)
            if i == 0:
                ax.set_title(r"$\alpha="+f"{alpha:g}"+"$")
            if j == 0:
                ax.set_ylabel(f"$L={L}$\nLearning cost (bits)")
            if i == 2:
                ax.set_xlabel("Transitions $t$")
    fig.legend(handles=[Line2D([], [], color=colors[0], label="$d=64$"),
                        Line2D([], [], color=colors[1], label="$d=256$"),
                        Line2D([], [], color=".2", marker="o", mfc="white", ls="none",
                               label=r"Bayes cost: mean $\pm 2$ SE"),
                        Line2D([], [], color=".2", label=r"Finite comparison $\mathcal{J}_d$")],
               loc="upper center", bbox_to_anchor=(.53, 1.005), ncol=2,
               frameon=False, columnspacing=1.5, handlelength=2)
    fig.subplots_adjust(left=.14, right=.975, bottom=.085, top=.875, wspace=.46, hspace=.28)
    fig.savefig(ROOT / "figures/context_finite_prediction.pdf")
    plt.close(fig)

    lines = [r"\begin{tabular}{ccrrrr}", r"\toprule",
             r"$L$ & $\alpha$ & Measured (bits) & $S_a=1$ (bits) & Finite (bits) & Difference (\%)\\",
             r"\midrule"]
    table_rows = []
    for L in DEPTHS:
        for alpha in ALPHAS:
            c = keyed[L, alpha, 256]
            old = c["comparisons"]["old_stationary"]
            finite = c["comparisons"]["joint_stationary"]
            measured, se = c["redundancy"]["mean"][-1], c["redundancy"]["SE"][-1]
            op, fp = old["prediction"]["mean"][-1], finite["prediction"]["mean"][-1]
            error = 100*finite["measured_over_prediction_minus_one"][-1]
            error_se = 100*finite["relative_difference_delta_method_SE"][-1]
            lines.append(f"{L} & {alpha:g} & ${measured:.0f}\\pm{se:.0f}$ & {op:.0f} & {fp:.0f} & ${error:+.2f}\\pm{error_se:.2f}$"+r"\\")
            table_rows.append(dict(L=L, alpha=alpha, d=256, transitions=262144,
                measured_bits=measured, measured_SE_bits=se,
                old_prediction_bits=op, finite_prediction_bits=fp,
                relative_difference_percent=error, paired_delta_method_SE_percent=error_se))
        if L != DEPTHS[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "results/mechanism/context_finite_prediction_table.tex").write_text("\n".join(lines)+"\n")
    summary = dict(source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        plotted_means=72, repetitions_per_setting=16, table_rows=table_rows)
    (ROOT / "results/mechanism/context_finite_prediction_figure.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("Wrote the 72-mean figure and 12-row comparison table.")


if __name__ == "__main__":
    main()

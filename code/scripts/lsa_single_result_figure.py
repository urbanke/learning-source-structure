#!/usr/bin/env python3
"""Figures and text for the single-distribution statement; no new measurements.

Run: python code/scripts/lsa_single_result_figure.py
Smoke: python code/scripts/lsa_single_result_figure.py --out-dir /tmp/lsa_single_figure
Inputs are the saved fixed-source and source-averaged validation JSON files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def key(cell):
    return cell["d"], cell["alpha"], cell["source_depth"], cell["k"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/mechanism")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "results/mechanism"
    validation = json.loads((data_dir / "lsa_single_validation.json").read_text())
    saved = json.loads((data_dir / "lsa_expected_profile.json").read_text())
    source = json.loads((data_dir / "lsa_single_source_validation.json").read_text())
    assert validation["complete"] and source["complete"]
    fixed = {key(cell): cell for cell in saved["cells"]}
    cells = validation["cells"]
    assert len(cells) == 12 and len({key(c)[:3] for c in cells}) == 4
    for cell in cells:
        p = np.array(fixed[key(cell)]["source_probabilities"])
        assert abs(p.sum() - 1) < 1e-12
        cell["entropy_bits"] = -float(np.dot(p[p > 0], np.log2(p[p > 0])))
        assert cell["concavity_lower"]["lower_bits"] <= cell["concavity_upper_bits"]

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    groups = sorted({key(c)[:3] for c in cells})
    colors = ["#2463a5", "#c46a23", "#3b8064", "#98528c"]
    for index, (group, color) in enumerate(zip(groups, colors)):
        selected = sorted((c for c in cells if key(c)[:3] == group), key=lambda c: c["k"])
        x = np.array([c["k"] for c in selected])
        entropy = np.array([c["k"] * c["entropy_bits"] for c in selected])
        predicted = np.array([c["deterministic_prediction"]["corrected_bits"] for c in selected])
        measured = np.array([c["saved_comparison"]["controlled_mean_bits"] for c in selected])
        se = np.array([c["saved_comparison"]["controlled_SE_bits"] for c in selected])
        lower = np.array([c["concavity_lower"]["lower_bits"] for c in selected])
        upper = np.array([c["concavity_upper_bits"] for c in selected])
        d, alpha, depth = group
        label = rf"$d={d},\ \alpha={alpha:g},\ L={depth}$"
        axes[0].fill_between(x, lower - entropy, upper - entropy, color=color, alpha=.09)
        axes[0].plot(x, predicted - entropy, color=color, lw=1.6, label=label)
        axes[0].errorbar(x, measured - entropy, yerr=2 * se, fmt="o", color=color,
                         markerfacecolor="white", ms=4, lw=1, capsize=2)
        # Small horizontal offsets make all four uncertainty bars visible.
        shifted = x + (index - 1.5) * 1.35
        axes[1].errorbar(shifted, predicted - measured, yerr=2 * se,
                         fmt="o", color=color, markerfacecolor="white", ms=4,
                         capsize=2, lw=1)
    for ax in axes:
        ax.set_xlabel(r"Number of observations $k$")
        ax.set_xticks([16, 64, 128])
        ax.set_xlim(9, 135)
        ax.grid(axis="y", color="#dddddd", lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Extra learning cost (bits)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.4,
                   borderaxespad=.1, labelspacing=.35)
    axes[1].set_ylabel("Formula minus simulation (bits)")
    axes[1].axhline(0, color="#777777", lw=.8)
    axes[1].set_ylim(-.18, .18)
    axes[1].set_yticks([-.15, -.10, -.05, 0, .05, .10, .15])
    fig.tight_layout(w_pad=1.5)
    fig.savefig(args.out_dir / "lsa_single_validation.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "lsa_single_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    max_error = max(abs(c["corrected_minus_controlled_mean_bits"]) for c in cells)
    widths = [c["concavity_width_bits"] for c in cells]
    outside = sum(abs(c["corrected_difference_in_controlled_SEs"]) > 2 for c in cells)
    summary = rf"""We first keep each source distribution fixed. The saved experiments use
four distributions: $d=64,256$ and $(\alpha,L)=(0.5,2),(1.5,6)$,
each checked at $k=16,64,128$, with 1,024 samples per case.
The formula uses the full LSA depth mixture, with exact means and
covariances from \eqref{{eq:mean}}--\eqref{{eq:cov}}; no parameters are fitted.
To reduce sampling noise, the saved estimates subtract a linear function
of $\Phi-\nu$, whose expectation is zero. This changes their variance,
not their expected value. Figure~\ref{{fig:single}} compares these estimates
with \eqref{{eq:estimate}}.

The largest observed discrepancy is {max_error:.3f} bits.
The intervals in \eqref{{eq:result}} have widths from {min(widths):.2f}
to {max(widths):.2f} bits. In {outside} of the 12 cases, the discrepancy
exceeds two sample standard errors: the remaining approximation error
is measurable despite being small in bits.
"""
    (args.out_dir / "lsa_single_validation_summary.tex").write_text(summary)

    rows = []
    for c in source["comparisons"]:
        rows.append(f"{c['alpha']:g} & {c['k']} & {c['prediction']['mean']:.2f}"
                    f" & {c['saved_rho_mean']:.2f}"
                    f" & {c['saved_rho_SE_converted_to_ddof1']:.2f} " + r"\\")
    table = r"""We also average over the source construction in \eqref{eq:source}.
The saved experiments use $d=256$, $L=3$, and 32 source draws for each
of $\alpha=0.75,1,1.25$. We reconstruct those source probabilities from
their recorded seeds and average the conditional formula at the same
32 distributions. No new token sequences are generated.
The table reports extra learning cost, in bits, for coding depths
$1,\ldots,26$.
\begin{center}
\begin{tabular}{rrrrr}
\toprule
$\alpha$ & $k$ & Formula & Simulation & Simulation SE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}
The standard errors include both source and sequence variation; the
historical values have been converted to the sample-variance convention.
Variation between the 32 sources also affects the averaged formula.
The saved data do not provide the paired standard error of the difference.
The source-averaged differences range from 0.71 to 6.23 bits; without
paired standard errors, they do not give a precise accuracy test.
"""
    (args.out_dir / "lsa_single_source_validation_table.tex").write_text(table)
    print(json.dumps({"fixed_cases": len(cells), "source_cases": len(source["comparisons"]),
                      "maximum_absolute_discrepancy_bits": max_error,
                      "outside_two_SE": outside, "bound_width_range_bits": [min(widths), max(widths)],
                      "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()

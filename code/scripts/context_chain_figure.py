"""Figure for the chain check of the per-context curve, depth 1.

Top: measured transition redundancy of the tilted row-product reference
on stationary-start chains (mean of repetitions, band two sample SE) for
alpha = 0.5, 1.0, 1.5, and the double sum sum_ab r_1(t pi_a q_b)
(black dotted), nothing fitted.  Bottom: measured divided by the
prediction.

    python code/scripts/context_chain_figure.py \
        --in results/mechanism/context_chain_check.json \
        --out figures/context_chain_check.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})

    r = json.loads(args.inp.read_text())
    n = np.array(r["n"], float)
    colors = {0.5: "#2b6a99", 1.0: "#2a8a84", 1.5: "#b0413e"}

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(4.4, 4.6),
        gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.30})
    first = True
    ratio_bounds = []
    for cell in r["alphas"]:
        a = cell["alpha"]
        m = np.array(cell["red_mean"])
        se = np.array(cell["red_se"])
        pred = np.array(cell["pred_pi_mean"])
        c = colors.get(a, "#666666")

        ax.plot(n, m, "-", lw=1.4, color=c, label=f"$\\alpha={a}$")
        ax.fill_between(n, m - 2 * se, m + 2 * se, color=c,
                        alpha=0.18, lw=0)
        ax.plot(n, pred, ":", lw=1.0, color="black", alpha=0.8,
                label="$\\sum_{a,b} r_1(t \\pi_a q_b)$"
                if first else None)
        first = False

        sel = n >= 64
        ratio = m[sel] / pred[sel]
        rse = 2 * se[sel] / pred[sel]
        ratio_bounds.extend([float(min(ratio-rse)),float(max(ratio+rse))])
        axr.plot(n[sel], ratio, "-", lw=1.2, color=c)
        axr.fill_between(n[sel], ratio - rse, ratio + rse,
                         color=c, alpha=0.18, lw=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$ (transitions)")
    ax.set_ylabel("redundancy [bits]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    axr.set_xscale("log")
    axr.set_ylim(min(ratio_bounds)-0.02,max(ratio_bounds)+0.02)
    axr.axhline(1.0, color="#999999", lw=0.6, ls=":")
    axr.set_xlabel("$t$ (transitions)")
    axr.set_ylabel("measured / predicted", fontsize=8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()

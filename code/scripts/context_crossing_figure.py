"""Figure: the difference of the two code lengths per transition,
(ell_F - ell_C)/t, on sampled chains at depth 1, d = 256, with the
reconstructed crossing marked (the balance using the
premium and finite-size terms from the measurement, i.e. the last zero
of G - P in the stored curves).

    python code/scripts/context_crossing_figure.py \
        --in results/mechanism/context_crossing_check.json \
        --out figures/crossing_anatomy.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def last_zero(n, y):
    idx = np.nonzero(np.diff(np.sign(y)) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[-1])
    f = y[i] / (y[i] - y[i + 1])
    return float(np.exp(np.log(n[i]) + f * math.log(n[i + 1] / n[i])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})

    r = json.loads(args.inp.read_text())
    n = np.array(r["n"], float)
    colors = {0.5: "#2b6a99", 1.0: "#2a8a84"}

    fig, axes = plt.subplots(1, len(r["alphas"]), figsize=(7.0, 2.9),
                             gridspec_kw={"wspace": 0.25})
    for col, cell in enumerate(r["alphas"]):
        a = cell["alpha"]
        c = colors.get(a, "#666666")
        diff = np.array(cell["diff_mean"])
        dse = np.array(cell["diff_se"])
        pred = last_zero(n, np.array(cell["G_mean"])
                         - np.array(cell["P_mean"]))

        ax = axes[col]
        sel = n >= 64
        ax.plot(n[sel], diff[sel] / n[sel], "-", lw=1.4, color=c)
        ax.fill_between(n[sel], (diff[sel] - 2 * dse[sel]) / n[sel],
                        (diff[sel] + 2 * dse[sel]) / n[sel],
                        color=c, alpha=0.18, lw=0)
        ax.axhline(0.0, color="#555555", lw=0.7)
        ax.axvline(pred, color="#888888", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"$\\alpha = {a}$", fontsize=9)
        ax.set_xlabel("$t$ (transitions)")
        if col == 0:
            ax.set_ylabel("$(\\ell_F - \\ell_C)/t$  [bits/transition]",
                          fontsize=8)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()

"""Figure: the optimal learning curve for one unknown distribution
drawn from the layered prior (depth 3), measured using the code
that mixes over depths, against the master
curve d r_3(k/d) of master_curve.py.

Top: measured curves for five alphabet sizes; black dotted, the
master curve d r_3(k/d); colored dashed, its tail, the two-term
expression (d/2) log2(k/d) + c_3 d (the simpler approximation,
kept deliberately). Bottom: measured divided by the master curve,
against samples per symbol k/d; the five curves collapse onto one
band around one.

    python code/scripts/necessary_curve_figure.py \
        --in results/mechanism/necessary \
        --master results/mechanism/master_curves.json \
        --out figures/necessary_iid.pdf
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=Path, required=True)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})

    mc = json.loads(args.master.read_text())
    mx = np.log(np.array(mc["x"]))
    mr3 = np.array(mc["r3_bits"])
    c3 = mc["c_bits"]["3"]
    slope0 = mc["slope0_bits_per_sample"]["3"]

    def r3(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        out = np.interp(np.log(x), mx, mr3)
        hi = np.log(x) > mx[-1]
        out[hi] = 0.5 * np.log2(x[hi]) + c3
        lo = np.log(x) < mx[0]
        out[lo] = slope0 * x[lo]
        return out

    files = sorted(args.indir.glob("nec_d*.json"),
                   key=lambda f: json.loads(f.read_text())["d"])
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(4.4, 4.6),
        gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.30})
    colors = {64: "#b0413e", 256: "#7a4f9e", 1024: "#2b6a99",
              4096: "#2a8a84", 16384: "#3d8a3d"}
    mlo, mhi = np.inf, 0.0
    first = True
    for f in files:
        r = json.loads(f.read_text())
        d = r["d"]
        k = np.array(r["k"], float)
        m = np.array(r["rho_mean"])
        se = np.array(r["rho_se"])
        c = colors.get(d, "#666666")

        ax.plot(k, m, "-", lw=1.4, color=c, label=f"$d={d}$")
        ax.fill_between(k, m - 2 * se, m + 2 * se, color=c,
                        alpha=0.18, lw=0)
        mlo = min(mlo, float(m[m > 0].min()))
        mhi = max(mhi, float((m + 2 * se).max()))

        master = d * r3(k / d)
        ax.plot(k, master, ":", lw=1.0, color="black", alpha=0.8,
                label="$d\\, r_3(k/d)$" if first else None)
        tail = d * (0.5 * np.log2(k / d) + c3)
        pos = tail > 0
        ax.plot(k[pos], tail[pos], "--", lw=0.8, color=c,
                alpha=0.5)
        first = False

        sel = k >= 8
        x = k[sel] / d
        ratio = m[sel] / master[sel]
        rse = 2 * se[sel] / master[sel]
        axr.plot(x, ratio, "-", lw=1.2, color=c)
        axr.fill_between(x, ratio - rse, ratio + rse, color=c,
                         alpha=0.18, lw=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(mlo * 0.7, mhi * 1.6)
    ax.set_xlabel("$k$ (samples)")
    ax.set_ylabel("redundancy [bits]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    axr.set_xscale("log")
    # Include the complete two-SE bands, including the early sparse samples.
    axr.set_ylim(0.80, 1.47)
    axr.axhline(1.0, color="#999999", lw=0.6, ls=":")
    axr.set_xlabel("$k/d$ (samples per symbol)")
    axr.set_ylabel("measured / $d\\, r_3(k/d)$", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()

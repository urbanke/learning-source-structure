"""Crossing position against alphabet size: measurement vs theory.

Left: median measured crossing n* against d, for a few depths at two
tilts.  Filled circles: the exact grid (256 seeds, exact to the
token, d <= 1024).  Open circles: the takeover estimates (16 seeds,
prior-guided grid, d up to 32768).  Guides: a pure d^2 slope, and for
alpha = 0.5, L = 3 the ceiling nubar(alpha, L) d^2 of the theorem.

Right: the effective exponent, the local slope of log n* against
log d, for the same series; the theorem says it approaches 2.

    python code/scripts/crossing_scaling_figure.py \
        --exact results/exact_grid --takeover results/takeover_summaries \
        --master results/mechanism/master_curves.json \
        --out figures/crossing_scaling.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EG = 0.5772156649015329
L2 = math.log(2.0)


def load_exact(dirpath: str):
    out = {}
    for f in glob.glob(f"{dirpath}/exact_*.json"):
        m = re.search(r"exact_d(\d+)_a([\d.]+)_L(\d+)", f)
        r = json.load(open(f))
        fp = [s["first_pos"] for s in r["seeds"] if s.get("first_pos")]
        if fp:
            out[(int(m.group(1)), float(m.group(2)), int(m.group(3)))] = (
                float(np.median(fp)),
                float(np.percentile(fp, 25)),
                float(np.percentile(fp, 75)))
    return out


def load_takeover(dirpath: str):
    out = {}
    for f in glob.glob(f"{dirpath}/summary_*.json"):
        r = json.load(open(f))
        if r.get("n_censored", 0) > 0:
            continue
        ns = r.get("n_star_last") or []
        ns = [x for x in ns if x]
        if ns:
            out[(r["d"], r["alpha"], r["L"])] = (
                float(np.median(ns)),
                float(np.percentile(ns, 25)),
                float(np.percentile(ns, 75)))
    return out


def nubar(alpha: float, L: int, master: dict) -> float | None:
    key = str(L)
    if f"r{L}_bits" not in master or key not in master["c_bits"]:
        return None
    xs = np.log(np.array(master["x"]))
    rl = np.array(master[f"r{L}_bits"])
    cL = master["c_bits"][key]
    s0 = master["slope0_bits_per_sample"][key]

    def r(x):
        x = np.asarray(x, float)
        out = np.interp(np.log(x), xs, rl)
        hi = np.log(x) > xs[-1]
        out[hi] = 0.5 * np.log2(x[hi]) + cL
        lo = np.log(x) < xs[0]
        out[lo] = s0 * x[lo]
        return out

    u = np.linspace(-14, 0, 400)
    s = np.exp(u)
    w = np.gradient(u) * s
    S, T = np.meshgrid(s, s, indexing="ij")
    W = np.outer(w, w)

    def gamma(nu):
        x = nu * (1 - alpha) ** 2 * (S * T) ** (-alpha)
        return float(((s0 * x - r(x.ravel()).reshape(x.shape)) * W).sum())

    target = -math.log2(1 - alpha) - alpha / L2
    lo, hi = 1e-4, 1e3
    while hi / lo > 1.001:
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if gamma(mid) < target else (lo, mid)
    return math.sqrt(lo * hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", default="results/exact_grid")
    ap.add_argument("--takeover", default="results/takeover_summaries")
    ap.add_argument("--master", type=Path,
                    default=Path("results/mechanism/master_curves.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/crossing_scaling.pdf"))
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})

    ex = load_exact(args.exact)
    tk = load_takeover(args.takeover)
    master = json.loads(args.master.read_text())

    # agreement between the two sources where both exist
    ratios = []
    for k in ex:
        if k in tk and ex[k][0] > 32:
            ratios.append(tk[k][0] / ex[k][0])
    print(f"overlap cells: {len(ratios)}, takeover/exact median "
          f"{np.median(ratios):.3f}, IQR "
          f"[{np.percentile(ratios,25):.3f}, {np.percentile(ratios,75):.3f}]")

    alphas = (0.5, 1.0)
    depths = (2, 3, 6, 12)
    dcol = {2: "#b0413e", 3: "#2b6a99", 6: "#2a8a84", 12: "#7a4f9e"}
    ds_all = sorted({k[0] for k in list(ex) + list(tk)})

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.26})
    for col, a in enumerate(alphas):
        ax, axk = axes[0, col], axes[1, col]
        for L in depths:
            c = dcol[L]
            dse = [d for d in ds_all if (d, a, L) in ex]
            de = [ex[(d, a, L)][0] for d in dse]
            dst = [d for d in ds_all if (d, a, L) in tk and d > 1024]
            dt = [tk[(d, a, L)][0] for d in dst]
            ax.plot(dse, de, "o", ms=4, color=c, label=f"$L={L}$")
            ax.plot(dst, dt, "o", ms=4, mfc="none", color=c)
            # effective exponent from the merged series
            dd = np.array(dse + dst, float)
            nn = np.array(de + dt, float)
            o = np.argsort(dd)
            dd, nn = dd[o], nn[o]
            if len(dd) > 1:
                ka = np.diff(np.log(nn)) / np.diff(np.log(dd))
                dmid = np.sqrt(dd[1:] * dd[:-1])
                # drop pairs touching the search floor (n* <= 16):
                # there the crossing is immediate and the slope is
                # meaningless
                keep = (nn[1:] > 16) & (nn[:-1] > 16)
                axk.plot(dmid[keep], ka[keep], "-o", ms=3, lw=1.0,
                         color=c)
        # d^2 slope guide through the alpha=.5 L=3 point at d=1024
        ref = ex.get((1024, a, 3), tk.get((1024, a, 3)))
        if ref:
            dg = np.array([64, 40000], float)
            ax.plot(dg, ref[0] * (dg / 1024) ** 2, ":", lw=1.0,
                    color="#555555", label="slope $2$")
        if a < 1:
            nb = nubar(a, 3, master)
            if nb:
                dg = np.array([64, 40000], float)
                ax.plot(dg, nb * dg ** 2, "--", lw=0.9, color="#999999",
                        label="$\\bar\\nu(\\alpha,3)\\, d^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$\\alpha = {a}$", fontsize=9)
        ax.set_ylabel("crossing $n^*$" if col == 0 else "")
        if col == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left",
                      ncol=2)
        axk.axhline(2.0, color="#999999", lw=0.7, ls=":")
        axk.set_xscale("log")
        axk.set_ylim(0.8, 2.4)
        axk.set_xlabel("$d$")
        axk.set_ylabel("local slope" if col == 0 else "")
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()

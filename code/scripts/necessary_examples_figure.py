"""Figure for the examples part of the appendix: the optimal curve
for the Zipf profile prior (plateau at log2 d!) and for the uniform
prior on the simplex. The dashed lines in the right panel are the
two-term expression (d-1)/2 log2 k + C(w), with the Clarke-Barron
constant C(w) evaluated in closed form for the uniform prior:
C(w) = -(d-1)/2 log2(2 pi e) - log2((d-1)!) + (d/2) H_{d-1} log2 e,
where H_{d-1} is the harmonic number. Computed values, not fits.

    python code/scripts/necessary_examples_figure.py \
        --in results/mechanism/necessary_examples.json \
        --out figures/necessary_examples.pdf
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


def c_uniform(d: int) -> float:
    """Clarke-Barron constant C(w) in bits for the uniform prior on
    the simplex over d symbols (s = d-1 dimensions)."""
    s = d - 1
    ln2 = math.log(2.0)
    h_w = -math.lgamma(d) / ln2
    harmonic = sum(1.0 / j for j in range(1, d))
    return (-(s / 2.0) * math.log2(2.0 * math.pi * math.e)
            + h_w + (d / 2.0) * harmonic / ln2)


def r_star_exact_bits(k: int, d: int) -> float:
    """Exact R*(k) = I(P; counts) for the uniform prior.
    The count vector is marginally uniform over the
    C(k+d-1, d-1) compositions, so
      R* = ln C(k+d-1,d-1) + ln k! - d E[ln M_1!] - k (H_d - 1)
    in nats, with Pr[M_1 = j] = C(k-j+d-2,d-2)/C(k+d-1,d-1)."""
    from scipy.special import gammaln
    ln2 = math.log(2.0)
    j = np.arange(k + 1, dtype=np.float64)
    lw = gammaln(k - j + d - 1) - gammaln(k - j + 1)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    eln = float(np.dot(w, gammaln(j + 1)))
    hd = float(np.sum(1.0 / np.arange(1, d + 1)))
    lnbin = math.lgamma(k + d) - math.lgamma(k + 1) - math.lgamma(d)
    return (lnbin + math.lgamma(k + 1) - d * eln
            - k * (hd - 1.0)) / ln2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    data = json.loads(args.inp.read_text())

    fig, (axp, axu) = plt.subplots(1, 2, figsize=(6.4, 2.6))

    pcol = {6: "#b0413e", 8: "#2b6a99"}
    for p in data["profile"]:
        d = p["d"]
        k = np.array(p["k"], float)
        m = np.array(p["rho_mean"])
        c = pcol.get(d, "#666666")
        axp.plot(k, m, "-", lw=1.4, color=c, label=f"$d={d}$")
        axp.axhline(p["plateau_log2_dfact"], color=c, lw=0.8, ls="--",
                    alpha=0.6)
    axp.set_xscale("log")
    axp.set_xlabel("$k$ (samples)")
    axp.set_ylabel("$R^*(k)$ [bits]")
    axp.set_title("Zipf profile prior: limit $\\log_2 d!$",
                  fontsize=8)
    axp.legend(frameon=False, fontsize=7, loc="lower right")

    ucol = {64: "#b0413e", 1024: "#2b6a99", 16384: "#3d8a3d"}
    for u in data["uniform"]:
        d = u["d"]
        k = np.array(u["k"], float)
        m = np.array(u["rho_mean"])
        c = ucol.get(d, "#666666")
        axu.plot(k, m, "-", lw=1.4, color=c, label=f"$d={d}$")
    axu.set_xscale("log")
    axu.set_yscale("log")
    ylo, yhi = axu.get_ylim()
    for u in data["uniform"]:
        d = u["d"]
        k = np.array(u["k"], float)
        c = ucol.get(d, "#666666")
        full = (d - 1) / 2 * np.log2(k) + c_uniform(d)
        pos = full > 0
        axu.plot(k[pos], full[pos], "--", lw=0.8, color=c,
                 alpha=0.55)
        exact = np.array([r_star_exact_bits(int(kk), d)
                          for kk in u["k"]])
        axu.plot(k, exact, ":", lw=0.9, color="black", alpha=0.8,
                 label="exact" if d == data["uniform"][0]["d"]
                 else None)
    axu.set_ylim(ylo, yhi)
    axu.set_xlabel("$k$ (samples)")
    axu.set_title("uniform prior: dashed "
                  "$\\frac{d-1}{2}\\log_2 k + C(w)$",
                  fontsize=8)
    axu.legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=160,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

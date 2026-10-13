"""Paper figures for the takeover-law discovery data.

Reads results/takeover_all.json (written by aggregate_takeover.py) and
produces vector PDFs in figures/ plus the LaTeX rows of the d-sweep
table on stdout.

    python code/scripts/paper_figures.py \
        --results results/takeover_all.json --out figures/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BLUE = "#345995"
C_RED = "#b0413e"
C_GREY = "#777777"


def law(d: float, c: float = 0.205) -> float:
    return c * d ** 1.5 * math.sqrt(math.log(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "mathtext.fontset": "cm"})

    def cell(d=None, L=None, alpha=None):
        out = [r for r in rows if
               (d is None or r["d"] == d) and
               (L is None or r["L"] == L) and
               (alpha is None or r["alpha"] == alpha)]
        return sorted(out, key=lambda r: (r["d"], r["L"], r["alpha"]))

    def value(r):
        """Crossing when defined, else median n* (open marker)."""
        if r["mean_lambda_crossing"]:
            return r["mean_lambda_crossing"], True
        return r["n_star_median"], False

    # ---------------- figure 1: the d-sweep and the law
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
    sweep = [r for r in cell(L=3, alpha=1.0)]
    ds = [r["d"] for r in sweep]
    vs = [value(r)[0] for r in sweep]
    for r in sweep:
        ax.vlines(r["d"], r["n_star_min"], r["n_star_max"],
                  color=C_BLUE, alpha=0.4, lw=1.2)
    ax.plot(ds, vs, "o", color=C_BLUE, ms=4.5, zorder=3,
            label="measured crossing")
    dd = [2 ** k for k in range(6, 16)]
    ax.plot(dd, [law(x) for x in dd], "--", color=C_GREY, lw=1,
            label=r"$0.205\,d^{3/2}\sqrt{\ln d}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"alphabet size $d$")
    ax.set_ylabel(r"takeover time $n^\star$")
    ax.legend(frameon=False, loc="upper left")

    ax2.plot(ds, [v / (d ** 1.5 * math.sqrt(math.log(d)))
                  for v, d in zip(vs, ds)],
             "o-", color=C_BLUE, ms=4.5, lw=1)
    ax2.axhline(0.205, color=C_GREY, ls="--", lw=1)
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0, 0.28)
    ax2.set_xlabel(r"alphabet size $d$")
    ax2.set_ylabel(r"$n^\star / \,(d^{3/2}\sqrt{\ln d}\,)$")
    for a in (ax, ax2):
        a.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.out / "takeover_d.pdf")
    print(f"wrote {args.out/'takeover_d.pdf'}")

    # ---------------- figure 2: depth and tilt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
    for d0, color in [(4096, C_RED), (16384, C_BLUE)]:
        pts = cell(d=d0, alpha=1.0)
        xs, ys = [], []
        for r in pts:
            v, is_cross = value(r)
            if r["n_censored"] >= r["n_seeds"]:
                ax.plot(r["L"], r["n_cap"], "^", color=color, ms=5)
                continue
            xs.append(r["L"]); ys.append(v)
            if not is_cross:
                ax.plot(r["L"], v, "o", mfc="none", mec=color, ms=4.5)
        ax.plot(xs, ys, "o-", color=color, ms=4.5, lw=1,
                label=rf"$d={d0}$")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"data depth $L_{\mathrm{s}}$")
    ax.set_ylabel(r"takeover time $n^\star$")
    ax.legend(frameon=False)

    for d0, color in [(4096, C_RED), (16384, C_BLUE)]:
        pts = cell(d=d0, L=3)
        xs, ys, opens = [], [], []
        for r in pts:
            v, is_cross = value(r)
            xs.append(r["alpha"]); ys.append(v); opens.append(not is_cross)
        ax2.plot(xs, ys, "-", color=color, lw=1)
        for x, y, o in zip(xs, ys, opens):
            ax2.plot(x, y, "o", mfc=("none" if o else color),
                     mec=color, ms=4.5)
        ax2.plot([], [], "o-", color=color, label=rf"$d={d0}$")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"tilt exponent $\alpha$")
    ax2.set_ylabel(r"takeover time $n^\star$")
    ax2.legend(frameon=False)
    for a in (ax, ax2):
        a.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.out / "takeover_L_alpha.pdf")
    print(f"wrote {args.out/'takeover_L_alpha.pdf'}")

    # ---------------- LaTeX table rows for the d-sweep
    print("\n% table rows (d-sweep, alpha=1, L=3):")
    for r in sweep:
        v, _ = value(r)
        c = v / (r["d"] ** 1.5 * math.sqrt(math.log(r["d"])))
        print(f"{r['d']} & {v:,.0f} & {r['n_star_median']:,.0f} & "
              f"[{r['n_star_min']:,} , {r['n_star_max']:,}] & "
              f"{c:.3f} \\\\".replace(",", "\\,"))


if __name__ == "__main__":
    main()

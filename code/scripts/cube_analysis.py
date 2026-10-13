"""Exponent analysis of the emergence-time cube.

For every (alpha, L) line of the cube, fit the d-exponent of the
emergence time and report the surface e(alpha, L).  The question the
cube was built to answer: does L bend the exponent, or only move the
constant?

Per cell, n* is the crossing of the seed-averaged Lambda when it
exists, else the median of per-seed emergence times.  Cells are
dropped from fits when censored, or when emergence is immediate
(n* < 64: the copier leads from the start, there is nothing to fit).
The tail exponent of a line is the least-squares slope of ln n* vs
ln d over the largest four usable d; the local exponent between the
two largest d is reported next to it as a finite-size indicator.

    python code/scripts/cube_analysis.py \
        --results results/takeover_all.json --out results/ \
        --figures figures/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MIN_NSTAR = 64.0


def n_star(row: dict):
    v = row.get("mean_lambda_crossing")
    est = "crossing"
    if v is None:
        v = row.get("n_star_median")
        est = "median"
    if v is None or v < MIN_NSTAR:
        return None, est
    if row.get("n_censored", 0) == row.get("n_seeds", 16):
        return None, est
    return float(v), est


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.results.read_text())
    lines: dict[tuple, list] = {}
    for r in rows:
        v, est = n_star(r)
        if v is not None:
            lines.setdefault((r["alpha"], r["L"]), []).append(
                (r["d"], v, est))

    # A line enters the fits if and only if it has at least five
    # usable d-values AND reaches n*(d_max) >= 10^4 -- below that
    # there is no dynamic range for a slope (emergence is nearly
    # immediate) and a fitted exponent is noise.  The criterion is
    # applied uniformly; nothing is picked by hand.
    MIN_POINTS, MIN_TOP = 5, 1e4
    surface = []
    for (a, L), pts in sorted(lines.items()):
        pts.sort()
        ds = np.array([p[0] for p in pts], float)
        ns = np.array([p[1] for p in pts], float)
        if len(ds) < 3:
            continue
        k = min(4, len(ds))
        fit, cov = np.polyfit(np.log(ds[-k:]), np.log(ns[-k:]), 1,
                              cov=True)
        tail = float(fit[0])
        err = float(math.sqrt(max(cov[0, 0], 0.0)))
        local = (math.log(ns[-1] / ns[-2]) / math.log(ds[-1] / ds[-2]))
        c = ns[-1] / ds[-1] ** tail
        surface.append({
            "alpha": a, "L": L, "n_d": int(len(ds)),
            "d_max": int(ds[-1]), "e_tail": round(tail, 3),
            "e_err": round(err, 3),
            "e_local_top": round(float(local), 3),
            "log10_C": round(float(math.log10(c)), 2),
            "included": bool(len(ds) >= MIN_POINTS and ns[-1] >= MIN_TOP),
            "estimators": sorted({p[2] for p in pts}),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cube_exponents.json").write_text(
        json.dumps(surface, indent=1) + "\n")

    # the verdict table: e_tail as a function of L, per alpha;
    # excluded lines (criterion above) shown in brackets
    alphas = sorted({s["alpha"] for s in surface})
    Ls = sorted({s["L"] for s in surface})
    by = {(s["alpha"], s["L"]): s for s in surface}
    print(f"e_tail(alpha, L); [brackets] = excluded "
          f"(fewer than {MIN_POINTS} points or top n* < {MIN_TOP:.0e})")
    print("alpha\\L " + " ".join(f"{L:>7}" for L in Ls))
    for a in alphas:
        row = []
        for L in Ls:
            c = by.get((a, L))
            if c is None:
                row.append("      -")
            elif c["included"]:
                row.append(f"{c['e_tail']:7.2f}")
            else:
                row.append(f"[{c['e_tail']:5.2f}]")
        vals = [by[(a, L)]["e_tail"] for L in Ls
                if (a, L) in by and by[(a, L)]["included"]]
        tail = (f"   incl: spread {max(vals)-min(vals):.3f} "
                f"(mean {np.mean(vals):.3f}, {len(vals)} lines)"
                if len(vals) >= 2 else "")
        print(f"{a:>6}  " + " ".join(row) + tail)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    inc = [s for s in surface if s["included"]]
    Ls_inc = sorted({s["L"] for s in inc})
    cmapL = plt.get_cmap("viridis")
    lcolor = {L: cmapL(i / max(len(Ls_inc) - 1, 1))
              for i, L in enumerate(Ls_inc)}
    cmapA = plt.get_cmap("plasma")
    acolor = {a: cmapA(0.85 * i / max(len(alphas) - 1, 1))
              for i, a in enumerate(alphas)}

    # (a) the exponent of every admitted line, with its fit error
    for L in Ls_inc:
        pts = sorted((s["alpha"], s["e_tail"], s["e_err"])
                     for s in inc if s["L"] == L)
        if pts:
            x, y, e = zip(*pts)
            axes[0].errorbar(x, y, yerr=e, fmt="o-", ms=2.5, lw=0.9,
                             elinewidth=0.7, capsize=1.5,
                             color=lcolor[L], label=f"$L={L}$")
    aa = np.linspace(0.5, 1.5, 200)
    axes[0].plot(aa, np.minimum(0.5 + 1 / aa, 1.75), "--", lw=1,
                 color="#888888",
                 label=r"$\min(\frac{1}{2}+\frac{1}{\alpha},\,1.75)$")
    axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel(r"$e(\alpha,L)$")
    axes[0].legend(frameon=False, fontsize=5.5, ncol=2)
    axes[0].set_title("exponent of every admitted line", fontsize=8)

    # (b) the claim itself: deviation from the per-alpha median, vs L
    for a in alphas:
        es = {s["L"]: s["e_tail"] for s in inc if s["alpha"] == a}
        if len(es) < 2:
            continue
        med = float(np.median(list(es.values())))
        Lx = sorted(es)
        axes[1].plot(Lx, [es[L] - med for L in Lx], "o-", ms=2.5,
                     lw=0.8, color=acolor[a], label=fr"$\alpha={a}$")
    axes[1].axhline(0.0, color="#888888", lw=0.7)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.4, 0.4)
    axes[1].set_xlabel("$L$")
    axes[1].set_ylabel(r"$e(\alpha,L)-\mathrm{med}_L\,e(\alpha,L)$")
    axes[1].set_title("does $L$ move the exponent?", fontsize=8)

    # (c) the constant of every admitted line
    for a in alphas:
        pts = sorted((s["L"], 10 ** s["log10_C"])
                     for s in inc if s["alpha"] == a)
        if len(pts) >= 2:
            axes[2].plot(*zip(*pts), "o-", ms=2.5, lw=0.8,
                         color=acolor[a])
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("$L$"); axes[2].set_ylabel(r"$C_{L,\alpha}$")
    axes[2].set_title("constant of every admitted line", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[2].legend(handles, labels, frameon=False, fontsize=5,
                   ncol=2, loc="lower left")

    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(wspace=0.42, bottom=0.18)
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures / "cube_surface.pdf")
    fig.savefig(args.figures / "cube_surface.png", dpi=160)
    print(f"wrote {args.figures / 'cube_surface.pdf'} and cube_exponents.json")


if __name__ == "__main__":
    main()

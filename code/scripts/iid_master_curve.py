"""The uniform (all-t) expression for the iid learning curve.

Starting point is an exact rearrangement of the expected KL of the
add-half estimator.  With K_a ~ Binomial(t, q_a) and
psi_t(q) := log2(t q) - E[log2(K + 1/2)],

    E[ D(q || hatq_t) ]
        = log2(1 + d/(2t)) + sum_a q_a psi_t(q_a)        (exact).

The first term is the cost of the pseudo-mass the estimator reserves
for letters regardless of the data.  Each summand depends on its
letter only through m = t q_a up to O(q_a) (the Poisson limit), so a
single one-letter master curve psi*(m) governs every alphabet size
and tilt: a letter contributes according to its own clock m = t q_a,
and the approximation log2(1 + d/(2t)) + sum_a q_a psi*(t q_a) is
accurate to O(max_a q_a) relative.  Limits: psi*(m) = log2(2m) + O(m)
for m << 1 (an unseen letter is overweighted); |psi*(m)| decays like
1/m^2 for m >> 1 -- beyond m ~ 30 a letter's own term is negligible,
and the classical (d-1)/(2 t ln 2) rate is carried by the
pseudo-mass term net of the aggregated O(q_a) corrections.

This script (1) tabulates psi* from small-q binomials, (2) verifies
the collapse psi_t(q) -> psi*(tq) across (d, alpha, t), and (3)
checks the uniform approximation
    log2(1 + d/(2t)) + sum_a q_a psi*(t q_a)
against the exact curves of iid_kl_curve.py over the whole t range.

    python code/scripts/iid_master_curve.py --out figures/iid_master.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

from mech_common import zipf


def e_log_half_scalar(t: int, p: float) -> float:
    m = t * p
    s = math.sqrt(max(t * p * (1 - p), 1e-30))
    lo = max(0, int(m - 12 * s) - 2)
    hi = min(t, int(m + 12 * s) + 25)
    k = np.arange(lo, hi + 1)
    w = binom.pmf(k, t, p)
    return float(w @ np.log2(k + 0.5))


def psi(t: int, p: float) -> float:
    return math.log2(t * p) - e_log_half_scalar(t, p)


def master_table(ms: np.ndarray, t: int = 10 ** 7) -> np.ndarray:
    """psi*(m): psi at tiny q = m/t (Poisson regime)."""

    return np.array([psi(t, m / t) for m in ms])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ms = 10 ** np.arange(-4, 6.01, 0.0625)
    tab = master_table(ms)
    lms = np.log(ms)

    def psi_star(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, float)
        out = np.interp(np.log(np.clip(m, ms[0], ms[-1])), lms, tab)
        lo = m < ms[0]          # analytic limit below the table
        out[lo] = np.log2(2 * m[lo])
        out[m > ms[-1]] = 0.0   # beyond the table psi* is < 1e-7
        return out

    # collapse check: psi_t(q) for scattered (t, q) against psi*(tq)
    rng = np.random.default_rng(0)
    worst = 0.0
    pts = []
    for t in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        for q in 10.0 ** rng.uniform(-6, -1.3, 12):
            if t * q < 1e-4 or t * q > 1e3:
                continue
            v = psi(t, q)
            m = t * q
            dev = abs(v - float(psi_star(np.array([m]))[0]))
            worst = max(worst, dev)
            pts.append((m, v, q))
    print(f"collapse: {len(pts)} scattered (t,q) points; worst absolute "
          f"deviation from the master curve {worst:.2e} bits")
    c0 = float(np.trapezoid(tab, ms))
    print(f"c0 = integral of psi* dm = {c0:.4f} bits (net lifetime "
          f"contribution of one letter to the cumulative redundancy)")

    # uniform approximation vs the exact curves (already computed and
    # stored by iid_kl_curve.py)
    import json
    stored = json.loads(Path("results/iid_kl.json").read_text())["mean"]
    fig_rows = []
    for r in stored:
        d, a = r["d"], r["alpha"]
        q = zipf(d, a)
        t = np.array(r["t"], float)
        exact = np.array(r["ekl_bits"])
        approx = np.array([math.log2(1 + d / (2 * tt))
                           + float(q @ psi_star(tt * q)) for tt in t])
        fig_rows.append((d, a, t, exact, approx))
        rel = np.max(np.abs(approx - exact)
                     / np.maximum(np.abs(exact), 1e-6))
        print(f"d={d:6d} alpha={a}: max |approx-exact|/|exact| "
              f"over t: {rel:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    keep = ms <= 1e3
    axes[0].plot(ms[keep], np.abs(tab)[keep], "-", color="#333333",
                 lw=1.2, label=r"$|\psi^*(m)|$ (master)")
    sc = [(p[0], abs(p[1])) for p in pts]
    axes[0].plot(*zip(*sorted(sc)), "o", ms=2.4, color="#b0413e",
                 alpha=0.6, label=r"$|\psi_t(q)|$, scattered $(t,q)$")
    mlo = ms[ms < 0.2]
    axes[0].plot(mlo, np.abs(np.log2(2 * mlo)), ":", color="#2b6a99",
                 lw=1.1, label=r"$|\log_2 2m|$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$m = t\,q_a$ (the letter's own clock)")
    axes[0].set_ylabel(r"$|\psi|$ [bits]")
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].set_title("one curve for every letter", fontsize=8)

    colors = {64: "#b0413e", 1024: "#2b6a99", 16384: "#3d8a3d"}
    styles = {0.5: ":", 1.0: "-", 1.5: "--"}
    for d, a, t, ex, apx in fig_rows:
        axes[1].plot(t, ex, styles[a], color=colors[d], lw=1.4, alpha=0.8)
        axes[1].plot(t, apx, styles[a], color="#000000", lw=0.6)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel(r"$\mathbb{E}\,D(q\,\|\,\hat q_t)$ [bits]")
    axes[1].set_title("exact (color) vs uniform formula (black)",
                      fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

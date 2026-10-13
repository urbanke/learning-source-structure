"""Exact expected KL of the add-1/2 estimator for an iid Zipf source.

The warm-up of the information-theoretic level: one unknown
distribution q on d letters, estimated after t iid samples by
hatq(a) = (K_a + 1/2) / (t + d/2), K_a ~ Binomial(t, q_a).  The
per-step excess of the expected code length over the entropy is

    E[D(q || hatq_t)] = sum_a q_a ( log2 q_a + log2(t + d/2)
                                    - E[log2(K_a + 1/2)] ),

a sum of one-dimensional binomial expectations -- computed exactly
here (pmf mass outside mean +- 12 sigma is discarded; it is far below
double precision).  No simulation.  The asymptote is the universal
(d-1) / (2 t ln 2) bits, independent of q; where the curve joins the
asymptote is where q lives (the discovery phase, set by the small
letter probabilities).  The cumulative panel R_n = sum_{t<n} E[D] is
obtained by integrating the exact per-step curve over the dense grid
(16 points per decade); only that integration step is numerical.

    python code/scripts/iid_kl_curve.py --out figures/iid_kl.pdf \
        --json results/iid_kl.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

from mech_common import zipf


def e_log_half(t: int, q: np.ndarray) -> np.ndarray:
    """E[log2(K + 1/2)] per letter, K ~ Binomial(t, q_a), exact."""

    out = np.empty(len(q))
    for i, p in enumerate(q):
        m = t * p
        s = math.sqrt(max(t * p * (1 - p), 1e-30))
        lo = max(0, int(m - 12 * s) - 2)
        hi = min(t, int(m + 12 * s) + 25)
        k = np.arange(lo, hi + 1)
        w = binom.pmf(k, t, p)
        out[i] = float(w @ np.log2(k + 0.5)) + (1.0 - w.sum()) * 0.0
        # discarded mass is < 1e-30; adding 0 for it is exact to double
    return out


def curve(d: int, alpha: float, ts: list[int]) -> dict:
    q = zipf(d, alpha)
    per_step = []
    for t in ts:
        e = float(np.sum(q * (np.log2(q) + math.log2(t + d / 2)))
                  - np.sum(q * e_log_half(t, q)))
        per_step.append(e)
    return {"d": d, "alpha": alpha, "t": ts, "ekl_bits": per_step}


def mc_check(d: int, alpha: float, t: int, reps: int = 4000) -> float:
    q = zipf(d, alpha)
    rng = np.random.default_rng(0)
    tot = 0.0
    for _ in range(reps):
        k = rng.multinomial(t, q)
        hat = (k + 0.5) / (t + d / 2)
        tot += float(np.sum(q * np.log2(q / hat)))
    return tot / reps


def sd_curve(d: int, alpha: float, ts: list[int], reps: int = 1000) -> dict:
    """Sample standard deviation of D(q || hatq_t) across samples.

    Each draw is a multinomial count vector; D is evaluated exactly on
    it, so only the spread estimate is statistical (its own standard
    error is about sd / sqrt(2 reps), 2 percent at reps = 1000)."""

    q = zipf(d, alpha)
    rng = np.random.default_rng(1)
    lq = np.log2(q)
    means, sds = [], []
    for t in ts:
        ks = rng.multinomial(t, q, size=reps)
        dd = ((q * (lq + math.log2(t + d / 2)))[None, :]
              - q[None, :] * np.log2(ks + 0.5)).sum(axis=1)
        means.append(float(dd.mean()))
        sds.append(float(dd.std(ddof=1)))
    return {"d": d, "alpha": alpha, "t": ts, "mean_mc": means, "sd": sds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    ts = sorted({int(round(10 ** e)) for e in np.arange(0, 6.01, 0.125)})
    ts_sd = sorted({int(round(10 ** e)) for e in np.arange(0, 6.01, 0.25)})
    ds = [64, 1024, 16384]
    alphas = [0.5, 1.0, 1.5]
    rows = [curve(d, a, ts) for d in ds for a in alphas]
    sds = [sd_curve(d, a, ts_sd) for d in ds for a in alphas]

    if args.check:
        ex = curve(64, 1.0, [1000])["ekl_bits"][0]
        mc = mc_check(64, 1.0, 1000)
        print(f"check d=64 a=1 t=1000: exact {ex:.6f}  MC {mc:.6f}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"mean": rows, "sd": sds}) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    colors = {64: "#b0413e", 1024: "#2b6a99", 16384: "#3d8a3d"}
    styles = {0.5: ":", 1.0: "-", 1.5: "--"}

    for r in rows:
        t = np.array(r["t"], float)
        e = np.array(r["ekl_bits"])
        axes[0].plot(t, e, styles[r["alpha"]], color=colors[r["d"]],
                     lw=1.3)
        axes[1].plot(t, np.cumsum(e * np.gradient(t)) /
                     ((r["d"] - 1) / 2 * np.log2(t + 1)),
                     styles[r["alpha"]], color=colors[r["d"]], lw=1.3)
    for r in sds:
        t = np.array(r["t"], float)
        rel = np.array(r["sd"]) / np.maximum(np.array(r["mean_mc"]), 1e-300)
        axes[2].plot(t, rel, styles[r["alpha"]], color=colors[r["d"]],
                     lw=1.3)
    for d in ds:
        t = np.array(ts, float)
        axes[0].plot(t, (d - 1) / (2 * math.log(2) * t), color=colors[d],
                     lw=0.7, alpha=0.45)
        axes[2].axhline(math.sqrt(2.0 / (d - 1)), color=colors[d],
                        lw=0.7, alpha=0.45)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("$t$ (samples)")
    axes[0].set_ylabel(r"$\mathbb{E}\,D(q\,\|\,\hat q_t)$ [bits]")
    axes[0].set_title("per step; thin: $(d-1)/(2t\\ln 2)$", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$n$ (samples)")
    axes[1].set_ylabel(r"$R_n \,/\, \frac{d-1}{2}\log_2 n$")
    axes[1].axhline(1.0, color="#999999", lw=0.7)
    axes[1].set_title("cumulative, against the half-log law",
                      fontsize=8)
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("$t$ (samples)")
    axes[2].set_ylabel(r"sd$\,/\,$mean of $D(q\,\|\,\hat q_t)$")
    axes[2].set_title(r"spread; thin: $\sqrt{2/(d-1)}$", fontsize=8)
    for d in ds:
        axes[0].plot([], [], "-", color=colors[d], label=f"$d={d}$")
    for a in alphas:
        axes[0].plot([], [], styles[a], color="#555555",
                     label=fr"$\alpha={a}$")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""The mechanism overlay test (Lambda decomposition against reality).

Claim under test: the copier's measured total splits, context by
context, into ideal part plus the universal maturation curve,

    l_C(t)  ~  log2 d + sum_a [ k_a(t) H(row_a) + rho(k_a(t)) ],

with k_a(t) the exact visit counts of the replayed chain, H(row_a) the
exact entropy of the seed's actual rule row, and rho the measured
mean maturation curve of mechanism_quantities.py (no free parameter).
From this and the measured l_F the predicted Lambda follows.

    python code/scripts/overlay_test.py \
        --seeds-dir <dir with d..._seedN.json> --rho <rho json> \
        --seeds 0-7 --out figures/overlay_d1024.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from takeover_experiment import parse_seeds


def rho_interp(rho_file: str):
    r = json.loads(Path(rho_file).read_text())
    ks = np.asarray(r["k"], dtype=float)
    vals = np.asarray(r["rho_mean"], dtype=float)
    lk, lv = np.log(ks), vals
    tail_slope = (lv[-1] - lv[-2]) / (lk[-1] - lk[-2])

    def rho(k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = np.interp(np.log(np.maximum(k, 1.0)), lk, lv)
        big = k > ks[-1]
        if np.any(big):
            out[big] = lv[-1] + tail_slope * (np.log(k[big]) - lk[-1])
        out[k < 1] = 0.0
        return out

    return rho, r


def entropy_bits(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", type=Path, required=True)
    ap.add_argument("--rho", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0-7")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import mech_common as mc

    rho, rmeta = rho_interp(args.rho)
    d, alpha, L = rmeta["d"], rmeta["alpha"], rmeta["L"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    rel_errs = []
    for s in parse_seeds(args.seeds):
        f = next(args.seeds_dir.glob(f"*_seed{s}.json"))
        rec = json.loads(f.read_text())
        assert rec["d"] == d and rec["L"] == L
        cps = rec["checkpoints"]
        horizon = max(cps)

        rng = np.random.default_rng(
            [s, d, L, int(round(alpha * 100)), 7])
        chain = mc.ChainSampler(d, alpha, L, rng)
        x = chain.extend_to(horizon)
        H_row = {a: entropy_bits(np.diff(np.concatenate([[0.0], cum])))
                 for a, cum in chain.rows.items()}

        pred_lc, meas_lc = [], []
        for t in cps:
            k = np.bincount(x[: t - 1], minlength=d)
            act = np.nonzero(k)[0]
            ideal = sum(k[a] * H_row[a] for a in act)
            debt = float(np.sum(rho(k[act])))
            pred_lc.append(math.log2(d) + ideal + debt)
        meas_lc = rec["l_C"]
        pred_lam = [lF - p for lF, p in zip(rec["l_F"], pred_lc)]
        rel = max(abs(p - m) / max(m, 1.0)
                  for p, m in zip(pred_lc, meas_lc))
        rel_errs.append(rel)

        kw = dict(lw=1, alpha=0.75)
        ax.plot(cps, rec["lam"], color="#345995", **kw)
        ax.plot(cps, pred_lam, "--", color="#b0413e", **kw)
        ax2.plot(cps, [m - p for m, p in zip(meas_lc, pred_lc)],
                 color="#777777", **kw)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Lambda(t)$ [bits]")
    ax.plot([], [], color="#345995", label="measured")
    ax.plot([], [], "--", color="#b0413e", label="mechanism prediction")
    ax.legend(frameon=False, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$\ell_C$: measured $-$ predicted [bits]")
    for a in (ax, ax2):
        a.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"overlay: {len(rel_errs)} seeds, worst relative error of "
          f"predicted l_C: {max(rel_errs):.3%} -> {args.out}")


if __name__ == "__main__":
    main()

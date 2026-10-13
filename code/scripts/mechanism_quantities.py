"""Measure the three mechanism quantities of the takeover decomposition.

    Lambda(t) = t*I - R_C(t) + R_F(t),      R_C(t) = sum_a rho(t*pi_a)

  I       mutual information between consecutive tokens (reward rate)
  rho(k)  expected extra bits for one unknown row after k iid samples,
          averaged over rows and samples. For a full-dimensional
          continuous prior this grows logarithmically at fixed d;
          the recorded D_estimate is a final value, not a plateau.
  pi      the visit law (= the tilt q, to first order)

Subcommands:

  rho   --d --alpha --L --rows M --k-max K --out FILE
        Draw M tilted rows, stream iid samples from each, and record
        block regret on a geometric k-grid, using the realized source
        log probability as the baseline. The code formula is evaluated
        conditional on each draw; sample standard errors cover both the
        row and sample randomness.

  info  --d --alpha --L --rows M
        Estimate I = H(q) - E_{a~q}[H(row_a)] by sampling rows.  At
        --exact-stationary (d <= 2048) also build the full transition
        matrix, compute its stationary distribution, and report how
        good the pi = q approximation is.

Evaluation goes through the anchor system (same configuration as
takeover_experiment; --store required).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from takeover_experiment import engage_anchor_mode


def entropy_bits(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def sample_row(d, L, rng, q):
    from lsa.product_simplex import sample_product_simplex_distribution

    if L == 0:
        return q.copy()
    v = sample_product_simplex_distribution(d, L, rng)
    v *= q
    return v / v.sum()


def cmd_rho(args) -> None:
    if args.rows < 2:
        raise ValueError("at least two independent rows are needed for a sample standard error")
    import mech_common as mc
    from lsa.codelength import (default_l_max, profile_of,
                                depth_averaged_codelength_profiles)

    d, L = args.d, args.L
    l_max = default_l_max(d)
    paired_available = 1 <= L <= l_max
    q = mc.zipf(d, args.alpha)
    ks = mc.geometric_grid(1, args.k_max, ratio=1.3)
    rows_regret, rows_fixed, rows_excess = [], [], []
    t0 = time.time()
    for m in range(args.rows):
        rng = np.random.default_rng([m, d, L, int(args.alpha * 100), 23])
        row = sample_row(d, L, rng, q)
        lrow = np.log2(row)
        x = rng.choice(d, size=args.k_max, p=row)
        profs, counts = {}, np.zeros(d, dtype=np.int64)
        nll = {}
        pos = 0
        for k in ks:
            counts += np.bincount(x[pos:k], minlength=d)
            pos = k
            profs[k] = profile_of(counts.tolist())
            # realized -log2 p_row(x^k), not k*H(row): same expectation
            # (the code is exchangeable, its length depends on counts
            # only), but the difference then has the variance of the
            # redundancy itself instead of a k-growing sampling term.
            # With k*H the noise reached +-1100 bits per row at k=1e6,
            # d=64, drowning a plateau of a few hundred bits and even
            # driving the mean negative.
            nll[k] = -float(counts @ lrow)
        ev = depth_averaged_codelength_profiles(
            {p: p for p in set(profs.values())}, d=d, l_max=l_max,
            jobs=args.jobs, tables_source="universal",
            universal_path=args.store)
        rows_regret.append(
            [(-ev[profs[k]].log2_q_avg) - nll[k] for k in ks])
        if paired_available:
            rows_fixed.append(
                [-ev[profs[k]].log2_q_by_depth[L - 1] - nll[k] for k in ks])
            rows_excess.append(
                [ev[profs[k]].log2_q_by_depth[L - 1] - ev[profs[k]].log2_q_avg
                 for k in ks])
        if (m + 1) % 8 == 0:
            print(f"  {m+1}/{args.rows} rows ({time.time()-t0:.0f}s)",
                  flush=True)
    R = np.asarray(rows_regret)          # rows x len(ks)
    mean, se = R.mean(0), R.std(0, ddof=1) / math.sqrt(len(R))
    fixed, excess = np.asarray(rows_fixed), np.asarray(rows_excess)
    out = {
        "d": d, "alpha": args.alpha, "L": L, "l_max": l_max,
        "rows": args.rows, "k": ks,
        "rho_mean": mean.tolist(), "rho_se": se.tolist(),
        "se_ddof": 1,
        "seed_protocol": "numpy.default_rng([row_index,d,L,int(alpha*100),23])",
        "D_estimate": float(mean[-1]),
        "note": "rho(k) estimates the depth-mixture redundancy relative to the realized source log probability. The fixed-depth estimate and paired depth-mixture excess use the same source and samples. D_estimate is the final recorded value, not a proved plateau.",
    }
    if paired_available:
        out.update({
            "fixed_depth_rho_mean": fixed.mean(0).tolist(),
            "fixed_depth_rho_se": (fixed.std(0, ddof=1) / math.sqrt(len(R))).tolist(),
            "paired_depth_excess_mean": excess.mean(0).tolist(),
            "paired_depth_excess_se": (excess.std(0, ddof=1) / math.sqrt(len(R))).tolist(),
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    tail = " ".join(f"{k}:{v:.1f}" for k, v in
                    list(zip(ks, mean))[-4:])
    print(f"d={d} L={L} alpha={args.alpha}: D ~ {mean[-1]:.1f} bits "
          f"(last points {tail}) -> {args.out}")


def cmd_info(args) -> None:
    import mech_common as mc

    d, L = args.d, args.L
    q = mc.zipf(d, args.alpha)
    hq = entropy_bits(q)
    # E_{a~q} H(row): rows are iid, so sample rows and weight by q via
    # sampling a ~ q implicitly: rows are exchangeable across a, hence
    # E_{a~q} H(row_a) = E[H(row)] (row law does not depend on a)
    rng = np.random.default_rng([d, L, int(args.alpha * 100), 29])
    hs = [entropy_bits(sample_row(d, L, rng, q)) for _ in range(args.rows)]
    hrow, se = float(np.mean(hs)), float(np.std(hs, ddof=1) / math.sqrt(len(hs)))
    print(f"d={d} L={L} alpha={args.alpha}: H(q) = {hq:.3f} bits, "
          f"E[H(row)] = {hrow:.3f} +- {se:.3f}, I ~ {hq - hrow:.3f} bits")

    if args.exact_stationary:
        assert d <= 2048, "full matrix only for small d"
        rngT = np.random.default_rng([7, d, L, int(args.alpha * 100), 31])
        T = np.stack([sample_row(d, L, rngT, q) for _ in range(d)])
        pi = np.full(d, 1.0 / d)
        for _ in range(2000):
            pi = pi @ T
        i_exact = entropy_bits(pi) - float(pi @ [entropy_bits(T[a])
                                                 for a in range(d)])
        tv = 0.5 * float(np.abs(pi - q).sum())
        print(f"  exact stationary: I = {i_exact:.3f} bits, "
              f"TV(pi, q) = {tv:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("rho")
    pr.add_argument("--d", type=int, required=True)
    pr.add_argument("--alpha", type=float, default=1.0)
    pr.add_argument("--L", type=int, default=3)
    pr.add_argument("--rows", type=int, default=32)
    pr.add_argument("--k-max", type=int, default=4096)
    pr.add_argument("--jobs", type=int, default=1)
    pr.add_argument("--store", type=str, required=True)
    pr.add_argument("--out", type=str, required=True)
    pi = sub.add_parser("info")
    pi.add_argument("--d", type=int, required=True)
    pi.add_argument("--alpha", type=float, default=1.0)
    pi.add_argument("--L", type=int, default=3)
    pi.add_argument("--rows", type=int, default=256)
    pi.add_argument("--exact-stationary", action="store_true")
    pi.add_argument("--store", type=str, default=None)
    args = ap.parse_args()

    if args.cmd == "rho":
        engage_anchor_mode(args.store)
        cmd_rho(args)
    else:
        if args.store:
            engage_anchor_mode(args.store)
        cmd_info(args)


if __name__ == "__main__":
    main()

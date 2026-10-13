"""Semi-analytic takeover time from the mechanism, no chains involved.

Solves  t * I  =  log2 d + sum_a rho(t * pi_a)  for t, using the
MEASURED maturation curve rho (mechanism_quantities.py) and the tilt q
as the visit law (mean field; the frequency strategy's own small
learning term is neglected here).  If the crossings so obtained
reproduce the measured exponents in d per alpha, the mechanism
accounts for the law and what remains analytic is rho itself.

    python code/scripts/semi_analytic_crossing.py \
        --rho-dir /root/work/mech_q --I "0.75:1.7,1.0:1.65,1.25:1.48"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def rho_fn(path: Path):
    r = json.loads(path.read_text())
    ks = np.asarray(r["k"], float)
    vals = np.asarray(r["rho_mean"], float)
    lk = np.log(ks)
    tail = (vals[-1] - vals[-2]) / (lk[-1] - lk[-2])

    def rho(k):
        k = np.asarray(k, float)
        out = np.interp(np.log(np.maximum(k, 1.0)), lk, vals)
        big = k > ks[-1]
        out = np.where(big, vals[-1] + tail * (np.log(k) - lk[-1]), out)
        return np.where(k < 0.5, 0.0, out)

    return rho, r["d"], r["alpha"], float(ks[-1])


def solve(d, alpha, rho, I):
    import mech_common as mc

    q = mc.zipf(d, alpha)
    lo, hi = 8.0, 1e12
    def gap(t):
        return t * I - (math.log2(d) + float(np.sum(rho(t * q))))
    if gap(hi) < 0:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-dir", type=Path, required=True)
    ap.add_argument("--I", type=str, required=True,
                    help="alpha:I pairs, e.g. 0.75:1.7,1.0:1.65")
    args = ap.parse_args()
    I_of = {float(a): float(v) for a, v in
            (kv.split(":") for kv in args.I.split(","))}

    rows = []
    for f in sorted(args.rho_dir.glob("rho_d*_L3.json")):
        rho, d, alpha, kmax = rho_fn(f)
        if alpha not in I_of:
            continue
        t = solve(d, alpha, rho, I_of[alpha])
        rows.append((alpha, d, t, kmax))
        top_k = t * mc_q_top(d, alpha) if t else 0
    import mech_common as mc
    print(f"{'alpha':>6} {'d':>6} {'t* (semi-analytic)':>20} "
          f"{'top-context k':>14} {'measured-rho range':>18}")
    for alpha, d, t, kmax in sorted(rows):
        q1 = mc.zipf(d, alpha)[0]
        note = "extrapolated" if t and t * q1 > kmax else "measured"
        print(f"{alpha:>6} {d:>6} {t:>20,.0f} {t*q1:>14,.0f} {note:>18}")
    print("\nexponents between consecutive d (per alpha):")
    for alpha in sorted({a for a, *_ in rows}):
        pts = sorted((d, t) for a, d, t, _ in rows if a == alpha and t)
        for (d1, t1), (d2, t2) in zip(pts, pts[1:]):
            print(f"  alpha={alpha}: {d1}->{d2}: "
                  f"{math.log(t2/t1)/math.log(d2/d1):.3f}")


def mc_q_top(d, alpha):
    import mech_common as mc

    return mc.zipf(d, alpha)[0]


if __name__ == "__main__":
    main()

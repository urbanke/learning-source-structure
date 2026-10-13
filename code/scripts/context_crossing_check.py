"""Anatomy of the measured transition-code crossing, depth 1.

On each sampled chain (rules T(b|a) = q_b E_ab / Z_a, q Zipf(alpha),
started from its numerically solved stationary law pi) four transition
code lengths are evaluated from explicit formulas on the same sequence.
The initial token is omitted. The tilted row product is a reference
code, not the full stationary-start Bayes mixture:

  true      : the code that knows the rules
  C reference: product of tilted row mixtures, context by context
              (tilted prior, one-dimensional integral)
  C deployed: symmetric prior per context (Dirichlet(1), closed form)
  F deployed: symmetric prior on one distribution over the targets

From these: the deployed difference  ell_F - ell_C,  the premium
P(n) = [ell_C - ell_C*] - [ell_F - n H(pi)],  and the theory gap
G(n) = n I - sum_ab r_1(n pi_a q_b)  with the exact per-seed pi and
exact per-seed I.  The crossing statement says the zero of the
deployed difference sits where G meets P.

    python code/scripts/context_crossing_check.py \
        --master results/mechanism/master_curves.json \
        --out results/mechanism/context_crossing_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_row_check import log_mixture, r1_interp, kgrid  # noqa: E402
from context_chain_check import stationary  # noqa: E402

L2 = math.log(2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=400_000)
    args = ap.parse_args()
    if args.seeds < 2:
        raise ValueError("at least two repetitions are needed for a sample standard error")

    r1 = r1_interp(json.loads(args.master.read_text()))
    d = args.d
    ns = kgrid(args.nmax, per_decade=8)
    ns = ns[ns >= 32]
    u_coarse = np.arange(-16.0, 27.0, 0.004)
    lgd = math.lgamma(d)

    out = {"d": d, "L": 1, "n": ns.tolist(), "seeds": args.seeds,
           "se_ddof": 1, "sample_count_unit": "transitions",
           "initialization": "sampled from the numerically solved stationary law of each transition table",
           "code_convention": "all lengths omit the initial token; RC_opt is the legacy name for a tilted row-product reference, not a stationary-start optimum",
           "alphas": []}
    for alpha in (0.5, 1.0):
        q = 1.0 / np.arange(1, d + 1) ** alpha
        q /= q.sum()
        s0_coarse = np.log1p(q[:, None] * np.exp(u_coarse)[None, :]).sum(0)
        D1 = float(np.sum(np.log2(1.0 / (d * q))))
        curves = {k: np.zeros((args.seeds, len(ns)))
                  for k in ("diff", "P", "PC", "G", "RC_opt", "RC_dep",
                            "RF_dep")}
        Is = []
        residuals = []
        t0 = time.time()
        for s in range(args.seeds):
            rng = np.random.default_rng([s, d, int(round(alpha * 100)), 47])
            T = q[None, :] * rng.exponential(size=(d, d))
            T /= T.sum(1, keepdims=True)
            pi = stationary(T)
            residuals.append(float(np.sum(np.abs(pi@T-pi))))
            Hrow = -(T * np.log2(T)).sum(1)
            Hpi = float(-(pi * np.log2(pi)).sum())
            I = Hpi - float((pi * Hrow).sum())
            Is.append(I)
            lam = np.outer(pi, q).ravel()

            Tcum = np.cumsum(T, axis=1)
            Tcum[:, -1] = 1.0
            logT = np.log(T)
            x = int(rng.choice(d, p=pi))
            prev_t = np.empty(args.nmax, dtype=np.int32)
            next_t = np.empty(args.nmax, dtype=np.int32)
            us = rng.random(args.nmax)
            for t in range(args.nmax):
                y = int(np.searchsorted(Tcum[x], us[t], side="right"))
                prev_t[t] = x
                next_t[t] = y
                x = y
            cum_true = np.concatenate(
                [[0.0], np.cumsum(logT[prev_t, next_t])])

            counts = np.zeros((d, d))
            prev = 0
            for j, n in enumerate(ns):
                np.add.at(counts, (prev_t[prev:n], next_t[prev:n]), 1.0)
                prev = n
                lm_tilt = 0.0
                lm_sym = 0.0
                for a in np.nonzero(counts.sum(1))[0]:
                    row = counts[a]
                    idx = np.nonzero(row)[0]
                    k = int(row.sum())
                    lm_tilt += log_mixture(idx, row[idx], k, q,
                                           u_coarse, s0_coarse)
                    lm_sym += (float(np.sum(gammaln(row[idx] + 1.0)))
                               + lgd - math.lgamma(k + d))
                tot = counts.sum(0)
                ti = np.nonzero(tot)[0]
                lm_F = (float(np.sum(gammaln(tot[ti] + 1.0)))
                        + lgd - math.lgamma(n + d))
                RC_opt = (cum_true[n] - lm_tilt) / L2
                RC_dep = (cum_true[n] - lm_sym) / L2
                RF_dep = (cum_true[n] - lm_F) / L2
                curves["RC_opt"][s, j] = RC_opt
                curves["RC_dep"][s, j] = RC_dep
                curves["RF_dep"][s, j] = RF_dep
                curves["diff"][s, j] = RF_dep - RC_dep
                curves["PC"][s, j] = RC_dep - RC_opt
                curves["P"][s, j] = (RC_dep - RC_opt) - (RF_dep - n * I)
                curves["G"][s, j] = n * I - float(np.sum(r1(n * lam)))
        rec = {"alpha": alpha, "D1_bits_per_context": D1,
               "maximum_stationary_residual_l1": max(residuals),
               "I_exact_mean": float(np.mean(Is)),
               "I_lead": (1 - 0.5772156649015329) / L2}
        for k, v in curves.items():
            rec[k + "_mean"] = v.mean(0).tolist()
            rec[k + "_se"] = (v.std(0, ddof=1) / math.sqrt(args.seeds)).tolist()
        balance=curves['G']-curves['P']
        residual=curves['diff']-balance
        rec['balance_se']=(balance.std(0,ddof=1)/math.sqrt(args.seeds)).tolist()
        rec['paired_balance_residual_mean']=residual.mean(0).tolist()
        rec['paired_balance_residual_se']=(residual.std(0,ddof=1)/math.sqrt(args.seeds)).tolist()
        out["alphas"].append(rec)
        dm = curves["diff"].mean(0)
        gz = np.array(rec["G_mean"]) - np.array(rec["P_mean"])
        def zero_of(y):
            # last sign change: the final crossing
            idx = np.nonzero(np.diff(np.sign(y)) != 0)[0]
            if len(idx) == 0:
                return float("nan")
            i = int(idx[-1])
            f = y[i] / (y[i] - y[i + 1])
            return float(np.exp(np.log(ns[i])
                                + f * math.log(ns[i + 1] / ns[i])))
        print(f"alpha={alpha}: last zero of diff at n≈{zero_of(dm):.0f}, "
              f"G=P at n≈{zero_of(gz):.0f}, I={np.mean(Is):.4f} "
              f"(lead {rec['I_lead']:.4f}), dD1={d*D1:.0f} bits, "
              f"{time.time()-t0:.0f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

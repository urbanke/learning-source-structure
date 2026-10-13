"""Check of the per-context curve on the chain itself, depth 1.

The row check (context_row_check.py) feeds one rule a fixed number
of iid samples.  Here the samples come from the chain: rules
T(b|a) = q_b E_ab / Z_a with E iid Exp(1), q Zipf(alpha), the chain
started from its numerically solved stationary law pi and run for n
transitions, with random visit counts and feedback included. The code
uses the product of unconditioned row-mixture factors as a transition
reference; under a stationary start this is not the full Bayes mixture.
The initial token is omitted from both code lengths. Each depth-one
factor uses the one-dimensional row integral, evaluated numerically.
The prediction, with
nothing fitted, is the double sum  sum_ab r_1(n pi_a q_b)  with the
exact pi of each repetition.

    python code/scripts/context_chain_check.py \
        --master results/mechanism/master_curves.json \
        --out results/mechanism/context_chain_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_row_check import log_mixture, r1_interp, kgrid  # noqa: E402

L2 = math.log(2.0)


def stationary(T: np.ndarray) -> np.ndarray:
    pi = np.full(T.shape[0], 1.0 / T.shape[0])
    for _ in range(4000):
        pin = pi @ T
        if np.max(np.abs(pin - pi)) < 1e-14:
            return pin/pin.sum()
        pi = pin
    raise RuntimeError("stationary iteration did not converge within 4000 steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=200_000)
    args = ap.parse_args()
    if args.seeds < 2:
        raise ValueError("at least two repetitions are needed for a sample standard error")

    r1 = r1_interp(json.loads(args.master.read_text()))
    d = args.d
    ns = kgrid(args.nmax, per_decade=6)
    ns = ns[ns >= 16]
    u_coarse = np.arange(-16.0, 27.0, 0.004)

    out = {"d": d, "L": 1, "n": ns.tolist(), "seeds": args.seeds,
           "se_ddof": 1, "sample_count_unit": "transitions",
           "initialization": "sampled from the numerically solved stationary law of each transition table",
           "code_convention": "transition-only row-product reference; initial token omitted; not the stationary-start Bayes mixture",
           "alphas": []}
    for alpha in (0.5, 1.0, 1.5):
        q = 1.0 / np.arange(1, d + 1) ** alpha
        q /= q.sum()
        s0_coarse = np.log1p(q[:, None] * np.exp(u_coarse)[None, :]).sum(0)
        red = np.zeros((args.seeds, len(ns)))
        pred_pi = np.zeros((args.seeds, len(ns)))
        residuals = []
        t0 = time.time()
        for s in range(args.seeds):
            rng = np.random.default_rng([s, d, int(round(alpha * 100)), 31])
            T = q[None, :] * rng.exponential(size=(d, d))
            T /= T.sum(1, keepdims=True)
            pi = stationary(T)
            residuals.append(float(np.sum(np.abs(pi@T-pi))))
            lam = np.outer(pi, q).ravel()
            pred_pi[s] = [float(np.sum(r1(n * lam))) for n in ns]

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
                tot_lm = 0.0
                for a in np.nonzero(counts.sum(1))[0]:
                    row = counts[a]
                    idx = np.nonzero(row)[0]
                    tot_lm += log_mixture(idx, row[idx],
                                          int(row.sum()), q,
                                          u_coarse, s0_coarse)
                red[s, j] = (cum_true[n] - tot_lm) / L2
        rec = {"alpha": alpha,
               "maximum_stationary_residual_l1": max(residuals),
               "red_mean": red.mean(0).tolist(),
               "red_se": (red.std(0, ddof=1) / math.sqrt(args.seeds)).tolist(),
               "pred_pi_se": (pred_pi.std(0, ddof=1) / math.sqrt(args.seeds)).tolist(),
               "paired_difference_mean": (red-pred_pi).mean(0).tolist(),
               "paired_difference_se": ((red-pred_pi).std(0, ddof=1)/math.sqrt(args.seeds)).tolist(),
               "pred_pi_mean": pred_pi.mean(0).tolist(),
               "pred_q": [float(np.sum(r1(n * np.outer(q, q).ravel())))
                          for n in ns]}
        out["alphas"].append(rec)
        m = red.mean(0)
        se = red.std(0, ddof=1) / math.sqrt(args.seeds)
        p = pred_pi.mean(0)
        z = np.abs(m - p) / np.maximum(2 * se, 1e-9)
        print(f"alpha={alpha}: max |diff|/(2se) = {z.max():.2f}, final "
              f"{m[-1]:.0f} vs {p[-1]:.0f} ({100*(m[-1]/p[-1]-1):+.1f}%), "
              f"{time.time()-t0:.0f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

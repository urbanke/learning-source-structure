"""One-step information of the tilted layered chain.

I = H(pi) - sum_a pi_a H(T(.|a)), with pi solved numerically for
a sampled transition table. The comparison L(1-gamma) log2 e is
motivated by diffuse frequencies. The recorded variance term concerns
row denominators alone; it is not an error bound for the stationary gain.

    python code/scripts/one_step_info.py \
        --out results/mechanism/one_step_info.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from context_chain_check import stationary

EG = 0.5772156649015329


def measure(d: int, alpha: float, L: int, seeds: int) -> dict:
    b = np.arange(1, d + 1, dtype=float)
    q = b ** (-alpha)
    q /= q.sum()
    s2 = float((q * q).sum())
    vals = []
    for seed in range(seeds):
        rng = np.random.default_rng([seed, d, L, int(round(alpha * 100)), 11])
        G = rng.exponential(size=(d, L, d)).prod(1)
        T = q[None, :] * G
        T /= T.sum(1, keepdims=True)
        pi = stationary(T)
        Hrow = -(T * np.log2(T)).sum(1)
        vals.append(float(-(pi * np.log2(pi)).sum() - (pi * Hrow).sum()))
    return {"alpha": alpha,
            "I_mean": float(np.mean(vals)),
            "I_sd": float(np.std(vals,ddof=1)),
            "I_se": float(np.std(vals,ddof=1)/math.sqrt(seeds)),
            "pred_bits": L * (1 - EG) / math.log(2),
            "var_term": (2 ** L - 1) * s2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.seeds<2:raise ValueError("at least two tables are needed for a sample standard error")
    out = {"d": args.d, "L": args.L, "seeds": args.seeds,
           "se_ddof":1,"sd_ddof":1,
           "convention":"stationary gain of independently sampled transition tables; stationary distributions solved numerically; I_sd is sample standard deviation, I_se is standard error of the mean",
           "alphas": [measure(args.d, a, args.L, args.seeds)
                      for a in (0.5, 1.0, 1.5)]}
    args.out.write_text(json.dumps(out, indent=1))
    for c in out["alphas"]:
        print(f"alpha={c['alpha']}: I = {c['I_mean']:.4f} "
              f"+- {c['I_se']:.4f} bits (one SE), comparison {c['pred_bits']:.4f}, "
              f"var term {c['var_term']:.4f}")


if __name__ == "__main__":
    main()

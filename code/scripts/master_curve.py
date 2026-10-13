"""Master curves of the layered simplex prior.

Structure: a depth-l sample multiplies l uniform simplex vectors
entrywise and renormalizes.  Each uniform vector is Exp(1) entries
divided by their sum; the sums are common factors and cancel in the
final renormalization, so exactly, at every d,

    V_b = G_b / sum_c G_c,   G_b iid,  G_b = prod_{j=1..l} Exp(1).

Hence d V_b -> W_l = prod of l Exp(1) (mean one), and in the limit
d -> infinity with x = k/d fixed the counts Poissonize and decouple:

    R*(k)/d  ->  r_l(x) = I(W; N),   N | W ~ Poisson(x W).

This script evaluates r_l(x) for l = 1 and 3 on a log grid of x,
together with the constants

    c_l = ( h(W_l) + l*gamma/2 - (1/2) ln(2 pi e) ) * log2 e,

which give the tail r_l(x) = (1/2) log2 x + c_l + o(1), and the
small-x slope r_l'(0) = l (1 - gamma) log2 e bits per sample.
For l = 1 the marginal of N is geometric and r_1 has the closed form
r_1(x) = (1+x) log2(1+x) - gamma x log2 e - E[log2 N!]; the Poisson
pipeline is validated against it.

    python code/scripts/master_curve.py --out results/mechanism/master_curves.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln

EG = 0.5772156649015329
L2 = math.log(2.0)
H_STEP = 1.0 / 512


def log_weight_density(L: int):
    """Density of S = ln W_L on a uniform grid (step H_STEP)."""
    s1 = np.arange(-45.0, 9.0, H_STEP)
    g1 = np.exp(s1 - np.exp(s1))
    g1 /= g1.sum() * H_STEP
    f, lo = g1.copy(), s1[0]
    for _ in range(L - 1):
        f = np.convolve(f, g1) * H_STEP
        lo += s1[0]
    s = lo + H_STEP * np.arange(len(f))
    f /= f.sum() * H_STEP
    keep = f > f.max() * 1e-18
    i0, i1 = np.argmax(keep), len(keep) - np.argmax(keep[::-1])
    return s[i0:i1], f[i0:i1]


def constants(L: int):
    s, f = log_weight_density(L)
    m = f * H_STEP
    ES = float(m @ s)
    hS = float(-(m @ np.log(np.maximum(f, 1e-300))))
    hW = hS + ES
    c = (hW - 0.5 * ES - 0.5 * math.log(2 * math.pi * math.e)) / L2
    return c, hW, ES


def r_poisson(x: float, s: np.ndarray, f: np.ndarray) -> float:
    """I(W; N), N|W ~ Poisson(x e^S), S on the given grid.  Bits."""
    step = s[1] - s[0]
    m = f * step
    m /= m.sum()
    w = np.exp(s)
    lam = x * w
    cum = np.cumsum(m[::-1])[::-1]
    whi = w[np.argmax(cum < 1e-12)] if (cum < 1e-12).any() else w[-1]
    nmax = int(x * whi + 12 * math.sqrt(max(x * whi, 1.0)) + 80)
    loglam = np.log(lam)
    pn = np.zeros(nmax + 1)
    b = 0.0
    for a in range(0, nmax + 1, 8192):
        n = np.arange(a, min(a + 8192, nmax + 1), dtype=float)
        lp = (n[:, None] * loglam[None, :] - lam[None, :]
              - gammaln(n + 1)[:, None])
        p = np.exp(lp)
        pn[a:a + len(n)] = p @ m
        b += float(np.sum((p * lp) @ m))
    pn = np.maximum(pn, 1e-300)
    return (-float(pn @ np.log(pn)) + b) / L2


def r1_geometric(x: float) -> float:
    """Closed form for depth 1 (uniform prior): geometric counts."""
    n = int(60 * (1 + x) + 200)
    j = np.arange(n)
    w = np.exp(j * math.log(x / (1 + x)) - math.log(1 + x))
    w /= w.sum()
    return ((1 + x) * math.log(1 + x) - EG * x
            - float(w @ gammaln(j + 1.0))) / L2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s3, f3 = log_weight_density(3)
    fine = slice(None, None, 4)      # step 1/128 for small x
    coarse = slice(None, None, 16)   # step 1/32 for large x

    print("validation, depth 1 (Poisson pipeline vs geometric closed form):")
    s1g, f1g = log_weight_density(1)
    for x in (1.0, 8.0, 61.0):
        a = r_poisson(x, s1g[fine], f1g[fine])
        cgrid = r_poisson(x, s1g[coarse], f1g[coarse])
        print(f"  x={x:>5}: fine {a:.6f}  coarse {cgrid:.6f}  "
              f"geometric {r1_geometric(x):.6f}")

    xs = np.geomspace(1e-4, 300.0, 141)
    r1 = np.array([r1_geometric(float(x)) for x in xs])
    r3 = np.empty_like(r1)
    for i, x in enumerate(xs):
        sl = fine if x < 8.0 else coarse
        r3[i] = r_poisson(float(x), s3[sl], f3[sl])

    cs = {str(L): constants(L)[0] for L in (1, 2, 3)}
    out = {
        "x": xs.tolist(),
        "r1_bits": r1.tolist(),
        "r3_bits": r3.tolist(),
        "c_bits": cs,
        "slope0_bits_per_sample": {str(L): L * (1 - EG) / L2
                                   for L in (1, 2, 3)},
        "doc": ("r_l(x) = I(W;N), N|W ~ Poisson(xW), "
                "W = product of l Exp(1); tail (1/2)log2 x + c_l; "
                "head l(1-gamma)x log2 e"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print("constants c_l [bits]:", {k: round(v, 5) for k, v in cs.items()})
    for d, kk in ((1024, 112), (4096, 112), (16384, 112)):
        val = d * float(np.interp(math.log(kk / d), np.log(xs), r3))
        print(f"prediction d*r_3(k/d) at d={d}, k={kk}: {val:.1f} bits")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

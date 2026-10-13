"""First check for the per-context section: one tilted row, depth 1.

The row prior is the tilt of the uniform prior: T(b) = q_b E_b / Z,
E_b iid Exp(1), Z = sum_c q_c E_c, q Zipf(alpha).  The Bayes mixture
over the row has an exact one-dimensional integral representation.
Insert Z^{-k} = (1/Gamma(k)) int s^{k-1} e^{-sZ} ds and use
E[E^m e^{-cE}] = m!/(1+c)^{m+1}:

  m(counts) = [prod_b q_b^{m_b} m_b!] / Gamma(k)
              * int_0^inf s^{k-1} prod_b (1+s q_b)^{-(m_b+1)} ds.

One repetition draws a row, draws k iid samples from it, and records
the realized redundancy log2 p(x^k) - log2 m(x^k) on a k grid, with
the integral evaluated by quadrature in u = ln s.  The prediction,
with no fitted parameters, is the master curve at each symbol's own
local density:  rho(k) ~ sum_b r_1(k q_b).

    python code/scripts/context_row_check.py \
        --master results/mechanism/master_curves.json \
        --out results/mechanism/context_row_check.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.optimize import brentq

EG = 0.5772156649015329
L2 = math.log(2.0)


def kgrid(kmax: int, per_decade: int = 8) -> np.ndarray:
    g = np.unique(np.round(np.logspace(0, math.log10(kmax),
                                       int(per_decade * math.log10(kmax)) + 1)
                           ).astype(int))
    return g[g >= 1]


def log_mixture(counts_idx, counts_val, k, q, u_coarse, s0_coarse):
    """log m(counts) in nats, via quadrature in u = ln s.
    s0_coarse = sum_b ln(1+q_b e^u) precomputed on u_coarse."""
    us, s0s = u_coarse[::5], s0_coarse[::5]
    qe = np.exp(us)[None, :] * q[counts_idx][:, None]
    lc = k * us - s0s - counts_val @ np.log1p(qe)
    i = int(np.argmax(lc))
    # Local fine grid around the peak. The full integrand decays like
    # e^{k u} to the left and e^{-d u} to the right. nnz below gives
    # a conservative integration interval.
    nnz = max(len(counts_val), 1)
    # the curvature at the peak is of order nnz, not k
    core = 12.0 / math.sqrt(nnz + 3.0) + 0.5
    lo = us[i] - core - 30.0 / k
    hi = us[i] + core + 30.0 / nnz
    npts = int(min(6001, max(1401, (hi - lo) * 6.0 * math.sqrt(k))))
    uf = np.linspace(lo, hi, npts)
    s0f = np.interp(uf, u_coarse, s0_coarse)
    qef = np.exp(uf)[None, :] * q[counts_idx][:, None]
    lf = k * uf - s0f - counts_val @ np.log1p(qef)
    mx = lf.max()
    integral = mx + math.log(np.trapezoid(np.exp(lf - mx), uf))
    return (float(np.dot(counts_val, np.log(q[counts_idx])))
            + float(np.sum(gammaln(counts_val + 1.0)))
            - math.lgamma(k) + integral)


def log_mixture_adaptive(counts, q, *, epsabs=2e-11, epsrel=2e-11):
    """Depth-one sequence mixture, with direct adaptive log-scale quadrature.

    This reference route avoids interpolating the precomputed kernel used by
    the large chain experiment. Its mode is found from the analytic derivative.
    The omitted log-concave tails are bounded by density/slope at each endpoint.
    Returns the log mixture and numerical/tail diagnostics, all in nats.
    """
    counts=np.asarray(counts,dtype=float);q=np.asarray(q,dtype=float)
    k=int(counts.sum())
    if k==0:return 0.0,dict(relative_quadrature_error=0.0,relative_tail_bound=0.0)
    idx=np.flatnonzero(counts);m=counts[idx];logq=np.log(q)
    def kernel(u):
        lp=np.logaddexp(0.0,u+logq)
        return float(k*u-lp.sum()-m@lp[idx])
    def slope(u):
        t=1/(1+np.exp(-(u+logq)))
        return float(k-t.sum()-m@t[idx])
    lo,hi=-32.,32.
    while slope(lo)<=0:lo-=16
    while slope(hi)>=0:hi+=16
    mode=brentq(slope,lo,hi,xtol=1e-12);peak=kernel(mode)
    left,right=mode-1,mode+1
    while kernel(left)-peak>-42:left-=1
    while kernel(right)-peak>-42:right+=1
    integral,error=quad(lambda u:math.exp(kernel(u)-peak),left,right,
                        points=[mode],epsabs=epsabs,epsrel=epsrel,limit=150)
    tails=(math.exp(kernel(left)-peak)/slope(left)
           +math.exp(kernel(right)-peak)/(-slope(right)))
    value=float(m@logq[idx]+gammaln(m+1).sum()-gammaln(k)+peak+math.log(integral))
    return value,dict(relative_quadrature_error=error/integral,
                      relative_tail_bound=tails/integral)


def r1_interp(master):
    xs = np.log(np.array(master["x"]))
    r1 = np.array(master["r1_bits"])
    c1 = master["c_bits"]["1"]
    s0 = master["slope0_bits_per_sample"]["1"]

    def f(lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam, float)
        out = np.interp(np.log(lam), xs, r1)
        hi = np.log(lam) > xs[-1]
        out[hi] = 0.5 * np.log2(lam[hi]) + c1
        lo = np.log(lam) < xs[0]
        out[lo] = s0 * lam[lo]
        return out
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--kmax", type=int, default=10 ** 5)
    args = ap.parse_args()
    if args.reps < 2:
        raise ValueError("at least two repetitions are needed for a sample standard error")

    r1 = r1_interp(json.loads(args.master.read_text()))
    rng = np.random.default_rng(23)
    d = args.d
    ks = kgrid(args.kmax)
    u_coarse = np.arange(-16.0, 27.0, 0.004)

    out = {"d": d, "L": 1, "k": ks.tolist(), "reps": args.reps,
           "se_ddof": 1, "seed": 23, "sample_count_unit": "iid row samples",
           "code_convention": "Bayes sequence mixture for one tilted iid row, evaluated by interpolated log-scale quadrature.",
           "alphas": []}
    for alpha in (0.5, 1.0, 1.5):
        q = 1.0 / np.arange(1, d + 1) ** alpha
        q /= q.sum()
        s0_coarse = np.log1p(q[:, None] * np.exp(u_coarse)[None, :]).sum(0)
        rho = np.zeros((args.reps, len(ks)))
        for r in range(args.reps):
            e = rng.exponential(size=d)
            t = q * e
            t /= t.sum()
            x = rng.choice(d, size=args.kmax, p=t)
            logt = np.log(t)
            cum = np.concatenate([[0.0], np.cumsum(logt[x])])
            counts = np.zeros(d)
            prev = 0
            for j, k in enumerate(ks):
                counts += np.bincount(x[prev:k], minlength=d)
                prev = k
                idx = np.nonzero(counts)[0]
                lm = log_mixture(idx, counts[idx], int(k), q,
                                 u_coarse, s0_coarse)
                rho[r, j] = (cum[k] - lm) / L2
        pred = np.array([float(np.sum(r1(k * q))) for k in ks])
        rec = {"alpha": alpha,
               "rho_mean": rho.mean(0).tolist(),
               "rho_se": (rho.std(0, ddof=1) / math.sqrt(args.reps)).tolist(),
               "pred_sum_r1": pred.tolist()}
        out["alphas"].append(rec)
        m, se = rho.mean(0), rho.std(0, ddof=1) / math.sqrt(args.reps)
        z = np.abs(m - pred) / np.maximum(2 * se, 1e-9)
        print(f"alpha={alpha}: max |diff|/(2se) = {z.max():.2f}, "
              f"final measured {m[-1]:.0f} vs predicted {pred[-1]:.0f} "
              f"({100 * (m[-1] / pred[-1] - 1):+.1f}%)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

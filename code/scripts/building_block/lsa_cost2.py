"""Gradient / Hessian of the fractional LSA cost, Laplace mixture measure, MC with control variate."""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, logsumexp
import lsa_cost as C

LN2 = C.LN2


def mu_measure(lsa: C.LSA, prof: dict, k: float, fine: float = 1.0 / 1024):
    """Exact (L,u)-measure proportional to exp(F_L(u)) on fine grids.  Returns lists."""
    grids, logw = [], []
    for L in range(1, lsa.Lmax + 1):
        uu = lsa.u
        f = lsa.F(L, prof, k, uu)
        m = f.max()
        keep = np.flatnonzero(f > m - 60.0)
        lo, hi = uu[max(keep[0] - 1, 0)], uu[min(keep[-1] + 1, len(uu) - 1)]
        fine = min(1.0 / 64, 0.25 / np.sqrt(max(k, 1.0)))
        ug = np.arange(lo, hi + fine / 2, fine)
        fg = lsa.F(L, prof, k, ug)
        grids.append(ug)
        logw.append(fg + np.log(fine))
    allw = np.concatenate(logw)
    Z = logsumexp(allw)
    weights = [np.exp(w - Z) for w in logw]
    return grids, weights, Z - gammaln(k)


def grad_hess(lsa: C.LSA, prof: dict, k: float, rs: list[int]):
    """E_mu[l_r] and Cov_mu(l_r, l_s) for r in rs (natural log units)."""
    grids, weights, _ = mu_measure(lsa, prof, k)
    R = len(rs)
    m1 = np.zeros(R)
    m2 = np.zeros((R, R))
    for L in range(1, lsa.Lmax + 1):
        ug, w = grids[L - 1], weights[L - 1]
        vals = np.array([lsa.spline(L, r)(ug) for r in rs])   # R x n
        m1 += vals @ w
        m2 += (vals * w[None, :]) @ vals.T
    cov = m2 - np.outer(m1, m1)
    return m1, cov


def mc_cost(lsa: C.LSA, p: np.ndarray, k: int, ndraw: int, rng, control=True):
    """MC estimate of E c(m), m ~ Mult(k,p), with the linear control variate."""
    nu = C.binom_mean_profile(p, k)
    rs = sorted(nu.keys())
    if control:
        g, _ = grad_hess(lsa, nu, k, rs)   # E_mu[l_r], natural units
        gmap = dict(zip(rs, g))
        nuvec = np.array([nu[r] for r in rs])
    vals = np.empty(ndraw)
    for i in range(ndraw):
        m = rng.multinomial(k, p)
        prof = C.profile_of_counts(m)
        c = lsa.cost_bits(prof, k)
        if control:
            # C(v) = -(1/ln2) ln Z(v) + const;  dC/dv_r = -E_mu[l_r]/ln2
            corr = sum((prof.get(r, 0) - nu[r]) * (-gmap[r]) for r in rs) / LN2
            # subtract the linear prediction: Y = c - grad.(Phi - nu)
            c = c - corr
        vals[i] = c
    return vals.mean(), vals.std(ddof=1) / np.sqrt(ndraw)


def multinomial_profile_cov(p: np.ndarray, k: int, rs: list[int]) -> np.ndarray:
    """Exact Cov(Phi_r, Phi_s) under Mult(k,p) for r,s in rs (Statement 1 formula)."""
    from scipy.stats import binom
    d = len(p)
    r = np.array(rs)
    R = len(rs)
    # marginals
    pm = np.array([binom.pmf(r, k, pb) for pb in p])   # d x R
    nu = pm.sum(axis=0)
    cov = np.diag(nu) - np.outer(nu, nu)
    # pair terms sum_{b != c} J_bc(r,s)
    lr = gammaln(r + 1.0)
    lk = gammaln(k + 1.0)
    rr, ss = np.meshgrid(r, r, indexing="ij")
    valid = (rr + ss) <= k
    lcoef = np.where(valid, lk - lr[:, None] - lr[None, :] - gammaln(np.maximum(k - rr - ss, 0) + 1.0), -np.inf)
    J = np.zeros((R, R))
    lp = np.log(p)
    for b in range(d):
        others = np.delete(np.arange(d), b)
        # vectorise over c
        lpc = lp[others]                                        # (d-1,)
        l1 = np.log1p(-(p[b] + p[others]))                      # (d-1,)
        # term[c, r, s] = exp(lcoef + r lp_b + s lp_c + (k-r-s) l1)
        t = (lcoef[None, :, :] + rr[None, :, :] * lp[b] + ss[None, :, :] * lpc[:, None, None]
             + (k - rr - ss)[None, :, :] * l1[:, None, None])
        J += np.exp(t).sum(axis=0)
    return cov + J, nu

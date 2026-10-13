"""Mean profiles of the tilted product-exponential source, with explicit approximations."""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, logsumexp
import lsa_phi as P
import lsa_cost as C

_dens_cache = {}


def lnW_density(Ls: int, ymin=-40.0, ymax=None, h=1.0 / 64):
    """Grid (y, ln f) for ln W_{Ls}."""
    key = (Ls, ymin, ymax, h)
    if key not in _dens_cache:
        if ymax is None:
            ymax = 3.0 * Ls + 6.0
        y = np.arange(ymin, ymax + h / 2, h)
        lf = P.log_density_lnW(Ls, y)
        _dens_cache[key] = (y, lf + np.log(h))
    return _dens_cache[key]


def log_binom_pmf(k: int, r: np.ndarray, logp: np.ndarray, log1mp: np.ndarray) -> np.ndarray:
    """ln Binom(k, p)(r) for arrays: r (R,), logp/log1mp (N,) -> (N, R)."""
    lc = gammaln(k + 1.0) - gammaln(r + 1.0) - gammaln(k - r + 1.0)
    return lc[None, :] + r[None, :] * logp[:, None] + (k - r)[None, :] * log1mp[:, None]


def frozen_profile(q: np.ndarray, Ls: int, k: int, second_order: bool = True):
    """nu_r = sum_b E_W Binom(k, P_b)(r), P_b = q_b W / (q_b W + Z_{-b}),
    Z_{-b} frozen at 1 - q_b, plus optional 1/2 sigma_b^2 d^2/dz^2 correction.
    Ls = 0 means P = q exactly."""
    r = np.arange(k + 1)
    nu = np.zeros(k + 1)
    if Ls == 0:
        for qb in q:
            nu += np.exp(log_binom_pmf(k, r, np.array([np.log(qb)]), np.array([np.log1p(-qb)]))[0])
        return nu
    y, lfh = lnW_density(Ls)
    W = np.exp(y)
    varW = 2.0 ** Ls - 1.0
    sumq2 = np.sum(q ** 2)
    for qb in q:
        x = qb * W
        z0 = 1.0 - qb
        def mean_pmf(z):
            p = x / (x + z)
            lp = np.log(p); l1 = np.log(z) - np.log(x + z)
            lpmf = log_binom_pmf(k, r, lp, l1)           # (N, R)
            return np.exp(logsumexp(lpmf + lfh[:, None], axis=0))
        base = mean_pmf(z0)
        nu += base
        if second_order:
            sig2 = varW * (sumq2 - qb ** 2)
            dz = 0.05 * z0
            d2 = (mean_pmf(z0 + dz) - 2 * base + mean_pmf(z0 - dz)) / dz ** 2
            nu += 0.5 * sig2 * d2
    return nu


def poisson_profile(lsa: C.LSA, q: np.ndarray, Ls: int, k: float, rmax: int):
    """nu_r = sum_b p_{Ls}(r; k q_b), mixed Poisson pmf from the phi tables (Ls>=1)."""
    r = np.arange(rmax + 1)
    nu = np.zeros(rmax + 1)
    lam = k * q
    lu = np.log(lam)
    for j, qb in enumerate(q):
        lp = np.array([r_ * lu[j] + lsa.spline(Ls, r_)(lu[j]) - gammaln(r_ + 1.0) for r_ in r])
        nu += np.exp(lp)
    return nu


def profile_dict(nu: np.ndarray, tol=1e-14) -> dict:
    return {int(i): float(v) for i, v in enumerate(nu) if v > tol}


def mc_profile_moments(q, Ls, k, ndraw, rng):
    """Monte Carlo mean and covariance of the profile (counts only, cheap)."""
    Phi = np.zeros((ndraw, k + 1))
    for i in range(ndraw):
        p = C.sample_source(q, Ls, rng)
        m = rng.multinomial(k, p)
        Phi[i] = np.bincount(m, minlength=k + 1)
    return Phi.mean(0), np.cov(Phi.T)


# ------------------------------------------------------------------ exact conditional profile
def mixed_poisson_pmfs(q: np.ndarray, Ls: int, k: float, R: int) -> np.ndarray:
    """p_b(r) = P(N_b = r), N_b ~ Poisson(k q_b W) mixed over W_{Ls}; shape (d, R+1).  Ls=0: Poisson."""
    r = np.arange(R + 1)
    lam = k * q
    if Ls == 0:
        return np.exp(r[None, :] * np.log(lam)[:, None] - lam[:, None] - gammaln(r + 1.0)[None, :])
    y, lfh = lnW_density(Ls)
    W = np.exp(y)
    out = np.empty((len(q), R + 1))
    for b, lb in enumerate(lam):
        lw = lb * W
        lp = r[None, :] * np.log(lw)[:, None] - lw[:, None] - gammaln(r + 1.0)[None, :]
        out[b] = np.exp(logsumexp(lp + lfh[:, None], axis=0))
    return out


def _conv(a, b, n):
    return np.convolve(a, b)[:n]


def conditional_profile(pm: np.ndarray, k: int):
    """Exact mean profile of Mult(k, P) via Poissonisation + conditioning on the total.
    pm: (d, R+1) mixed-Poisson pmfs.  Returns nu (length k+1) and the ratio matrix rows."""
    d, R1 = pm.shape
    n = R1
    prefix = [None] * (d + 1)
    suffix = [None] * (d + 2)
    prefix[0] = np.zeros(n); prefix[0][0] = 1.0
    suffix[d + 1] = np.zeros(n); suffix[d + 1][0] = 1.0
    for i in range(1, d + 1):
        prefix[i] = _conv(prefix[i - 1], pm[i - 1], n)
    for i in range(d, 0, -1):
        suffix[i] = _conv(suffix[i + 1], pm[i - 1], n)
    PS = prefix[d]
    nu = np.zeros(k + 1)
    r = np.arange(k + 1)
    for b in range(1, d + 1):
        Pmb = _conv(prefix[b - 1], suffix[b + 1], n)
        nu += pm[b - 1, :k + 1] * Pmb[k - r] / PS[k]
    return nu, PS


def exact_profile_scale_integral(q: np.ndarray, Ls: int, k: int, R: int | None = None,
                                 xgrid=None, return_parts=False):
    """Exact mean profile of Mult(k,P), P = q*W/Z, by the scale integral
    Pr(m_b = r) = k int_0^inf s^{-1} p_b(r; s) P_{-b}(k-r; s) ds.
    Also returns the scale density w(s) = k P_S(k; s)/s on the grid."""
    if R is None:
        R = 4 * k + 10
    if xgrid is None:
        xgrid = np.arange(-1.6, 1.6001, 0.04)
    s_vals = k * np.exp(xgrid)
    r = np.arange(k + 1)
    nu = np.zeros(k + 1)
    wlist, nulist = [], []
    for x, s in zip(xgrid, s_vals):
        pm = mixed_poisson_pmfs(q, Ls, s, R)
        d = len(q); n = R + 1
        prefix = [np.zeros(n)]; prefix[0][0] = 1.0
        for i in range(d):
            prefix.append(_conv(prefix[-1], pm[i], n))
        suffix = [None] * (d + 2); suffix[d + 1] = np.zeros(n); suffix[d + 1][0] = 1.0
        for i in range(d, 0, -1):
            suffix[i] = _conv(suffix[i + 1], pm[i - 1], n)
        PS = prefix[d]
        # unnormalised conditional profile at this s: sum_b p_b(r) P_{-b}(k-r)
        acc = np.zeros(k + 1)
        for b in range(1, d + 1):
            Pmb = _conv(prefix[b - 1], suffix[b + 1], n)
            acc += pm[b - 1, :k + 1] * Pmb[k - r]
        # integrate over s with measure k ds/s = k dx
        w = k * PS[k]            # density of x = ln(s/k): w(s) ds = k P_S(k;s)/s ds = k P_S(k;s) dx
        nu += k * acc * 0.04
        wlist.append(w); nulist.append(acc / max(PS[k], 1e-300))
    if return_parts:
        return nu, np.array(wlist), np.array(nulist), xgrid
    return nu

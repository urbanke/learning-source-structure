"""LSA code costs from the phi tables: exact integral, Laplace, profile tools."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.interpolate import CubicSpline
from uspline import USpline
from scipy.optimize import brentq
import lsa_phi as P

LN2 = np.log(2.0)


class LSA:
    def __init__(self, Lmax: int, prefix: str = "phitab_L"):
        self.Lmax = Lmax
        self.tabs = [None] + [np.load(f"{prefix}{L}.npy") for L in range(1, Lmax + 1)]
        self.rmax = self.tabs[1].shape[0] - 1
        self.u = P.u
        self._spl = {}

    # ---- ln phi as a function of u (cubic spline) ----------------------
    def spline(self, L: int, r: int) -> CubicSpline:
        key = (L, r)
        if key not in self._spl:
            self._spl[key] = USpline(self.u, self.tabs[L][r])
        return self._spl[key]

    def F(self, L: int, prof: dict, k: float, uu: np.ndarray, deriv: int = 0) -> np.ndarray:
        """F_L(u) = k u + sum_r prof[r] ln phi_{L,r}(e^u)  (or its derivative)."""
        out = np.zeros_like(uu, dtype=float)
        if deriv == 0:
            out += k * uu
        elif deriv == 1:
            out += k
        for r, v in prof.items():
            if v == 0:
                continue
            out += v * self.spline(L, r)(uu, nu=deriv)
        return out

    # ---- exact depth-L log probability of a (possibly fractional) profile
    def logQ_L(self, L: int, prof: dict, k: float, fine: float = None) -> float:
        uu = self.u
        f = self.F(L, prof, k, uu)
        m = f.max()
        keep = np.flatnonzero(f > m - 60.0)
        lo, hi = uu[max(keep[0] - 1, 0)], uu[min(keep[-1] + 1, len(uu) - 1)]
        if fine is None:
            fine = min(1.0 / 64, 0.25 / np.sqrt(max(k, 1.0)))
        ug = np.arange(lo, hi + fine / 2, fine)
        fg = self.F(L, prof, k, ug)
        return logsumexp(fg) + np.log(fine) - gammaln(k)

    def cost_bits(self, prof: dict, k: float) -> float:
        """LSA cost -log2 (1/Lmax) sum_L Q_L for a profile (exact integral)."""
        lq = np.array([self.logQ_L(L, prof, k) for L in range(1, self.Lmax + 1)])
        return -(logsumexp(lq) - np.log(self.Lmax)) / LN2

    def cost_bits_by_depth(self, prof: dict, k: float) -> np.ndarray:
        return -np.array([self.logQ_L(L, prof, k) for L in range(1, self.Lmax + 1)]) / LN2

    # ---- Laplace approximation of the u integral ------------------------
    def laplace_L(self, L: int, prof: dict, k: float):
        uu = self.u
        f = self.F(L, prof, k, uu)
        j = int(np.argmax(f))
        a, b = uu[max(j - 2, 0)], uu[min(j + 2, len(uu) - 1)]
        g = lambda x: self.F(L, prof, k, np.array([x]), 1)[0]
        try:
            ustar = brentq(g, a, b)
        except ValueError:
            ustar = uu[j]
        fpp = self.F(L, prof, k, np.array([ustar]), 2)[0]
        val = self.F(L, prof, k, np.array([ustar]))[0]
        return ustar, val, fpp

    def laplace_logQ_L(self, L: int, prof: dict, k: float) -> float:
        ustar, val, fpp = self.laplace_L(L, prof, k)
        return val + 0.5 * np.log(2 * np.pi / max(-fpp, 1e-300)) - gammaln(k)

    def laplace_cost_bits(self, prof: dict, k: float) -> float:
        lq = np.array([self.laplace_logQ_L(L, prof, k) for L in range(1, self.Lmax + 1)])
        return -(logsumexp(lq) - np.log(self.Lmax)) / LN2


# ---------------------------------------------------------------- profiles
def profile_of_counts(m: np.ndarray) -> dict:
    vals, cnt = np.unique(m, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def zipf(d: int, alpha: float) -> np.ndarray:
    q = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return q / q.sum()


def sample_source(q: np.ndarray, Ls: int, rng: np.random.Generator) -> np.ndarray:
    """P_b = q_b W_b / Z, W_b product of Ls exponentials (Ls = 0: P = q)."""
    if Ls == 0:
        return q.copy()
    W = np.prod(rng.exponential(size=(Ls, len(q))), axis=0)
    p = q * W
    return p / p.sum()


def binom_mean_profile(p: np.ndarray, k: int) -> dict:
    """nu_r = sum_b Binomial(k, p_b)(r), r = 0..k (exact, fixed p)."""
    from scipy.stats import binom
    r = np.arange(k + 1)
    nu = np.zeros(k + 1)
    for pb in p:
        nu += binom.pmf(r, k, pb)
    return {int(i): float(v) for i, v in enumerate(nu) if v > 1e-300}

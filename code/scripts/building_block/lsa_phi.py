"""Independent numerical machinery for the LSA code (feasibility check only).

phi_{L,r}(s) = E[W_L^r exp(-s W_L)],  W_L = product of L iid Exp(1).

The density f_L of Y = ln W_L is computed by saddle-point Mellin inversion
(uniform relative accuracy in both tails):
    f_L(y) = (1/2 pi) int exp(L lnGamma(1 + c + i t) - (c + i t) y) dt,
with c = c(y) solving L psi(1 + c) = y.  Then
    ln phi_{L,r}(e^u) = ln sum_y h f_L(y) exp(r y - e^{u+y}).
All logs natural.
"""
from __future__ import annotations

import numpy as np
from scipy.special import loggamma, gammaln, digamma, polygamma, logsumexp

EULER = 0.57721566490153286
HU = 1.0 / 64                      # u = ln s grid step
U_MIN, U_MAX = -45.0, 15.0
u = np.arange(U_MIN, U_MAX + HU / 2, HU)
NU = len(u)

HY = 1.0 / 256
Y_MIN = -90.0


def _y_grid(L: int) -> np.ndarray:
    ymax = 3.0 * L + 45.0
    return np.arange(Y_MIN, ymax + HY / 2, HY)


def _series_exp(b: np.ndarray, order: int) -> np.ndarray:
    """Taylor coefficients of exp(sum_k b_k e^k), b_0 = 0, up to e^order."""
    c = np.zeros(order + 1)
    c[0] = 1.0
    for n in range(1, order + 1):
        c[n] = sum(k * b[k] * c[n - k] for k in range(1, n + 1)) / n
    return c


def log_density_lnW_series(L: int, y: np.ndarray, M: int = 25) -> np.ndarray:
    """Residue series for f_L(y) = sum_m e^{m y} R_m(y); exact, use for y <= -2."""
    from scipy.special import zeta
    total = np.zeros_like(y)
    for m in range(1, M + 1):
        b = np.zeros(L)
        js = np.arange(1, m, dtype=float)
        if L >= 2:
            b[1] = L * (np.sum(1.0 / js) - EULER)
            for k in range(2, L):
                b[k] = L * (zeta(k) * (-1) ** k / k + np.sum(1.0 / (k * js ** k)))
        a = _series_exp(b, L - 1)
        poly = np.zeros_like(y)
        for j in range(L):
            poly += a[L - 1 - j] * (-y) ** j / np.exp(gammaln(j + 1.0))
        sgn = (-1.0) ** ((m - 1) * L)
        total += sgn * np.exp(m * y - L * gammaln(m)) * poly
    return np.log(np.maximum(total, 1e-300))


def log_density_lnW(L: int, y: np.ndarray) -> np.ndarray:
    """ln f_L(y): residue series for y <= -2, saddle-point inversion otherwise."""
    if L == 1:
        return y - np.exp(y)
    return _log_density_saddle(L, y)


def _log_density_saddle(L: int, y: np.ndarray) -> np.ndarray:
    # solve L psi(1+c) = y for c > -1 by Newton on x = 1 + c > 0 (log-parametrised)
    x = np.exp(np.clip(y / L, -30, 30)) + 0.0  # rough start: psi(x) ~ ln x
    x = np.maximum(x, 1e-12)
    for _ in range(60):
        g = L * digamma(x) - y
        x_new = x - g / (L * polygamma(1, x))
        x = np.where(x_new > 0, x_new, x / 2)
    c = x - 1.0
    sd = 1.0 / np.sqrt(L * polygamma(1, x))
    out = np.empty(len(y))
    chunk = 256
    for a in range(0, len(y), chunk):
        cc = c[a:a + chunk]
        ss = sd[a:a + chunk]
        yy = y[a:a + chunk]
        T = np.maximum(12.0 * ss, 14.0)
        nt = 2 * int(np.ceil(np.max(T / (ss / 8.0)))) + 1
        nt = min(nt, 20001)
        tt = np.linspace(-1.0, 1.0, nt)
        t = T[:, None] * tt[None, :]
        z = (1.0 + cc)[:, None] + 1j * t
        lg = L * loggamma(z) - (cc[:, None] + 1j * t) * yy[:, None]
        ref = lg[:, nt // 2].real
        vals = np.exp(lg - ref[:, None]).real
        integ = np.trapezoid(vals, t, axis=1)
        out[a:a + chunk] = ref + np.log(np.maximum(integ, 1e-300)) - np.log(2 * np.pi)
    return out


class PhiTables:
    """ln phi_{L,r}(e^u) on the u grid for L = 1..Lmax, r = 0..rmax."""

    def __init__(self, Lmax: int, rmax: int, verbose: bool = False):
        self.Lmax, self.rmax = Lmax, rmax
        self.tab = np.empty((Lmax + 1, rmax + 1, NU))
        eu = np.exp(u)
        for L in range(1, Lmax + 1):
            y = _y_grid(L)
            if L == 1:
                lf = y - np.exp(y)
            else:
                lf = log_density_lnW(L, y)
            lfh = lf + np.log(HY)
            ey = np.exp(y)
            for r in range(rmax + 1):
                base = lfh + r * y
                res = np.empty(NU)
                for a in range(0, NU, 16):
                    uu = eu[a:a + 16]
                    res[a:a + 16] = logsumexp(base[None, :] - uu[:, None] * ey[None, :], axis=1)
                self.tab[L, r] = res
            if verbose:
                print(f"L={L} done", flush=True)

    def log_phi(self, L: int, r: int, uu: np.ndarray) -> np.ndarray:
        """Cubic interpolation in u."""
        return np.interp(uu, u, self.tab[L, r])

    def save(self, path):
        np.savez_compressed(path, tab=self.tab, u=u)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls.__new__(cls)
        obj.tab = d["tab"]
        obj.Lmax, obj.rmax = obj.tab.shape[0] - 1, obj.tab.shape[1] - 1
        return obj


if __name__ == "__main__":
    import time
    from scipy.special import kv, exp1
    y = _y_grid(2)
    t0 = time.time()
    lf1 = log_density_lnW(1, y)
    print("L=1 density max err", np.max(np.abs(lf1 - (y - np.exp(y)))), time.time() - t0)
    lf2 = log_density_lnW(2, y)
    ex2 = y + np.log(2 * kv(0, 2 * np.exp(y / 2)))
    ok = np.isfinite(ex2)
    print("L=2 density max err (where kv finite)", np.max(np.abs(lf2[ok] - ex2[ok])))
    # tail check: y=-60, L=2: f ~ e^{y}(-y-2gamma)
    print("L=2 tail", lf2[0], Y_MIN + np.log(-Y_MIN - 2 * 0.5772156649))
    t0 = time.time()
    T = PhiTables(2, 40, verbose=True)
    print("tables L<=2 r<=40:", time.time() - t0)
    s = np.exp(u)
    for r in (0, 1, 2, 5, 20, 40):
        exact = gammaln(r + 1.0) - (r + 1) * np.log1p(s)
        print("L=1", r, np.max(np.abs(T.tab[1, r] - exact)))
    ok = s > 0.02
    ex = np.log(np.exp(1 / s[ok]) * exp1(1 / s[ok]) / s[ok])
    print("L=2 r=0 max err", np.max(np.abs(T.tab[2, 0][ok] - ex)))
    x = np.linspace(0, 80, 800001)[1:]
    for r in (1, 3, 10, 40):
        for j in (0, 300, 700, NU - 1):
            ss = s[j]
            val = gammaln(r + 1) + np.log(np.trapezoid(np.exp(r * np.log(x) - x - (r + 1) * np.log1p(ss * x)), x))
            print(f"L=2 r={r} s={ss:.3g}: table {T.tab[2, r, j]:.9f} direct {val:.9f}")

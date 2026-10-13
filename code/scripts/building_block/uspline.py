"""Natural cubic spline on a uniform grid with float32 second derivatives (memory-light)."""
import numpy as np


class USpline:
    __slots__ = ("x0", "h", "y", "M", "n")

    def __init__(self, x, y):
        self.x0 = float(x[0]); self.h = float(x[1] - x[0]); self.n = len(y)
        self.y = y
        h = self.h
        n = self.n
        # natural spline: solve tridiagonal system for second derivatives M
        rhs = np.zeros(n)
        rhs[1:-1] = 6.0 * (y[2:] - 2 * y[1:-1] + y[:-2]) / (h * h)
        # Thomas algorithm for the interior (diag 4, off-diag 1), M[0] = M[-1] = 0
        m = n - 2
        c = np.full(m, 1.0); dgn = np.full(m, 4.0)
        b = rhs[1:-1].copy()
        cp = np.empty(m); dp = np.empty(m)
        cp[0] = c[0] / dgn[0]; dp[0] = b[0] / dgn[0]
        for i in range(1, m):
            den = dgn[i] - c[i] * cp[i - 1]
            cp[i] = c[i] / den
            dp[i] = (b[i] - c[i] * dp[i - 1]) / den
        Mi = np.empty(m)
        Mi[-1] = dp[-1]
        for i in range(m - 2, -1, -1):
            Mi[i] = dp[i] - cp[i] * Mi[i + 1]
        M = np.zeros(n); M[1:-1] = Mi
        self.M = M.astype(np.float32)

    def __call__(self, x, nu=0):
        x = np.asarray(x, dtype=float)
        t = (x - self.x0) / self.h
        i = np.clip(np.floor(t).astype(int), 0, self.n - 2)
        a = t - i                      # in [0,1)
        b = 1.0 - a
        h = self.h
        y0 = self.y[i]; y1 = self.y[i + 1]; M0 = self.M[i].astype(float); M1 = self.M[i + 1].astype(float)
        if nu == 0:
            return b * y0 + a * y1 + ((b ** 3 - b) * M0 + (a ** 3 - a) * M1) * h * h / 6.0
        if nu == 1:
            return (y1 - y0) / h + ((3 * a * a - 1) * M1 - (3 * b * b - 1) * M0) * h / 6.0
        if nu == 2:
            return b * M0 + a * M1
        raise ValueError(nu)

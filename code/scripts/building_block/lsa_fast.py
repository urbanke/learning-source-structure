"""Spline-free LSA evaluator on the memory-mapped tables ln phi_{L,r}(e^u).

Values at arbitrary u come from a six-point Lagrange stencil on the uniform u grid; u-derivatives come from
the exact identities  d/du ln phi_r = -e^u phi_{r+1}/phi_r  and its derivative (rows r+1, r+2).  No per-row
objects are kept, so the memory does not grow with the number of (L, r) pairs used.  Importing this module
also replaces the spline-based hot spots of bb.py / lsa_cost2.py with vectorised versions (same results).

Table files: phitab_L{L}_r{r0}_{r1}_full.npy (columns = u grid -45..40 step 1/64) as written by
merge_tables.py / build_tables_r.py, or produced from the anchor store by store_rows.py.
"""
from __future__ import annotations
import glob, re, os
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq
import lsa_cost as C
import lsa_cost2 as C2
import bb

LN2 = np.log(2.0)
STENCIL = 6


def _lagrange_weights(tt):
    """Weights of the 6-point Lagrange stencil at local coordinate tt in [0, 5]: shape (6, m)."""
    tt = np.asarray(tt, float)
    W = np.empty((STENCIL, len(tt)))
    for j in range(STENCIL):
        w = np.ones_like(tt)
        for m in range(STENCIL):
            if m != j:
                w *= (tt - m) / (j - m)
        W[j] = w
    return W


class LSAfast(C.LSA):
    def __init__(self, Lmax: int, rmax: int = 10 ** 9, table_dir: str = "."):
        self.Lmax = Lmax
        HU = C.P.HU
        self.u = np.concatenate([C.P.u, np.arange(15.0 + HU, 40.0 + HU / 2, HU)])
        self.u0 = float(self.u[0]); self.h = float(self.u[1] - self.u[0]); self.nu = len(self.u)
        self.u_extended = True
        self._blocks = {}      # L -> list of (r0, r1, array, col_offset); col_offset = index of the block's first u
        self.rfull = 10 ** 9   # counts covered by full-grid blocks at every level
        covered = []
        for L in range(1, Lmax + 1):
            blocks = []
            for f in glob.glob(os.path.join(table_dir, f"phitab_L{L}_r*.npy")):
                m = re.match(rf".*phitab_L{L}_r(\d+)_(\d+)_(full|u(-?\d+)_(-?\d+))\.npy$", os.path.basename(f))
                if not m:
                    continue
                r0 = int(m.group(1))
                a = np.load(f, mmap_mode="r")
                if m.group(3) == "full":
                    off = 0
                    assert a.shape[1] == self.nu, (f, a.shape)
                else:
                    lo = float(m.group(4)); off = int(round((lo - self.u0) / self.h))
                    assert off >= 0 and off + a.shape[1] <= self.nu, (f, a.shape, off)
                blocks.append((r0, r0 + a.shape[0] - 1, a, off))
            blocks.sort(key=lambda b: b[0])
            if not blocks or blocks[0][0] != 0 or blocks[0][3] != 0:
                raise FileNotFoundError(f"no full-grid phitab_L{L}_r0_*_full.npy in {table_dir}")
            top = blocks[0][1]; topfull = blocks[0][1]
            for r0, r1, _, off in blocks[1:]:
                if r0 == top + 1:
                    top = r1
                    if off == 0 and topfull == r0 - 1:
                        topfull = r1
                else:
                    break
            covered.append(top)
            self.rfull = min(self.rfull, topfull)
            self._blocks[L] = blocks
        self.rmax = min(min(covered), rmax)
        self._dens = {}
        self._extra = {}          # (L, r) -> row computed directly (counts beyond the tables)

    # ---- rows -----------------------------------------------------------
    def _direct_row(self, L, r):
        key = (L, r)
        if key not in self._extra:
            if L not in self._dens:
                y = C.P._y_grid(L)
                lf = C.P.log_density_lnW(L, y) + np.log(C.P.HY)
                self._dens[L] = (y, lf)
            y, lfh = self._dens[L]
            base = lfh + r * y
            ey = np.exp(y)
            vals = np.empty(self.nu)
            for a in range(0, self.nu, 32):
                uu = np.exp(self.u[a:a + 32])
                g = base - np.exp(self.u[min(a + 16, self.nu - 1)]) * ey
                keep = np.flatnonzero(g > g.max() - 70.0)
                lo, hi = max(keep[0] - 300, 0), min(keep[-1] + 300, len(y) - 1)
                vals[a:a + 32] = logsumexp(base[None, lo:hi + 1] - uu[:, None] * ey[None, lo:hi + 1], axis=1)
            self._extra[key] = vals
        return self._extra[key]

    def rows(self, L, rs):
        """Table rows for the counts rs: (len(rs), nu) array (a copy only for rows outside the tables)."""
        rs = np.asarray(rs, dtype=int)
        out = np.empty((len(rs), self.nu))
        done = np.zeros(len(rs), dtype=bool)
        for r0, r1, a, off in self._blocks[L]:
            sel = np.flatnonzero((rs >= r0) & (rs <= r1) & ~done)
            if len(sel):
                out[sel] = np.nan
                out[sel, off:off + a.shape[1]] = a[rs[sel] - r0]
                done[sel] = True
        for i in np.flatnonzero(~done):
            out[i] = self._direct_row(L, int(rs[i]))
        return out

    def _gather(self, L, rs, cols):
        """(len(rs), len(cols)) values table[r, col]."""
        rs = np.asarray(rs, dtype=int); cols = np.asarray(cols, dtype=int)
        out = np.empty((len(rs), len(cols)))
        done = np.zeros(len(rs), dtype=bool)
        for r0, r1, a, off in self._blocks[L]:
            sel = np.flatnonzero((rs >= r0) & (rs <= r1) & ~done)
            if len(sel):
                if off > 0 or a.shape[1] < self.nu:
                    if cols.min() < off or cols.max() >= off + a.shape[1]:
                        raise ValueError(f"u outside the stored window for L={L}, r in {rs[sel].min()}..{rs[sel].max()}")
                out[sel] = a[np.ix_(rs[sel] - r0, cols - off)]
                done[sel] = True
        for i in np.flatnonzero(~done):
            out[i] = self._direct_row(L, int(rs[i]))[cols]
        return out

    # ---- values and derivatives ----------------------------------------
    def lnphi(self, L, rs, u, deriv=0):
        """ln phi_{L,r}(e^u) for r in rs at the points u: array (len(rs), len(u)); deriv 1, 2 in u."""
        rs = np.asarray(rs, dtype=int); u = np.atleast_1d(np.asarray(u, float))
        if deriv == 0:
            return self._interp(L, rs, u)
        a = self._interp(L, rs, u); b = self._interp(L, rs + 1, u)
        g = -np.exp(u[None, :] + b - a)
        if deriv == 1:
            return g
        c = self._interp(L, rs + 2, u)
        g1 = -np.exp(u[None, :] + c - b)
        return g * (1.0 + g1 - g)

    def _interp(self, L, rs, u):
        t = (u - self.u0) / self.h
        i_near = np.rint(t).astype(int)
        on_grid = np.abs(t - i_near) < 1e-9
        out = np.empty((len(rs), len(u)))
        if on_grid.all():
            return self._gather(L, rs, np.clip(i_near, 0, self.nu - 1))
        i0 = np.clip(np.floor(t).astype(int) - 2, 0, self.nu - STENCIL)
        tt = t - i0
        W = _lagrange_weights(tt)                                       # (6, m)
        cols = (i0[None, :] + np.arange(STENCIL)[:, None]).ravel()      # (6 m,)
        # chunk over u to bound memory
        CH = max(1, 200000 // max(len(rs), 1))
        for a in range(0, len(u), CH):
            sl = slice(a, a + CH)
            m = len(u[sl])
            cc = (i0[sl][None, :] + np.arange(STENCIL)[:, None]).ravel()
            Y = self._gather(L, rs, cc).reshape(len(rs), STENCIL, m)
            out[:, sl] = np.einsum("rjm,jm->rm", Y, W[:, sl])
        return out

    def spline(self, L, r):
        """Compatibility shim: a callable (u, nu=0) like the spline objects."""
        lsa = self
        class _S:
            def __call__(self_, u, nu=0):
                u = np.atleast_1d(np.asarray(u, float))
                return lsa.lnphi(L, [r], u, deriv=nu)[0]
        return _S()

    # ---- the code integrals ----------------------------------------------
    def F(self, L, prof, k, uu, deriv=0):
        uu = np.atleast_1d(np.asarray(uu, float))
        rs = np.array([r for r, v in prof.items() if v != 0], dtype=int)
        vs = np.array([prof[r] for r in rs], dtype=float)
        out = vs @ self.lnphi(L, rs, uu, deriv) if len(rs) else np.zeros_like(uu)
        if deriv == 0:
            out = out + k * uu
        elif deriv == 1:
            out = out + k
        return out

    def _coarse(self, prof):
        if max(prof) > self.rfull:
            return self.u[self.u >= -10.0 + 1e-9][::4]
        return self.u[::4]

    def logQ_L(self, L, prof, k, fine=None):
        coarse = self._coarse(prof)
        f = self.F(L, prof, k, coarse)
        m = f.max(); keep = np.flatnonzero(f > m - 60.0)
        lo, hi = coarse[max(keep[0] - 1, 0)], coarse[min(keep[-1] + 1, len(coarse) - 1)]
        if fine is None:
            fine = min(1.0 / 64, 0.25 / np.sqrt(max(k, 1.0)))
        ug = np.arange(lo, hi + fine / 2, fine)
        fg = self.F(L, prof, k, ug)
        return logsumexp(fg) + np.log(fine) - gammaln(k)

    def laplace_L(self, L, prof, k):
        coarse = self._coarse(prof)
        f = self.F(L, prof, k, coarse)
        j = int(np.argmax(f))
        a, b = coarse[max(j - 1, 0)], coarse[min(j + 1, len(coarse) - 1)]
        g = lambda x: float(self.F(L, prof, k, np.array([x]), 1)[0])
        try:
            ustar = brentq(g, a, b)
        except ValueError:
            ustar = coarse[j]
        fpp = float(self.F(L, prof, k, np.array([ustar]), 2)[0])
        val = float(self.F(L, prof, k, np.array([ustar]))[0])
        return ustar, val, fpp


# ---- vectorised replacements for the spline-based hot spots -------------------------------------
def _pmfs_mixed_fast(lsa, q, Ls, s, R, d, lam, lu, r):
    rt = min(lsa.rmax, R)
    out = np.zeros((d, R + 1))
    r1 = min(rt, lsa.rfull)
    rr = np.arange(r1 + 1)
    lp = rr[:, None] * lu[None, :] + lsa.lnphi(Ls, rr, lu) - gammaln(rr + 1.0)[:, None]   # (r1+1, d)
    out[:, :r1 + 1] = np.exp(lp).T
    if rt > r1:
        # counts beyond the full-grid blocks: stored on u >= -10 only; the pmf is zero elsewhere
        rr = np.arange(r1 + 1, rt + 1)
        ok = np.flatnonzero(lu >= -10.0 + 1e-9)
        if len(ok):
            lp = rr[:, None] * lu[None, ok] + lsa.lnphi(Ls, rr, lu[ok]) - gammaln(rr + 1.0)[:, None]
            out[ok, r1 + 1:rt + 1] = np.exp(lp).T
    if R > rt:
        y, lfh = bb.S.lnW_density(Ls)
        W = np.exp(y)
        r2 = np.arange(rt + 1, R + 1)
        for b in range(d):
            lw = lam[b] * W
            lp2 = r2[None, :] * np.log(lw)[:, None] - lw[:, None] - gammaln(r2 + 1.0)[None, :]
            out[b, rt + 1:] = np.exp(logsumexp(lp2 + lfh[:, None], axis=0))
    return out


def eta_laplace_fast(lsa, nu_d, k, Sigma, rs, us, fpp, w):
    M = lsa.Lmax
    rs = np.asarray(rs, dtype=int)
    R = len(rs)
    m1 = np.zeros(R); quad = 0.0; V = np.zeros((M, R))
    for L in range(1, M + 1):
        vals = lsa.lnphi(L, rs, [us[L - 1]])[:, 0]
        d1 = lsa.lnphi(L, rs, [us[L - 1]], 1)[:, 0]
        V[L - 1] = vals
        m1 += w[L - 1] * vals
        quad += w[L - 1] * (d1 @ Sigma @ d1) / max(-fpp[L - 1], 1e-300)
    for L in range(1, M + 1):
        dv = V[L - 1] - m1
        quad += w[L - 1] * (dv @ Sigma @ dv)
    return 0.5 * quad / LN2


def mu_measure_fast(lsa, prof, k, fine=None):
    grids, logw = [], []
    for L in range(1, lsa.Lmax + 1):
        coarse = lsa._coarse(prof)
        f = lsa.F(L, prof, k, coarse)
        m = f.max(); keep = np.flatnonzero(f > m - 60.0)
        lo, hi = coarse[max(keep[0] - 1, 0)], coarse[min(keep[-1] + 1, len(coarse) - 1)]
        fn = min(1.0 / 64, 0.25 / np.sqrt(max(k, 1.0)))
        ug = np.arange(lo, hi + fn / 2, fn)
        fg = lsa.F(L, prof, k, ug)
        grids.append(ug); logw.append(fg + np.log(fn))
    allw = np.concatenate(logw); Z = logsumexp(allw)
    return grids, [np.exp(x - Z) for x in logw], Z - gammaln(k)


def grad_hess_fast(lsa, prof, k, rs):
    grids, weights, _ = mu_measure_fast(lsa, prof, k)
    rs = np.asarray(rs, dtype=int); R = len(rs)
    m1 = np.zeros(R); m2 = np.zeros((R, R))
    for L in range(1, lsa.Lmax + 1):
        ug, w = grids[L - 1], weights[L - 1]
        vals = lsa.lnphi(L, rs, ug)
        m1 += vals @ w
        m2 += (vals * w[None, :]) @ vals.T
    return m1, m2 - np.outer(m1, m1)


bb._pmfs_mixed = _pmfs_mixed_fast
bb.eta_laplace = eta_laplace_fast
C2.mu_measure = mu_measure_fast
C2.grad_hess = grad_hess_fast
bb.LSAx = LSAfast

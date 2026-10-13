"""Building block: expected LSA cost of k iid samples from a tilted product-exponential row.

E c(m) ~ int w(s) [ C_Lap(nu^(s)) - eta^(s) ] ds
  w(s) ds = k P_s(S=k) ds/s        (scale law; exact identity)
  nu^(s)  = conditioned mean profile at scale s (exact given the mixed-Poisson pmfs)
  C_Lap   = Laplace-in-scale, log-sum-exp over depths, at the mean profile
  eta^(s) = second-order fluctuation term with the independent-symbol covariance
"""
from __future__ import annotations
import sys, json, time
import numpy as np
from scipy.special import gammaln, logsumexp
import lsa_cost as C
import source_profiles as S

LN2 = np.log(2.0)
KEXACT = 32


class LSAx(C.LSA):
    """LSA tables ln phi_{L,r}(e^u) on the full u grid [-45, 40], memory-light.

    Blocks phitab_L{L}_r{r0}_{r1}_full.npy (columns = full u grid) are memory-mapped; rows are
    handed to the spline as views, so only the spline's second derivatives (float32) live in RAM.
    Counts beyond the tabulated range fall back to direct evaluation (slow, exact)."""

    def __init__(self, Lmax: int, rmax: int = 10 ** 9):
        import glob, re
        self.Lmax = Lmax
        HU = C.P.HU
        self.u = np.concatenate([C.P.u, np.arange(15.0 + HU, 40.0 + HU / 2, HU)])
        self.u_extended = True
        self._blocks = {}
        covered = []
        for L in range(1, Lmax + 1):
            blocks = []
            for f in glob.glob(f"phitab_L{L}_r*_full.npy"):
                m = re.match(rf"phitab_L{L}_r(\d+)_(\d+)_full\.npy$", f)
                if not m:
                    continue
                r0, r1 = int(m.group(1)), int(m.group(2))
                a = np.load(f, mmap_mode="r")
                assert a.shape[1] == len(self.u), (f, a.shape)
                blocks.append((r0, r0 + a.shape[0] - 1, a))
            blocks.sort()
            if not blocks or blocks[0][0] != 0:
                raise FileNotFoundError(f"no phitab_L{L}_r0_*_full.npy")
            top = blocks[0][1]
            for r0, r1, _ in blocks[1:]:
                if r0 == top + 1:
                    top = r1
                else:
                    break
            covered.append(top)
            self._blocks[L] = blocks
        self.rmax = min(min(covered), rmax)
        self._spl = {}
        self._dens = {}

    def _row(self, L: int, r: int):
        for r0, r1, a in self._blocks[L]:
            if r0 <= r <= r1:
                return a[r - r0]
        return None

    def spline(self, L, r):
        key = (L, r)
        if key in self._spl:
            return self._spl[key]
        if len(self._spl) > 80000:
            self._spl.clear()
        row = self._row(L, r)
        if row is not None:
            self._spl[key] = C.USpline(self.u, row)
        else:
            # direct evaluation on the u grid (rare large counts)
            if L not in self._dens:
                y = C.P._y_grid(L)
                lf = C.P.log_density_lnW(L, y) + np.log(C.P.HY)
                self._dens[L] = (y, lf)
            y, lfh = self._dens[L]
            base = lfh + r * y
            ey = np.exp(y)
            vals = np.empty(len(self.u))
            for a in range(0, len(self.u), 32):
                uu = np.exp(self.u[a:a + 32])
                g = base - np.exp(self.u[min(a + 16, len(self.u) - 1)]) * ey
                keep = np.flatnonzero(g > g.max() - 70.0)
                lo, hi = max(keep[0] - 300, 0), min(keep[-1] + 300, len(y) - 1)
                vals[a:a + 32] = logsumexp(base[None, lo:hi + 1] - uu[:, None] * ey[None, lo:hi + 1], axis=1)
            self._spl[key] = C.USpline(self.u, vals)
        return self._spl[key]
        if r <= self.rmax:
            self._spl[key] = C.USpline(self.u, self.tabs[L][r])
        else:
            # direct evaluation on the u grid (rare large counts)
            if L not in self._dens:
                y = C.P._y_grid(L)
                lf = C.P.log_density_lnW(L, y) + np.log(C.P.HY)
                self._dens[L] = (y, lf)
            y, lfh = self._dens[L]
            base = lfh + r * y
            ey = np.exp(y)
            vals = np.empty(len(self.u))
            for a in range(0, len(self.u), 32):
                uu = np.exp(self.u[a:a + 32])
                g = base - np.exp(self.u[min(a + 16, len(self.u) - 1)]) * ey
                keep = np.flatnonzero(g > g.max() - 70.0)
                lo, hi = max(keep[0] - 300, 0), min(keep[-1] + 300, len(y) - 1)
                vals[a:a + 32] = logsumexp(base[None, lo:hi + 1] - uu[:, None] * ey[None, lo:hi + 1], axis=1)
            self._spl[key] = C.USpline(self.u, vals)
        return self._spl[key]


def pmfs_at_scale(lsa, q, Ls, s, R, pin=None):
    """p_b(r; s) = P(Poisson(s q_b W) = r), r = 0..R, from the tables (r <= rmax) else direct.
    pin = (a, w): symbol a has the fixed weight w (plain Poisson with mean s q_a w)."""
    d = len(q)
    lam = s * q
    lu = np.log(lam)
    r = np.arange(R + 1)
    if Ls == 0:
        return np.exp(r[None, :] * lu[:, None] - lam[:, None] - gammaln(r + 1.0)[None, :])
    out = _pmfs_mixed(lsa, q, Ls, s, R, d, lam, lu, r)
    if pin is not None:
        a, w = pin
        la = s * q[a] * w
        out[a] = np.exp(r * np.log(la) - la - gammaln(r + 1.0))
    return out


def _pmfs_mixed(lsa, q, Ls, s, R, d, lam, lu, r):
    out = np.zeros((d, R + 1))
    rt = min(lsa.rmax, R)
    for rr in range(rt + 1):
        out[:, rr] = np.exp(rr * lu + lsa.spline(Ls, rr)(lu) - gammaln(rr + 1.0))
    if R > rt:
        y, lfh = S.lnW_density(Ls)
        W = np.exp(y)
        rr = np.arange(rt + 1, R + 1)
        for b in range(d):
            lw = lam[b] * W
            lp = rr[None, :] * np.log(lw)[:, None] - lw[:, None] - gammaln(rr + 1.0)[None, :]
            out[b, rt + 1:] = np.exp(logsumexp(lp + lfh[:, None], axis=0))
    return out


def _fftconv(a, b, n):
    m = len(a) + len(b) - 1
    nf = 1 << (m - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]
    return np.maximum(c, 0.0)


def conditioned_profile(pm, k):
    """Conditional marginals ptil[b, r] = P(N_b = r | S = k) and P_S(k), for independent N_b ~ pm[b]."""
    d, R1 = pm.shape
    n = k + 1
    conv = _fftconv if k > 64 else (lambda a, b, n: np.convolve(a, b)[:n])
    prefix = [np.zeros(n)]; prefix[0][0] = 1.0
    for i in range(d):
        prefix.append(conv(prefix[-1], pm[i, :n], n))
    suffix = [None] * (d + 2); suffix[d + 1] = np.zeros(n); suffix[d + 1][0] = 1.0
    for i in range(d, 0, -1):
        suffix[i] = conv(suffix[i + 1], pm[i - 1, :n], n)
    PS = prefix[d][k]
    r = np.arange(k + 1)
    ptil = np.empty((d, k + 1))
    for b in range(1, d + 1):
        Pmb = conv(prefix[b - 1], suffix[b + 1], n)
        ptil[b - 1] = pm[b - 1, :n] * Pmb[k - r] / max(PS, 1e-300)
    return ptil, PS


def rate_law(lsa, q, Ls, k, xgrid, xstep):
    """Weights g_k(s) d ln s on x = ln(s/k) over xgrid (only P_s(k) is needed, no costs)."""
    w = np.zeros(len(xgrid))
    for i, x in enumerate(xgrid):
        s = k * np.exp(x)
        pm = pmfs_at_scale(lsa, q, Ls, s, k)
        n = k + 1
        conv = _fftconv if k > 64 else (lambda a, b, n: np.convolve(a, b)[:n])
        tot = np.zeros(n); tot[0] = 1.0
        for b in range(len(q)):
            tot = conv(tot, pm[b, :n], n)
        w[i] = k * tot[k] * xstep
    return w


def rate_window(lsa, q, Ls, k, tail=1e-5, xmax=12.0, xstep=0.1):
    """[x_lo, x_hi] leaving less than `tail` of the rate law on each side, from a wide grid."""
    xg = np.arange(-xmax, xmax + 1e-9, xstep)
    w = rate_law(lsa, q, Ls, k, xg, xstep); c = np.cumsum(w) / w.sum()
    lo = xg[max(int(np.searchsorted(c, tail)) - 1, 0)]; hi = xg[min(int(np.searchsorted(c, 1.0 - tail)) + 1, len(xg) - 1)]
    return float(lo), float(hi), float(w.sum())


def laplace_pieces(lsa, nu_d, k):
    """Per depth: u*, F(u*), F''(u*); mixture weights and the Laplace cost (bits)."""
    M = lsa.Lmax
    us = np.zeros(M); vals = np.zeros(M); fpp = np.zeros(M)
    for L in range(1, M + 1):
        u_, v_, f_ = lsa.laplace_L(L, nu_d, k)
        us[L - 1], vals[L - 1], fpp[L - 1] = u_, v_, f_
    lq = vals + 0.5 * np.log(2 * np.pi / np.maximum(-fpp, 1e-300)) - gammaln(k)
    cost = -(logsumexp(lq) - np.log(M)) / LN2
    w = np.exp(lq - logsumexp(lq))
    return us, vals, fpp, w, cost, lq


def eta_laplace(lsa, nu_d, k, Sigma, rs, us, fpp, w):
    """Second-order term (bits): 1/2 sum_rs Sigma_rs Cov_mu(l_r, l_s) with the Laplace mixture measure."""
    M = lsa.Lmax
    R = len(rs)
    m1 = np.zeros(R)
    quad = 0.0
    V = np.zeros((M, R))
    for L in range(1, M + 1):
        vals = np.array([lsa.spline(L, r)(us[L - 1]) for r in rs])
        d1 = np.array([lsa.spline(L, r)(us[L - 1], 1) for r in rs])
        V[L - 1] = vals
        m1 += w[L - 1] * vals
        quad += w[L - 1] * (d1 @ Sigma @ d1) / max(-fpp[L - 1], 1e-300)
    for L in range(1, M + 1):
        dv = V[L - 1] - m1
        quad += w[L - 1] * (dv @ Sigma @ dv)
    return 0.5 * quad / LN2


def building_block(lsa, q, Ls, k, xhalf=1.6, xstep=0.1, rcap=None, verbose=False, pin=None, xcenter=0.0):
    """Returns dict with U (bits), eta (bits), cost = U - eta, mass, and the scale-law moments.
    pin = (a, w): the weight of symbol a is held fixed at w (pinned building block).
    The rate window is x = ln(s/k) in [xcenter - xhalf, xcenter + xhalf] in steps of xstep."""
    d = len(q)
    if rcap is None:
        rcap = min(k, lsa.rmax)
    xgrid = np.arange(xcenter - xhalf, xcenter + xhalf + 1e-9, xstep)
    rho = np.arange(k + 1, dtype=float)
    tot_w = 0.0; acc_U = 0.0; acc_eta = 0.0; acc_x = 0.0; acc_x2 = 0.0
    for x in xgrid:
        s = k * np.exp(x)
        pm = pmfs_at_scale(lsa, q, Ls, s, k, pin=pin)
        ptil, PS = conditioned_profile(pm, k)
        w = k * PS * xstep
        if w <= 0 or (tot_w > 0 and w < 1e-10 * tot_w and x > 0):
            continue
        nu = ptil.sum(axis=0)
        keep = np.flatnonzero((nu > 1e-12) & (np.arange(k + 1) <= rcap))
        nu_d = {int(r): float(nu[r]) for r in keep}
        rs = [int(r) for r in keep]
        Sig = np.diag(nu[keep]) - ptil[:, keep].T @ ptil[:, keep]
        v = Sig @ rho[keep]
        denom = rho[keep] @ v
        if denom > 1e-12:
            Sig = Sig - np.outer(v, v) / denom
        if k <= KEXACT:
            # small k: exact scale integral and exact (L,u)-measure Hessian
            import lsa_cost2 as C2
            U = lsa.cost_bits(nu_d, k)
            _, cov_mu = C2.grad_hess(lsa, nu_d, k, rs)
            eta = 0.5 * np.sum(Sig * cov_mu) / LN2
        else:
            us, vals, fpp, wL, U, lq = laplace_pieces(lsa, nu_d, k)
            eta = eta_laplace(lsa, nu_d, k, Sig, rs, us, fpp, wL)
        tot_w += w; acc_U += w * U; acc_eta += w * eta; acc_x += w * x; acc_x2 += w * x * x
    U = acc_U / tot_w; eta = acc_eta / tot_w
    mx = acc_x / tot_w
    return dict(k=k, U=U, eta=eta, cost=U - eta, mass=tot_w, x_mean=mx, x_sd=float(np.sqrt(max(acc_x2 / tot_w - mx * mx, 0.0))))


KGRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 700,
         800, 960, 1120, 1280, 1440, 1700, 2000, 2500, 3000, 3500, 4000, 5000]


def bb_grid(d, alpha, Ls, Lmax, kgrid=KGRID, out=None, xhalf=None, xcenter=0.0):
    q = C.zipf(d, alpha)
    lsa = LSAx(Lmax, rmax=max(kgrid) + 5)
    if xhalf is None:
        xhalf = 1.6 if Ls <= 2 else 2.2
    res = []
    for k in kgrid:
        t0 = time.time()
        r = building_block(lsa, q, Ls, k, xhalf=xhalf, xcenter=xcenter)
        r["seconds"] = time.time() - t0
        res.append(r)
        print(f"d={d} a={alpha} Ls={Ls} k={k}: U={r['U']:.3f} eta={r['eta']:.3f} cost={r['cost']:.3f} mass={r['mass']:.4f} xsd={r['x_sd']:.3f} ({r['seconds']:.0f}s)", flush=True)
        if out:
            json.dump(dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, xhalf=xhalf, xcenter=xcenter, rows=res), open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    import lsa_fast  # noqa
    d, alpha, Ls, Lmax = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    kmax = int(sys.argv[5]) if len(sys.argv) > 5 else 700
    bb_grid(d, alpha, Ls, Lmax, kgrid=[k for k in KGRID if k <= kmax], out=f"bb_d{d}_a{alpha}_L{Ls}.json")


# ------------------------------------------------------------------ exit-conditioned building block
def conditioned_profile_exit(pm, e, a):
    """Condition independent N_b on sum_{b != a} N_b = e; symbol a unconditioned.
    Returns ptil (d x R+1), P(sum_{b!=a} = e)."""
    d, R1 = pm.shape
    n = e + 1
    conv = _fftconv if e > 64 else (lambda x, y, n: np.convolve(x, y)[:n])
    idx = [b for b in range(d) if b != a]
    prefix = [np.zeros(n)]; prefix[0][0] = 1.0
    for b in idx:
        prefix.append(conv(prefix[-1], pm[b, :n], n))
    suffix = [None] * (len(idx) + 2); suffix[len(idx) + 1] = np.zeros(n); suffix[len(idx) + 1][0] = 1.0
    for i in range(len(idx), 0, -1):
        suffix[i] = conv(suffix[i + 1], pm[idx[i - 1], :n], n)
    PS = prefix[len(idx)][e]
    r = np.arange(n)
    ptil = np.zeros((d, R1))
    for i, b in enumerate(idx, start=1):
        Pmb = conv(prefix[i - 1], suffix[i + 1], n)
        ptil[b, :n] = pm[b, :n] * Pmb[e - r] / max(PS, 1e-300)
    ptil[a] = pm[a]
    return ptil, PS


def building_block_exit(lsa, q, Ls, e, a, xhalf=1.6, xstep=0.05, verbose=False):
    """Expected LSA cost of the counts of row a when sampling stops after e exits (non-a draws)."""
    d = len(q)
    xgrid = np.arange(-xhalf, xhalf + 1e-9, xstep)
    tot_w = 0.0; acc_U = 0.0; acc_eta = 0.0; acc_K = 0.0
    for x in xgrid:
        s = e * np.exp(x)
        R = int(min(lsa.rmax, e + 12 * s * q[a] + 60))
        pm = pmfs_at_scale(lsa, q, Ls, s, R)
        ptil, PS = conditioned_profile_exit(pm, e, a)
        w = e * PS * xstep
        if w <= 0 or (tot_w > 0 and w < 1e-10 * tot_w and x > 0):
            continue
        nu = ptil.sum(axis=0)
        rho_all = np.arange(R + 1, dtype=float)
        Kbar = float(rho_all @ nu)
        keep = np.flatnonzero(nu > 1e-12)
        nu_d = {int(r): float(nu[r]) for r in keep}
        rs = [int(r) for r in keep]
        # covariance: non-a symbols with the exit-total projection, plus symbol a alone
        pn = ptil[[b for b in range(d) if b != a]][:, keep]
        Sig = np.diag(nu[keep] - ptil[a, keep]) - pn.T @ pn
        v = Sig @ rho_all[keep]; den = rho_all[keep] @ v
        if den > 1e-12:
            Sig = Sig - np.outer(v, v) / den
        pa = ptil[a, keep]
        Sig = Sig + np.diag(pa) - np.outer(pa, pa)
        if Kbar <= KEXACT:
            import lsa_cost2 as C2
            U = lsa.cost_bits(nu_d, Kbar)
            _, cov_mu = C2.grad_hess(lsa, nu_d, Kbar, rs)
            eta = 0.5 * np.sum(Sig * cov_mu) / LN2
        else:
            us, vals, fpp, wL, U, lq = laplace_pieces(lsa, nu_d, Kbar)
            eta = eta_laplace(lsa, nu_d, Kbar, Sig, rs, us, fpp, wL)
        tot_w += w; acc_U += w * U; acc_eta += w * eta; acc_K += w * Kbar
    return dict(e=e, a=a, U=acc_U / tot_w, eta=acc_eta / tot_w, cost=(acc_U - acc_eta) / tot_w, mass=tot_w, Kbar=acc_K / tot_w)

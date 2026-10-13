"""Sampling-free prediction of both costs and the crossing (Step 5): the two-state reduction with one-row inputs.

Given the matrix, the visits to context a follow a two-state chain: stay with probability P_a = mu_a + x_a (own row),
enter from outside with probability e_a; the balance pi_a (1 - P_a) = mu_a (1 - pi_a) + eps_a gives
pi_a = (mu_a + eps_a)/(1 - x_a) and e_a = mu_a + eps_a/(1 - pi_a), where eps_a = sum_{c != a} pi_c x_a^{(c)} collects the
other rows.  The law of x_a is the one-row marginal (Fourier inversion); eps_a is a shifted gamma with the cumulants
kappa_m = sum_{c != a} mu_c^m kappa_m(x_a/(1 - x_c)) (one-row integrals).  For each component (x_a, eps_a):
  lambda = P - e,  E K = m pi + (q_a - pi)(1 - lambda^m)/(1 - lambda),  Var K = m pi(1-pi)(1+lambda)/(1-lambda) - 2 pi(1-pi) lambda (1-lambda^m)/(1-lambda)^2
with m = n - 1 positions for the copier's visits and m = n for the token counts.  Costs as in Section 3:
copier = log2 d + sum_a E B(K_a) + sum_a Cov(E[K_a|T], H_a),  unigram = Laplace cost of the mean profile - fluctuation term.
Usage: python3 sf_predict.py d alpha Ls bbfile n1,n2,... out.json"""
from __future__ import annotations
import sys, time, json
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma as gam
from scipy.interpolate import CubicSpline
import bb, lsa_fast  # noqa
import chain_sim as CS
from onerow import OneRow
from table_mc import BBInterp

LN2 = np.log(2.0)


def two_moment_logpmf(m, V, R):
    """log pmf on 0..R for arrays m, V (shape (N,)): Poisson / negative binomial / binomial (as table_mc.two_moment_pmf)."""
    m = np.asarray(m, float); V = np.asarray(V, float); r = np.arange(R + 1, dtype=float)[None, :]
    lp = np.empty((len(m), R + 1))
    nb = V > m * (1 + 1e-6); bi = V < m * (1 - 1e-6); po = ~(nb | bi)
    if nb.any():
        nu = (m[nb] ** 2 / (V[nb] - m[nb]))[:, None]; pr = (m[nb] / (nu[:, 0] + m[nb]))[:, None]
        lp[nb] = gammaln(r + nu) - gammaln(nu) - gammaln(r + 1.0) + nu * np.log1p(-pr) + r * np.log(pr)
    if bi.any():
        kk = np.maximum(np.round(m[bi] ** 2 / (m[bi] - V[bi])), np.ceil(m[bi]))[:, None]; pp = np.minimum(m[bi][:, None] / kk, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = gammaln(kk + 1.0) - gammaln(r + 1.0) - gammaln(np.maximum(kk - r, 0) + 1.0) + r * np.log(np.maximum(pp, 1e-300)) + (kk - r) * np.log1p(-pp)
        lp[bi] = np.where(r <= kk, v, -np.inf)
    if po.any():
        mm = m[po][:, None]; lp[po] = r * np.log(mm) - mm - gammaln(r + 1.0)
    lp = lp - logsumexp(lp, axis=1, keepdims=True)
    return lp


def mixed_pmf(w, m, V, rmax):
    """sum_k w_k pmf(r; m_k, V_k) on 0..rmax, each component evaluated on its own support m +- 9 sqrt(V) (plus 20)."""
    out = np.zeros(rmax + 1)
    sd = np.sqrt(np.maximum(V, m)); lo = np.maximum(np.floor(m - 7 * sd - 20), 0).astype(int); hi = np.minimum(np.ceil(m + 7 * sd + 20), rmax).astype(int)
    width = hi - lo
    # group components by support width class to vectorize: evaluate on a window of length W = max width in the group
    order = np.argsort(width)
    for start in range(0, len(order), 256):
        idx = order[start:start + 256]; W = int(width[idx].max()) + 1
        r = lo[idx][:, None] + np.arange(W)[None, :]                    # (k, W) absolute counts
        mm = m[idx][:, None]; VV = V[idx][:, None]; rr = r.astype(float)
        nb = VV > mm * (1 + 1e-6); bi = VV < mm * (1 - 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = mm ** 2 / np.where(nb, VV - mm, 1.0); pr = mm / (nu + mm)
            lp_nb = gammaln(rr + nu) - gammaln(nu) - gammaln(rr + 1.0) + nu * np.log1p(-pr) + rr * np.log(pr)
            kk = np.maximum(np.round(mm ** 2 / np.where(bi, mm - VV, 1.0)), np.ceil(mm)); pp = np.minimum(mm / kk, 1.0)
            lp_bi = np.where(rr <= kk, gammaln(kk + 1.0) - gammaln(rr + 1.0) - gammaln(np.maximum(kk - rr, 0) + 1.0) + rr * np.log(np.maximum(pp, 1e-300)) + (kk - rr) * np.log1p(-pp), -np.inf)
            lp_po = rr * np.log(mm) - mm - gammaln(rr + 1.0)
        lp = np.where(nb, lp_nb, np.where(bi, lp_bi, lp_po))
        lp = np.where(r <= rmax, lp, -np.inf)
        p = np.exp(lp - logsumexp(lp, axis=1, keepdims=True)) * w[idx][:, None]
        for k in range(len(idx)):
            out[lo[idx[k]]:lo[idx[k]] + W] += p[k, :min(W, rmax + 1 - lo[idx[k]])]
    return out


def bins_from_density(x, f, NQ=60, ntail=30, tail=0.99):
    """Equal-mass bins (NQ up to mass `tail`, ntail above) of a density on a grid; nodes = bin means."""
    w = f * (x[1] - x[0]); w = w / w.sum(); cw = np.cumsum(w)
    edges = np.concatenate([np.linspace(0, tail, NQ + 1)[:-1], np.linspace(tail, 1.0, ntail + 1)])
    idx = np.searchsorted(cw, edges[1:-1]); idx = np.concatenate([[0], idx, [len(x)]])
    nodes = []; masses = []
    for lo, hi in zip(idx[:-1], idx[1:]):
        if hi <= lo:
            continue
        mass = w[lo:hi].sum()
        if mass <= 0:
            continue
        nodes.append(np.sum(w[lo:hi] * x[lo:hi]) / mass); masses.append(mass)
    masses = np.array(masses); return np.array(nodes), masses / masses.sum()


def gamma_nodes(k1, k2, k3, NQ=40, ntail=20, tail=0.99):
    """Shifted gamma with mean k1, variance k2, third cumulant k3 (nodes = bin means)."""
    sd = np.sqrt(max(k2, 1e-30)); skew = max(k3 / sd ** 3, 0.05)
    k = 4.0 / skew ** 2; theta = sd / np.sqrt(k); shift = k1 - k * theta
    edges = np.concatenate([np.linspace(0, tail, NQ + 1)[:-1], np.linspace(tail, 1.0, ntail + 1)]); edges[-1] = 1 - 1e-10
    qs = gam.ppf(edges, k, scale=theta); w = np.diff(edges)
    Fk1 = gam.cdf(qs, k + 1, scale=theta); Fk = gam.cdf(qs, k, scale=theta)
    nodes = shift + theta * k * np.diff(Fk1) / np.maximum(np.diff(Fk), 1e-300)
    return nodes, w


def eta_laplace_op(lsa, nu_d, k, qf, rs, us, fpp, w):
    """bb.eta_laplace with the covariance given as a quadratic form qf(z) = z^T Sigma z (no K x K matrix)."""
    M = lsa.Lmax; rs = np.asarray(rs, dtype=int); R = len(rs)
    m1 = np.zeros(R); quad = 0.0; V = np.zeros((M, R))
    for L in range(1, M + 1):
        vals = lsa.lnphi(L, rs, [us[L - 1]])[:, 0]
        d1 = lsa.lnphi(L, rs, [us[L - 1]], 1)[:, 0]
        V[L - 1] = vals
        m1 += w[L - 1] * vals
        quad += w[L - 1] * qf(d1) / max(-fpp[L - 1], 1e-300)
    for L in range(1, M + 1):
        dv = V[L - 1] - m1
        quad += w[L - 1] * qf(dv)
    return 0.5 * quad / LN2


class LSAext(lsa_fast.LSAfast):
    """LSAfast with rows beyond the tables (r > 6000) from Laplace's method in y = ln W, with the second-order
    correction: ln phi_r(e^u) = ln int exp(f(y)) dy, f(y) = ln f_L(y) + r y - e^{u+y}; the maximizer solves
    (ln f_L)'(y) + r = e^{u+y}."""

    def __init__(self, Lmax):
        super().__init__(Lmax)
        self._lf = {}

    def _density(self, L):
        if L not in self._lf:
            y = np.arange(-40.0, max(3.0 * L + 30.0, 62.0), 1.0 / 32)
            lf = lsa_fast.C.P.log_density_lnW(L, y)
            self._lf[L] = (CubicSpline(y, lf), y[0], y[-1])
        return self._lf[L]

    def ensure(self, rmax_needed):
        return

    G = 50   # step of the grid of counts on which the Laplace rows are computed (cubic interpolation in between)

    def _direct_row(self, L, r):
        if r <= self.rmax:
            return super()._direct_row(L, r)
        return self._gather(L, np.array([r]), np.arange(self.nu))[0]

    def _gather(self, L, rs, cols):
        """Table values for the counts rs at the columns cols; counts beyond the tables by cubic interpolation in r
        between the cached Laplace rows (only the grid rows are stored: one row per (L, r) would need gigabytes)."""
        rs = np.asarray(rs, dtype=int); cols = np.asarray(cols, dtype=int)
        extra = rs > self.rmax
        if not extra.any():
            return super()._gather(L, rs, cols)
        out = np.empty((len(rs), len(cols)))
        if (~extra).any():
            out[~extra] = super()._gather(L, rs[~extra], cols)
        g = self.G; base = self.rmax + 1; idx = np.flatnonzero(extra); re = rs[extra]
        k = (re - base) // g; t = (re - base - g * k) / g
        for kk in np.unique(k):
            sel = np.flatnonzero(k == kk)
            p0, p1, p2, p3 = [self._laplace_row(L, base + g * (kk + i))[cols] for i in (-1, 0, 1, 2)]
            tt = t[sel][:, None]
            out[idx[sel]] = 0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt * tt + (-p0 + 3 * p1 - 3 * p2 + p3) * tt ** 3)
        return out

    def _laplace_row(self, L, r):
        key = ("lap", L, r)
        if key in self._extra:
            return self._extra[key]
        lf, ylo, yhi = self._density(L); u = self.u
        # maximizer of f(y) = lf(y) + r y - e^{u+y}: f' is decreasing (lf is concave), so bisection on [ylo, yhi]
        a = np.full_like(u, ylo); b = np.full_like(u, yhi)
        for _ in range(50):
            y = 0.5 * (a + b); g = lf(y, 1) + r - np.exp(u + y)
            a = np.where(g > 0, y, a); b = np.where(g > 0, b, y)
        y = 0.5 * (a + b); inside = (y > ylo + 1.0) & (y < yhi - 1.0)
        yc = np.clip(y, ylo, yhi); ey = np.exp(u + yc)
        f0 = lf(yc) + r * yc - ey; f2 = lf(yc, 2) - ey; f3 = lf(yc, 3) - ey; f4 = lf(yc, 4) - ey
        val = f0 + 0.5 * np.log(2 * np.pi / (-f2)) + (f4 / (8 * f2 ** 2) - 5 * f3 ** 2 / (24 * (-f2) ** 3))
        bound = L * gammaln(r + 1.0)                          # ln E[W^r], the value as u -> -infinity
        val = np.where(inside, np.minimum(val, bound), bound)
        self._extra[key] = val
        return val


class SFPredict:
    def __init__(self, d, alpha, Ls, bbfile, ns, verbose=True):
        t0 = time.time()
        self.d, self.alpha, self.Ls, self.ns = d, alpha, Ls, [int(n) for n in ns]
        self.o = OneRow(d, alpha, Ls); o = self.o; self.q = o.q; self.mu = o.mu(); mu = self.mu
        self.S, self.Q2, self.Q3 = o.moments_T()
        k2T = self.Q2 - self.S ** 2; k3T = self.Q3 - 3 * self.S * self.Q2 + 2 * self.S ** 3
        Soff = self.S.copy(); np.fill_diagonal(Soff, 0.0); k2o = k2T.copy(); np.fill_diagonal(k2o, 0.0); k3o = k3T.copy(); np.fill_diagonal(k3o, 0.0)
        self.k1 = mu @ Soff; self.k2 = (mu ** 2) @ k2o; self.k3 = (mu ** 3) @ k3o
        if verbose: print(f"one-row moments: {time.time()-t0:.0f}s", flush=True)
        t0 = time.time(); self.Hbar, self.EHinv = o.entropy_terms()
        self.cov1 = (self.EHinv - (1.0 + np.diag(self.S)) * self.Hbar) / LN2      # Cov(1/(1-x_a), H) in bits
        if verbose: print(f"entropy terms: {time.time()-t0:.0f}s  Hbar = {self.Hbar/LN2:.5f} bits", flush=True)
        t0 = time.time()
        self.xnodes = []; self.xw = []; self.xnodes_u = []; self.xw_u = []
        for a in range(d):
            x, f = o.law_P(a)
            nodes, w = bins_from_density(x, f, 60, 30); P = 1.0 / (1.0 + np.exp(-nodes)); self.xnodes.append(P - mu[a]); self.xw.append(w)
            nodes, w = bins_from_density(x, f, 32, 12); P = 1.0 / (1.0 + np.exp(-nodes)); self.xnodes_u.append(P - mu[a]); self.xw_u.append(w)
        self.enodes = []; self.ew = []; self.enodes_u = []; self.ew_u = []
        for a in range(d):
            nodes, w = gamma_nodes(self.k1[a], self.k2[a], self.k3[a], 40, 20); self.enodes.append(nodes); self.ew.append(w)
            nodes, w = gamma_nodes(self.k1[a], self.k2[a], self.k3[a], 16, 8); self.enodes_u.append(nodes); self.ew_u.append(w)
        if verbose: print(f"laws: {time.time()-t0:.0f}s", flush=True)
        self.BB = BBInterp(bbfile); nmax = max(self.ns); k = np.arange(0, nmax + 1, dtype=float); self.BBk = self.BB(k)
        self.spl = CubicSpline(k[1:], self.BBk[1:])
        self.Lmax = CS.lmax_of(d); self.lsa = None

    def components(self, a, npos, uni=False):
        xn, xw = (self.xnodes_u[a], self.xw_u[a]) if uni else (self.xnodes[a], self.xw[a])
        en, ew = (self.enodes_u[a], self.ew_u[a]) if uni else (self.enodes[a], self.ew[a])
        x = np.repeat(xn, len(en)); eps = np.tile(en, len(xn)); w = np.outer(xw, ew).ravel()
        mu, q = self.mu[a], self.q[a]
        P = np.clip(mu + x, 1e-12, 1 - 1e-12)
        pi = np.clip((mu + eps) / (1.0 - x), 1e-12, 1 - 1e-12)
        e = np.clip(mu + eps / (1.0 - pi), 1e-12, 1 - 1e-12)
        lam = P - e; geo = (1.0 - lam ** npos) / (1.0 - lam)
        m = npos * pi + (q - pi) * geo
        V = npos * pi * (1 - pi) * (1 + lam) / (1 - lam) - 2 * pi * (1 - pi) * lam * geo / (1 - lam) ** 2
        return w, np.maximum(m, 1e-9), np.maximum(V, 1e-9)

    def EB(self, m, V, n, mcut=64):
        """E B(K) for two-moment laws with means m and variances V (arrays)."""
        V = np.maximum(V, 0.2 * m)
        out = np.empty_like(m)
        big = m >= mcut
        out[big] = self.spl(m[big]) + 0.5 * V[big] * self.spl(m[big], 2)
        sm = ~big
        if sm.any():
            R = int(min(n, mcut + 10 * np.sqrt(max(V[sm].max(), mcut)) + 20))
            lp = two_moment_logpmf(m[sm], V[sm], R)
            out[sm] = np.exp(lp) @ self.BBk[:R + 1]
        return out

    def scale(self, npos, uni=False):
        """Common factor that makes the model's mean counts sum to the number of positions (they fall short by ~0.01%)."""
        tot = sum(float(w @ m) for w, m, V in (self.components(a, npos, uni=uni) for a in range(self.d)))
        return npos / tot

    def copier(self, n):
        d = self.d; tot = np.log2(d); cov = 0.0; EK = np.zeros(d); VA = np.zeros(d)
        sc = self.scale(n - 1)
        for a in range(d):
            w, m, V = self.components(a, n - 1); m = m * sc; V = V * sc
            tot += float(w @ self.EB(m, V, n))
            EK[a] = w @ m; VA[a] = w @ (m * m) - EK[a] ** 2
            cov += (n - 1) * (self.mu[a] + self.k1[a]) * self.cov1[a]
        return tot, cov, EK, VA

    def unigram(self, n):
        if self.lsa is None:
            self.lsa = LSAext(self.Lmax)
        d = self.d
        sc = self.scale(n, uni=True)
        comps = [(w, m * sc, V * sc) for w, m, V in (self.components(b, n, uni=True) for b in range(d))]
        rmax = int(min(n, max(np.max(m + 10 * np.sqrt(np.maximum(V, m)) + 20) for _, m, V in comps)))
        self.lsa.ensure(rmax)
        pm = np.zeros((d, rmax + 1))
        for b in range(d):
            w, m, V = comps[b]; V = np.maximum(V, 0.2 * m)
            pm[b] = mixed_pmf(w, m, V, rmax)
        nu = pm.sum(axis=0); keep = np.flatnonzero(nu > 1e-9); nu_d = {int(r): float(nu[r]) for r in keep}; rs = [int(r) for r in keep]
        us, vals, fpp, wL, U, lq = bb.laplace_pieces(self.lsa, nu_d, n)
        # covariance of the profile, Sigma = diag(nu) - pm^T pm, conditioned on the total count (rank-one correction),
        # used only through quadratic forms z^T Sigma z (the K x K matrix itself would need gigabytes at d = 512)
        pk = pm[:, keep]; nk = nu[keep]
        rho = np.arange(rmax + 1, dtype=float)[keep]; v = nk * rho - pk.T @ (pk @ rho); den = float(rho @ v)

        def qf(z):
            val = float(nk @ (z * z)) - float(np.sum((pk @ z) ** 2))
            if den > 1e-12:
                val -= float(v @ z) ** 2 / den
            return val
        eta = eta_laplace_op(self.lsa, nu_d, n, qf, rs, us, fpp, wL)
        return float(U) - float(eta)


def crossing(ns, gap):
    ns = np.asarray(ns, float); g = np.asarray(gap, float)
    if g[-1] <= 0:
        return None
    i = max(j for j in range(len(g)) if g[j] <= 0) if np.any(g <= 0) else None
    if i is None or i == len(g) - 1:
        return None
    return float(ns[i] + (0 - g[i]) * (ns[i + 1] - ns[i]) / (g[i + 1] - g[i]))


if __name__ == "__main__":
    d, alpha, Ls = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    bbfile = sys.argv[4]; ns = [int(a) for a in sys.argv[5].split(",")]; out = sys.argv[6]
    t0 = time.time()
    sf = SFPredict(d, alpha, Ls, bbfile, ns)
    rows = []
    for n in ns:
        t1 = time.time()
        C, cov, EK, VA = sf.copier(n)
        t2 = time.time()
        F = sf.unigram(n)
        rows.append(dict(n=n, C=C + cov, C_bb=C, cov=cov, F=F, gap=F - C - cov, EK_top=EK[:5].tolist(), VA_top=VA[:5].tolist()))
        print(f"n={n:6d}: copier {C+cov:10.1f} (bb {C:10.1f}, cov {cov:7.1f}) [{t2-t1:.0f}s] | unigram {F:10.1f} [{time.time()-t2:.0f}s] | gap {F-C-cov:+9.1f}", flush=True)
    x = crossing(ns, [r["gap"] for r in rows])
    res = dict(d=d, alpha=alpha, Ls=Ls, ns=ns, rows=rows, x_sf=x, Hbar_bits=sf.Hbar / LN2, seconds=time.time() - t0)
    json.dump(res, open(out, "w"), indent=1)
    print(f"crossing (sampling-free): {x}   total {time.time()-t0:.0f}s")

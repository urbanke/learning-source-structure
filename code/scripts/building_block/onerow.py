"""One-row expectations of the tilted product-exponential row P_b = q_b W_b / Z, Z = sum_c q_c W_c, W iid ~ W_{Ls},
computed without sampling (Step 5).  Notation: x_a = P_a - mu_a, y_a = x_a/(1 - x_a), Y_i = (1 + mu_i) Z - q_i W_i.

  mu_a       = E P_a = q_a int_0^inf phi_1(s q_a) prod_{c != a} phi_0(s q_c) ds
  S[i,a]     = E[x_a/(1 - x_i)] = int_0^inf E[(q_a W_a - mu_a Z) e^{-u Y_i}] du
  Q2[i,a]    = E[(x_a/(1 - x_i))^2] = int_0^inf u E[(q_a W_a - mu_a Z)^2 e^{-u Y_i}] du
  law of P_a : density of the logit ln(q_a W_a) - ln Z_{-a} by Fourier inversion, with
               E[(q_a W_a)^{i w}] = q_a^{i w} Gamma(1 + i w)^Ls and
               E[Z_{-a}^{-i w}] = [1 + i w int e^{i w t}(G_{-a}(e^t) - 1_{t<0}) dt] / Gamma(1 + i w)
  E[y_a H(P)]: two-dimensional integrals (gamma trick for 1/(Z Y_a) and Frullani for ln Z).
All integrals are quadratures on log grids of products of tabulated functions phi_r(t) = E[W^r e^{-tW}], r = 0, 1, 2,
and psi_r(t) = E[W^r ln W e^{-tW}], r = 1, 2, obtained from the density of ln W_{Ls}."""
from __future__ import annotations
import numpy as np
from scipy.special import loggamma, logsumexp
from scipy.interpolate import CubicSpline
import lsa_cost as C
import source_profiles as SP


class OneRow:
    def __init__(self, d, alpha, Ls, tmin=-16.0, tmax=14.0, ht=0.02):
        self.d, self.alpha, self.Ls = d, alpha, Ls
        self.q = C.zipf(d, alpha)
        y, lfh = SP.lnW_density(Ls, ymin=-45.0, h=1.0 / 32)
        W = np.exp(y)
        self.lt = np.arange(tmin, tmax + ht / 2, ht)          # ln t grid for the tabulated functions
        t = np.exp(self.lt)
        # ln phi_r(t) for r = 0,1,2 and psi_r(t) = E[W^r ln W e^{-tW}] (split by the sign of ln W)
        lphi = []
        for r in range(4):
            lphi.append(logsumexp(lfh[:, None] + r * y[:, None] - t[None, :] * W[:, None], axis=0))
        self.lphi = [CubicSpline(self.lt, v) for v in lphi]
        pos = y > 0
        psi = []
        for r in (1, 2):
            lp = logsumexp(lfh[pos, None] + r * y[pos, None] + np.log(y[pos])[:, None] - t[None, :] * W[pos, None], axis=0)
            ln_ = logsumexp(lfh[~pos, None] + r * y[~pos, None] + np.log(-y[~pos])[:, None] - t[None, :] * W[~pos, None], axis=0)
            psi.append(np.exp(lp) - np.exp(ln_))
        self.psi = [CubicSpline(self.lt, v) for v in psi]
        self.EW = float(np.sum(np.exp(lfh + y)))               # = 1
        self._mu = None

    # ---- tabulated functions at arbitrary arguments (arrays) ------------------------------------------------
    def lphi_r(self, r, t):
        lt = np.log(np.maximum(t, 1e-300))
        out = self.lphi[r](np.clip(lt, self.lt[0], self.lt[-1]))
        small = lt < self.lt[0]
        if np.any(small):                                       # phi_r(t) -> E W^r as t -> 0
            out = np.where(small, self.lphi[r](self.lt[0]), out)
        return out

    def g(self, r, t):
        """phi_r(t)/phi_0(t)"""
        return np.exp(self.lphi_r(r, t) - self.lphi_r(0, t))

    def psi_r(self, r, t):
        lt = np.log(np.maximum(t, 1e-300))
        return self.psi[r - 1](np.clip(lt, self.lt[0], self.lt[-1]))

    # ---- mean row ----------------------------------------------------------------------------------------------
    def lnG(self, s):
        """ln prod_c phi_0(s q_c) for an array of s"""
        return np.sum(self.lphi_r(0, np.outer(s, self.q)), axis=1)

    def mu(self):
        if self._mu is None:
            v = np.arange(-14.0, 12.0, 0.01); s = np.exp(v)
            lG = self.lnG(s)
            g1 = self.g(1, np.outer(s, self.q))                  # (ns, d)
            self._mu = self.q * np.sum(np.exp(lG)[:, None] * g1 * s[:, None] * 0.01, axis=0)
        return self._mu

    # ---- S, second and third moments of T_{ia} = x_a/(1 - x_i): for each i, integrate over u ------------------
    def moments_T(self, hv=0.02, vmin=-14.0, vmax=12.0):
        """Returns (S, Q2, Q3) with S[i,a] = E[T_ia], Q2[i,a] = E[T_ia^2], Q3[i,a] = E[T_ia^3],
        T_ia = (q_a W_a - mu_a Z)/Y_i, 1/Y^m = int_0^inf u^{m-1}/(m-1)! e^{-uY} du.  Under the tilt e^{-uY_i} the
        variables q_c W_c are independent with raw moments m_r(c) = q_c^r phi_r(t_c)/phi_0(t_c)."""
        d, q, mu = self.d, self.q, self.mu()
        v = np.arange(vmin, vmax, hv); u = np.exp(v); w = u * hv
        S = np.zeros((d, d)); Q2 = np.zeros((d, d)); Q3 = np.zeros((d, d))
        for i in range(d):
            targ = np.outer(u, (1.0 + mu[i]) * q); targ[:, i] = u * mu[i] * q[i]
            l0 = self.lphi_r(0, targ); F = np.exp(np.sum(l0, axis=1))
            m1 = q[None, :] * np.exp(self.lphi_r(1, targ) - l0)
            m2 = (q[None, :] ** 2) * np.exp(self.lphi_r(2, targ) - l0)
            m3 = (q[None, :] ** 3) * np.exp(self.lphi_r(3, targ) - l0)
            # cumulants of Z = sum_c q_c W_c under the tilt, and of R = Z - q_a W_a for each a
            k1 = m1.sum(axis=1); k2 = (m2 - m1 ** 2).sum(axis=1); k3 = (m3 - 3 * m2 * m1 + 2 * m1 ** 3).sum(axis=1)
            R1 = k1[:, None] - m1; R2c = k2[:, None] - (m2 - m1 ** 2); R3c = k3[:, None] - (m3 - 3 * m2 * m1 + 2 * m1 ** 3)
            ER2 = R2c + R1 ** 2; ER3 = R3c + 3 * R2c * R1 + R1 ** 3
            A1, A2, A3 = m1, m2, m3                                      # moments of A = q_a W_a
            ca = (1.0 - mu)[None, :]; cb = (-mu)[None, :]                 # T Y = ca A + cb R
            E1 = ca * A1 + cb * R1
            E2 = ca ** 2 * A2 + 2 * ca * cb * A1 * R1 + cb ** 2 * ER2
            E3 = ca ** 3 * A3 + 3 * ca ** 2 * cb * A2 * R1 + 3 * ca * cb ** 2 * A1 * ER2 + cb ** 3 * ER3
            S[i] = np.sum((F * w)[:, None] * E1, axis=0)
            Q2[i] = np.sum((F * w * u)[:, None] * E2, axis=0)
            Q3[i] = np.sum((F * w * u ** 2 / 2.0)[:, None] * E3, axis=0)
        return S, Q2, Q3

    def S_Q2(self, **kw):
        S, Q2, Q3 = self.moments_T(**kw); return S, Q2

    # ---- entropy: Hbar = E H(P) and E[H(P)/(1 - x_a)] (nats) -------------------------------------------------
    def entropy_terms(self, hv=0.02, vmin=-14.0, vmax=12.0):
        """Returns (Hbar, EHinv) in nats: Hbar = E H(P), EHinv[a] = E[H(P) Z/Y_a] = E[H(P)/(1 - x_a)].
        H(P) = ln Z - (1/Z) sum_b q_b W_b ln(q_b W_b).  With V(v) := sum_b q_b [ln q_b g_1(b; v q_b) + psi_1(v q_b)/phi_0(v q_b)]:
          E H = int (e^{-s} - G(s))/s ds - int G(s) V(s) ds
          E[(sum_b q_b W_b ln q_b W_b)/Y_a] = int du F_a(u) V_a(u)           (1D, V_a with the a-th argument mu_a u q_a)
          E[Z ln Z / Y_a] = int du int ds/s [e^{-s} K_a(0,u) - K_a(s,u)],  K_a(s,u) = E[Z e^{-sZ - uY_a}]
        where K_a(s,u) = G(v) phi_0(t_a)/phi_0(v q_a) [Z1(v) - q_a g_1(a; v q_a) + q_a g_1(a; t_a)], v = s + (1+mu_a) u,
        t_a = (s + mu_a u) q_a, and Z1(v) = sum_b q_b g_1(b; v q_b)."""
        d, q, mu = self.d, self.q, self.mu()
        vv = np.arange(vmin - 1.0, vmax + 2.0, hv); sg = np.exp(vv)                 # single-variable functions on a log grid
        targ = np.outer(sg, q); l0 = self.lphi_r(0, targ); l1 = self.lphi_r(1, targ)
        lnG = np.sum(l0, axis=1); g1 = np.exp(l1 - l0)
        Z1 = g1 @ q
        V = (g1 * (q * np.log(q))[None, :]).sum(axis=1) + (self.psi_r(1, targ) / np.exp(l0) * q[None, :]).sum(axis=1)
        lnG_i = CubicSpline(vv, lnG); Z1_i = CubicSpline(vv, Z1); V_i = CubicSpline(vv, V)
        # Hbar
        s = np.exp(np.arange(vmin, vmax, hv)); ds = s * hv
        Hbar = np.sum((np.exp(-s) - np.exp(lnG_i(np.log(s)))) / s * ds) - np.sum(np.exp(lnG_i(np.log(s))) * V_i(np.log(s)) * ds)
        # per-a terms
        EHinv = np.zeros(d)
        u = np.exp(np.arange(vmin, vmax, hv)); du = u * hv
        sgrid = np.exp(np.arange(vmin, vmax, hv)); dsg = sgrid * hv
        for a in range(d):
            # term B: int du F_a(u) V_a(u), F_a(u) = G(v) phi_0(t_a)/phi_0(v q_a), v = (1+mu_a) u, t_a = mu_a u q_a
            v0 = (1.0 + mu[a]) * u; ta0 = mu[a] * u * q[a]
            lF0 = lnG_i(np.log(v0)) + self.lphi_r(0, ta0) - self.lphi_r(0, v0 * q[a])
            Va = V_i(np.log(v0)) - q[a] * (np.log(q[a]) * self.g(1, v0 * q[a]) + self.psi_r(1, v0 * q[a]) / np.exp(self.lphi_r(0, v0 * q[a]))) \
                 + q[a] * (np.log(q[a]) * self.g(1, ta0) + self.psi_r(1, ta0) / np.exp(self.lphi_r(0, ta0)))
            termB = np.sum(np.exp(lF0) * Va * du)
            # term A: 2D
            K0 = np.exp(lF0) * (Z1_i(np.log(v0)) - q[a] * self.g(1, v0 * q[a]) + q[a] * self.g(1, ta0))       # K_a(0,u)
            termA = 0.0
            for sv, dsv in zip(sgrid, dsg):
                v = sv + v0; ta = (sv + mu[a] * u) * q[a]
                lF = lnG_i(np.log(v)) + self.lphi_r(0, ta) - self.lphi_r(0, v * q[a])
                K = np.exp(lF) * (Z1_i(np.log(v)) - q[a] * self.g(1, v * q[a]) + q[a] * self.g(1, ta))
                termA += np.sum((np.exp(-sv) * K0 - K) * du) * dsv / sv
            EHinv[a] = termA - termB
        return Hbar, EHinv

    # ---- law of one entry --------------------------------------------------------------------------------------
    def law_P(self, a, ht=0.02, tmin=-40.0, tmax=30.0, c=0.3):
        """Density of the logit l = ln(q_a W_a) - ln Z_{-a} of P_a on a grid; returns (xgrid, density).
        Uses the tilted transform E[e^{(c + i w) l}] = q_a^{c+iw} Gamma(1+c+iw)^Ls M_c(w)/Gamma(c+iw) with
        M_c(w) = int e^{(c+iw) t} G_{-a}(e^t) dt (a smooth integrand, trapezoidal rule), and inverts it."""
        q, Ls = self.q, self.Ls
        t = np.arange(tmin, tmax, ht); s = np.exp(t)
        lG = np.sum(self.lphi_r(0, np.outer(s, np.delete(q, a))), axis=1)
        h = np.exp(lG + c * t)                                  # e^{c t} G_{-a}(e^t)
        N = len(t); om = 2 * np.pi * np.fft.fftfreq(N, d=ht)
        M = ht * np.exp(1j * om * t[0]) * np.fft.ifft(h) * N     # int e^{i w t} h(t) dt
        z = c + 1j * om
        phi = np.exp(z * np.log(q[a]) + Ls * loggamma(1 + z) - loggamma(z)) * M   # E[e^{z l}]
        dom = om[1] - om[0]
        f = np.real(np.fft.fft(phi * np.exp(-1j * om * t[0])) * dom / (2 * np.pi)) * np.exp(-c * t)
        return t, np.maximum(f, 0.0)

    def P_nodes(self, a, **kw):
        """Bins of the law of P_a: returns (P values at bin means, weights) from the logit density."""
        x, f = self.law_P(a, **kw)
        w = f * (x[1] - x[0]); w = w / w.sum()
        P = 1.0 / (1.0 + np.exp(-x))
        return P, w

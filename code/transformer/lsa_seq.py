"""Sequential evaluation of the two mechanisms of Section 2.2 on a token sequence: the per-position loss (bits) of the
unigram predictor and of the copier, their predictive distributions, and the learner's posterior weight on copying.

Both mechanisms use the LSA code of Appendix A (uniform mixture over coding depths 1..Lmax). For a table with counts
m (total k) the code probability at depth L is
    Q_L(m) = (1/Gamma(k)) int exp(k u) prod_b phi_{L, m_b}(e^u) du,      phi_{L,r}(s) = E[W_L^r e^{-s W_L}],
and Q(m) = (1/Lmax) sum_L Q_L(m). The predictive probability of a symbol b with current count r is Q(m+e_b)/Q(m).
We keep, per table and depth, the log-integrand A_L(u) = k u + sum_b ln phi_{L,m_b}(e^u) on a fixed u grid (step
1/64), and update it by one table row when a symbol is observed:  A_L += u + ln phi_{L,r+1} - ln phi_{L,r}.  The
integral is a plain sum on the grid; for the peaked integrand this rectangle rule is accurate to far better than
1e-6 bits at every k up to a few thousand (its error decays like exp(-2 pi^2 (sigma/h)^2) with sigma ~ 1/sqrt(k)).
The grid is restricted to u in [u_lo, u_hi], which covers the peak at u* ~ ln(k/d) for d up to 512 and k up to 2048
with a margin of 12 on the left (where the integrand decays like e^u) and 5 on the right (double-exponential decay).

Requires the ln phi rows: the tables phitab_L*_r*_full.npy with lsa_fast.py, or the anchor store (see phi_source.py).
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_source import open_tables

LN2 = np.log(2.0)
EULER = 0.57721566490153286


def lmax_of(d):
    """The depth cap used throughout the paper: round(2 ln d / (1 - gamma))."""
    return int(round(2.0 * np.log(d) / (1.0 - EULER)))


class SeqLSA:
    def __init__(self, d, Lmax=None, u_lo=-19.0, u_hi=10.0, rmax=1100, table_dir=".", depths=None):
        """depths: the coding depths of the mixture (default 1..Lmax, the paper's predictor); a single depth [L] gives
        the predictor whose prior sits at that depth."""
        self.d = int(d)
        self.Lmax = int(Lmax if Lmax is not None else lmax_of(d))
        self.depths = list(range(1, self.Lmax + 1)) if depths is None else [int(L) for L in depths]
        lsa = open_tables(max(max(self.depths), 1), table_dir, u_lo=u_lo, u_hi=u_hi)
        cols = np.flatnonzero((lsa.u >= u_lo - 1e-9) & (lsa.u <= u_hi + 1e-9))
        self.u = lsa.u[cols].copy()
        self.h = float(lsa.h)
        self.rmax = int(rmax)
        rows = np.arange(self.rmax + 2)
        LN = np.stack([lsa.rows(L, rows)[:, cols] for L in self.depths])                  # (n_depths, rmax+2, nu)
        self.LN0 = np.ascontiguousarray(LN[:, 0, :])                                      # ln phi_{L,0}
        D = LN[:, 1:, :] - LN[:, :-1, :] + self.u[None, None, :]                         # u + ln phi_{r+1} - ln phi_r
        self.D = np.ascontiguousarray(np.transpose(D, (1, 0, 2)))                         # D[r] is (Lmax, nu), r = 0..rmax
        del LN, D
        self.lgam = gammaln(np.arange(0, self.rmax + 4, dtype=float).clip(1e-300))       # ln Gamma(k), k>=1 used

    # ---- one table ------------------------------------------------------------------------------------------
    def new_table(self):
        return _Table(self)

    # ---- whole sequence, both mechanisms ------------------------------------------------------------------------
    def run(self, x, full_probs=False, positions=None):
        """x: token sequence (ints in 0..d-1). Returns per-position losses in bits of the unigram (lF) and of the copier
        (lC), the learner's weight on copying before each position (wC), and, if full_probs, the predictive
        distributions pF, pC (n x d) at the positions given (default all)."""
        x = np.asarray(x, dtype=int); n = len(x); d = self.d
        lF = np.empty(n); lC = np.empty(n)
        want = np.zeros(n, dtype=bool)
        if full_probs:
            want[np.arange(n) if positions is None else np.asarray(positions)] = True
            pF = np.zeros((int(want.sum()), d)); pC = np.zeros((int(want.sum()), d)); j = 0
        uni = _Table(self)
        tabs = [None] * d
        for t in range(n):
            b = int(x[t])
            if want[t]:
                pF[j] = uni.predictive()
            lF[t] = uni.observe(b)
            if t == 0:
                if want[t]:
                    pC[j] = 1.0 / d
                lC[t] = np.log2(d)          # the copier pays log2 d for the first token (no context)
            else:
                a = int(x[t - 1])
                if tabs[a] is None:
                    tabs[a] = _Table(self)
                if want[t]:
                    pC[j] = tabs[a].predictive()
                lC[t] = tabs[a].observe(b)
            if want[t]:
                j += 1
        gap = np.cumsum(lF) - np.cumsum(lC)                 # Lambda(t) = l_F(x^t) - l_C(x^t)
        wC = np.empty(n); wC[0] = 0.5; wC[1:] = 1.0 / (1.0 + 2.0 ** (-gap[:-1]))   # weight before position t
        out = dict(lF=lF, lC=lC, wC=wC)
        if full_probs:
            out["pF"] = pF; out["pC"] = pC; out["positions"] = np.flatnonzero(want)
        return out


def _lse_rows(A, drop=40.0):
    """log sum exp along the last axis of a 2-d array (no scipy overhead). Only the columns where some row is within
    `drop` nats of its maximum are exponentiated; the rest contribute less than e^-40 relative and are skipped."""
    m = A.max(axis=1)
    keep = (A > (m[:, None] - drop)).any(axis=0)
    lo = int(np.argmax(keep)); hi = len(keep) - int(np.argmax(keep[::-1]))
    return np.log(np.exp(A[:, lo:hi] - m[:, None]).sum(axis=1)) + m


def _lse(v):
    m = v.max()
    return np.log(np.exp(v - m).sum()) + m


class _Table:
    """One count table with its log-integrands A_L(u) and the current log Q_L."""
    __slots__ = ("S", "A", "k", "m", "logQ")

    def __init__(self, S: SeqLSA):
        self.S = S
        self.A = S.d * S.LN0.copy()              # all d symbols at count 0
        self.k = 0
        self.m = np.zeros(S.d, dtype=int)
        self.logQ = None                         # log Q_L(m) per depth (k >= 1)

    def _logQ_after(self, r):
        """log Q_L(m + e_b) per depth for a symbol b with current count r (array over depths)."""
        S = self.S
        if r >= S.rmax:
            raise ValueError(f"count {r} exceeds the table range {S.rmax}")
        return _lse_rows(self.A + S.D[r]) + np.log(S.h) - S.lgam[self.k + 1]

    def predictive(self):
        """The predictive distribution over the d symbols."""
        S = self.S
        if self.k == 0:
            return np.full(S.d, 1.0 / S.d)
        p = np.empty(S.d)
        base = _lse(self.logQ)
        for r in np.unique(self.m):
            lq = _lse(self._logQ_after(int(r))) - base
            p[self.m == r] = np.exp(lq)
        return p

    def observe(self, b):
        """Observe symbol b: returns its loss in bits under the table's predictive distribution, then updates."""
        S = self.S; r = int(self.m[b])
        new = self._logQ_after(r)
        if self.k == 0:
            loss = np.log2(S.d)
        else:
            loss = -(_lse(new) - _lse(self.logQ)) / LN2
        self.A += S.D[r]
        self.k += 1; self.m[b] += 1; self.logQ = new
        return float(loss)

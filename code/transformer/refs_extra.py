"""Four further reference predictors, all built from the LSA code of the paper (same tables, same code integral), to
locate what a trained transformer does relative to the two mechanisms:

  joint     the joint-pair copier: one LSA table over the d^2 pairs (x_t, x_{t+1}); the next token given the context a
            is predicted by conditioning the pair code on a, p(b | a) ~ g(M_ab) with g the pair code's predictive
            probability of a cell seen M_ab times. Exchangeable over cells: it learns how skewed the rows are from the
            profile of the whole matrix, but nothing about columns.
  depth     the copier of the paper with its prior at the true source depth L_s instead of the depth mixture.
  backoff   the copier whose row prior is centred on the running frequency estimate of the sequence: the LSA row code
            with per-symbol weights q_b, Q_L(m | q) = (1/Gamma(k)) int exp(k u) prod_b q_b^{m_b} phi_{L,m_b}(q_b e^u) du,
            where q is the unigram predictor's current predictive distribution (refreshed at every visit of the row).
  oracle    the same with the true base law q of the source (relabelled like the sequence).
  zipf      the product-model copier: the same with the base law estimated from the token counts so far, knowing only
            that it is a Zipf law with an unknown exponent in [0.5, 1.5] and an unknown labelling (symbols ranked by
            count, ties share their block of ranks, the exponent integrated out). This is what a predictor can know
            that has learned the family and the product structure, but not the base law itself.

Usage: python3 refs_extra.py --d 32 --settings "1,2" --refs refs/d32 --out refs/d32 --table_dir <tables>
Writes extra_<tag>.npz with the per-position losses lJ, lD, lB, lO (bits) for the first n_seq sequences of the
evaluation set (the same sequences as the references).
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_sets import eval_set, md5_of, parse_settings, tag_of
from source import zipf

LN2 = np.log(2.0); H = 1.0 / 64


class FullTables:
    """ln phi_{L,r}(e^u) on the full u grid of the tables for the given depths (float32), plus the window used for
    the integrals and the shifted-lookup helper for per-symbol weights."""
    def __init__(self, depths, rmax, table_dir, u_lo=-19.0, u_hi=10.0):
        from phi_source import open_tables
        self.depths = list(depths)
        lsa = open_tables(max(depths), table_dir, u_lo=-45.0, u_hi=u_hi + 2 * H)   # the shifted lookups need u below the window, never above it
        self.ufull = lsa.u.copy(); self.h = float(lsa.h); self.rmax = int(rmax)
        rows = np.arange(rmax + 2)
        self.LN = np.stack([lsa.rows(L, rows) for L in depths]).astype(np.float32)        # (nL, rmax+2, nu_full)
        self.c0 = int(np.flatnonzero(self.ufull >= u_lo - 1e-9)[0]); self.c1 = int(np.flatnonzero(self.ufull <= u_hi + 1e-9)[-1]) + 1
        self.u = self.ufull[self.c0:self.c1].astype(np.float64)                               # the window
        self.lgam = gammaln(np.arange(0, rmax + 4, dtype=float).clip(1e-300))

    def shifts(self, q):
        """Integer and fractional grid shifts for the weights q: ln q_b = (j_b + f_b) h."""
        delta = np.log(np.clip(q, 1e-12, None))
        j = np.floor(delta / self.h).astype(int); f = delta / self.h - j
        return delta, j, f

    def row_terms(self, r, j, f):
        """ln phi_{L,r}(q_b e^u) on the window for one symbol with count r and shift (j, f): (nL, nu)."""
        a = self.c0 + j; b = self.c1 + j
        return (1.0 - f) * self.LN[:, r, a:b] + f * self.LN[:, r, a + 1:b + 1]


def _lse_rows(A, drop=40.0):
    m = A.max(axis=1)
    keep = (A > (m[:, None] - drop)).any(axis=0)
    lo = int(np.argmax(keep)); hi = len(keep) - int(np.argmax(keep[::-1]))
    return np.log(np.exp(A[:, lo:hi] - m[:, None]).sum(axis=1)) + m


def _lse(v):
    m = v.max(); return np.log(np.exp(v - m).sum()) + m


class WeightedRow:
    """One row of the copier with per-symbol weights q (fixed for the oracle; refreshed by the caller for the backoff)."""
    def __init__(self, T: FullTables, d):
        self.T = T; self.d = d; self.k = 0; self.m = np.zeros(d, dtype=int); self.A = None; self.logQ = None; self.q_id = None

    def rebuild(self, q, q_id):
        """A_L(u) = k u + sum_b [m_b ln q_b + ln phi_{L,m_b}(q_b e^u)] for the weights q."""
        T = self.T; delta, j, f = T.shifts(q)
        A = self.k * T.u[None, :] + np.zeros((len(T.depths), len(T.u)))
        for b in range(self.d):
            A += self.m[b] * delta[b] + T.row_terms(int(self.m[b]), int(j[b]), float(f[b]))
        self.A = A; self.delta, self.j, self.f = delta, j, f; self.q_id = q_id
        self.logQ = _lse_rows(self.A) + np.log(T.h) - T.lgam[max(self.k, 1)] if self.k >= 1 else None

    def _after(self, b):
        T = self.T; r = int(self.m[b])
        add = T.u + self.delta[b] + T.row_terms(r + 1, int(self.j[b]), float(self.f[b])) - T.row_terms(r, int(self.j[b]), float(self.f[b]))
        return add, _lse_rows(self.A + add) + np.log(T.h) - T.lgam[self.k + 1]

    def _base(self):
        """log of the code probability of the current counts, mixture over depths: log Q(m | w); Q(empty) = 1."""
        return np.log(len(self.T.depths)) if self.k == 0 else _lse(self.logQ)

    def observe(self, b):
        """Observe symbol b: its loss in bits under the predictive distribution Q(m + e_b | w) / Q(m | w), then the
        update. For an empty table (k = 0) the predictive is the prior's: E[w_b W_b / sum_c w_c W_c], which is the base
        law smoothed by the prior (uniform only for uniform weights)."""
        add, new = self._after(b)
        loss = -(_lse(new) - self._base()) / LN2
        self.A += add; self.k += 1; self.m[b] += 1; self.logQ = new
        return float(loss)

    def predictive(self):
        """The predictive distribution over the d symbols (normalised)."""
        base = self._base()
        p = np.array([np.exp(_lse(self._after(b)[1]) - base) for b in range(self.d)])
        return p / p.sum()


ALPHAS = np.linspace(0.5, 1.5, 21)


def zipf_estimate(counts):
    """Estimate of the base law from the token counts, knowing only that it is a Zipf law with an exponent in
    [0.5, 1.5] and an unknown labelling: symbols are ranked by their counts, tied symbols share a block of ranks and
    get the mean Zipf mass of the block, and the exponent is integrated out with a uniform prior on the grid ALPHAS
    (posterior weights from the multinomial likelihood of the counts)."""
    d = len(counts); order = np.argsort(-counts, kind="stable"); sorted_counts = counts[order]
    ranks = np.arange(1, d + 1, dtype=float)
    # blocks of tied counts, in rank order
    blocks = []; i = 0
    while i < d:
        j = i
        while j + 1 < d and sorted_counts[j + 1] == sorted_counts[i]:
            j += 1
        blocks.append((i, j + 1)); i = j + 1
    qhat = np.zeros(d); loglik = np.empty(len(ALPHAS)); qs = []
    for k, a in enumerate(ALPHAS):
        q = ranks ** (-a); q /= q.sum()
        qb = np.empty(d)
        for lo, hi in blocks:
            qb[lo:hi] = q[lo:hi].mean()
        qs.append(qb); loglik[k] = float((sorted_counts * np.log(qb)).sum())
    w = np.exp(loglik - loglik.max()); w /= w.sum()
    qsorted = sum(wk * qk for wk, qk in zip(w, qs))
    qhat[order] = qsorted
    return qhat / qhat.sum()


def run_extra(x, q_true, S_pairs, S_depth, T_full, S_uni, which="ZO", positions=None, refresh=16, warm=64):
    """Per-position losses (bits) of the reference predictors on one sequence x (the whole sequence; position t
    predicts x[t+1] from x[..t]); which: letters among J (joint pairs), D (true depth), B (backoff), O (oracle),
    Z (product model). Returns a dict of arrays of length n-1 aligned with the transformer's positions, and, if
    positions is given, the product model's predictive distributions at those positions ('pZ').
    The base-law estimates (B, Z) are refreshed at every visit during the first `warm` tokens and every `refresh`
    tokens afterwards; between refreshes the row is updated incrementally with the last estimate."""
    import lsa_seq
    x = np.asarray(x, dtype=int); n = len(x); d = len(q_true)
    out = {k: np.empty(n - 1) for k in which}
    want = np.zeros(n, dtype=bool)
    if positions is not None:
        want[np.asarray(positions)] = True
        pZ = np.zeros((int(want.sum()), d)); jj = 0
    pairs = lsa_seq._Table(S_pairs) if "J" in which else None
    Mrow = np.zeros((d, d), dtype=int)
    tabsD = [None] * d; rowsB = [None] * d; rowsO = [None] * d; rowsZ = [None] * d
    lastB = np.full(d, -10 ** 9); lastZ = np.full(d, -10 ** 9)
    uni = lsa_seq._Table(S_uni)                          # the unigram, for the running frequency estimate
    uni.observe(int(x[0]))
    qZ = None; qZ_t = -1
    for t in range(1, n):
        a = int(x[t - 1]); b = int(x[t])
        if "J" in which:
            if pairs.k == 0:
                out["J"][t - 1] = np.log2(d)
            else:
                base = _lse(pairs.logQ)
                g = {int(r): np.exp(_lse(pairs._logQ_after(int(r))) - base) for r in np.unique(Mrow[a])}
                row = np.array([g[int(r)] for r in Mrow[a]])
                out["J"][t - 1] = -np.log2(row[b] / row.sum())
            pairs.observe(a * d + b); Mrow[a, b] += 1
        if "D" in which:
            if tabsD[a] is None:
                tabsD[a] = lsa_seq._Table(S_depth)
            out["D"][t - 1] = tabsD[a].observe(b)
        if "B" in which:
            if rowsB[a] is None:
                rowsB[a] = WeightedRow(T_full, d)
            if t < warm or t - lastB[a] >= refresh:
                rowsB[a].rebuild(uni.predictive(), t); lastB[a] = t
            out["B"][t - 1] = rowsB[a].observe(b)
        if "O" in which:
            if rowsO[a] is None:
                rowsO[a] = WeightedRow(T_full, d); rowsO[a].rebuild(q_true, 0)
            out["O"][t - 1] = rowsO[a].observe(b)
        if "Z" in which:
            if rowsZ[a] is None:
                rowsZ[a] = WeightedRow(T_full, d)
            if t < warm or t - lastZ[a] >= refresh:
                if qZ_t != t:
                    qZ = zipf_estimate(uni.m.astype(float)); qZ_t = t
                rowsZ[a].rebuild(qZ, t); lastZ[a] = t
            if want[t]:                                   # loop position t predicts x[t]: the transformer position t-1
                pZ[jj] = rowsZ[a].predictive(); jj += 1
            out["Z"][t - 1] = rowsZ[a].observe(b)
        uni.observe(b)
    if positions is not None:
        out["pZ"] = pZ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--settings", default="1,2")
    ap.add_argument("--n_eval", type=int, default=512, help="size of the evaluation set (must match the references)")
    ap.add_argument("--n_seq", type=int, default=64, help="how many of its sequences to run")
    ap.add_argument("--n_full", type=int, default=64, help="sequences with the product model's predictive distributions")
    ap.add_argument("--full_every", type=int, default=4)
    ap.add_argument("--which", default="ZO", help="predictors: letters among J D B O Z")
    ap.add_argument("--refresh", type=int, default=16)
    ap.add_argument("--table_dir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.table_dir)
    import lsa_seq
    d = args.d
    t0 = time.time()
    S_uni = lsa_seq.SeqLSA(d, table_dir=args.table_dir)
    S_pairs = lsa_seq.SeqLSA(d * d, Lmax=min(lsa_seq.lmax_of(d * d), 30), table_dir=args.table_dir) if "J" in args.which else None
    print(f"tables: unigram/copier Lmax {S_uni.Lmax}; predictors {args.which}; {time.time()-t0:.0f} s", flush=True)
    positions = np.arange(0, args.n_ctx, args.full_every)
    os.makedirs(args.out, exist_ok=True)
    for alpha, ls in parse_settings(args.settings):
        tag = tag_of(d, alpha, ls); fn = os.path.join(args.out, f"extra_{tag}.npz")
        if os.path.exists(fn):
            print(f"  {tag}: exists", flush=True); continue
        t0 = time.time()
        S_depth = lsa_seq.SeqLSA(d, table_dir=args.table_dir, depths=[max(ls, 1)]) if "D" in args.which else None
        T_full = FullTables(S_uni.depths, S_uni.rmax, args.table_dir)
        x, meta = eval_set(d, alpha, ls, args.n_ctx, args.n_eval, permute=True)
        q = zipf(d, alpha)
        N = min(args.n_seq, x.shape[0]); NF = min(args.n_full, N)
        res = {k: np.empty((N, args.n_ctx), np.float32) for k in args.which}
        pZ = np.empty((NF, len(positions), d), np.float16) if "Z" in args.which else None
        for i in range(N):
            perm = meta["perm"][i]; q_true = np.empty(d); q_true[perm] = q           # symbol perm[b] has probability q_b
            r = run_extra(x[i].astype(int), q_true, S_pairs, S_depth, T_full, S_uni, which=args.which,
                          positions=(positions + 1) if (i < NF and "Z" in args.which) else None, refresh=args.refresh)
            for k in args.which:
                res[k][i] = r[k]
            if i < NF and "Z" in args.which:
                pZ[i] = r["pZ"]
            if i == 0:
                print("    first sequence: " + " ".join(f"{k} {r[k].mean():.3f}" for k in args.which) + f" bits/token; {time.time()-t0:.0f} s", flush=True)
        save = {f"l{k}": res[k] for k in args.which}
        if pZ is not None:
            save["pZ"] = pZ; save["positions"] = positions
        np.savez_compressed(fn, tokens=x[:N], alpha=alpha, ls=ls, d=d, md5=md5_of(x), n_seq=N, n_full=NF, **save)
        print(f"  {tag}: " + ", ".join(f"{k} {res[k].mean():.3f}" for k in args.which) + f" bits/token over {N} sequences; {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

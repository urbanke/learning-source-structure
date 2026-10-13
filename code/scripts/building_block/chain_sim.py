"""Simulate the Markov source of Section 2 and compute exact LSA costs of both mechanisms.

Per repetition: one table T (rows q*W normalised, W product of Ls exponentials), one path.
At each checkpoint n: unigram cost l_F(n), copier cost l_C(n) (log2 d + sum over contexts),
and control variates with known conditional means:
   cvF = -sum_b N_b log2 pi^(n)_b(T),   E[cvF|T] = n H2(pi^(n)(T))
   cvC = sum_a K_a H2(T_a),              E[cvC|T] = sum_a E[K_a|T] H2(T_a)
where pi^(n)(T) = (1/n) sum_{t<=n} q T^{t-1} and K_a counts visits at positions 1..n-1.
"""
from __future__ import annotations
import sys, time, json
import numpy as np
import lsa_cost as C
import bb
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)

LN2 = np.log(2.0)
EULER = 0.57721566490153286


def lmax_of(d):
    return int(round(2.0 * np.log(d) / (1.0 - EULER)))


def h2(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def draw_table(q, Ls, rng):
    d = len(q)
    if Ls == 0:
        return np.tile(q, (d, 1))
    W = np.prod(rng.exponential(size=(Ls, d, d)), axis=0)
    T = q[None, :] * W
    return T / T.sum(axis=1, keepdims=True)


def occupation(q, T, nmax):
    """p[t-1] = q T^{t-1} for t = 1..nmax  (shape nmax x d)."""
    d = len(q)
    p = np.empty((nmax, d))
    v = q.copy()
    for t in range(nmax):
        p[t] = v
        v = v @ T
    return p


def simulate_path(q, T, nmax, rng):
    d = len(q)
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    x = np.empty(nmax, dtype=np.int32)
    x[0] = min(int(np.searchsorted(np.cumsum(q), rng.random(), side="right")), d - 1)
    u = rng.random(nmax)
    for t in range(1, nmax):
        x[t] = min(int(np.searchsorted(cum[x[t - 1]], u[t], side="right")), d - 1)
    return x


class CostCache:
    def __init__(self, lsa):
        self.lsa = lsa
        self.cache = {}

    def cost(self, prof: dict, k: int) -> float:
        if k == 0:
            return 0.0
        key = (k, tuple(sorted(prof.items())))
        v = self.cache.get(key)
        if v is None:
            v = self.lsa.cost_bits(prof, k)
            self.cache[key] = v
        return v


def run(d, alpha, Ls, R, seed, checkpoints, out):
    q = C.zipf(d, alpha)
    Lmax = lmax_of(d)
    lsa = bb.LSAx(Lmax)
    print(f"tables cover r <= {lsa.rmax}", flush=True)
    cc = CostCache(lsa)
    rng = np.random.default_rng(seed)
    nmax = max(checkpoints)
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, R=R, seed=seed, checkpoints=list(checkpoints),
               lF=[], lC=[], cvF=[], cvC=[], EcvF=[], EcvC=[], Hpi=[], seconds=[],
               K=[], costA=[], rowH=[], N=[])
    for rep in range(R):
        t0 = time.time()
        T = draw_table(q, Ls, rng)
        p = occupation(q, T, nmax)                    # nmax x d
        cp = np.cumsum(p, axis=0)                     # cumulative occupation
        Hrows = np.array([h2(T[a]) for a in range(d)])
        x = simulate_path(q, T, nmax, rng)
        lF = []; lC = []; cvF = []; cvC = []; EcvF = []; EcvC = []; Hpi = []
        Klist = []; costAlist = []; Nlist = []
        for n in checkpoints:
            N = np.bincount(x[:n], minlength=d)
            profF = C.profile_of_counts(N)
            lF.append(cc.cost(profF, n))
            # transitions 1..n-1
            src = x[:n - 1]; dst = x[1:n]
            M = np.zeros((d, d), dtype=np.int64)
            np.add.at(M, (src, dst), 1)
            K = M.sum(axis=1)
            tot = np.log2(d)
            costA = np.zeros(d)
            for a in range(d):
                if K[a] > 0:
                    costA[a] = cc.cost(C.profile_of_counts(M[a]), int(K[a]))
                    tot += costA[a]
            lC.append(tot)
            Klist.append(K.tolist()); costAlist.append(costA.tolist()); Nlist.append(N.tolist())
            pin = cp[n - 1] / n                       # pi^(n)(T)
            cvF.append(float(-np.sum(N * np.log2(pin))))
            EcvF.append(float(n * h2(pin)))
            cvC.append(float(np.sum(K * Hrows)))
            EK = cp[n - 2] if n >= 2 else np.zeros(d)  # E[K_a|T] over positions 1..n-1
            EcvC.append(float(np.sum(EK * Hrows)))
            Hpi.append(float(h2(pin)))
        for key, val in (("lF", lF), ("lC", lC), ("cvF", cvF), ("cvC", cvC), ("EcvF", EcvF), ("EcvC", EcvC), ("Hpi", Hpi),
                         ("K", Klist), ("costA", costAlist), ("N", Nlist)):
            res[key].append(val)
        res["rowH"].append(Hrows.tolist())
        res["seconds"].append(time.time() - t0)
        json.dump(res, open(out, "w"))
        print(f"rep {rep}: {time.time()-t0:.0f}s, cache {len(cc.cache)}; lF-cvF at last checkpoint {lF[-1]-cvF[-1]:.1f}, lC-cvC {lC[-1]-cvC[-1]:.1f}", flush=True)
    return res


if __name__ == "__main__":
    d, alpha, Ls, R, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    checkpoints = [int(a) for a in sys.argv[6].split(",")]
    out = sys.argv[7]
    run(d, alpha, Ls, R, seed, checkpoints, out)

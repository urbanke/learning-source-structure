"""Rare contexts: shifted-gamma law of eps_a from its first three cumulants vs the true (row-sampled) heavy law."""
import sys, numpy as np
from scipy.stats import gamma as gam
from table_mc import BBInterp, two_moment_pmf, support
from sf_lib import RowTheory
tag, bbf, j = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(f'sfbig_{tag}.npz'); d, alpha, Ls = int(z["d"]), float(z["alpha"]), int(z["Ls"]); ns = [int(n) for n in z["ns"]]
muK, VK = z["muK"].astype(float), z["VK"].astype(float); NT = min(muK.shape[0], 2000)
th = RowTheory(d, alpha, Ls, 100000); q, mu = th.q, th.mu; NR = th.NR
BB = BBInterp(bbf); BBk = BB(np.arange(0, max(ns) + 1, dtype=float))
def bins(vals, NQ=60, ntail=30):
    edges = np.concatenate([np.linspace(0, 0.99, NQ + 1)[:-1], np.linspace(0.99, 1.0, ntail + 1)])
    v = np.sort(vals); N = len(v); nodes = []; w = np.diff(edges)
    for k in range(len(edges) - 1):
        lo, hi = int(round(edges[k] * N)), int(round(edges[k + 1] * N)); nodes.append(v[lo:hi].mean())
    return np.array(nodes), w
def gamma_nodes(m1, var, skew, NQ=60, ntail=30):
    """shifted gamma with given mean, variance, skewness; nodes at bin means (analytic bins by quantiles)"""
    if skew < 0.05: skew = 0.05
    k = 4.0 / skew ** 2; theta = np.sqrt(var / k); shift = m1 - k * theta
    edges = np.concatenate([np.linspace(0, 0.99, NQ + 1)[:-1], np.linspace(0.99, 1.0, ntail + 1)])
    edges[-1] = 1 - 1e-9
    qs = gam.ppf(edges, k, scale=theta)
    w = np.diff(edges); nodes = []
    for lo, hi in zip(qs[:-1], qs[1:]):        # bin mean of a gamma: theta*k*[F_{k+1}(hi)-F_{k+1}(lo)]/[F_k(hi)-F_k(lo)]
        num = gam.cdf(hi, k + 1, scale=theta) - gam.cdf(lo, k + 1, scale=theta); den = gam.cdf(hi, k, scale=theta) - gam.cdf(lo, k, scale=theta)
        nodes.append(shift + theta * k * num / max(den, 1e-300))
    return np.array(nodes), w
def EB(m, V, n):
    R = support(m, V, n); pk = two_moment_pmf(m, max(V, 0.2 * m), R); return float(pk @ BBk[:R + 1])
def two_state(a, xq, eps, npos):
    P = min(max(mu[a] + xq, 1e-12), 1 - 1e-12); pi = min(max((mu[a] + eps) / (1 - xq), 1e-12), 1 - 1e-12)
    e = min(max(mu[a] + eps / (1 - pi), 1e-12), 1 - 1e-12); lam = P - e; geo = (1 - lam ** npos) / (1 - lam)
    m = npos * pi + (q[a] - pi) * geo; V = npos * pi * (1 - pi) * (1 + lam) / (1 - lam) - 2 * pi * (1 - pi) * lam * geo / (1 - lam)
    return max(m, 1e-9), max(V, 1e-9)
n = ns[j]; rng = np.random.default_rng(5); NE = 40000
idx = rng.integers(0, NR, size=(d, NE)); xc = th.x[idx, np.arange(d)[:, None]]; inv = mu[:, None] / (1.0 - xc)
print(f"{tag} n={n}: context | sampled E_T[EB] (SE) | heavy eps (true) | gamma eps (3 cumulants) | gaussian eps")
for a in [0, 1, 3, 7, 15, 30, 60, 120, 250]:
    vals = np.array([EB(muK[i, j, a], VK[i, j, a], n) for i in range(NT)]); s_, se = vals.mean(), vals.std(ddof=1) / np.sqrt(NT)
    xn, xw = bins(th.x[:, a])
    xa = th.x[idx, a]; eps = (inv * xa).sum(axis=0) - inv[a] * xa[a]
    m1, var = eps.mean(), eps.var(); skew = ((eps - m1) ** 3).mean() / var ** 1.5
    res = []
    for en, ew in [bins(eps), gamma_nodes(m1, var, skew), gamma_nodes(m1, var, 0.0)]:
        h = 0.0
        for xq, wq in zip(xn, xw):
            for e_, we in zip(en, ew):
                mm, V = two_state(a, xq, e_, n - 1); h += wq * we * EB(mm, V, n)
        res.append(h)
    print(f"  a={a+1:4d} | {s_:9.2f} ({se:.2f}) | {res[0]:9.2f} ({res[0]-s_:+6.2f}) | {res[1]:9.2f} ({res[1]-s_:+6.2f}) | {res[2]:9.2f} ({res[2]-s_:+6.2f})   [skew {skew:.2f}]")

"""Two-state reduction (Step 5 theory). Given the matrix, the visits to context a follow a two-state chain: stay with
probability P_a = T(a|a) = mu_a + x_a (own row), enter from outside with probability e_a = mu_a + eps_a, where
eps_a = sum_{c != a} pi_c x_a^{(c)} ~ N(E eps_a, v_rest,a) collects the other rows (independent of row a).
  pi_a = e/(1 - P + e),  lambda = P - e,
  E[K_a|T] = (n-1) pi + (q_a - pi)(1 - lambda^(n-1))/(1 - lambda),
  Var(K_a|T) = (n-1) pi (1-pi) (1+lambda)/(1-lambda) - 2 pi (1-pi) lambda (1 - lambda^(n-1))/(1-lambda)^2.
The law of x_a is the one-row law (here from sampled rows); K_a | T two-moment.  Copier: log2 d + sum_a E B(K_a).
Unigram: N_b over n positions, same with n; mixed pmfs -> profile -> Laplace cost - eta.
Usage: python3 sf_two.py sf_<tag>.npz bb2.json NR NQ [sfbig.npz] [uni js]"""
import sys, time, json
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
import bb, lsa_fast  # noqa
import chain_sim as CS
from table_mc import BBInterp, two_moment_pmf, support
from sf_lib import RowTheory

z = np.load(sys.argv[1]); bbfile = sys.argv[2]; NR = int(sys.argv[3]); NQ = int(sys.argv[4])
big = np.load(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
ujs = [int(a) for a in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
d, alpha, Ls = int(z["d"]), float(z["alpha"]), int(z["Ls"]); ns = [int(n) for n in z["ns"]]
th = RowTheory(d, alpha, Ls, NR); q, mu = th.q, th.mu
BB = BBInterp(bbfile); BBk = BB(np.arange(0, max(ns) + 1, dtype=float))
# own-row law of x_a: NQ equal-probability bins plus 40 bins in the top one percent, node = bin mean
edges = np.concatenate([np.linspace(0, 0.99, NQ + 1)[:-1], np.linspace(0.99, 1.0, 41)])
Xq = np.zeros((len(edges) - 1, d)); Wq = np.diff(edges)
xs_sorted = np.sort(th.x, axis=0); NRr = xs_sorted.shape[0]
for k in range(len(edges) - 1):
    lo, hi = int(round(edges[k] * NRr)), int(round(edges[k + 1] * NRr))
    Xq[k] = xs_sorted[lo:hi].mean(axis=0)
S = th.S.copy(); np.fill_diagonal(S, 0.0); Eeps = mu @ S
M2 = th.M2.copy(); np.fill_diagonal(M2, 0.0); vrest = mu ** 2 @ M2
gh_x, gh_w = hermegauss(6); gh_w = gh_w / gh_w.sum()
ref = {}
if len(sys.argv) > 5 and sys.argv[5] != "-":
    try:
        ref = {int(k): v for k, v in json.load(open(sys.argv[5].replace(".npz", "_sampled.json"))).items()}
    except FileNotFoundError:
        pass


def EB(m, V, n):
    R = support(m, V, n); pk = two_moment_pmf(m, max(V, 0.2 * m), R); return float(pk @ BBk[:R + 1])


def two_state(a, xq, eps, npos):
    P = min(max(mu[a] + xq, 1e-12), 1 - 1e-12)
    pi = min(max((mu[a] + eps) / (1 - xq), 1e-12), 1 - 1e-12)        # exact balance: pi (1 - P) = mu (1 - pi) + eps
    e = min(max(mu[a] + eps / (1 - pi), 1e-12), 1 - 1e-12)          # entry probability per outside step
    lam = P - e
    geo = (1 - lam ** npos) / (1 - lam)
    m = npos * pi + (q[a] - pi) * geo
    V = npos * pi * (1 - pi) * (1 + lam) / (1 - lam) - 2 * pi * (1 - pi) * lam * geo / (1 - lam)
    return max(m, 1e-9), max(V, 1e-9)


def components(a, npos):
    sd = np.sqrt(max(vrest[a], 0.0))
    for xq, wq in zip(Xq[:, a], Wq):
        for gx, gw in zip(gh_x, gh_w):
            m, V = two_state(a, xq, Eeps[a] + sd * gx, npos)
            yield gw * wq, m, V


def copier(n):
    tot = np.log2(d); EK = np.zeros(d); VA = np.zeros(d); VW = np.zeros(d)
    for a in range(d):
        s1 = s2 = sv = sb = 0.0
        for w, m, V in components(a, n - 1):
            sb += w * EB(m, V, n); s1 += w * m; s2 += w * m * m; sv += w * V
        tot += sb; EK[a] = s1; VA[a] = s2 - s1 * s1; VW[a] = sv
    return tot, EK, VA, VW


def unigram(n):
    lsa = bb.LSAx(CS.lmax_of(d))
    comps = [list(components(b, n)) for b in range(d)]
    rmax = int(min(n, max(max(m + 10 * np.sqrt(max(V, m)) + 20 for _, m, V in comps[b]) for b in range(d))))
    pm = np.zeros((d, rmax + 1))
    for b in range(d):
        for w, m, V in comps[b]:
            pm[b] += w * two_moment_pmf(m, max(V, 0.2 * m), rmax)
    nu = pm.sum(axis=0); keep = np.flatnonzero(nu > 1e-9); nu_d = {int(r): float(nu[r]) for r in keep}; rs = [int(r) for r in keep]
    us, vals, fpp, wL, U, lq = bb.laplace_pieces(lsa, nu_d, n)
    Sig = np.diag(nu[keep]) - pm[:, keep].T @ pm[:, keep]
    rho = np.arange(rmax + 1, dtype=float)[keep]; v = Sig @ rho; den = rho @ v
    if den > 1e-12: Sig = Sig - np.outer(v, v) / den
    eta = bb.eta_laplace(lsa, nu_d, n, Sig, rs, us, fpp, wL)
    return float(U) - float(eta)


out = []
print("n | copier: two-state theory | sampled (SE) | diff || sum|EK err| (sum SE) | across var sampled/theory | within var sampled/theory")
for j, n in enumerate(ns):
    t0 = time.time()
    ct, EK, VA, VW = copier(n)
    src = big if big is not None else z
    bm = src["muK"][:, j].astype(float); mb = bm.mean(axis=0); se = bm.std(axis=0) / np.sqrt(bm.shape[0]); Va = bm.var(axis=0); Vw = src["VK"][:, j].astype(float).mean(axis=0)
    cs, cse = (ref[n]["C"], ref[n]["Cse"]) if n in ref else (float("nan"), float("nan"))
    print(f"{n:6d} | {ct:9.1f} | {cs:9.1f} ({cse:.1f}) | {ct-cs:+6.1f} || {np.abs(mb-EK).sum():.1f} ({se.sum():.1f}) | {Va.sum():.0f}/{VA.sum():.0f} | {Vw.sum():.0f}/{VW.sum():.0f}   [{time.time()-t0:.0f}s]", flush=True)
    row = dict(n=n, ct=ct, cs=cs, cse=cse, EKerr=float(np.abs(mb - EK).sum()), EKse=float(se.sum()), Va=float(Va.sum()), VA=float(VA.sum()), Vw=float(Vw.sum()), VW=float(VW.sum()))
    if j in ujs:
        t0 = time.time(); ut = unigram(n)
        us = ref[n].get("U", float("nan")) if n in ref else float("nan"); use = ref[n].get("Use", float("nan")) if n in ref else float("nan")
        print(f"        unigram: two-state theory {ut:.1f} | sampled {us:.1f} ({use:.1f}) | diff {ut-us:+.1f}   [{time.time()-t0:.0f}s]", flush=True)
        row.update(ut=ut, us=us, use=use)
    out.append(row)
json.dump(out, open(sys.argv[1].replace(".npz", "_two.json"), "w"), indent=1)

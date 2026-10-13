"""Table Monte Carlo: predictions of both mechanisms' expected costs from the building block.

Per table T (no paths):
  occupation p_t = q T^{t-1}; fundamental matrix Z = (I - T + 1 pi)^{-1}; g_b = Z_bb - 1.
  Count moments given T (positions 1..n): mean mu_b = sum_t p_t(b), var V_b = sum_t p_t(1-p_t) + 2 n pi_b g_b.
  Unigram prediction: per-symbol two-moment closure -> mean profile -> Laplace cost - eta.
  Copier prediction: log2 d + sum_a E[BB(K_a) | T] with K_a two-moment closure and BB interpolated.
Control variates: E[cvF|T] = n H2(pi^(n)), E[cvC|T] = sum_a E[K_a|T] H2(T_a).
"""
from __future__ import annotations
import sys, time, json
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.interpolate import CubicSpline
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)

LN2 = np.log(2.0)


def two_moment_pmf(mu, V, rmax):
    """pmf on 0..rmax with mean mu and variance V: Poisson / negative binomial / binomial."""
    r = np.arange(rmax + 1)
    if mu <= 0:
        p = np.zeros(rmax + 1); p[0] = 1.0; return p
    if V > mu * (1 + 1e-6):
        nu = mu * mu / (V - mu)
        pr = mu / (nu + mu)
        lp = gammaln(r + nu) - gammaln(nu) - gammaln(r + 1.0) + nu * np.log1p(-pr) + r * np.log(pr)
    elif V < mu * (1 - 1e-6):
        kk = max(int(round(mu * mu / (mu - V))), int(np.ceil(mu)))
        pp = min(mu / kk, 1.0)
        lp = np.where(r <= kk, gammaln(kk + 1.0) - gammaln(r + 1.0) - gammaln(np.maximum(kk - r, 0) + 1.0)
                      + r * np.log(max(pp, 1e-300)) + (kk - r) * np.log1p(-pp), -np.inf)
    else:
        lp = r * np.log(mu) - mu - gammaln(r + 1.0)
    p = np.exp(lp - logsumexp(lp))
    return p


class BBInterp:
    """B(k) from a grid file: cubic spline in ln k inside the grid; beyond the largest grid point,
    h k + c ln k + const with (h, c) fitted on the upper part of the grid (k >= kmax/2.5)."""

    def __init__(self, path):
        j = json.load(open(path))
        ks = np.array([r["k"] for r in j["rows"]], dtype=float)
        cs = np.array([r["cost"] for r in j["rows"]])
        self.kmax = ks.max()
        self.a = (cs[-1] - cs[-2]) / (ks[-1] - ks[-2])
        g = cs - self.a * ks
        self.spl = CubicSpline(np.log(ks), g)
        self.c_end = cs[-1]
        sel = ks >= self.kmax / 2.5
        A = np.stack([ks[sel], np.log(ks[sel]), np.ones(int(sel.sum()))], axis=1)
        self.h, self.c, _ = np.linalg.lstsq(A, cs[sel], rcond=None)[0]

    def __call__(self, k):
        k = np.asarray(k, dtype=float)
        out = np.zeros_like(k)
        pos = (k > 0) & (k <= self.kmax)
        out[pos] = self.spl(np.log(k[pos])) + self.a * k[pos]
        big = k > self.kmax
        out[big] = self.c_end + self.h * (k[big] - self.kmax) + self.c * np.log(k[big] / self.kmax)
        return out


def support(mu, V, n):
    """pmf support length for a count with mean mu and variance V (total n)."""
    return int(min(n, np.ceil(mu + 10.0 * np.sqrt(max(V, mu, 1.0)) + 20)))


def unigram_prediction(lsa, mu, V, n, rmax=None):
    """Returns U, eta, depth-1 weight, and the explicit depth-one/Stirling formula (bits).
    rmax (support of the count laws) defaults to the largest support needed by any symbol."""
    from scipy.special import polygamma
    d = len(mu)
    if rmax is None:
        rmax = max(support(mu[b], V[b], n) for b in range(d))
    pm = np.array([two_moment_pmf(mu[b], V[b], rmax) for b in range(d)])
    nu = pm.sum(axis=0)
    keep = np.flatnonzero(nu > 1e-9)
    nu_d = {int(r): float(nu[r]) for r in keep}
    rs = [int(r) for r in keep]
    us, vals, fpp, wL, U, lq = bb.laplace_pieces(lsa, nu_d, n)
    Sig = np.diag(nu[keep]) - pm[:, keep].T @ pm[:, keep]
    rho = np.arange(rmax + 1, dtype=float)[keep]
    v = Sig @ rho
    den = rho @ v
    if den > 1e-12:
        Sig = Sig - np.outer(v, v) / den
    eta = bb.eta_laplace(lsa, nu_d, n, Sig, rs, us, fpp, wL)
    # explicit depth-one formula with E ln N! from the two-moment law, plus the mixture overhead
    r = np.arange(rmax + 1)
    ElnNf = float(np.sum(pm @ gammaln(r + 1.0)))
    depth1 = (gammaln(n + d) - gammaln(d) - ElnNf) / LN2 + np.log2(lsa.Lmax)
    # Stirling version: E ln N! ~ lnGamma(mu+1) + 0.5 V psi'(mu+1)
    ElnNf_st = float(np.sum(gammaln(mu + 1.0) + 0.5 * V * polygamma(1, mu + 1.0)))
    depth1_st = (gammaln(n + d) - gammaln(d) - ElnNf_st) / LN2 + np.log2(lsa.Lmax)
    return U, eta, float(wL[0]), depth1, depth1_st


def occupation_sums(q, T, ns, tol=1e-14, steps=20):
    """S1[n] = sum_{t=1}^n p_t and S2[n] = sum_{t=1}^n p_t (1 - p_t), p_t = q T^{t-1}, for n in ns and n - 1;
    and the stationary law pi. The iteration stops once p_t has converged to pi (L1 change below tol for
    `steps` consecutive steps) and continues exactly with pi from there. Same values as the full iteration
    (chain_sim.occupation + cumsum) to floating-point accuracy, without the nmax x d arrays."""
    need = sorted(set([int(n) for n in ns] + [int(n) - 1 for n in ns])); nmax = max(need)
    v = q.copy(); s1 = np.zeros_like(v); s2 = np.zeros_like(v); S1 = {}; S2 = {}
    t = 0; conv = 0; pi = None
    while t < nmax:
        t += 1
        s1 = s1 + v; s2 = s2 + v * (1.0 - v)
        if t in need:
            S1[t] = s1.copy(); S2[t] = s2.copy()
        w = v @ T
        conv = conv + 1 if np.abs(w - v).sum() < tol else 0
        v = w
        if conv >= steps:
            pi = v.copy(); break
    if pi is None:
        pi = v @ np.linalg.matrix_power(T, 50)
    pi = pi / pi.sum()
    for n in need:
        if n > t:
            S1[n] = s1 + (n - t) * pi; S2[n] = s2 + (n - t) * pi * (1.0 - pi)
    return S1, S2, pi


def run(d, alpha, Ls, NT_pred, NT_occ, seed, checkpoints, bbfile, out):
    q = C.zipf(d, alpha)
    Lmax = CS.lmax_of(d)
    lsa = bb.LSAx(Lmax)
    BB = BBInterp(bbfile)
    rng = np.random.default_rng(seed)
    nmax = max(checkpoints)
    rmax = nmax                      # count laws live on 0..n; the tables cover r <= lsa.rmax, beyond that direct
    print(f"tables cover r <= {lsa.rmax}", flush=True)
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, checkpoints=list(checkpoints), NT_pred=NT_pred, NT_occ=NT_occ,
               predF_U=[], predF_eta=[], predC=[], EcvF=[], EcvC=[], EcvF_occ=[], EcvC_occ=[], Hpi_occ=[], seconds=[],
               w1=[], depth1=[], depth1_st=[])
    kgrid = np.arange(0, rmax + 1, dtype=float)
    BBk = BB(kgrid)
    # mean row entropy (bits) over many rows, for the visits-entropy covariance term
    rr = np.random.default_rng(seed + 777)
    Wm = np.prod(rr.exponential(size=(max(Ls, 1), 20000, d)), axis=0) if Ls > 0 else np.ones((20000, d))
    Pm = q[None, :] * Wm; Pm /= Pm.sum(axis=1, keepdims=True)
    Hbar = float(np.mean(-np.sum(Pm * np.log2(Pm), axis=1)))
    res["Hbar"] = Hbar
    res["covC"] = []
    for it in range(NT_occ):
        t0 = time.time()
        T = CS.draw_table(q, Ls, rng)
        S1, S2, pi = occupation_sums(q, T, checkpoints)     # cumulative occupation sums at the checkpoints
        Hrows = np.array([CS.h2(T[a]) for a in range(d)])
        # fundamental matrix
        Z = np.linalg.inv(np.eye(d) - T + np.outer(np.ones(d), pi))
        g = np.diag(Z) - 1.0
        EcvF = []; EcvC = []; Hpi = []; pU = []; peta = []; pC = []; w1 = []; d1 = []; d1s = []; cC = []
        for n in checkpoints:
            pin = S1[n] / n
            EcvF.append(float(n * CS.h2(pin)))
            EK = S1[n - 1]
            EcvC.append(float(np.sum(EK * Hrows)))
            Hpi.append(float(CS.h2(pin)))
            # copier: visits over positions 1..n-1 (all tables)
            muK = S1[n - 1]
            VK = S2[n - 1] + 2.0 * (n - 1) * pi * g
            VK = np.maximum(VK, 0.2 * muK)
            tot = np.log2(d)
            for a in range(d):
                R = support(muK[a], VK[a], n)
                pk = two_moment_pmf(muK[a], VK[a], R)
                tot += float(pk @ BBk[:R + 1])
            pC.append(tot)
            cC.append(float(np.sum(muK * (Hrows - Hbar))))
            if it < NT_pred:
                # unigram: counts over positions 1..n
                mu = S1[n]
                V = S2[n] + 2.0 * n * pi * g
                V = np.maximum(V, 0.2 * mu)
                U, eta, w1_, d1_, d1s_ = unigram_prediction(lsa, mu, V, n)
                pU.append(float(U)); peta.append(float(eta)); w1.append(w1_); d1.append(d1_); d1s.append(d1s_)
        res["EcvF_occ"].append(EcvF); res["EcvC_occ"].append(EcvC); res["Hpi_occ"].append(Hpi)
        res["predC"].append(pC); res["covC"].append(cC)
        if it < NT_pred:
            res["predF_U"].append(pU); res["predF_eta"].append(peta)
            res["EcvF"].append(EcvF); res["EcvC"].append(EcvC)
            res["w1"].append(w1); res["depth1"].append(d1); res["depth1_st"].append(d1s)
        res["seconds"].append(time.time() - t0)
        if it % 10 == 0 or it < NT_pred:
            json.dump(res, open(out, "w"))
            print(f"table {it}: {time.time()-t0:.1f}s  predC-EcvC {pC[-1]-EcvC[-1]:.1f}" + (f"  predF-EcvF {pU[-1]-peta[-1]-EcvF[-1]:.1f}" if it < NT_pred else ""), flush=True)
    json.dump(res, open(out, "w"))
    return res


if __name__ == "__main__":
    d, alpha, Ls = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    NT_pred, NT_occ, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    checkpoints = [int(a) for a in sys.argv[7].split(",")]
    bbfile, out = sys.argv[8], sys.argv[9]
    run(d, alpha, Ls, NT_pred, NT_occ, seed, checkpoints, bbfile, out)

"""Test: eta with the exact conditioned profile covariance (pairwise terms) vs the independent-symbol
covariance with the total-count projection, inside the building block (small k, exact (L,u)-measure)."""
import sys, time, numpy as np
import bb, lsa_cost as C, lsa_cost2 as C2
LN2 = np.log(2)

def conv(a, b, n):
    return np.convolve(a, b)[:n]

def pair_profile_cov(pm, k):
    """Exact Cov(Phi_r, Phi_r') under N | S=k with independent mixed-Poisson N_b ~ pm[b]."""
    d, _ = pm.shape; n = k + 1; r = np.arange(n)
    # full prefix/suffix
    ptil, PS = bb.conditioned_profile(pm, k)
    nu = ptil.sum(0)
    Sig = np.diag(nu) - ptil.T @ ptil          # within-symbol part: sum_b [diag(ptil_b) - ptil_b ptil_b^T]
    # pair part: sum_{b != c} [P(N_b=r, N_c=r') - ptil_b(r) ptil_c(r')]
    pair = np.zeros((n, n))
    for b in range(d):
        pmb = np.delete(pm, b, axis=0)          # d-1 pmfs
        prefix = [np.zeros(n)]; prefix[0][0] = 1.0
        for i in range(d - 1):
            prefix.append(conv(prefix[-1], pmb[i, :n], n))
        suffix = [None] * (d + 1); suffix[d] = np.zeros(n); suffix[d][0] = 1.0
        for i in range(d - 1, 0, -1):
            suffix[i] = conv(suffix[i + 1], pmb[i - 1, :n], n)
        for cc in range(d - 1):                  # cc indexes pmb; actual symbol c = cc if cc < b else cc+1
            c = cc if cc < b else cc + 1
            Pmbc = conv(prefix[cc], suffix[cc + 2], n)   # sum without b and c
            # joint[r, r'] = pm[b, r] pm[c, r'] Pmbc[k - r - r'] / PS
            idx = k - r[:, None] - r[None, :]
            J = np.where(idx >= 0, pm[b, :n][:, None] * pm[c, :n][None, :] * Pmbc[np.clip(idx, 0, k)], 0.0) / PS
            pair += J - np.outer(ptil[b], ptil[c])
    return Sig, Sig + pair, ptil, PS

def run(d, a, Ls, k, Lmax, xhalf, xstep=0.1):
    q = C.zipf(d, a); lsa = bb.LSAx(Lmax, rmax=k + 5)
    rho = np.arange(k + 1, dtype=float)
    xgrid = np.arange(-xhalf, xhalf + 1e-9, xstep)
    tot = 0.0; accU = 0.0; acc_ind = 0.0; acc_ex = 0.0; acc_ex0 = 0.0
    t0 = time.time()
    for x in xgrid:
        s = k * np.exp(x)
        pm = bb.pmfs_at_scale(lsa, q, Ls, s, k)
        Sig_ind, Sig_ex, ptil, PS = pair_profile_cov(pm, k)
        w = k * PS * xstep
        if w <= 0: continue
        nu = ptil.sum(0); keep = np.flatnonzero(nu > 1e-12); rs = [int(r_) for r_ in keep]
        nu_d = {int(r_): float(nu[r_]) for r_ in keep}
        Si = Sig_ind[np.ix_(keep, keep)]; v = Si @ rho[keep]; den = rho[keep] @ v
        if den > 1e-12: Si = Si - np.outer(v, v) / den
        Se = Sig_ex[np.ix_(keep, keep)]
        U = lsa.cost_bits(nu_d, k)
        _, cov_mu = C2.grad_hess(lsa, nu_d, k, rs)
        eta_ind = 0.5 * np.sum(Si * cov_mu) / LN2
        eta_ex = 0.5 * np.sum(Se * cov_mu) / LN2
        tot += w; accU += w * U; acc_ind += w * eta_ind; acc_ex += w * eta_ex
        # sanity: exact covariance should give zero variance of the total count
        tv = rho[keep] @ Se @ rho[keep]
        if abs(x) < 1e-9:
            print(f"   x=0: total-count variance under exact cov {tv:.2e}; eta_ind {eta_ind:.4f} eta_exact {eta_ex:.4f}", flush=True)
    print(f"d={d} a={a} Ls={Ls} k={k}: U {accU/tot:.4f}  eta_indep(projected) {acc_ind/tot:.4f}  eta_exact {acc_ex/tot:.4f}  -> cost_indep {accU/tot - acc_ind/tot:.4f}  cost_exact {accU/tot - acc_ex/tot:.4f}  mass {tot:.5f} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    for (d, a, Ls, k, Lmax) in [(128, 0.75, 2, 32, 23), (128, 0.75, 3, 12, 23), (128, 0.75, 2, 8, 23)]:
        run(d, a, Ls, k, Lmax, xhalf=max(1.6 if Ls <= 2 else 2.2, 1 + 3 / np.sqrt(k)))

"""Recompute the exact E[H(P)] for all settings with a corrected quadrature (exact log-grid weights ds = s h, wider range).
The original info_gain.exact_EHP used np.gradient weights on a log grid, which are 0.027 percent too large."""
import json, numpy as np, time
import lsa_cost as C, source_profiles as S
from scipy.special import logsumexp
def EHP_fixed(q, Ls, h=0.02):
    y, lfh = S.lnW_density(Ls, ymin=-45.0, h=1.0/32); W = np.exp(y); lnW = y
    s = np.exp(np.arange(-14.0, 12.0, h)); ds = s * h
    lphi0 = np.array([logsumexp(lfh[:, None] - (qc * s)[None, :] * W[:, None], axis=0) for qc in q]); lnG = lphi0.sum(axis=0); G = np.exp(lnG)
    ElnZ = np.sum((np.exp(-s) - G) / s * ds); pos = lnW > 0; tot = 0.0
    for b, qb in enumerate(q):
        lphi1 = logsumexp(lfh[:, None] + lnW[:, None] - (qb * s)[None, :] * W[:, None], axis=0)
        lpp = logsumexp(lfh[pos, None] + lnW[pos, None] + np.log(lnW[pos])[:, None] - (qb * s)[None, :] * W[pos, None], axis=0)
        with np.errstate(divide="ignore"):
            lpn = logsumexp(lfh[~pos, None] + lnW[~pos, None] + np.log(-lnW[~pos])[:, None] - (qb * s)[None, :] * W[~pos, None], axis=0)
        psi = np.exp(lpp) - np.exp(lpn); Gmb = np.exp(lnG - lphi0[b])
        tot += np.sum(qb * (np.log(qb) * np.exp(lphi1) + psi) * Gmb * ds)
    return ElnZ - tot
old = json.load(open('exact_EH_archive.json')); new = {}
for key in old:
    d, a, L = key.split(','); d, a, L = int(d), float(a), int(L); t0 = time.time()
    v = EHP_fixed(C.zipf(d, a), L) / np.log(2); new[key] = v
    print(f"{key:14s}: old {old[key]:.6f} new {v:.6f} diff {v-old[key]:+.6f} bits  [{time.time()-t0:.0f}s]", flush=True)
json.dump(new, open('exact_EH_archive_fixed.json', 'w'), indent=1)
# the own simulations S1..S8 (compare.py, make_tables.py; not used in the paper) have the same (d, alpha, Ls) as archive settings
S = {"S1": "128,0.75,2", "S2": "128,1,2", "S3": "128,0.75,3", "S4": "128,1,3", "S5": "256,0.75,2", "S6": "256,1,2", "S7": "256,0.75,3", "S8": "256,1,3"}
json.dump({k: round(new[v], 5) for k, v in S.items()}, open('exact_EH.json', 'w'), indent=1)

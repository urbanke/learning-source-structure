"""Extend ln phi_{L,r}(e^u) tables to u in (15, 40] for r = 0..rmax."""
import sys, time, numpy as np
import lsa_phi as P
from scipy.special import logsumexp
L0, L1, rmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
HU = P.HU
u_ext = np.arange(15.0 + HU, 40.0 + HU / 2, HU)
eu = np.exp(u_ext)
CH = 16
for L in range(L0, L1 + 1):
    t0 = time.time()
    y = P._y_grid(L)
    lf = P.log_density_lnW(L, y)
    lfh = lf + np.log(P.HY)
    ey = np.exp(y)
    tab = np.empty((rmax + 1, len(u_ext)))
    for r in range(rmax + 1):
        base = lfh + r * y
        for a in range(0, len(u_ext), CH):
            umid = u_ext[min(a + CH // 2, len(u_ext) - 1)]
            g = base - np.exp(umid) * ey
            m = g.max()
            keep = np.flatnonzero(g > m - 70.0)
            lo, hi = max(keep[0] - 300, 0), min(keep[-1] + 300, len(y) - 1)
            uu = eu[a:a + CH]
            tab[r, a:a + CH] = logsumexp(base[None, lo:hi + 1] - uu[:, None] * ey[None, lo:hi + 1], axis=1)
    np.save(f"phitab_L{L}_u15_40.npy", tab)
    print(f"L={L} u-ext done in {time.time()-t0:.0f}s", flush=True)

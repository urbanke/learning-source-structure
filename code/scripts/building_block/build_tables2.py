"""Fast table build: ln phi_{L,r}(e^u), r = r0..r1, narrow y-window (60 nats below the max)."""
import sys, time, numpy as np
import lsa_phi as P
from scipy.special import logsumexp
L0, L1, r0, r1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
eu = np.exp(P.u)
CH = 16
for L in range(L0, L1 + 1):
    t0 = time.time()
    y = P._y_grid(L)
    lf = P.log_density_lnW(L, y)
    lfh = lf + np.log(P.HY)
    ey = np.exp(y)
    tab = np.empty((r1 - r0 + 1, P.NU))
    for r in range(r0, r1 + 1):
        base = lfh + r * y
        for a in range(0, P.NU, CH):
            umid = P.u[min(a + CH // 2, P.NU - 1)]
            g = base - np.exp(umid) * ey
            m = g.max()
            keep = np.flatnonzero(g > m - 70.0)
            lo, hi = max(keep[0] - 300, 0), min(keep[-1] + 300, len(y) - 1)
            uu = eu[a:a + CH]
            tab[r - r0, a:a + CH] = logsumexp(base[None, lo:hi + 1] - uu[:, None] * ey[None, lo:hi + 1], axis=1)
    np.save(f"phitab_L{L}_r{r0}_{r1}.npy", tab)
    print(f"L={L} r={r0}..{r1} done in {time.time()-t0:.0f}s", flush=True)

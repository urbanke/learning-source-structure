"""Check of the rows beyond the tables: Laplace rows against table rows at the same count (r = 3000, 6000) and the derivative identity."""
import numpy as np
from sf_predict import LSAext
lsa = LSAext(30)
cols = np.arange(0, lsa.nu, 5); cb = cols[lsa.u[cols] >= -9.9]
for L in (1, 2, 7, 30):
    for r in (3000, 6000):
        tab = lsa._gather(L, np.array([r]), cb)[0]; lap = lsa._laplace_row(L, r)[cb]
        ok = np.isfinite(tab)
        print(f"L={L:2d} r={r}: |Laplace - table| median {np.median(np.abs(lap-tab)[ok]):.1e}, max {np.abs(lap-tab)[ok].max():.1e} nats (u in [{lsa.u[cb][ok].min():.1f},{lsa.u[cb][ok].max():.1f}])")
    rs = np.array([5990, 5995, 6003, 6010, 6500]); u = np.array([-3.0, 0.5, 2.0]); h = 1e-3
    v = lsa.lnphi(L, rs, u, 1); fd = (lsa.lnphi(L, rs, u + h) - lsa.lnphi(L, rs, u - h)) / (2 * h)
    print(f"   derivative: rel. diff to finite difference, table rows {np.abs((v-fd)/v)[:2].max():.1e}, Laplace rows {np.abs((v-fd)/v)[2:].max():.1e}")

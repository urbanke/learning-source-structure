"""Materialise ln phi_{L,r}(e^u) rows from the sealed anchor store (product_model_with_memory) in the table
format of lsa_fast.py: phitab_L{L}_r0_2000_full.npy (u in [-45, 40], step 1/64) and
phitab_L{L}_r2001_6000_u-10_40.npy (u in [-10, 40]).  Usage: python3 store_rows.py L0 L1 outdir storedir pmm_src"""
import sys, os, time, numpy as np
L0, L1, outdir, store, src = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
os.environ.setdefault("PMM_UNIVERSAL_TABLES", store); os.environ.setdefault("PMM_PHI_LADDER_EVERY", "1")
os.environ.setdefault("PMM_PHI_LADDER_DEGREE", "11"); os.environ.setdefault("PMM_PHI_SADDLE_MIN_L", "54")
sys.path.insert(0, src)
from product_model_with_memory import universal_tables as UT
ut = UT.UniversalTables(store, read_only=True)
HU = 1.0 / 64
u_full = np.arange(-45.0, 40.0 + HU / 2, HU); u_part = np.arange(-10.0, 40.0 + HU / 2, HU)
os.makedirs(outdir, exist_ok=True)
from scipy.special import gammaln
def rows(L, rs, u):
    if L == 1:   # closed form, not in the store: ln phi_{1,r}(s) = ln r! - (r+1) ln(1+s)
        rs = np.asarray(rs, float)
        return gammaln(rs + 1.0)[:, None] - (rs + 1.0)[:, None] * np.log1p(np.exp(u))[None, :]
    return ut.log_phi_matrix(L, list(rs), u)
for L in range(L0, L1 + 1):
    t0 = time.time()
    f1 = os.path.join(outdir, f"phitab_L{L}_r0_2000_full.npy")
    if not os.path.exists(f1):
        M = np.empty((2001, len(u_full)))
        for a in range(0, 2001, 250):
            rs = list(range(a, min(a + 250, 2001)))
            M[a:a + len(rs)] = rows(L, rs, u_full)
        np.save(f1 + ".tmp.npy", M); os.replace(f1 + ".tmp.npy", f1)
    t1 = time.time()
    f2 = os.path.join(outdir, f"phitab_L{L}_r2001_6000_u-10_40.npy")
    if not os.path.exists(f2):
        M = np.empty((4000, len(u_part)))
        for a in range(0, 4000, 250):
            rs = list(range(2001 + a, 2001 + min(a + 250, 4000)))
            M[a:a + len(rs)] = rows(L, rs, u_part)
        np.save(f2 + ".tmp.npy", M); os.replace(f2 + ".tmp.npy", f2)
    print(f"L={L}: full block {t1-t0:.0f}s, large-count block {time.time()-t1:.0f}s", flush=True)

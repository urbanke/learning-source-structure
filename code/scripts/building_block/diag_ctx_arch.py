"""Per-context residual of the row-averaged copier formula on archived paths (diag_<tag>.npz: matrices and sparse
transition counts of R paths at a few lengths): exact cost of each context's continuation counts minus
B(K_a) + K_a (H2(T_a) - Hbar), binned by self-transition probability and by visits.
Usage: python3 diag_ctx_arch.py diag_file bb2_file d alpha Ls Lmax"""
import sys, json, time
import numpy as np
import lsa_cost as C, bb, chain_sim as CS, lsa_fast  # noqa
from table_mc import BBInterp
f, bbf, d, alpha, Ls, Lmax = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
z = np.load(f); R = int(z["R"]); ns = [int(n) for n in z["ns"]]; T_all = z["T"].astype(np.float64)
BB = BBInterp(bbf); Hbar = json.load(open("exact_EH_archive.json"))[f"{d},{alpha:g},{Ls}"]
lsa = bb.LSAx(Lmax); cc = CS.CostCache(lsa); t0 = time.time()
for n in ns:
    rows = []; tot_meas = []; tot_avg = []
    for i in range(R):
        T = T_all[i]; T = T / T.sum(axis=1, keepdims=True)
        M = np.zeros(d * d, np.int64); M[z[f"ind_{i}_{n}"]] = z[f"val_{i}_{n}"]; M = M.reshape(d, d); K = M.sum(axis=1)
        m = a_ = 0.0
        for a in np.flatnonzero(K > 0):
            k = int(K[a]); meas = cc.cost(C.profile_of_counts(M[a]), k)
            avg = float(BB(np.array([float(k)]))[0]) + k * (CS.h2(T[a]) - Hbar)
            rows.append((float(T[a, a]), k, meas - avg)); m += meas; a_ += avg
        tot_meas.append(m); tot_avg.append(a_)
    rows = np.array(rows); th, kk, res = rows[:, 0], rows[:, 1], rows[:, 2]
    tm, ta = np.array(tot_meas), np.array(tot_avg)
    print(f"== n={n} ({R} archived paths, {time.time()-t0:.0f}s): meas - avg per path {np.mean(tm-ta):+.1f} (SE {np.std(tm-ta, ddof=1)/np.sqrt(R):.1f}); mean copier cost {tm.mean():.0f}", flush=True)
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]:
        s = (th >= lo) & (th < hi)
        if s.sum(): print(f"   theta [{lo},{hi}): {s.sum()/R:6.1f} ctx/path  meanK {kk[s].mean():7.1f}  meas-avg {res[s].mean():+7.2f}/ctx  {res[s].sum()/R:+7.1f}/path")
    for lo, hi in [(1, 4), (4, 16), (16, 33), (33, 100), (100, 300), (300, 1000), (1000, 10**6)]:
        s = (kk >= lo) & (kk < hi)
        if s.sum(): print(f"   K [{lo},{hi}): {s.sum()/R:6.1f} ctx/path  mean theta {th[s].mean():.3f}  meas-avg {res[s].mean():+7.2f}/ctx  {res[s].sum()/R:+7.1f}/path")

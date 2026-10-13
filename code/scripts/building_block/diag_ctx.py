"""Per-context residual of the row-averaged copier formula on paths of chain_sim.py (which store the exact cost of
every context): meas_a - [B(K_a) + K_a (H2(T_a) - Hbar)], binned by the self-transition probability and by the visits.
The matrices are regenerated from the seed. Usage: python3 diag_ctx.py sim_file bb2_file d alpha Ls j1,j2,..."""
import sys, json
import numpy as np
import lsa_cost as C, bb, chain_sim as CS, lsa_fast  # noqa
from table_mc import BBInterp

simf, bbf, d, alpha, Ls, js = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), [int(x) for x in sys.argv[6].split(",")]
s = json.load(open(simf)); ns = s["checkpoints"]; R = s["R"]; q = C.zipf(d, alpha)
BB = BBInterp(bbf); Hbar = json.load(open("exact_EH_archive.json"))[f"{d},{alpha:g},{Ls}"]
rng = np.random.default_rng(s["seed"]); nmax = max(ns); Ts = []
for i in range(R):
    T = CS.draw_table(q, Ls, rng); x = CS.simulate_path(q, T, nmax, rng)
    assert np.bincount(x[:ns[0]], minlength=d).tolist() == s["N"][i][0], "replay mismatch"
    Ts.append(T)
for j in js:
    n = ns[j]; rows = []
    for i in range(R):
        T = Ts[i]; K = np.array(s["K"][i][j], float); cA = np.array(s["costA"][i][j], float); rowH = np.array(s["rowH"][i])
        for a in np.flatnonzero(K > 0):
            k = int(K[a]); avg = float(BB(np.array([float(k)]))[0]) + k * (rowH[a] - Hbar)
            rows.append((float(T[a, a]), k, float(cA[a] - avg)))
    rows = np.array(rows); th, kk, res = rows[:, 0], rows[:, 1], rows[:, 2]
    print(f"== n={n}: meas - avg total per path {res.sum()/R:+.1f} bits (paths {R}; copier cost {np.mean([s['lC'][i][j] for i in range(R)]):.0f})")
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]:
        m = (th >= lo) & (th < hi)
        if m.sum(): print(f"   theta [{lo},{hi}): {m.sum()/R:6.1f} ctx/path  meanK {kk[m].mean():7.1f}  meas-avg {res[m].mean():+7.2f}/ctx  {res[m].sum()/R:+7.1f}/path")
    for lo, hi in [(1, 4), (4, 16), (16, 33), (33, 100), (100, 300), (300, 1000), (1000, 10**6)]:
        m = (kk >= lo) & (kk < hi)
        if m.sum(): print(f"   K [{lo},{hi}): {m.sum()/R:6.1f} ctx/path  mean theta {th[m].mean():.3f}  meas-avg {res[m].mean():+7.2f}/ctx  {res[m].sum()/R:+7.1f}/path")

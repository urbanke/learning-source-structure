"""Residual of the row-averaged copier formula per context, binned by the self-transition probability and by the visits,
over all paths of a setting (no profile costs; needs the matrices: archive npz + replayed extra paths)."""
import sys, json, os
import numpy as np
import lsa_cost as C, chain_sim as CS
from table_mc import BBInterp
from diag_rowknown import matrices_for

tag = sys.argv[1]; js = [int(x) for x in sys.argv[2].split(",")]
sim = json.load(open(f"sim_A_{tag}.json")); d, alpha, Ls = sim["d"], sim["alpha"], sim["Ls"]; ns = sim["checkpoints"]
K = list(sim["K"]); costA = list(sim["costA"]); rowH = [np.array(r) for r in sim["rowH"]]
if os.path.exists(f"sim_X_{tag}.json"):
    x = json.load(open(f"sim_X_{tag}.json"))
    K += x["K"]; costA += x["costA"]; rowH += [np.array(r) for r in x["rowH"]]
R = len(K)
Ts = matrices_for(tag, sim, R)
BB = BBInterp(f"bb_d{d}_a{alpha:g}_L{Ls}.json"); Hbar = json.load(open("exact_EH_archive.json"))[f"{d},{alpha:g},{Ls}"]
print(f"{tag}: d={d} alpha={alpha} Ls={Ls} paths {R}")
for j in js:
    n = ns[j]; th = []; kk = []; ra = []; tot = []
    for i in range(R):
        Ki = np.array(K[i][j], float); ci = np.array(costA[i][j], float); T = Ts[i]
        sel = Ki > 0
        avg = BB(Ki[sel]) + Ki[sel] * (rowH[i][sel] - Hbar)
        r = ci[sel] - avg
        th += list(np.diag(T)[sel]); kk += list(Ki[sel]); ra += list(r); tot.append(r.sum())
    th, kk, ra, tot = map(np.array, (th, kk, ra, tot))
    print(f"  n={n}: meas - avg per path {tot.mean():+.1f} bits (SE {tot.std(ddof=1)/np.sqrt(R):.1f}); copier regret meas ~ {np.mean([ (np.array(costA[i][j]) - np.array(K[i][j])*rowH[i]).sum() for i in range(R)]):.0f}")
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]:
        s = (th >= lo) & (th < hi)
        if s.sum():
            print(f"     theta in [{lo:.2f},{hi:.2f}): contexts/path {s.sum()/R:6.1f}, mean K {kk[s].mean():7.1f}, meas - avg per context {ra[s].mean():+.3f} (SE {ra[s].std(ddof=1)/np.sqrt(s.sum()):.3f}), per path {ra[s].sum()/R:+.2f}")
    for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1e9)]:
        s = (kk >= lo) & (kk < hi)
        if s.sum():
            print(f"     K in [{int(lo)},{int(hi) if hi < 1e9 else 'inf'}): contexts/path {s.sum()/R:6.1f}, mean theta {th[s].mean():.3f}, meas - avg per context {ra[s].mean():+.3f} (SE {ra[s].std(ddof=1)/np.sqrt(s.sum()):.3f}), per path {ra[s].sum()/R:+.2f}")

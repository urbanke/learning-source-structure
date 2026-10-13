"""Copier costs at the actual visit counts of simulated paths: measured, row averaged, row known.

For a setting (tag), per path i, checkpoint j and context a with K_a > 0:
  meas_a  = stored exact LSA cost of the continuation counts after a
  avg_a   = B(K_a) + K_a (H2(P_a) - Hbar)      the current formula, conditional on the actual visits
  known_a = C(nu) - theta                      the count approximation with the row known:
                                               nu = mean profile of Multinomial(K_a, P_a), theta = fluctuation term
Sums over contexts per path give the copier cost under the three views; their differences are the
row-averaging error (avg - meas) and the error of the count approximation with the row known (known - meas).
Usage: python3 diag_rowknown.py TAG NPATHS j1,j2,... out.json   (NPATHS paths: archive first, then extra)
"""
import sys, json, time
import numpy as np
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa
from table_mc import BBInterp, unigram_prediction

ARCH = "/mnt/user-data/uploads/layered_architectures/output/crossing_finite_count_prediction"


def matrices_for(tag, sim, need):
    """transition matrices of the first `need` paths of sim_AX/sim_A (archive paths, then replayed extra paths)"""
    z = np.load(f"{ARCH}/{tag}.npz"); Ts = [z["transition_matrices"][i] for i in range(min(need, z["transition_matrices"].shape[0]))]
    if need > len(Ts):
        x = json.load(open(f"sim_X_{tag}.json")); q = C.zipf(sim["d"], sim["alpha"]); rng = np.random.default_rng(x["seed"])
        nmax = max(x["checkpoints"])
        for i in range(need - len(Ts)):
            T = CS.draw_table(q, x["Ls"], rng); path = CS.simulate_path(q, T, nmax, rng)
            N = np.bincount(path[:x["checkpoints"][0]], minlength=sim["d"])
            assert N.tolist() == x["N"][i][0], "replay mismatch"
            Ts.append(T)
    return Ts


def run(tag, npaths, js, out):
    import os
    sim = json.load(open(f"sim_A_{tag}.json"))          # archived paths (K, costA, rowH per path)
    d, alpha, Ls = sim["d"], sim["alpha"], sim["Ls"]; ns = sim["checkpoints"]
    K = [r for r in sim["K"]]; costA = [r for r in sim["costA"]]; rowH = [np.array(r) for r in sim["rowH"]]
    if npaths > len(K) and os.path.exists(f"sim_X_{tag}.json"):
        x = json.load(open(f"sim_X_{tag}.json")); assert x["checkpoints"] == ns[:len(x["checkpoints"])]
        for i in range(npaths - len(K)):
            K.append(x["K"][i]); costA.append(x["costA"][i]); rowH.append(np.array(x["rowH"][i]))
    Ts = matrices_for(tag, sim, npaths)
    BB = BBInterp(f"bb_d{d}_a{alpha:g}_L{Ls}.json")
    Hbar = json.load(open("exact_EH_archive.json"))[f"{d},{alpha:g},{Ls}"]
    lsa = bb.LSAx(CS.lmax_of(d))
    res = dict(tag=tag, d=d, alpha=alpha, Ls=Ls, npaths=npaths, js=js, ns=[ns[j] for j in js], Hbar=Hbar,
               meas=[], avg=[], known=[], theta=[], Kbin=[], resid_avg=[], resid_known=[])
    for j in js:
        n = ns[j]; meas_p = []; avg_p = []; known_p = []; rows = []
        t0 = time.time()
        for i in range(npaths):
            T = Ts[i]; Ki = np.array(K[i][j], float); ci = np.array(costA[i][j], float)
            assert len(ci) == d, "costs not available at this checkpoint"
            m = a = 0.0; kn = 0.0
            for aa in np.flatnonzero(Ki > 0):
                k = int(Ki[aa]); P = T[aa]
                avg_a = BB(np.array([float(k)]))[0] + k * (rowH[i][aa] - Hbar)
                U, eta, _, _, _ = unigram_prediction(lsa, k * P, k * P * (1 - P), k)
                known_a = U - eta
                m += ci[aa]; a += avg_a; kn += known_a
                rows.append((float(P[aa]), k, float(ci[aa] - avg_a), float(known_a - ci[aa])))
            meas_p.append(m); avg_p.append(a); known_p.append(kn)
        res["meas"].append(meas_p); res["avg"].append(avg_p); res["known"].append(known_p)
        rows = np.array(rows)
        res["theta"].append(rows[:, 0].tolist()); res["Kbin"].append(rows[:, 1].tolist())
        res["resid_avg"].append(rows[:, 2].tolist()); res["resid_known"].append(rows[:, 3].tolist())
        mp, ap, kp = map(np.array, (meas_p, avg_p, known_p))
        R = len(mp)
        print(f"{tag} n={n}: paths {R}; copier cost meas {mp.mean():.1f}; avg - meas {np.mean(ap-mp):+.1f} (SE {np.std(ap-mp, ddof=1)/np.sqrt(R):.1f}); "
              f"known - meas {np.mean(kp-mp):+.1f} (SE {np.std(kp-mp, ddof=1)/np.sqrt(R):.1f}); {time.time()-t0:.0f}s", flush=True)
        # residual of the row-averaged formula by self-transition probability and by visits
        th = rows[:, 0]; kk = rows[:, 1]; ra = rows[:, 2]
        for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]:
            sel = (th >= lo) & (th < hi)
            if sel.sum():
                print(f"    theta in [{lo},{hi}): contexts {sel.sum():5d}, mean K {kk[sel].mean():7.1f}, meas - avg per context {ra[sel].mean():+.3f}, per path {ra[sel].sum()/R:+.2f}")
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4])

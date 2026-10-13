"""Row-known correction for the copier, for contexts with a large self-transition probability.

For each sampled matrix T and each context a with T(a|a) >= theta0, the correction is
   rk_a(n) = [c_known(kbar) - kbar H2(T(.|a))] - [B(kbar) - kbar Hbar],   kbar = round(E[K_a | T]),
the regret of the row itself (profile cost of Multinomial(kbar, T(.|a)) minus the fluctuation term) in place of
the row-averaged regret. It is computed at three lengths (first, middle, last checkpoint) and interpolated
linearly in ln n at the others. Output: rkC[table][length] (bits), nrk[table]; to be added to the copier
prediction of table_mc.py.  Usage: python3 table_rk.py d alpha Ls NT seed n1,n2,... bbfile theta0 Hbar out.json
"""
import sys, time, json
import numpy as np
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa
from table_mc import BBInterp, unigram_prediction, occupation_sums


def run(d, alpha, Ls, NT, seed, checkpoints, bbfile, theta0, Hbar, out):
    q = C.zipf(d, alpha); Lmax = CS.lmax_of(d); lsa = bb.LSAx(Lmax); BB = BBInterp(bbfile)
    rng = np.random.default_rng(seed)
    ns = list(checkpoints); J = len(ns); jsel = sorted(set([0, J // 2, J - 1])); lnn = np.log(np.array(ns, float))
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, checkpoints=ns, NT=NT, seed=seed, theta0=theta0, Hbar=Hbar, bbfile=bbfile,
               lengths_computed=[ns[j] for j in jsel], rkC=[], nrk=[], seconds=[])
    for it in range(NT):
        t0 = time.time()
        T = CS.draw_table(q, Ls, rng)
        S1, S2, pi = occupation_sums(q, T, ns)
        heavy = np.flatnonzero(np.diag(T) >= theta0)
        kmax_last = S1[ns[-1] - 1]
        heavy = np.array([a for a in heavy if kmax_last[a] <= lsa.rmax - 1000], dtype=int)   # beyond the tables the row-averaged block is kept at every length
        corr = np.zeros(J); done = np.zeros(J, bool)
        for j in jsel:
            n = ns[j]; muK = S1[n - 1]; c = 0.0
            for a in heavy:
                kb = int(round(muK[a]))
                if kb < 2:
                    continue
                P = T[a]
                U, eta, _, _, _ = unigram_prediction(lsa, kb * P, kb * P * (1 - P), kb)
                c += ((U - eta) - kb * CS.h2(P)) - (float(BB(np.array([float(kb)]))[0]) - kb * Hbar)
            corr[j] = c; done[j] = True
        corr = np.interp(lnn, lnn[done], corr[done])
        res["rkC"].append(corr.tolist()); res["nrk"].append(int(len(heavy))); res["seconds"].append(time.time() - t0)
        if it % 10 == 0:
            json.dump(res, open(out, "w"))
            print(f"table {it}: {time.time()-t0:.1f}s heavy {len(heavy)} correction {np.round(corr, 1).tolist()}", flush=True)
    json.dump(res, open(out, "w"))
    rk = np.array(res["rkC"])
    print("mean correction per length:", np.round(rk.mean(axis=0), 1).tolist(), " SE:", np.round(rk.std(axis=0, ddof=1) / np.sqrt(NT), 1).tolist())
    return res


if __name__ == "__main__":
    d, alpha, Ls, NT, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    checkpoints = [int(a) for a in sys.argv[6].split(",")]
    bbfile, theta0, Hbar, out = sys.argv[7], float(sys.argv[8]), float(sys.argv[9]), sys.argv[10]
    run(d, alpha, Ls, NT, seed, checkpoints, bbfile, theta0, Hbar, out)

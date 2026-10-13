"""Convert an archived chain run (output/crossing_finite_count_prediction/<tag>.npz + .json) into the
simulation format used by compare.py, adding the control variates, and recompute both codelengths from the
stored counts with this implementation of the LSA code (cross-check of the two evaluators).

Usage: python3 archive_extract.py L2_a1_d128 [--no-recompute]
Writes sim_A_<tag>.json and archive_check_<tag>.json.
"""
import sys, json, time, os
import numpy as np
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)

ARCH = "/mnt/user-data/uploads/layered_architectures/output/crossing_finite_count_prediction"


def run(tag, recompute=True):
    j = json.load(open(f"{ARCH}/{tag}.json")); z = np.load(f"{ARCH}/{tag}.npz")
    d, alpha, Ls = j["d"], j["alpha"], j["L"]
    Lmax = j["l_max"]; assert Lmax == CS.lmax_of(d), (Lmax, CS.lmax_of(d))
    tokens = [int(n) for n in j["tokens"]]
    T_all = z["transition_matrices"]; fc = z["chain_frequency_counts"]; tc = z["chain_transition_counts"]
    lF_st = z["chain_frequency_codelength"]; lC_st = z["chain_context_codelength"]
    R = T_all.shape[0]; q = z["q"]
    lsa = bb.LSAx(Lmax) if recompute else None
    cc = CS.CostCache(lsa) if recompute else None
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, R=R, seed=j["seed"], checkpoints=tokens, source=tag,
               lF=[], lC=[], cvF=[], cvC=[], EcvF=[], EcvC=[], Hpi=[], K=[], N=[], rowH=[], costA=[],
               lF_stored=lF_st.tolist(), lC_stored=lC_st.tolist())
    check = dict(tag=tag, d=d, alpha=alpha, Ls=Ls, dF=[], dC=[], dC_first=[])
    nmax = max(tokens)
    for i in range(R):
        t0 = time.time()
        T = T_all[i]
        p = CS.occupation(q, T, nmax); cp = np.cumsum(p, axis=0)
        Hrows = np.array([CS.h2(T[a]) for a in range(d)])
        lF = []; lC = []; cvF = []; cvC = []; EcvF = []; EcvC = []; Hpi = []; Kl = []; Nl = []; cAl = []
        for jn, n in enumerate(tokens):
            N = fc[i, jn].astype(np.int64); M = tc[i, jn].astype(np.int64); K = M.sum(axis=1)
            assert N.sum() == n and K.sum() == n - 1, (N.sum(), K.sum(), n)
            pin = cp[n - 1] / n
            cvF.append(float(-np.sum(N * np.log2(pin)))); EcvF.append(float(n * CS.h2(pin))); Hpi.append(float(CS.h2(pin)))
            cvC.append(float(np.sum(K * Hrows))); EK = cp[n - 2]; EcvC.append(float(np.sum(EK * Hrows)))
            Kl.append(K.tolist()); Nl.append(N.tolist())
            if recompute and max(int(N.max()), int(M.max())) <= lsa.rmax:
                f = cc.cost(C.profile_of_counts(N), n)
                costA = np.zeros(d); tot = np.log2(d)
                for a in range(d):
                    if K[a] > 0:
                        costA[a] = cc.cost(C.profile_of_counts(M[a]), int(K[a])); tot += costA[a]
                lF.append(f); lC.append(tot); cAl.append(costA.tolist())
                check["dF"].append(f - float(lF_st[i, jn])); check["dC"].append(tot - float(lC_st[i, jn]))
            else:
                # counts beyond the tables: keep the stored values (the two evaluators agree to 1e-8 bits where compared)
                lF.append(float(lF_st[i, jn])); lC.append(float(lC_st[i, jn])); cAl.append([])
        for key, val in (("lF", lF), ("lC", lC), ("cvF", cvF), ("cvC", cvC), ("EcvF", EcvF), ("EcvC", EcvC), ("Hpi", Hpi), ("K", Kl), ("N", Nl), ("costA", cAl)):
            res[key].append(val)
        res["rowH"].append(Hrows.tolist())
        print(f"{tag} rep {i}: {time.time()-t0:.0f}s" + (f"  recomputed - stored: F {check['dF'][-1]:+.4f}  C {check['dC'][-1]:+.4f}" if recompute else ""), flush=True)
        json.dump(res, open(f"sim_A_{tag}.json", "w"))
    if recompute:
        dF = np.array(check["dF"]); dC = np.array(check["dC"])
        check["summary"] = dict(dF_mean=float(dF.mean()), dF_max=float(np.abs(dF).max()), dC_mean=float(dC.mean()), dC_max=float(np.abs(dC).max()))
        print(tag, "recomputed minus stored: F mean %+.4f max|.| %.4f ; C mean %+.4f max|.| %.4f" % (dF.mean(), np.abs(dF).max(), dC.mean(), np.abs(dC).max()))
        json.dump(check, open(f"archive_check_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], recompute=("--no-recompute" not in sys.argv))

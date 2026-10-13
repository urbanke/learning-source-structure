"""Extension runs (output/crossing_lsa_extension, .../refined256, output/crossing_lsa_alpha2) -> simulation
format of archive_extract.py (control variates from the stored matrices and counts; codelengths as stored).
Runs where the archives are (numpy only). Usage: python3 ext_extract.py <dir> <tag> <outdir>"""
import sys, json, time
import numpy as np

def h2(p):
    p = p[p > 0]; return float(-np.sum(p * np.log2(p)))

def run(src, tag, outdir):
    j = json.load(open(f"{src}/{tag}.json")); z = np.load(f"{src}/{tag}.npz")
    d, alpha, Ls, Lmax = j["d"], j["alpha"], j["L"], j["l_max"]
    tokens = [int(n) for n in z["tokens"]]; J = len(tokens)
    T_all = z["transition_matrices"]; fc = z["chain_frequency_counts"]
    indptr, ind, val = z["chain_count_indptr"], z["chain_count_indices"], z["chain_count_values"]
    lF_st = z["chain_frequency_codelength"]; lC_st = z["chain_context_codelength"]
    R = T_all.shape[0]; q = z["q"]; nmax = max(tokens)
    assert len(indptr) - 1 == R * J
    res = dict(d=d, alpha=alpha, Ls=Ls, Lmax=Lmax, R=R, seed=j["seed"], checkpoints=tokens, source=f"{src}/{tag}",
               lF=[], lC=[], cvF=[], cvC=[], EcvF=[], EcvC=[], Hpi=[], K=[], N=[], rowH=[], costA=[],
               lF_stored=lF_st.tolist(), lC_stored=lC_st.tolist(),
               archive=dict(archived_mean_curve_crossing=j.get("archived_mean_curve_crossing"),
                            chain=j["methods"].get("chain", {}).get("mean_curve_crossing"),
                            count=j["methods"].get("count", {}).get("mean_curve_crossing")))
    t0 = time.time()
    need = sorted(set([n - 1 for n in tokens] + [n - 2 for n in tokens if n >= 2]))
    for i in range(R):
        T = T_all[i]
        v = q.copy(); s = np.zeros(d); cp = {}
        for t in range(nmax):
            s = s + v
            if t in need: cp[t] = s.copy()
            v = v @ T
        Hrows = np.array([h2(T[a]) for a in range(d)])
        lF = []; lC = []; cvF = []; cvC = []; EcvF = []; EcvC = []; Hpi = []; Kl = []; Nl = []
        for jn, n in enumerate(tokens):
            N = fc[i, jn].astype(np.int64)
            row = i * J + jn; sl = slice(indptr[row], indptr[row + 1])
            M = np.zeros(d * d, np.int64); M[ind[sl]] = val[sl]; M = M.reshape(d, d); K = M.sum(axis=1)
            assert N.sum() == n and K.sum() == n - 1, (N.sum(), K.sum(), n)
            pin = cp[n - 1] / n
            cvF.append(float(-np.sum(N * np.log2(pin)))); EcvF.append(float(n * h2(pin))); Hpi.append(h2(pin))
            cvC.append(float(np.sum(K * Hrows))); EK = cp[n - 2] if n >= 2 else np.zeros(d); EcvC.append(float(np.sum(EK * Hrows)))
            Kl.append(K.tolist()); Nl.append(N.tolist())
            lF.append(float(lF_st[i, jn])); lC.append(float(lC_st[i, jn]))
        for key, v_ in (("lF", lF), ("lC", lC), ("cvF", cvF), ("cvC", cvC), ("EcvF", EcvF), ("EcvC", EcvC), ("Hpi", Hpi), ("K", Kl), ("N", Nl)):
            res[key].append(v_)
        res["costA"].append([[] for _ in tokens]); res["rowH"].append(Hrows.tolist())
        if i % 16 == 0:
            print(f"{tag}: draw {i} of {R}, {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"{outdir}/sim_A_{tag}.json", "w"))
    print(f"{tag}: done, R={R}, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])

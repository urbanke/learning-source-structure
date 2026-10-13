"""Sampled-matrix prediction (per-matrix costs averaged) with many matrices: copier for all checkpoints, unigram for js."""
import sys, time, json, numpy as np
import bb, lsa_fast  # noqa
import chain_sim as CS
from table_mc import BBInterp, two_moment_pmf, support, unigram_prediction
z = np.load(sys.argv[1]); bbfile = sys.argv[2]; NTC = int(sys.argv[3]); NTU = int(sys.argv[4]); js = [int(a) for a in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
d = int(z["d"]); ns = [int(n) for n in z["ns"]]; muK, VK, muN, VN = z["muK"], z["VK"], z["muN"], z["VN"]
BB = BBInterp(bbfile); BBk = BB(np.arange(0, max(ns) + 1, dtype=float))
def EB(m, V, n):
    R = support(m, V, n); pk = two_moment_pmf(m, max(V, 0.2 * m), R); return float(pk @ BBk[:R + 1])
out = {}
for j, n in enumerate(ns):
    t0 = time.time()
    c = np.array([np.log2(d) + sum(EB(float(muK[i, j, a]), float(VK[i, j, a]), n) for a in range(d)) for i in range(NTC)])
    out[n] = dict(C=c.mean(), Cse=c.std(ddof=1) / np.sqrt(NTC))
    print(f"n={n}: copier {c.mean():.2f} ({c.std(ddof=1)/np.sqrt(NTC):.2f})  [{time.time()-t0:.0f}s]", flush=True)
lsa = bb.LSAx(CS.lmax_of(d))
for j in js:
    n = ns[j]; t0 = time.time()
    u = np.array([(lambda r: r[0] - r[1])(unigram_prediction(lsa, muN[i, j].astype(float), np.maximum(VN[i, j].astype(float), 0.2 * muN[i, j]), n)) for i in range(NTU)])
    Hpi = np.array([CS.h2(muN[i, j].astype(float) / n) for i in range(muN.shape[0])])
    reg = u - n * Hpi[:NTU]
    out[n].update(U=u.mean(), Use=u.std(ddof=1) / np.sqrt(NTU), EH=Hpi.mean(), EHse=Hpi.std(ddof=1) / np.sqrt(len(Hpi)), reg=reg.mean(), regse=reg.std(ddof=1) / np.sqrt(NTU))
    print(f"n={n}: unigram cost {u.mean():.2f} ({u.std(ddof=1)/np.sqrt(NTU):.2f}); via entropy+regret {n*Hpi.mean()+reg.mean():.2f} ({np.sqrt((n*Hpi.std(ddof=1)/np.sqrt(len(Hpi)))**2 + (reg.std(ddof=1)/np.sqrt(NTU))**2):.2f})  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(sys.argv[1].replace(".npz", "_sampled.json"), "w"), indent=1)

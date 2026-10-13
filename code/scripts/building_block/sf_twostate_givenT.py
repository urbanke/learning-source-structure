"""Given the sampled matrices: two-state moments from the true (P_a, pi_a) per matrix vs the exact moments; costs."""
import sys, time, numpy as np
import bb, lsa_fast  # noqa
import chain_sim as CS, lsa_cost as C
from table_mc import unigram_prediction, occupation_sums
tag, d, alpha, Ls, j, NT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
z = np.load(f'sf_{tag}.npz'); ns = [int(n) for n in z['ns']]; n = ns[j]; seed = int(z['seed'])
muN, VN, PI, Tdiag = z['muN'][:NT, j].astype(float), z['VN'][:NT, j].astype(float), z['pi'][:NT], z['Tdiag'][:NT]
q = C.zipf(d, alpha); lsa = bb.LSAx(CS.lmax_of(d))
# two-state moments per matrix from (P_a, pi_a)
P = np.clip(Tdiag, 1e-12, 1 - 1e-12); pi = np.clip(PI, 1e-12, 1 - 1e-12); e = np.clip(pi * (1 - P) / (1 - pi), 1e-12, 1 - 1e-12)
lam = P - e; geo = (1 - lam ** n) / (1 - lam)
m2 = n * pi + (q[None, :] - pi) * geo; V2 = n * pi * (1 - pi) * (1 + lam) / (1 - lam) - 2 * pi * (1 - pi) * lam * geo / (1 - lam) ** 2
print(f"{tag} n={n}, {NT} matrices: mean over matrices of (two-state - exact) per symbol: mean count | variance   (and totals)")
for b in [0, 1, 3, 7, 15, 31, 63, 127]:
    print(f"  b={b+1:4d}: dm {np.mean(m2[:, b] - muN[:, b]):+8.3f} (exact mean {muN[:, b].mean():8.2f}) | dV {np.mean(V2[:, b] - VN[:, b]):+8.2f} (exact var {VN[:, b].mean():8.2f})")
print(f"  totals: sum dm {np.mean(m2.sum(axis=1) - muN.sum(axis=1)):+.3f}, sum dV {np.mean(V2.sum(axis=1) - VN.sum(axis=1)):+.1f}")
t0 = time.time(); c_ex = []; c_2s = []
for i in range(min(NT, 40)):
    U, eta, *_ = unigram_prediction(lsa, muN[i], np.maximum(VN[i], 0.2 * muN[i]), n); c_ex.append(U - eta)
    mm = np.maximum(m2[i], 1e-9) * (n / m2[i].sum()); U, eta, *_ = unigram_prediction(lsa, mm, np.maximum(V2[i], 0.2 * mm), n); c_2s.append(U - eta)
c_ex, c_2s = np.array(c_ex), np.array(c_2s)
print(f"  unigram cost per matrix: two-state - exact inputs: mean {np.mean(c_2s - c_ex):+.2f} (SE {np.std(c_2s - c_ex, ddof=1)/np.sqrt(len(c_ex)):.2f})   [{time.time()-t0:.0f}s]")

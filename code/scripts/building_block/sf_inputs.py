"""Step 5 investigation: per-matrix chain inputs for NT sampled matrices (same generator as table_mc.py).
Saves muK[it, j, a] = E[K_a | T] at checkpoint n_j (positions 1..n-1), VK (variance given T, before the floor),
muN, VN (token counts, positions 1..n), Hrows[it, a] (bits), pi[it, a], g[it, a] = Z_aa - 1.
Usage: python3 sf_inputs.py d alpha Ls NT seed n1,n2,... out.npz"""
import sys, time
import numpy as np
import lsa_cost as C
import chain_sim as CS
from table_mc import occupation_sums

d, alpha, Ls, NT, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ns = [int(a) for a in sys.argv[6].split(",")]
out = sys.argv[7]
q = C.zipf(d, alpha)
rng = np.random.default_rng(seed)
J = len(ns)
muK = np.zeros((NT, J, d)); VK = np.zeros((NT, J, d)); muN = np.zeros((NT, J, d)); VN = np.zeros((NT, J, d))
Hrows = np.zeros((NT, d)); PI = np.zeros((NT, d)); G = np.zeros((NT, d)); Tdiag = np.zeros((NT, d))
t0 = time.time()
for it in range(NT):
    T = CS.draw_table(q, Ls, rng)
    S1, S2, pi = occupation_sums(q, T, ns)
    Hrows[it] = np.array([CS.h2(T[a]) for a in range(d)])
    Z = np.linalg.inv(np.eye(d) - T + np.outer(np.ones(d), pi))
    g = np.diag(Z) - 1.0
    PI[it] = pi; G[it] = g; Tdiag[it] = np.diag(T)
    for j, n in enumerate(ns):
        muK[it, j] = S1[n - 1]; VK[it, j] = S2[n - 1] + 2.0 * (n - 1) * pi * g
        muN[it, j] = S1[n]; VN[it, j] = S2[n] + 2.0 * n * pi * g
    if it % 50 == 0:
        print(f"{it}: {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(out, d=d, alpha=alpha, Ls=Ls, seed=seed, ns=np.array(ns), muK=muK.astype(np.float32), VK=VK.astype(np.float32), muN=muN.astype(np.float32), VN=VN.astype(np.float32),
                    Hrows=Hrows, pi=PI, g=G, Tdiag=Tdiag)
print("saved", out, f"{time.time()-t0:.0f}s")

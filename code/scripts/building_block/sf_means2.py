"""Store E[N_b|T] for the top 8 symbols and E[K_b|T], over many matrices, at one checkpoint."""
import sys, time, numpy as np
import lsa_cost as C, chain_sim as CS
from table_mc import occupation_sums
d, alpha, Ls, NT, seed, n = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]); out = sys.argv[7]
q = C.zipf(d, alpha); rng = np.random.default_rng(seed); vals = np.zeros((NT, 8)); valsV = np.zeros((NT, 8)); t0 = time.time()
for it in range(NT):
    T = CS.draw_table(q, Ls, rng); S1, S2, pi = occupation_sums(q, T, [n])
    Z = np.linalg.inv(np.eye(d) - T + np.outer(np.ones(d), pi)); g = np.diag(Z) - 1.0
    vals[it] = S1[n][:8]; valsV[it] = (S2[n] + 2.0 * n * pi * g)[:8]
np.savez(out, n=n, vals=vals, valsV=valsV); print("done", f"{time.time()-t0:.0f}s")

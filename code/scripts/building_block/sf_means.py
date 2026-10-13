"""Accumulate E[N_b|T] and E[K_a|T] over many matrices (sums only) at given checkpoints.  Usage: d alpha Ls NT seed n1,n2 out.npz"""
import sys, time, numpy as np
import lsa_cost as C, chain_sim as CS
from table_mc import occupation_sums
d, alpha, Ls, NT, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ns = [int(a) for a in sys.argv[6].split(",")]; out = sys.argv[7]
q = C.zipf(d, alpha); rng = np.random.default_rng(seed); J = len(ns)
s1 = np.zeros((J, d)); s2 = np.zeros((J, d)); k1 = np.zeros((J, d)); k2 = np.zeros((J, d)); hs = np.zeros(J); hs2 = np.zeros(J)
t0 = time.time()
for it in range(NT):
    T = CS.draw_table(q, Ls, rng); S1, S2, pi = occupation_sums(q, T, ns)
    for j, n in enumerate(ns):
        s1[j] += S1[n]; s2[j] += S1[n] ** 2; k1[j] += S1[n - 1]; k2[j] += S1[n - 1] ** 2
        h = CS.h2(S1[n] / n); hs[j] += h; hs2[j] += h * h
    if it % 10000 == 0: print(f"{it}: {time.time()-t0:.0f}s", flush=True)
np.savez(out, ns=np.array(ns), NT=NT, s1=s1, s2=s2, k1=k1, k2=k2, hs=hs, hs2=hs2)
print("done", f"{time.time()-t0:.0f}s")

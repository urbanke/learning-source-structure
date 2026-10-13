"""Monte Carlo check of the row-known Laplace prediction (known = C(nu) - theta) for heavy contexts:
exact expected LSA cost of Multinomial(K, P_a) profiles vs the formula, and the same with runs known."""
import sys, json, time
import numpy as np
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa
from diag_runaware import counts_for

tag, nsel, ndraw, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
D = json.load(open(f"diag_ra_{tag}.json")); sim = json.load(open(f"sim_A_{tag}.json"))
d = D["d"]; lsa = bb.LSAx(CS.lmax_of(d)); rng = np.random.default_rng(seed)
data = counts_for(tag, sim, D["npaths"], D["ns"])
rows = [r for r in D["rows"] if r["n"] == D["ns"][-1] and 0.02 <= r["theta"] < 0.4]
rng.shuffle(rows); rows = rows[:nsel]
def profile_cost(m):
    prof = {}
    for v in m[m > 0]: prof[int(v)] = prof.get(int(v), 0) + 1
    prof[0] = int((m == 0).sum()); return lsa.cost_bits(prof, float(m.sum()))
print("theta    K    meas    known   MC(known)  se    known-MC | runaware  MC(runaware)  runaware-MC")
t0 = time.time()
for r in rows:
    T, _ = data[r["path"]]; P = T[r["a"]]; K = r["K"]; S = r["S"]; R = r["R"]
    cs = [profile_cost(rng.multinomial(K, P)) for _ in range(ndraw)]
    Pe = P.copy(); Pe[r["a"]] = 0; Pe /= Pe.sum()
    cr = []
    for _ in range(ndraw):
        m = rng.multinomial(R, Pe); m[r["a"]] += S; cr.append(profile_cost(m))
    mk, sk = np.mean(cs), np.std(cs, ddof=1) / np.sqrt(ndraw); mr, sr = np.mean(cr), np.std(cr, ddof=1) / np.sqrt(ndraw)
    print(f"{r['theta']:.3f} {K:5d} {r['meas']:8.1f} {r['known']:8.1f} {mk:8.1f} {sk:5.2f} {r['known']-mk:+7.2f} | {r['runaware']:8.1f} {mr:8.1f} {r['runaware']-mr:+7.2f}", flush=True)
print(f"{time.time()-t0:.0f}s")

"""K-law check on chain_sim.py paths (matrices regenerated from the seed): sum_a B(K_a) at the actual visits vs
sum_a E[B(K_a) | T] under the two-moment law of table_mc. Usage: python3 diag_klaw2.py sim_file bb2_file j1,j2,.."""
import sys, json
import numpy as np
import lsa_cost as C, chain_sim as CS
from table_mc import BBInterp, two_moment_pmf, support, occupation_sums
simf, bbf, js = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
s = json.load(open(simf)); d, alpha, Ls, ns, R = s["d"], s["alpha"], s["Ls"], s["checkpoints"], s["R"]; q = C.zipf(d, alpha)
rng = np.random.default_rng(s["seed"]); nmax = max(ns); Ts = []
for i in range(R):
    T = CS.draw_table(q, Ls, rng); x = CS.simulate_path(q, T, nmax, rng)
    assert np.bincount(x[:ns[0]], minlength=d).tolist() == s["N"][i][0]; Ts.append(T)
BB = BBInterp(bbf); BBk = BB(np.arange(0, nmax + 1, dtype=float))
for j in js:
    n = ns[j]; act = []; pred = []; heavy_act = []; heavy_pred = []
    for i in range(R):
        T = Ts[i]; Kj = np.array(s["K"][i][j], float)
        S1, S2, pi = occupation_sums(q, T, [n]); Z = np.linalg.inv(np.eye(d) - T + np.outer(np.ones(d), pi)); g = np.diag(Z) - 1.0
        muK = S1[n - 1]; VK = np.maximum(S2[n - 1] + 2.0 * (n - 1) * pi * g, 0.2 * muK)
        tot = 0.0; th = 0.0; ta = 0.0
        for a in range(d):
            Rr = support(muK[a], VK[a], n); pk = two_moment_pmf(muK[a], VK[a], Rr); e = float(pk @ BBk[:Rr + 1]); tot += e
            if T[a, a] >= 0.2: th += e; ta += float(BB(np.array([Kj[a]]))[0]) if Kj[a] > 0 else 0.0
        act.append(float(BB(Kj[Kj > 0]).sum())); pred.append(tot); heavy_act.append(ta); heavy_pred.append(th)
    act, pred, ha, hp = map(np.array, (act, pred, heavy_act, heavy_pred))
    print(f"n={n}: sum_a B(K_a): two-moment law {pred.mean():.1f} vs actual visits {act.mean():.1f}: diff {np.mean(pred-act):+.1f} (SE {np.std(pred-act, ddof=1)/np.sqrt(R):.1f}); "
          f"contexts with theta >= 0.2: {np.mean(hp-ha):+.1f} (SE {np.std(hp-ha, ddof=1)/np.sqrt(R):.1f})", flush=True)

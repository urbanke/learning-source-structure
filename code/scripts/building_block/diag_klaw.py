"""K-law check: on the paths of a setting (same matrices), compare sum_a B(K_a) at the actual visits with
sum_a E[B(K_a) | T] under the two-moment (Poisson/binomial/NB) law used by table_mc, per checkpoint."""
import sys, json, os
import numpy as np
import lsa_cost as C, chain_sim as CS
from table_mc import BBInterp, two_moment_pmf, support, occupation_sums
from diag_rowknown import matrices_for

tag, bbfile = sys.argv[1], sys.argv[2]; js = [int(x) for x in sys.argv[3].split(",")]
sim = json.load(open(f"sim_A_{tag}.json")); d, alpha, Ls = sim["d"], sim["alpha"], sim["Ls"]; ns = sim["checkpoints"]
K = list(sim["K"])
if os.path.exists(f"sim_X_{tag}.json"):
    x = json.load(open(f"sim_X_{tag}.json")); K += x["K"]
R = len(K); Ts = matrices_for(tag, sim, R); q = C.zipf(d, alpha)
BB = BBInterp(bbfile); nmax = max(ns[j] for j in js); BBk = BB(np.arange(0, nmax + 1, dtype=float))
print(f"{tag}: {R} paths, grid {bbfile}")
for j in js:
    n = ns[j]; act = []; pred = []; predP = []
    for i in range(R):
        T = Ts[i]; Kj = np.array(K[i][j], float)
        act.append(float(BB(Kj[Kj > 0]).sum()))
        S1, S2, pi = occupation_sums(q, T, [n])
        Z = np.linalg.inv(np.eye(d) - T + np.outer(np.ones(d), pi)); g = np.diag(Z) - 1.0
        muK = S1[n - 1]; VK = np.maximum(S2[n - 1] + 2.0 * (n - 1) * pi * g, 0.2 * muK)
        tot = 0.0; totP = 0.0
        for a in range(d):
            Rr = support(muK[a], VK[a], n); pk = two_moment_pmf(muK[a], VK[a], Rr); tot += float(pk @ BBk[:Rr + 1])
            pkP = two_moment_pmf(muK[a], muK[a], Rr); totP += float(pkP @ BBk[:Rr + 1])   # Poisson (no Markov variance)
        pred.append(tot); predP.append(totP)
    act, pred, predP = map(np.array, (act, pred, predP))
    print(f"  n={n}: E_T sum_a B(K_a): two-moment law {pred.mean():.1f}, actual visits {act.mean():.1f}; "
          f"two-moment - actual {np.mean(pred-act):+.1f} (SE {np.std(pred-act, ddof=1)/np.sqrt(R):.1f}); Poisson - actual {np.mean(predP-act):+.1f}", flush=True)

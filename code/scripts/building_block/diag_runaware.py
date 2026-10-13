"""Count effect for the heavy contexts: does the dependence between the visits K_a and the self-loops
explain the residual of the row-known prediction?

Per path, checkpoint and heavy context a (T(a|a) >= theta0, K_a >= kmin), three predictions of the exact
cost of the continuation counts after a (meas, from the stored exact costs):
  avg      = B(K) + K (H2(P_a) - Hbar)                       row averaged (the formula of the paper)
  known    = C(nu) - theta with nu = Multinomial(K, P_a)     row known, independent draws
  runaware = C(nu) - theta with the self-loop count fixed at its actual value S = N[a,a] and the
             R = K - S exits drawn from P_a(.|not a)          row known, runs known
Usage: python3 diag_runaware.py TAG NPATHS j1,j2,... theta0 kmin out.json
"""
import sys, json, time, os
import numpy as np
import lsa_cost as C
import bb, chain_sim as CS
import lsa_fast  # noqa
from table_mc import BBInterp, unigram_prediction

ARCH = "/mnt/user-data/uploads/layered_architectures/output/crossing_finite_count_prediction"


def counts_for(tag, sim, need, ns):
    """(T, counts[j] (d x d)) for the first `need` paths: archived paths from the npz, extra paths replayed."""
    z = np.load(f"{ARCH}/{tag}.npz"); toks = [int(t) for t in z["tokens"]]
    jmap = [toks.index(n) for n in ns]
    out = []
    for i in range(min(need, z["transition_matrices"].shape[0])):
        out.append((z["transition_matrices"][i], [z["chain_transition_counts"][i][jj] for jj in jmap]))
    if need > len(out):
        x = json.load(open(f"sim_X_{tag}.json")); q = C.zipf(sim["d"], sim["alpha"]); rng = np.random.default_rng(x["seed"])
        nmax = max(x["checkpoints"]); d = sim["d"]
        for i in range(need - len(out)):
            T = CS.draw_table(q, x["Ls"], rng); path = CS.simulate_path(q, T, nmax, rng)
            N0 = np.bincount(path[:x["checkpoints"][0]], minlength=d)
            assert N0.tolist() == x["N"][i][0], "replay mismatch"
            cnts = []
            for n in ns:
                M = np.zeros((d, d), dtype=np.int64)
                np.add.at(M, (path[:n - 1], path[1:n]), 1)
                cnts.append(M)
            out.append((T, cnts))
    return out


def run(tag, npaths, js, theta0, kmin, out):
    sim = json.load(open(f"sim_A_{tag}.json"))
    d, alpha, Ls = sim["d"], sim["alpha"], sim["Ls"]; ns_all = sim["checkpoints"]; ns = [ns_all[j] for j in js]
    K = [r for r in sim["K"]]; costA = [r for r in sim["costA"]]; rowH = [np.array(r) for r in sim["rowH"]]
    if npaths > len(K) and os.path.exists(f"sim_X_{tag}.json"):
        x = json.load(open(f"sim_X_{tag}.json")); assert x["checkpoints"] == ns_all[:len(x["checkpoints"])]
        for i in range(npaths - len(K)):
            K.append(x["K"][i]); costA.append(x["costA"][i]); rowH.append(np.array(x["rowH"][i]))
    data = counts_for(tag, sim, npaths, ns)
    BB = BBInterp(f"bb2_d{d}_a{alpha:g}_L{Ls}.json")
    Hbar = json.load(open("exact_EH_archive.json"))[f"{d},{alpha:g},{Ls}"]
    lsa = bb.LSAx(CS.lmax_of(d))
    res = dict(tag=tag, d=d, alpha=alpha, Ls=Ls, npaths=npaths, js=js, ns=ns, theta0=theta0, kmin=kmin, rows=[])
    t0 = time.time()
    for jj, j in enumerate(js):
        n = ns[jj]
        for i in range(npaths):
            T, cnts = data[i]; M = cnts[jj]; Ki = np.array(K[i][j], float); ci = np.array(costA[i][j], float)
            assert len(ci) == d, "costs not available at this checkpoint"
            assert np.array_equal(M.sum(axis=1), Ki.astype(np.int64)), "count / visit mismatch"
            for a in np.flatnonzero((np.diag(T) >= theta0) & (Ki >= kmin)):
                k = int(Ki[a]); P = T[a]; th = float(P[a]); S = int(M[a, a]); R = k - S
                avg = float(BB(np.array([float(k)]))[0]) + k * (rowH[i][a] - Hbar)
                U, eta, _, _, _ = unigram_prediction(lsa, k * P, k * P * (1 - P), k)
                known = U - eta
                Pe = P.copy(); Pe[a] = 0.0; Pe = Pe / Pe.sum()
                mu = R * Pe; V = R * Pe * (1 - Pe); mu[a] = S; V[a] = 0.75
                U2, eta2, _, _, _ = unigram_prediction(lsa, mu, V, k)
                runaware = U2 - eta2
                res["rows"].append(dict(n=n, path=i, a=int(a), theta=th, K=k, S=S, R=R, ES=k * th, meas=float(ci[a]),
                                        avg=float(avg), known=float(known), runaware=float(runaware)))
        json.dump(res, open(out, "w"))
        print(f"{tag} n={n}: {sum(1 for r in res['rows'] if r['n'] == n)} heavy contexts, {time.time()-t0:.0f}s", flush=True)
    return res


def report(res):
    rows = res["rows"]; npaths = res["npaths"]
    edges = [0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    for n in res["ns"]:
        rr = [r for r in rows if r["n"] == n]
        print(f"== {res['tag']} n={n}: {len(rr)/npaths:.1f} heavy contexts per path")
        print("   theta bin        ctx/path  meanK   S-ES/K   meas-avg   meas-known   meas-runaware   (bits per path; per context in brackets)")
        tot = np.zeros(3)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in rr if lo <= r["theta"] < hi]
            if not sel: continue
            A = np.array([[r["meas"] - r["avg"], r["meas"] - r["known"], r["meas"] - r["runaware"]] for r in sel])
            dev = np.mean([(r["S"] - r["ES"]) / r["K"] for r in sel])
            s = A.sum(axis=0) / npaths; m = A.mean(axis=0); tot += s
            print(f"   [{lo:.2f},{hi:.2f})   {len(sel)/npaths:6.2f}  {np.mean([r['K'] for r in sel]):6.0f}   {dev:+.3f}   {s[0]:+7.1f} ({m[0]:+6.2f})  {s[1]:+7.1f} ({m[1]:+6.2f})  {s[2]:+7.1f} ({m[2]:+6.2f})")
        print(f"   total per path                                  {tot[0]:+7.1f}           {tot[1]:+7.1f}            {tot[2]:+7.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(json.load(open(sys.argv[2])))
    else:
        r = run(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
        report(r)

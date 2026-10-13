"""Summary statistics of the crossing comparison over all settings with a crossing (summary_all.json)."""
import json, os, sys
import numpy as np
S = json.load(open(os.environ.get("SUMMARY", "summary_all.json")))
rows = [v for v in S.values() if v["x_meas"] is not None and v["x_pred"] is not None]
E = np.array([100 * (v["x_pred"] - v["x_meas"]) / v["x_meas"] for v in rows])
SE = np.array([100 * np.sqrt(v["x_se"] ** 2 + v["x_pse"] ** 2) / v["x_meas"] for v in rows])
SEm = np.array([100 * v["x_se"] / v["x_meas"] for v in rows])
print(f"{len(rows)} settings with a crossing: median |e| {np.median(abs(E)):.1f}%, max {abs(E).max():.1f}%, mean {E.mean():+.1f}%, "
      f"within combined SE {(abs(E) <= SE).sum()}, within 2 combined SE {(abs(E) <= 2*SE).sum()}, within the simulation's SE {(abs(E) <= SEm).sum()}, within 5% {(abs(E) <= 5).sum()}, within 10% {(abs(E) <= 10).sum()}")
for sel, nm in [(lambda v: v["alpha"] <= 1, "alpha <= 1"), (lambda v: v["alpha"] > 1, "alpha > 1"), (lambda v: v["Ls"] <= 3, "Ls <= 3"), (lambda v: v["Ls"] >= 6, "Ls >= 6")]:
    idx = [i for i, v in enumerate(rows) if sel(v)]
    if not idx: continue
    e = E[idx]; print(f"   {nm:10s}: {len(idx)} settings, median |e| {np.median(abs(e)):.1f}%, max {abs(e).max():.1f}%, mean {e.mean():+.1f}%, within combined SE {(abs(e) <= SE[idx]).sum()}")
for v, e, se in sorted(zip(rows, E, SE), key=lambda t: -abs(t[1]))[:6]:
    print(f"   largest: {v['tag']:14s} {e:+.1f}% (combined SE {se:.1f}%), slope of the predicted gap at the crossing:", end=" ")
    ns = np.array(v["ns"], float); gp = np.array([r["gap_pred"] for r in v["rows"]]); i = max(j for j in range(len(ns)-1) if ns[j] <= v["x_pred"])
    print(f"{(gp[i+1]-gp[i])/(ns[i+1]-ns[i]):.2f} bits/token")
# slope range over all settings (predicted gap at the predicted crossing)
sl = []
for v in rows:
    ns = np.array(v["ns"], float); gp = np.array([r["gap_pred"] for r in v["rows"]]); i = max(j for j in range(len(ns)-1) if ns[j] <= v["x_pred"]); sl.append((v["tag"], (gp[i+1]-gp[i])/(ns[i+1]-ns[i]), v["x_pred"]))
sl.sort(key=lambda t: t[1]); print("slopes (bits/token) at the crossing:", [(t[0], round(t[1], 2)) for t in sl[:4]], "...", [(t[0], round(t[1], 2)) for t in sl[-3:]])
print("ten bits within (tokens):", [(t[0], round(10/t[1]), f"{100*10/t[1]/t[2]:.1f}% of n*") for t in sl[:5]])

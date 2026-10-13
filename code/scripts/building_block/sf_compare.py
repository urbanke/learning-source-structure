"""Compare the sampling-free prediction (sfp_<tag>.json) with the sampled-matrix prediction of the paper (tab_M_<tag>.json,
summary_all.json) and the simulation.  The sampled copier cost is mean(predC) + mean(covC) + (n-1)(Hbar_MC - Hbar), with
the corrected exact Hbar of the sampling-free file; the sampled unigram cost is the paper's construction (mean regret over
the unigram tables plus the 600-matrix mean of n H2(pi^(n))).  Usage: python3 sf_compare.py [tag ...]"""
import sys, json, os
import numpy as np
from compare import crossing



def load_tables(tag):
    """The sampled-matrix predictions: tab_M_<tag>.json, or the "tables" part of results/building_block/chains_<tag>.json."""
    if os.path.exists(f"tab_M_{tag}.json"):
        return json.load(open(f"tab_M_{tag}.json"))
    for p in [f"chains_{tag}.json", f"../../../results/building_block/chains_{tag}.json"]:
        if os.path.exists(p):
            return json.load(open(p))["tables"]
    raise FileNotFoundError(f"tab_M_{tag}.json or chains_{tag}.json")


s = json.load(open("summary_all.json"))
tags = sys.argv[1:] or [t for t in s if os.path.exists(f"sfp_{t}.json")]
out = {}
print("setting          |  n* sim (SE) |  n* sampled (SE) | n* sampling-free | sf-sampled | sf-sim  || at the crossing: copier sf-sampled (bits) | unigram sf-sampled | gap slope")
for tag in tags:
    v = s[tag]; f = json.load(open(f"sfp_{tag}.json")); t = load_tables(tag)
    ns = [r["n"] for r in f["rows"]]; assert ns == v["ns"], (tag, ns, v["ns"])
    Hb = f["Hbar_bits"]
    pC = np.array(t["predC"]); covC = np.array(t["covC"]); pU = np.array(t["predF_U"]); pe = np.array(t["predF_eta"])
    EcvF = np.array(t["EcvF"]); EcvF_occ = np.array(t["EcvF_occ"])
    Cs = [pC[:, j].mean() + covC[:, j].mean() + (n - 1) * (t["Hbar"] - Hb) for j, n in enumerate(ns)]
    Fs = [(pU[:, j] - pe[:, j] - EcvF[:, j]).mean() + EcvF_occ[:, j].mean() for j, n in enumerate(ns)]
    gs = [F - C for F, C in zip(Fs, Cs)]
    gap_sf = [r["gap"] for r in f["rows"]]
    gm = [r["gap_meas"] for r in v["rows"]]
    xs = crossing(ns, gs); xf = crossing(ns, gap_sf); xm = crossing(ns, gm)
    # differences at the bracketing lengths of the sampling-free crossing
    if xf is not None:
        i = max(j for j in range(len(ns) - 1) if ns[j] <= xf)
        dC = 0.5 * ((f["rows"][i]["C"] - Cs[i]) + (f["rows"][i + 1]["C"] - Cs[i + 1]))
        dF = 0.5 * ((f["rows"][i]["F"] - Fs[i]) + (f["rows"][i + 1]["F"] - Fs[i + 1]))
        slope = (gap_sf[i + 1] - gap_sf[i]) / (ns[i + 1] - ns[i])
    else:
        dC = dF = slope = float("nan")
    xse = v["x_se"]; xpse = v["x_pse"]
    fmt = lambda x: f"{x:8.0f}" if x is not None else "    none"
    rel = lambda a, b: f"{100 * (a - b) / b:+6.1f}%" if (a is not None and b is not None) else "   n/a "
    print(f"{tag:16s} | {fmt(xm)} ({xse:4.0f}) | {fmt(xs)} ({xpse:4.0f}) | {fmt(xf)} | {rel(xf, xs)} | {rel(xf, xm)} || {dC:+7.1f} | {dF:+7.1f} | {slope:6.3f}")
    out[tag] = dict(d=v["d"], alpha=v["alpha"], Ls=v["Ls"], ns=ns, x_meas=xm, x_se=xse, x_sampled=xs, x_pse=xpse, x_sf=xf, dC=dC, dF=dF, slope=slope,
                    C_sf=[r["C"] for r in f["rows"]], C_sampled=Cs, F_sf=[r["F"] for r in f["rows"]], F_sampled=Fs, gap_sf=gap_sf, gap_sampled=gs, gap_meas=gm,
                    cov_sf=[r["cov"] for r in f["rows"]], cov_sampled=[covC[:, j].mean() + (n - 1) * (t["Hbar"] - Hb) for j, n in enumerate(ns)],
                    seconds=f["seconds"], Hbar=Hb)
json.dump(out, open("sf_compare.json", "w"), indent=1)

"""Merge prediction files of the same setting and checkpoints (different seeds) into one: the per-table lists are
concatenated; the covariance term is re-centred at the exact mean row entropy (Hbar := exact E[H(P)]).
Usage: python3 merge_tabs.py out.json in1.json in2.json [...]"""
import sys, json
import numpy as np
from compare_all import exact_EH
out, ins = sys.argv[1], sys.argv[2:]
ts = [json.load(open(f)) for f in ins]
t0 = ts[0]; ns = t0["checkpoints"]; EH = exact_EH(t0["d"], t0["alpha"], t0["Ls"])
m = {k: t0[k] for k in ["d", "alpha", "Ls", "Lmax", "checkpoints"]}
m["sources"] = ins; m["Hbar"] = EH; m["NT_pred"] = 0; m["NT_occ"] = 0
for k in ["predF_U", "predF_eta", "predC", "EcvF", "EcvC", "EcvF_occ", "EcvC_occ", "Hpi_occ", "seconds", "w1", "depth1", "depth1_st", "covC"]:
    m[k] = []
for t in ts:
    assert t["checkpoints"] == ns and (t["d"], t["alpha"], t["Ls"]) == (t0["d"], t0["alpha"], t0["Ls"])
    shift = [(n - 1.0) * (t["Hbar"] - EH) for n in ns]
    for k in ["predF_U", "predF_eta", "predC", "EcvF", "EcvC", "EcvF_occ", "EcvC_occ", "Hpi_occ", "seconds", "w1", "depth1", "depth1_st"]:
        m[k] += t[k]
    m["covC"] += [[c + s for c, s in zip(row, shift)] for row in t["covC"]]
    m["NT_pred"] += t["NT_pred"]; m["NT_occ"] += len(t["predC"])
json.dump(m, open(out, "w"))
print(out, "tables:", m["NT_pred"], "unigram,", m["NT_occ"], "copier")

"""Merge the archived paths of a setting (sim_A_<tag>.json) with the extra paths of chain_sim.py (sim_X_<tag>.json and
sim_X2_<tag>.json if present) into sim_AX_<tag>.json, keeping the checkpoints of the extra paths (a subset of the
archive's, matched by value). Usage: python3 merge_ext.py <tag>"""
import sys, json, os
tag = sys.argv[1]
a = json.load(open(f"sim_A_{tag}.json")); xs = [json.load(open(f)) for f in [f"sim_X_{tag}.json", f"sim_X2_{tag}.json"] if os.path.exists(f)]
nx = xs[0]["checkpoints"]; idx = [a["checkpoints"].index(n) for n in nx]
out = {k: a[k] for k in ["d", "alpha", "Ls", "Lmax"]}
out["source"] = a.get("source", tag); out["checkpoints"] = nx
for k in ["lF", "lC", "cvF", "cvC", "EcvF", "EcvC", "Hpi"]:
    out[k] = [[row[i] for i in idx] for row in a[k]]
out["rowH"] = list(a["rowH"]); out["R"] = a["R"]; out["R_archive"] = a["R"]; out["R_extra"] = 0; out["seeds"] = [a["seed"]]
for x in xs:
    assert x["checkpoints"] == nx and (a["d"], a["alpha"], a["Ls"]) == (x["d"], x["alpha"], x["Ls"]), "mismatch"
    for k in ["lF", "lC", "cvF", "cvC", "EcvF", "EcvC", "Hpi"]:
        out[k] += x[k]
    out["rowH"] += x["rowH"]; out["R"] += x["R"]; out["R_extra"] += x["R"]; out["seeds"].append(x["seed"])
if "archive" in a: out["archive"] = a["archive"]
json.dump(out, open(f"sim_AX_{tag}.json", "w"))
print(tag, "merged: R =", out["R"], "(archive", out["R_archive"], "+ extra", out["R_extra"], ")")

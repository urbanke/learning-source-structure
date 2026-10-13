"""Merge an archived setting (sim_A_<tag>.json, 10 checkpoints) with extra paths (sim_X_<tag>.json, first 9
checkpoints) into sim_AX_<tag>.json used by archive_compare.py."""
import sys, json
tag = sys.argv[1]
a = json.load(open(f"sim_A_{tag}.json")); x = json.load(open(f"sim_X_{tag}.json"))
nx = x["checkpoints"]; assert nx == a["checkpoints"][:len(nx)], (nx, a["checkpoints"])
out = {k: a[k] for k in ["d", "alpha", "Ls", "Lmax", "source"]}
out["checkpoints"] = nx
keys = ["lF", "lC", "cvF", "cvC", "EcvF", "EcvC", "Hpi"]
for k in keys:
    out[k] = [row[:len(nx)] for row in a[k]] + x[k]
out["rowH"] = a["rowH"] + x["rowH"]
out["R"] = a["R"] + x["R"]; out["R_archive"] = a["R"]; out["R_extra"] = x["R"]; out["seeds"] = [a["seed"], x["seed"]]
json.dump(out, open(f"sim_AX_{tag}.json", "w"))
print(tag, "merged: R =", out["R"])

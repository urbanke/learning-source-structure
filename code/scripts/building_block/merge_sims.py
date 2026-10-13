"""Merge chain simulation files with the same settings (different seeds) into one."""
import sys, json
out = sys.argv[1]; files = sys.argv[2:]
base = json.load(open(files[0]))
for f in files[1:]:
    s = json.load(open(f))
    assert (s["d"], s["alpha"], s["Ls"], s["checkpoints"]) == (base["d"], base["alpha"], base["Ls"], base["checkpoints"])
    for key in ["lF", "lC", "cvF", "cvC", "EcvF", "EcvC", "Hpi", "seconds", "K", "costA", "rowH", "N"]:
        base[key] = base[key] + s[key]
    base["R"] += s["R"]
base["seeds"] = [json.load(open(f))["seed"] for f in files]
json.dump(base, open(out, "w"))
print(out, "R =", base["R"])

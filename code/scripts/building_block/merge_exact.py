"""Merge the exact small-k building block (bbx_d{d}_a{alpha}_L{Ls}.json, k <= K0) into a grid file:
rows with k <= K0 are replaced by the exact values (one row per integer k), rows with k > K0 are kept.
Usage: python3 merge_exact.py bb_d128_a1.25_L2.json bbx_d128_a1.25_L2.json bb2_d128_a1.25_L2.json"""
import sys, json
grid, exact, out = sys.argv[1:4]
g = json.load(open(grid)); x = json.load(open(exact))
K0 = x["K0"]
old = {r["k"]: r["cost"] for r in g["rows"]}
rows = [dict(k=r["k"], cost=r["cost"], exact=True) for r in x["rows"]] + [r for r in g["rows"] if r["k"] > K0]
rows.sort(key=lambda r: r["k"])
g2 = dict(g); g2["rows"] = rows; g2["exact_upto"] = K0; g2["source_exact"] = exact
json.dump(g2, open(out, "w"), indent=1)
diffs = [(r["k"], r["cost"] - old[r["k"]]) for r in x["rows"] if r["k"] in old]
print(out, "exact - formula at common k:", ", ".join(f"{k}: {d:+.3f}" for k, d in diffs))

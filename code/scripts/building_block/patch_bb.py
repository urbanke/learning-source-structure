"""Recompute small-k rows (exact scale integral) and fill missing k in a bb grid file."""
import sys, json, time
import numpy as np
import bb, lsa_cost as C
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)
path = sys.argv[1]; kmax = int(sys.argv[2])
j = json.load(open(path))
d, alpha, Ls, Lmax = j["d"], j["alpha"], j["Ls"], j["Lmax"]
q = C.zipf(d, alpha); lsa = bb.LSAx(Lmax, rmax=kmax + 5)
if lsa.rmax < kmax:
    print(f'tables cover r <= {lsa.rmax}; capping kmax {kmax} -> {lsa.rmax}', flush=True); kmax = lsa.rmax
xhalf = 1.6 if Ls <= 2 else 2.2
rows = {r["k"]: r for r in j["rows"]}
todo = [k for k in bb.KGRID if k <= kmax and k not in rows]
for k in todo:
    t0 = time.time()
    r = bb.building_block(lsa, q, Ls, k, xhalf=xhalf); r["seconds"] = time.time() - t0
    rows[k] = r
    print(f"{path}: k={k} cost={r['cost']:.4f} ({r['seconds']:.0f}s)", flush=True)
    j["rows"] = [rows[k] for k in sorted(rows)]
    json.dump(j, open(path, "w"), indent=1)
print("done", path)

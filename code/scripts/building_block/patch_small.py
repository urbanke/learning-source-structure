"""Recompute the small-k rows of a grid file with a scale window wide enough for the broad
scale density at small k: xhalf(k) = max(base, 1 + 3/sqrt(k))."""
import sys, json, time
import numpy as np
import bb, lsa_cost as C
import lsa_fast  # noqa: spline-free evaluator (replaces bb.LSAx)
path = sys.argv[1]; kcut = int(sys.argv[2]) if len(sys.argv) > 2 else 24
j = json.load(open(path))
d, alpha, Ls, Lmax = j["d"], j["alpha"], j["Ls"], j["Lmax"]
q = C.zipf(d, alpha); lsa = bb.LSAx(Lmax, rmax=kcut + 5)
base = 1.6 if Ls <= 2 else 2.2
rows = {r["k"]: r for r in j["rows"]}
for k in [k for k in bb.KGRID if k <= kcut]:
    xh = max(base, 1.0 + 3.0 / np.sqrt(k))
    t0 = time.time()
    r = bb.building_block(lsa, q, Ls, k, xhalf=xh); r["seconds"] = time.time() - t0; r["xhalf"] = xh
    old = rows.get(k, {}).get("cost", float("nan")); oldm = rows.get(k, {}).get("mass", float("nan"))
    rows[k] = r
    print(f"{path}: k={k} xhalf={xh:.2f} cost={r['cost']:.4f} mass={r['mass']:.5f} (was {old:.4f}, mass {oldm:.4f}; change {r['cost']-old:+.4f}) [{r['seconds']:.0f}s]", flush=True)
    j["rows"] = [rows[k] for k in sorted(rows)]
    json.dump(j, open(path, "w"), indent=1)
print("done", path)

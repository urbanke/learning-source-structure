"""Monte Carlo truths of B(k) at (128, 1.25, 2) for small k, against the grid (bb_d128_a1.25_L2.json)."""
import sys, json, numpy as np
from single_val2 import truth
from table_mc import BBInterp
ndraw = int(sys.argv[1]); out = sys.argv[2]
BB = BBInterp("bb_d128_a1.25_L2.json")
rows = []
for k in [2, 4, 8, 16, 32, 64]:
    r = truth(128, 1.25, 2, k, 23, ndraw, 3000 + k, 2.2)
    r["formula"] = float(BB(np.array([float(k)]))[0]); r["diff"] = r["formula"] - r["mc"]
    rows.append(r); json.dump(rows, open(out, "w"), indent=1)
    print(f"k={k:3d}: MC {r['mc']:.3f} +- {r['se']:.3f} (raw SE {r['se_raw']:.3f}); formula {r['formula']:.3f}; formula - MC {r['diff']:+.3f}; mass {r['mass']:.4f}; {r['t_profile']:.0f}+{r['t_mc']:.0f}s", flush=True)

"""Tail exponents from EXACT crossings (jed rerun, d 2048-32768).

Reads the per-seed exact files (exact_<cell>_s<k>.json) and fits, per
(alpha, L) line, the slope of log2 n* against log2 d over the top-4
available d points -- the same convention as cube_analysis.py, but on
exact crossings instead of grid checkpoints, and one decade higher.

Three statistics per seed, per RU's open averaging question:
  last  = first_pos       (the exact last crossing)
  first = first_cross     (gap-rule first crossing; = last if undefined)
  mid   = (first + last)/2
Cell value = median over seeds; censored seeds excluded (and counted).

    python code/scripts/exact_verdict.py /tmp/jed_exact
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONJ = lambda a: min(0.5 + 1 / a, 1.75)


def load(root: Path):
    cells = defaultdict(list)
    for f in root.glob("exact_*.json"):
        r = json.loads(f.read_text())
        for s in r["seeds"]:
            cells[(r["d"], r["alpha"], r["L"])].append(
                (s, r.get("prior_median")))
    out = {}
    for key, rows in cells.items():
        last, first, mid = [], [], []
        n_cens = n_start = 0
        for s, _ in rows:
            if s.get("note", "").startswith("censored"):
                n_cens += 1
                continue
            if s.get("note") == "positive from the start":
                n_start += 1
                continue
            lv = s["first_pos"]
            fv = s.get("first_cross") or lv
            last.append(lv)
            first.append(fv)
            mid.append(0.5 * (fv + lv))
        if last:
            out[key] = dict(
                n=len(last), n_cens=n_cens, n_start=n_start,
                last=float(np.median(last)),
                first=float(np.median(first)),
                mid=float(np.median(mid)),
                last_se=float(np.std(last) / np.sqrt(len(last))),
                prior=rows[0][1])
    return out


def fit(ds, vs):
    x, y = np.log2(ds), np.log2(vs)
    (sl, _), cov = np.polyfit(x, y, 1, cov=True)
    return sl, float(np.sqrt(cov[0, 0]))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/jed_exact")
    cells = load(root)
    print(f"{len(cells)} cells", flush=True)

    # seed counts and prior agreement
    per_d = defaultdict(list)
    ratios = []
    for (d, a, L), c in cells.items():
        per_d[d].append(c["n"])
        if c["prior"]:
            ratios.append(c["last"] / c["prior"])
    for d in sorted(per_d):
        ns = per_d[d]
        print(f"  d={d:6d}: {len(ns):3d} cells, seeds "
              f"min/med {min(ns)}/{int(np.median(ns))}")
    ratios = np.array(ratios)
    print(f"exact/prior median ratio: median {np.median(ratios):.3f}, "
          f"central 90pc [{np.quantile(ratios, .05):.3f}, "
          f"{np.quantile(ratios, .95):.3f}]")

    # per-line fits, top-4 d points, for each statistic
    lines = defaultdict(dict)
    for (d, a, L), c in cells.items():
        lines[(a, L)][d] = c
    alphas = sorted({a for a, _ in lines})
    for stat in ("last", "first", "mid"):
        print(f"\n== exponent fits on {stat} (top-4 d, "
              f"+- stderr; * = window ends below 32768) ==")
        print("alpha   conj " + " ".join(f"{'L=' + str(L):>12}"
              for L in (2, 3, 4, 5, 6, 8, 10, 12, 16, 24)))
        for a in alphas:
            row = [f"{a:5.2f}  {CONJ(a):.3f}"]
            meds = []
            for L in (2, 3, 4, 5, 6, 8, 10, 12, 16, 24):
                c = lines.get((a, L))
                if not c or len(c) < 3:
                    row.append(f"{'-':>12}")
                    continue
                ds = sorted(c)[-4:]
                sl, se = fit(ds, [c[d][stat] for d in ds])
                mark = "" if max(ds) == 32768 else "*"
                row.append(f"{sl:6.3f}+-{se:.2f}{mark}")
                if max(ds) == 32768 and L <= 6:
                    meds.append(sl)
            tail = (f"  | med(L2-6)={np.median(meds):.3f} "
                    f"spread={max(meds) - min(meds):.2f}"
                    if len(meds) >= 3 else "")
            print(" ".join(row) + tail)

    # the verdict block: alpha > 1, last statistic, full-window lines
    print("\n== alpha > 1 verdict (last, full top window 4096-32768) ==")
    for a in (1.0, 1.1, 1.25, 1.5):
        es = []
        for L in (2, 3, 4, 5, 6):
            c = lines.get((a, L))
            if c and len(c) >= 4 and max(c) == 32768:
                ds = sorted(c)[-4:]
                es.append(fit(ds, [c[d]["last"] for d in ds])[0])
        if es:
            print(f"  a={a}: measured {np.median(es):.3f} "
                  f"(L2-6 range {min(es):.3f}-{max(es):.3f})  "
                  f"conjectured {CONJ(a):.3f}")


if __name__ == "__main__":
    main()

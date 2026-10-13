"""Generate the 10x10x10 cube grid (emergence-time surface).

Axes (1000 cells):
  d      64 .. 32768, geometric factor 2 (10 values) -- power laws are
         straight lines on this grid.
  alpha  0.5 .. 1.5, near-uniform with extra density at 0.7--0.9 where
         the conjectured kink 1/2 + 1/alpha = 1 + kappa sits
         (alpha_c ~ 0.8 for kappa ~ 0.75); reuses the five already
         measured values so old and new cells line up.
  L      2 .. 24, geometric-ish.  L=1 is excluded: at d=4096 it never
         emerged within 2^21 tokens, so an L=1 row would pay full
         price for censored cells.

Per-cell token cap: 4x the emergence time predicted from the measured
law (anchored on results/takeover_all.json), rounded up to a power of
two, floored at 2^17 and capped at 2^25 (d<=8192) / 2^26 (d>=16384).
The 4x margin means a cell that violates the law by less than 4x in
time still shows its crossing; the few corner cells whose prediction
exceeds the ceiling run capped and report a lower bound instead.
Cells already measured (present in results/takeover_all.json) are
dropped; their results are merged back at aggregation time.

Outputs two task files, each sorted cheapest-first with ids 0..N-1:
  code/cluster/takeover_cube_mac.json     d <= 8192   (Mac)
  code/cluster/takeover_cube_scitas.json  d >= 16384  (SCITAS array)

    python code/cluster/make_cube_grid.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

D_GRID = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
A_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
L_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]

SEEDS = 16
MARGIN = 4.0            # cap = MARGIN * predicted n*
FLOOR = 2 ** 17
CEIL_SMALL = 2 ** 25    # d <= 8192
CEIL_LARGE = 2 ** 26    # d >= 16384
TOK_PER_SEC = 30_000    # measured Mac throughput (12 jobs), conservative
STOP_OVERHEAD = 1.7     # adaptive horizon processes ~1.7x n* per seed


def e_of_alpha(a: float) -> float:
    """Conjectured exponent, with 2.0 as the safe saturated value."""
    return min(0.5 + 1.0 / a, 2.0)


def load_anchors() -> tuple[dict, set]:
    rows = json.loads((REPO / "results" / "takeover_all.json").read_text())
    measured = {(r["d"], r["alpha"], r["L"]) for r in rows}
    # constant of the L=3 line per measured alpha, taken at the largest
    # measured d for that alpha (least finite-size drift)
    anchors = {}
    for r in rows:
        if r["L"] != 3:
            continue
        n = r["mean_lambda_crossing"] or r["n_star_median"]
        if not n:
            continue
        a = r["alpha"]
        if a not in anchors or r["d"] > anchors[a][0]:
            anchors[a] = (r["d"], n)
    const = {a: math.log(n) - e_of_alpha(a) * math.log(d)
             for a, (d, n) in anchors.items()}
    return const, measured


def ln_const(a: float, const: dict) -> float:
    """ln C_alpha at L=3: piecewise-linear in alpha between anchors."""
    ks = sorted(const)
    if a <= ks[0]:
        return const[ks[0]]
    if a >= ks[-1]:
        return const[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= a <= hi:
            w = (a - lo) / (hi - lo)
            return (1 - w) * const[lo] + w * const[hi]
    raise AssertionError


# measured n* ratios to L=3 at (d=4096, alpha=1); ln m, piecewise-linear
# in ln L, extrapolated with the last slope on each side
_LM_ANCHORS = [(2, math.log(3.298)), (3, 0.0),
               (6, math.log(0.1051)), (12, math.log(0.006448))]


def ln_mult(L: int) -> float:
    xs = [(math.log(l), v) for l, v in _LM_ANCHORS]
    x = math.log(L)
    if x <= xs[0][0]:
        (x0, y0), (x1, y1) = xs[0], xs[1]
    elif x >= xs[-1][0]:
        (x0, y0), (x1, y1) = xs[-2], xs[-1]
    else:
        for (x0, y0), (x1, y1) in zip(xs, xs[1:]):
            if x0 <= x <= x1:
                break
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def predict(d: int, a: float, L: int, const: dict) -> float:
    return math.exp(ln_const(a, const)
                    + e_of_alpha(a) * math.log(d) + ln_mult(L))


def main() -> None:
    const, measured = load_anchors()
    cells, dropped = [], []
    for d in D_GRID:
        for a in A_GRID:
            for L in L_GRID:
                if (d, a, L) in measured:
                    dropped.append((d, a, L))
                    continue
                pred = predict(d, a, L, const)
                ceil = CEIL_LARGE if d >= 16384 else CEIL_SMALL
                cap = min(max(MARGIN * pred, FLOOR), ceil)
                cap = 2 ** math.ceil(math.log2(cap))
                cap = min(cap, ceil)
                cells.append({"d": d, "alpha": a, "L": L,
                              "n_cap": int(cap),
                              "pred_n_star": int(round(pred)),
                              "censor_expected": pred * 1.2 > ceil})
    cells.sort(key=lambda c: c["pred_n_star"])

    for name in ("mac", "scitas"):
        tier = [c for c in cells if (c["d"] <= 8192) == (name == "mac")]
        tier = [{"task": i, **c} for i, c in enumerate(tier)]
        tokens = sum(min(c["pred_n_star"] * STOP_OVERHEAD, c["n_cap"])
                     * SEEDS for c in tier)
        hours = tokens / TOK_PER_SEC / 3600
        n_censor = sum(c["censor_expected"] for c in tier)
        out = REPO / "code" / "cluster" / f"takeover_cube_{name}.json"
        doc = {
            "comment": (f"Cube tier '{name}': {len(tier)} cells, "
                        f"cheapest first. Estimated "
                        f"{tokens/1e9:.2f}G tokens ~ {hours:.0f} h at "
                        f"{TOK_PER_SEC//1000}k tok/s (one 12-worker "
                        f"process); {n_censor} cells expected censored "
                        f"at their ceiling (recorded as lower bounds). "
                        f"{len(dropped)} already-measured cells dropped "
                        f"grid-wide; merged back at aggregation."),
            "defaults": {"seeds": "0-15", "jobs": 16},
            "tasks": tier,
        }
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{name}: {len(tier)} cells, est {tokens/1e9:.2f}G tokens"
              f" ~ {hours:.0f} h single-process, {n_censor} censored"
              f" -> {out.name}")
    print(f"dropped (already measured): {len(dropped)}")


if __name__ == "__main__":
    main()

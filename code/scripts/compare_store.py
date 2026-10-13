#!/usr/bin/env python3
"""Compare two anchor stores column by column.

    python3 code/scripts/compare_store.py STORE_A STORE_B

A store is a directory of ``level_LL.bin`` (float64 columns, one after the
other) and ``level_LL.index.json`` (``{r: [offset, length, crc]}``).  Both
stores are read directly; the LSA package is not needed.  Only levels present
in both stores are compared.

For every level the script reports the number of columns, how many of them are
byte-identical (equal crc), and the largest difference between the two stores
relative to the column's scale (``max |a - b| / max |a|`` over the column).
Two stores built from the same plan on different platforms typically agree to
about 1e-15 of the column scale without being byte-identical: the last bit of
``exp`` and ``log`` differs between platforms, and the builder writes columns
in the order the workers finish, so offsets differ as well.

Exit status 0 when every shared column has the same length and agrees to
within ``--tol`` (default 1e-12) of its scale, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_level(store: Path, L: int):
    index = json.loads((store / f"level_{L:02d}.index.json").read_text())
    values = np.fromfile(store / f"level_{L:02d}.bin", dtype=np.float64)
    return index, values


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("store_a", type=Path)
    p.add_argument("store_b", type=Path)
    p.add_argument("--tol", type=float, default=1e-12,
                   help="largest allowed |a-b| relative to the column scale")
    args = p.parse_args()

    levels = sorted(
        int(q.stem.split("_")[1].split(".")[0])
        for q in args.store_a.glob("level_*.index.json")
        if (args.store_b / q.name).exists())
    if not levels:
        print("no level shared by the two stores")
        return 1

    print(f"{'level':>5} {'columns':>8} {'identical':>10} {'max rel diff':>13}")
    ok = True
    for L in levels:
        ia, a = load_level(args.store_a, L)
        ib, b = load_level(args.store_b, L)
        shared = sorted(set(ia) & set(ib), key=int)
        missing = (set(ia) ^ set(ib))
        if missing:
            print(f"L={L}: {len(missing)} columns present in one store only")
            ok = False
        identical = 0
        worst = 0.0
        for r in shared:
            oa, na, ca = ia[r]
            ob, nb, cb = ib[r]
            if na != nb:
                print(f"L={L} r={r}: length {na} vs {nb}")
                ok = False
                continue
            if ca == cb:
                identical += 1
            x = a[oa:oa + na]
            y = b[ob:ob + nb]
            scale = np.abs(x).max()
            if scale > 0:
                worst = max(worst, float(np.abs(x - y).max() / scale))
            elif np.any(y != 0):
                worst = float("inf")
        if worst > args.tol:
            ok = False
        print(f"{L:>5} {len(shared):>8} {identical:>10} {worst:>13.2e}")
    print("agree to within tolerance" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

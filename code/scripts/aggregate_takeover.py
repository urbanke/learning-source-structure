"""Aggregate takeover-grid summaries into one table and the scaling plots.

Reads every ``summary_*.json`` below the given directories (the
takeover_experiment.py outputs), writes one compact JSON table, and
draws the discovery plots:

  panel 1: takeover vs d (log-log), alpha=1, L=3, with a d^{3/2} guide
  panel 2: takeover vs L (semi-log y), per d
  panel 3: takeover vs alpha (semi-log y), per d

The plotted point per configuration is the crossing of the
seed-averaged Lambda when it exists (filled marker); otherwise the
median per-seed takeover n*_last (open marker) — the mean-Lambda
crossing degenerates when Lambda starts positive (large alpha or L).
Censored configurations (no takeover within n_cap) are drawn as
upward-pointing triangles at n_cap.

    python code/scripts/aggregate_takeover.py \
        --in output/takeover --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load_rows(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for f in sorted(root.rglob("summary_*.json")):
            s = json.loads(f.read_text())
            ns = s.get("n_star_last") or []
            n_seeds = len(s.get("seeds", [])) or (len(ns) + s["n_censored"])
            rows.append({
                "d": s["d"], "alpha": s["alpha"], "L": s["L"],
                "n_cap": s["n_cap"],
                "mean_lambda_crossing": s["mean_lambda_crossing"],
                "n_star_median": median(ns) if ns else None,
                "n_star_min": min(ns) if ns else None,
                "n_star_max": max(ns) if ns else None,
                "n_seeds": n_seeds,
                "n_censored": s["n_censored"],
                "total_seconds": s["total_seconds"],
            })
    rows.sort(key=lambda r: (r["d"], r["L"], r["alpha"]))
    return rows


def point(row: dict) -> tuple[float | None, bool]:
    """(value to plot, is_crossing).  None if fully censored."""

    if row["n_censored"] >= row["n_seeds"]:
        return None, False
    if row["mean_lambda_crossing"]:
        return row["mean_lambda_crossing"], True
    return row["n_star_median"], False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="roots", nargs="+", type=Path,
                    required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.roots)
    if not rows:
        raise SystemExit("no summary_*.json found")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "takeover_all.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} configurations -> {args.out / 'takeover_all.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # panel 1: d-sweep at (alpha=1, L=3)
    ax = axes[0]
    sweep = [r for r in rows if r["alpha"] == 1.0 and r["L"] == 3]
    for r in sweep:
        v, is_cross = point(r)
        if v is None:
            ax.plot(r["d"], r["n_cap"], "^", color="#888")
            continue
        ax.plot(r["d"], v, "o" if is_cross else "o",
                mfc=("#1f77b4" if is_cross else "none"),
                mec="#1f77b4", ms=7)
        if r["n_star_min"]:
            ax.vlines(r["d"], r["n_star_min"], r["n_star_max"],
                      color="#1f77b4", alpha=0.35)
    ds = [r["d"] for r in sweep if point(r)[0]]
    if ds:
        import numpy as np

        dd = np.array(sorted(ds), dtype=float)
        anchor = next(r for r in reversed(sweep) if point(r)[0])
        c = point(anchor)[0] / anchor["d"] ** 1.5
        ax.plot(dd, c * dd ** 1.5, "--", color="#999",
                label=f"{c:.2f} d^1.5")
        ax.legend(frameon=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("d"); ax.set_ylabel("takeover time")
    ax.set_title("alpha=1, L=3")

    # panel 2: L-sweep per d
    ax = axes[1]
    for d0, color in [(4096, "#d62728"), (16384, "#2ca02c")]:
        pts = [r for r in rows if r["d"] == d0 and r["alpha"] == 1.0]
        xs, ys = [], []
        for r in sorted(pts, key=lambda r: r["L"]):
            v, _ = point(r)
            if v is None:
                ax.plot(r["L"], r["n_cap"], "^", color=color)
            else:
                xs.append(r["L"]); ys.append(v)
        ax.plot(xs, ys, "o-", color=color, label=f"d={d0}")
    ax.set_yscale("log")
    ax.set_xlabel("data depth L"); ax.set_title("alpha=1")
    ax.legend(frameon=False)

    # panel 3: alpha-sweep per d
    ax = axes[2]
    for d0, color in [(4096, "#d62728"), (16384, "#2ca02c")]:
        pts = [r for r in rows if r["d"] == d0 and r["L"] == 3]
        xs, ys, opens = [], [], []
        for r in sorted(pts, key=lambda r: r["alpha"]):
            v, is_cross = point(r)
            if v is not None:
                xs.append(r["alpha"]); ys.append(v); opens.append(not is_cross)
        ax.plot(xs, ys, "-", color=color, alpha=0.6)
        for x, y, o in zip(xs, ys, opens):
            ax.plot(x, y, "o", mfc=("none" if o else color), mec=color)
        ax.plot([], [], "o-", color=color, label=f"d={d0}")
    ax.set_yscale("log")
    ax.set_xlabel("alpha"); ax.set_title("L=3")
    ax.legend(frameon=False)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Takeover time of the copier (open markers: median n*; "
                 "filled: mean-Lambda crossing; triangles: censored at cap)",
                 fontsize=10)
    fig.tight_layout()
    out_png = args.out / "takeover_scaling.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

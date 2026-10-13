"""The two-faces figure: probe loss vs total loss through the takeover.

Reads the unchanged probe_experiment.py outputs in results/probe/
and draws two measured paths. The figure is sized for a 5.5-inch
text column; sparse labeled ticks keep the log axes readable.

    python code/scripts/two_faces_figure.py \
        --probe-dir results/probe --out figures/two_faces.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


def w_c(lam: float) -> float:
    if abs(lam) < 50:
        return 1.0 / (1.0 + 2.0 ** (-lam))
    return 1.0 if lam > 0 else 0.0


def panel(ax, rec, t_min, title, ticks):
    W = [w for w in rec["windows"] if w["t"] >= t_min]
    ts = [w["t"] for w in W]
    ax.plot(ts, [w["loss_total"] for w in W], "-", color="#333333",
            lw=1.5, label="all positions")
    ax.plot(ts, [w["loss_probe"] for w in W], "-", color="#b0413e",
            lw=1.5, label="probe positions")
    if any(w["loss_nonprobe"] is not None for w in W):
        ax.plot(ts, [w["loss_nonprobe"] for w in W], "-",
                color="#8aa2c8", lw=1.1, label="fresh contexts")
    ax2 = ax.twinx()
    ax2.plot(ts, [w_c(w["Lambda"]) for w in W], ":", color="#2ca02c",
             lw=1.3)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_yticks([0, 1])
    ax2.tick_params(labelsize=7, colors="#2ca02c")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("$n$ (tokens observed)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.15)
    return ax2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})

    early = json.loads(
        (args.probe_dir / "probe_d4096_a1.0_L12_seed0.json").read_text())
    late = json.loads(
        (args.probe_dir / "probe_d1024_a1.0_L3_seed0.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    panel(axes[0], early, 150,
          r"$d=4096$, $L_{\mathrm{s}}=12$", [200, 1000, 5000])
    axes[0].set_ylabel("bits per token")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    ax2 = panel(axes[1], late, 1000,
                r"$d=1024$, $L_{\mathrm{s}}=3$", [1000, 10000, 40000])
    ax2.set_ylabel(r"$w_C$ (dotted)", color="#2ca02c", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

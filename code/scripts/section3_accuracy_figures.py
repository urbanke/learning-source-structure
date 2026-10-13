"""Make the Section 3 figures from the already validated accuracy summary.

No model inference, new sequences, posterior computation, or data adjustment.
Run with --out DIR for an alternative output location, or --check-only to
verify an existing figure receipt and the hashes of its inputs and outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/transformer/context_accuracy_audit"
DEFAULT_OUT = ROOT / "results/transformer/section3_accuracy_curves"
CASES = {
    "coin_twopoint": (4096, 257),
    "coin_uniform": (4096, 257),
    "product_8x8": (128, 1025),
    "uniform_base_64": (1024, 1025),
}
FIGURES = {
    "coins": {
        "cases": ["coin_twopoint", "coin_uniform"],
        "titles": ["(a) Two-point coin prior", "(b) Uniform coin prior"],
        "y_limits": [0.0, 0.22],
        "y_ticks": [0.0, 0.05, 0.10, 0.15, 0.20],
    },
    "chains": {
        "cases": ["product_8x8", "uniform_base_64"],
        "titles": ["(a) Product Markov chains\n$8\\times8$", "(b) Uniform-base control\n$d=64$"],
        "y_limits": [0.0, 3.0],
        "y_ticks": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
}
STYLES = {
    "transformer": dict(color="#1764AB", linestyle="-", marker="D", markersize=3.1),
    "informed_bayes": dict(color="#D56A15", linestyle="--", marker="o", markersize=6.0),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_validated_panels():
    source_receipt = read_json(SOURCE / "receipt.json")
    require(source_receipt.get("passed") is True, "Source accuracy audit did not pass")
    require(
        source_receipt.get("outputs", {}).get("summary.json") == sha_file(SOURCE / "summary.json"),
        "Source summary does not match its validation receipt",
    )
    summary = read_json(SOURCE / "summary.json")
    require(summary.get("schema") == "context-accuracy-audit-v1", "Unexpected source schema")
    require(summary.get("units") == "bits per next-token prediction", "Unexpected source units")
    raw_panels = summary.get("panels", [])
    require(len(raw_panels) == len(CASES), "Unexpected number of source panels")
    panels = {panel["case"]: panel for panel in raw_panels}
    require(set(panels) == set(CASES), "Unexpected or duplicate source cases")
    for case, (n_sequences, n_tokens) in CASES.items():
        panel = panels[case]
        require(panel["n_sequences"] == n_sequences, f"{case}: unexpected sample count")
        require(panel["n_tokens"] == n_tokens, f"{case}: unexpected sequence length")
        expected_times = [2**k for k in range((n_tokens - 1).bit_length())]
        require(panel["times"] == expected_times, f"{case}: unexpected observation times")
        for predictor in ["transformer", "informed_bayes", "transformer_minus_bayes"]:
            for quantity in ["mean", "se"]:
                values = np.asarray(panel[predictor][quantity], dtype=float)
                require(values.shape == (len(expected_times),), f"{case}: incorrect array length")
                require(np.isfinite(values).all(), f"{case}: nonfinite {predictor} {quantity}")
                if quantity == "se" or predictor != "transformer_minus_bayes":
                    require(np.all(values >= 0), f"{case}: negative {predictor} {quantity}")
    for spec in FIGURES.values():
        low, high = spec["y_limits"]
        for case in spec["cases"]:
            for predictor in STYLES:
                mean = np.asarray(panels[case][predictor]["mean"])
                se = np.asarray(panels[case][predictor]["se"])
                require(np.all(mean - se >= low) and np.all(mean + se <= high),
                        f"{case}: y limits would hide data or error bars")
    return summary, panels


def receipt_base(summary, panels):
    return {
        "schema": "section3-accuracy-figures-v1",
        "passed": True,
        "script_sha256": sha_file(__file__),
        "inputs": {
            str(path.relative_to(ROOT)): sha_file(path)
            for path in [SOURCE / "summary.json", SOURCE / "receipt.json"]
        },
        "source_validation_scope": "Consume the four validated curve panels and inherit the existing numerical reference validation; no new convergence audit.",
        "quantity": summary["quantity"],
        "time": summary["time"],
        "uncertainty": summary["uncertainty"],
        "figures": FIGURES,
        "data": {case: {
            "n_sequences": panels[case]["n_sequences"],
            "n_tokens": panels[case]["n_tokens"],
            "times": panels[case]["times"],
            "transformer": panels[case]["transformer"],
            "informed_bayes": panels[case]["informed_bayes"],
        } for case in CASES},
        "display": {
            "size_inches": [6.5, 3.0],
            "x_scale": "log base 2; all saved times shown and labelled",
            "error_bars": "Mean plus/minus one sample standard error, without clipping",
            "connecting_lines": "Straight segments join the displayed means as a visual guide; no new data values are inferred between observation times.",
            "data_changes": "None: all saved mean and standard-error values are plotted unchanged.",
            "styles": STYLES,
        },
        "new_experiments": False,
        "checkpoint_inference": False,
    }


def make_figure(name, spec, panels, out):
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19,
                            top=0.75 if name == "chains" else 0.79, wspace=0.15)
        for ax, case, title in zip(axes, spec["cases"], spec["titles"]):
            panel = panels[case]
            times = np.asarray(panel["times"])
            ax.set_xscale("log", base=2)
            ax.set_xlim(times[0] / 1.2, times[-1] * 1.2)
            ax.set_xticks(times, [str(t) for t in times])
            ax.minorticks_off()
            ax.set_ylim(spec["y_limits"])
            ax.set_yticks(spec["y_ticks"])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#D8DEE5", linewidth=0.55)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#76808B")
                ax.spines[spine].set_linewidth(0.65)
            ax.tick_params(length=3, width=0.65, color="#76808B")
            for predictor, style in STYLES.items():
                ax.errorbar(times, panel[predictor]["mean"], yerr=panel[predictor]["se"],
                            fmt="none", ecolor=style["color"], elinewidth=0.85,
                            capsize=2, capthick=0.85, alpha=0.8, zorder=2)
            for predictor, style in STYLES.items():
                ax.plot(times, panel[predictor]["mean"], **style,
                        linewidth=1.45, markerfacecolor="none", markeredgewidth=0.95,
                        clip_on=False, zorder=3)
            ax.set_title(title, loc="left", pad=9)
            ax.set_xlabel("Observed tokens, $t$", labelpad=6)
        axes[0].set_ylabel("Extra loss (bits per prediction)", labelpad=7)
        handles = [Line2D([], [], **style, linewidth=1.45, markerfacecolor="none",
                          markeredgewidth=0.95)
                   for style in STYLES.values()]
        fig.legend(handles, ["Transformer", "Bayes with the source prior"],
                   loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=2,
                   frameon=False, handlelength=3.3, columnspacing=2.5)
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / f"{name}.png", dpi=240, metadata={"Software": "section3_accuracy_figures.py"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary, panels = read_validated_panels()
    expected = receipt_base(summary, panels)
    outputs = [f"{name}.{extension}" for name in FIGURES for extension in ["pdf", "png"]]
    if args.check_only:
        actual = read_json(args.out / "receipt.json")
        hashes = actual.pop("outputs", None)
        require(actual == expected, "Figure receipt differs from its validated inputs or script")
        require(hashes == {name: sha_file(args.out / name) for name in outputs},
                "Figure files differ from their recorded hashes")
        print("Section 3 figure inputs, unchanged numerical values, and output hashes verified.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for name, spec in FIGURES.items():
        make_figure(name, spec, panels, args.out)
    expected["outputs"] = {name: sha_file(args.out / name) for name in outputs}
    (args.out / "receipt.json").write_text(json.dumps(expected, indent=2, allow_nan=False) + "\n")
    print(f"Wrote two Section 3 figures and their provenance receipt to {args.out}")


if __name__ == "__main__":
    main()

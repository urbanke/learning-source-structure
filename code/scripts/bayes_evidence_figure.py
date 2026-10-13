"""Plot validated selected-prefix Bayesian/transformer divergences.

This is a figure-only consumer of shared_depth_curated_v1. It plots every
selected prefix, without smoothing, interpolation, or cumulative inference.

    MPLCONFIGDIR=/private/tmp/layered-mpl-cache python code/scripts/bayes_evidence_figure.py

For a smoke render, direct --pdf, --png and --provenance to an ignored output
directory and use --dpi 80. The smoke render uses the same complete data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_SCHEMA = "shared_depth_curated_v1"
METRIC = "kl_bayes_to_transformer"
PANELS = (
    ("product_8x8", r"(a) Product, $8\times8$"),
    ("uniform_d64", r"(b) Uniform-base LSA, $d=64$"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_panels(path: Path) -> list[dict]:
    report = json.loads(path.read_text())
    if report.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"Expected validated input schema {INPUT_SCHEMA}")
    cases = report["cases"]
    by_name = {case["case"]: case for case in cases}
    if len(by_name) != len(cases) or set(by_name) != {name for name, _ in PANELS}:
        raise ValueError("Expected exactly the two distinct validated source cases")
    panels = []
    for name, title in PANELS:
        case = by_name[name]
        audit = case["independent_validation"]
        if audit.get("passed") is not True:
            raise ValueError(f"Independent validation has not passed for {name}")
        if case["units"] != "bits" or case["first_token_scored"] is not False:
            raise ValueError(f"Unexpected scoring convention for {name}")
        if (case["n_sequences"] != 128 or audit["sequences_checked"] != 128
                or case["source"]["n_ctx"] != 1024
                or audit["scored_positions_per_sequence"] != 1024):
            raise ValueError(f"Unexpected validated sample count or horizon for {name}")
        prefixes = case["prefixes"]
        measured = case["by_prefix"][METRIC]
        means, errors = measured["mean"], measured["se"]
        if prefixes != [2**i for i in range(11)]:
            raise ValueError(f"Expected all measured prefixes 1 through 1024 for {name}")
        if len(means) != len(prefixes) or len(errors) != len(prefixes):
            raise ValueError(f"Metric/prefix lengths disagree for {name}")
        values = np.asarray([means, errors], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid KL mean or standard error for {name}")
        # Preserve the JSON values exactly; arrays are not rounded or resampled.
        panels.append(dict(case=name, title=title, n_sequences=case["n_sequences"],
                           observed_prefix=prefixes, mean_bits=means,
                           standard_error_bits=errors,
                           checkpoint_sha256=case["model"]["checkpoint_sha256"],
                           training_seed=case["model"]["config"]["seed"]))
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/transformer/shared_depth/summary.json")
    parser.add_argument("--pdf", type=Path, default=ROOT / "figures/bayes_evidence.pdf")
    parser.add_argument("--png", type=Path, default=ROOT / "output/bayes_evidence_figure/bayes_evidence.png")
    parser.add_argument("--provenance", type=Path,
                        default=ROOT / "results/transformer/bayes_evidence_figure.json")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi < 40:
        parser.error("--dpi must be at least 40")
    destinations = [args.pdf.resolve(), args.png.resolve(), args.provenance.resolve()]
    if len(set(destinations)) != 3 or args.input.resolve() in destinations:
        parser.error("Input and output paths must be distinct")
    panels = load_panels(args.input)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.65), sharey=True)
    maximum = max(max(np.asarray(p["mean_bits"]) + np.asarray(p["standard_error_bits"]))
                  for p in panels)
    upper = np.ceil(1.06 * maximum / 0.01) * 0.01
    for ax, panel in zip(axes, panels):
        ax.errorbar(panel["observed_prefix"], panel["mean_bits"],
                    yerr=panel["standard_error_bits"], fmt="o-", color="#256B8F",
                    linewidth=1.1, markersize=3.4, markerfacecolor="white",
                    markeredgewidth=0.85, capsize=2.2, elinewidth=0.75,
                    capthick=0.75, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 1280)
        ax.set_xticks([1, 4, 16, 64, 256, 1024])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper + 0.005, 0.02))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel(r"Observed prefix length $t$")
        ax.set_title(panel["title"], loc="left", pad=8)
        ax.grid(axis="y", color="0.9", linewidth=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.65)
    axes[0].set_ylabel(r"Mean $\mathrm{KL}(B\Vert\mathrm{TF})$ [bits]")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.23, top=0.85, wspace=0.16)
    for path in (args.pdf, args.png, args.provenance):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, metadata={"CreationDate": None, "ModDate": None,
                                   "Title": "Selected-prefix Bayesian predictive divergence"})
    fig.savefig(args.png, dpi=args.dpi)
    plt.close(fig)
    provenance = dict(
        schema="bayes_evidence_figure_v1", input_schema=INPUT_SCHEMA,
        input=display_path(args.input), input_sha256=sha256(args.input),
        script=display_path(Path(__file__)), script_sha256=sha256(Path(__file__)),
        metric=METRIC, units="bits", panels=panels,
        uncertainty="One standard error across 128 independent sequence/source draws per family; training-seed uncertainty is not measured.",
        scoring="Prefix t predicts X(t+1); X1 is supplied free. Each source draw contains 1025 tokens.",
        plot=dict(width_inches=6.4, height_inches=2.65, x_scale="log2",
                  y_scale="linear", shared_y_limits=[0.0, float(upper)],
                  error_bar_standard_errors=1, all_selected_prefixes=True,
                  connecting_lines="guide to the eye; no smoothing or cumulative KL inference"),
        output_pdf=display_path(args.pdf), output_pdf_sha256=sha256(args.pdf),
        runtime=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
    )
    args.provenance.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Plotted {sum(len(p['observed_prefix']) for p in panels)} validated points: {args.pdf}")


if __name__ == "__main__":
    main()

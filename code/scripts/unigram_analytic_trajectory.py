"""Compare the sampling-free unigram formula with saved simulations.

The figure is a fixed 3 by 3 grid for d=256. Its columns use
alpha in {0.5, 1.0, 1.5}; its rows use source depth L_s in {2, 3, 6}.
Every cell averages the same 16 saved, independent source-table/path runs.
It plots expected cumulative codelength divided by sequence length.  This
average number of bits per token exposes learning that is hidden by an
almost linear cumulative-codelength curve.  No new random source table or
path is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
)
import numpy as np

from tilted_row_reference import DEFAULT_STORE, configure_store
from unigram_analytic_approximation import (
    EULER,
    beta_binomial_mean_profile,
    evaluate,
    finite_occupation_moments,
    fractional_profile_cost,
    source_row_moments,
    trim_profile,
    zipf_law,
)


ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (0.5, 1.0, 1.5)
SOURCE_DEPTHS = (2, 3, 6)
ALPHABET_SIZE = 256
REPETITIONS = 16

# These 13 points are a roughly logarithmically spaced subset of the 50
# checkpoints common to every saved run between k=8 and k=2048.
SELECTED_K = (8, 13, 20, 31, 49, 77, 121, 214, 337, 530, 834, 1312, 2048)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def alpha_directory(alpha: float) -> str:
    return f"{alpha:.1f}"


def seed_number(path: Path) -> int:
    match = re.search(r"_seed(\d+)\.json$", path.name)
    if match is None:
        raise ValueError(f"Cannot read the seed number from {path}")
    return int(match.group(1))


def load_saved_cell(alpha: float, source_depth: int) -> dict:
    relative_directory = Path(
        f"output/takeover/d{ALPHABET_SIZE}_a{alpha_directory(alpha)}_L{source_depth}")
    directory = ROOT / relative_directory
    paths = sorted(directory.glob("*_seed*.json"), key=seed_number)
    if len(paths) != REPETITIONS:
        raise ValueError(
            f"Expected {REPETITIONS} saved runs in {directory}, found {len(paths)}")

    expected_seeds = list(range(REPETITIONS))
    actual_seeds = [seed_number(path) for path in paths]
    if actual_seeds != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, found {actual_seeds}")

    codelengths = []
    input_records = []
    common_checkpoints: set[int] | None = None
    estimators = set()
    for path, seed in zip(paths, actual_seeds):
        saved = json.loads(path.read_text())
        if (saved["d"], float(saved["alpha"]), saved["L"], saved["seed"]) != (
                ALPHABET_SIZE, alpha, source_depth, seed):
            raise ValueError(f"Metadata does not match the directory name in {path}")
        checkpoints = [int(value) for value in saved["checkpoints"]]
        if len(checkpoints) != len(saved["l_F"]):
            raise ValueError(f"Checkpoint and codelength arrays differ in {path}")
        checkpoint_set = set(checkpoints)
        common_checkpoints = (
            checkpoint_set if common_checkpoints is None
            else common_checkpoints.intersection(checkpoint_set))
        values = dict(zip(checkpoints, map(float, saved["l_F"])))
        missing = set(SELECTED_K).difference(values)
        if missing:
            raise ValueError(f"Missing checkpoints {sorted(missing)} in {path}")
        codelengths.append([values[k] for k in SELECTED_K])
        estimators.add(saved["sequence_estimator"])
        relative_path = str(path.relative_to(ROOT))
        input_records.append({
            "seed": seed,
            "path": relative_path,
            "sha256": sha256(path),
        })

    if common_checkpoints is None:
        raise AssertionError("The saved-run list was unexpectedly empty")
    common_in_range = sorted(k for k in common_checkpoints if 8 <= k <= 2048)
    if common_in_range[0] != 8 or common_in_range[-1] != 2048:
        raise ValueError("Saved runs do not cover the requested common range")

    values = np.asarray(codelengths, dtype=float)
    mean = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / math.sqrt(REPETITIONS)
    return {
        "d": ALPHABET_SIZE,
        "alpha": alpha,
        "source_depth": source_depth,
        "saved_repetitions": REPETITIONS,
        "tokens": list(SELECTED_K),
        "measured_mean_bits": mean.tolist(),
        "measured_standard_error_bits": standard_error.tolist(),
        "common_saved_checkpoints_8_through_2048": common_in_range,
        "simulation_inputs": input_records,
        "simulation_input_set_sha256": combined_hash(input_records),
        "sequence_estimators": sorted(estimators),
    }


def evaluate_cell(alpha: float, source_depth: int, tables, *, source_step: float,
                  cost_step: float) -> dict:
    cell = load_saved_cell(alpha, source_depth)
    coding_cap = round(2 * math.log(ALPHABET_SIZE) / (1 - EULER))
    q = zipf_law(ALPHABET_SIZE, alpha)
    row = source_row_moments(q, source_depth, tables, step=source_step)
    predictions = []
    point_diagnostics = []
    tail_sensitivity = None
    maximum_nested_error = 0.0
    for k in SELECTED_K:
        result, _ = evaluate(
            ALPHABET_SIZE, alpha, source_depth, k, coding_cap, tables,
            source_step=source_step, cost_step=cost_step, row=row)
        predictions.append(result["predicted_codelength_bits"])
        integral_diagnostics = result["lsa_integral_diagnostics"]
        occupation_diagnostics = result["finite_occupation_diagnostics"]
        tail_diagnostics = result["profile_tail_diagnostics"]
        maximum_nested_error = max(
            maximum_nested_error,
            integral_diagnostics["maximum_nested_grid_difference_bits"],
        )
        point_diagnostics.append({
            "k": k,
            "profile_tail_diagnostics": tail_diagnostics,
            "moment_clipping_diagnostics": {
                "negative_variance_clipping_detected":
                    occupation_diagnostics["minimum_variance"] == 0.0,
                "upper_probability_bound_clipping_detected":
                    occupation_diagnostics["maximum_variance_fraction"]
                    >= 1.0 - 2e-12,
                "minimum_retained_variance":
                    occupation_diagnostics["minimum_variance"],
                "maximum_retained_variance_fraction":
                    occupation_diagnostics["maximum_variance_fraction"],
            },
        })
        if k == SELECTED_K[-1]:
            occupation = finite_occupation_moments(q, k, row)
            profile, _ = beta_binomial_mean_profile(
                occupation["mean"], occupation["variance"], k)
            tighter_profile, tighter_tail = trim_profile(
                profile, weighted_tail_tolerance=1e-11)
            tighter_cost, _ = fractional_profile_cost(
                tighter_profile, k, coding_cap, tables, step=cost_step)
            tail_sensitivity = {
                "k": k,
                "baseline_weighted_tail_tolerance":
                    tail_diagnostics["weighted_tail_tolerance"],
                "tighter_weighted_tail_tolerance":
                    tighter_tail["weighted_tail_tolerance"],
                "baseline_codelength_bits":
                    result["predicted_codelength_bits"],
                "tighter_codelength_bits": tighter_cost,
                "absolute_codelength_change_bits": abs(
                    tighter_cost - result["predicted_codelength_bits"]),
                "baseline_profile_tail_diagnostics": tail_diagnostics,
                "tighter_profile_tail_diagnostics": tighter_tail,
            }
    predictions_array = np.asarray(predictions)
    measured = np.asarray(cell["measured_mean_bits"])
    difference = predictions_array - measured
    absolute_relative = np.abs(difference) / measured
    cell.update({
        "coding_depth_cap": coding_cap,
        "predicted_codelength_bits": predictions_array.tolist(),
        "prediction_minus_measurement_bits": difference.tolist(),
        "absolute_relative_difference": absolute_relative.tolist(),
        "maximum_absolute_difference_bits": float(np.max(np.abs(difference))),
        "maximum_absolute_relative_difference": float(np.max(absolute_relative)),
        "maximum_nested_grid_difference_bits": maximum_nested_error,
        "source_row_integral_diagnostics": row["diagnostics"],
        "displayed_point_numerical_diagnostics": point_diagnostics,
        "profile_tail_tolerance_sensitivity": tail_sensitivity,
    })
    return cell


def plot(cells: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.size": 6.6,
        "axes.titlesize": 7.4,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(
        len(SOURCE_DEPTHS), len(ALPHAS), figsize=(5.5, 4.65),
        sharex=True, sharey=True)
    analytic_color = "#2166ac"
    measured_color = "#b2182b"
    lookup = {(cell["source_depth"], cell["alpha"]): cell for cell in cells}
    largest_difference = maximum_relative_difference_point(cells)

    for row_index, source_depth in enumerate(SOURCE_DEPTHS):
        for column_index, alpha in enumerate(ALPHAS):
            axis = axes[row_index, column_index]
            cell = lookup[(source_depth, alpha)]
            k = np.asarray(cell["tokens"], dtype=float)
            predicted = np.asarray(cell["predicted_codelength_bits"]) / k
            measured = np.asarray(cell["measured_mean_bits"]) / k
            two_standard_errors = (
                2 * np.asarray(cell["measured_standard_error_bits"]) / k)

            axis.plot(k, predicted, color=analytic_color, zorder=3)
            axis.errorbar(
                k, measured, yerr=two_standard_errors, fmt="o", ms=2.6,
                markerfacecolor="white", markeredgewidth=0.75,
                color=measured_color, ecolor=measured_color,
                elinewidth=0.65, capsize=1.25, zorder=2)
            axis.set_xscale("log")
            axis.set_xlim(7.3, 2250)
            axis.set_ylim(4.0, 8.65)
            axis.grid(which="major", color="#d9d9d9", linewidth=0.5)
            axis.grid(which="minor", color="#eeeeee", linewidth=0.35)
            axis.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{int(value):d}"))
            axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.yaxis.set_major_locator(MultipleLocator(1.0))
            axis.yaxis.set_minor_locator(MultipleLocator(0.5))
            axis.tick_params(which="major", length=2.4, width=0.6)
            axis.tick_params(which="minor", length=1.4, width=0.45)
            if (source_depth == largest_difference["source_depth"]
                    and alpha == largest_difference["alpha"]):
                largest_index = cell["tokens"].index(largest_difference["k"])
                axis.annotate(
                    "largest difference:\n"
                    f"{100 * largest_difference['absolute_relative_difference']:.1f}% "
                    rf"at $k={largest_difference['k']}$",
                    xy=(k[largest_index],
                        0.5 * (predicted[largest_index]
                               + measured[largest_index])),
                    xytext=(14, 8.36), textcoords="data",
                    ha="left", va="center", fontsize=5.9,
                    arrowprops={"arrowstyle": "-", "color": "#555555",
                                "linewidth": 0.6},
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.88, "pad": 0.7})
            if row_index == 0:
                axis.set_title(rf"$\alpha={alpha:g}$", pad=4)
            if column_index == 0:
                axis.set_ylabel(
                    rf"$L_{{\rm s}}={source_depth}$", rotation=0, labelpad=22,
                    va="center")

    legend = [
        Line2D([0], [0], color=analytic_color, label="scalable approximation"),
        Line2D([0], [0], marker="o", linestyle="none", color=measured_color,
               markerfacecolor="white", markeredgewidth=0.75,
               label=r"simulation mean $\pm\,2$ standard errors"),
    ]
    figure.legend(
        handles=legend, loc="upper center", ncol=2,
        bbox_to_anchor=(0.57, 0.995), frameon=False, handlelength=2.2,
        columnspacing=1.6)
    figure.text(0.155, 0.968, r"$d=256$", ha="left", va="center", fontsize=6.8)
    figure.supylabel(
        "expected codelength per token (bits)", fontsize=6.8, x=0.012)
    figure.supxlabel("sequence length $k$", fontsize=6.8, y=0.012)
    figure.subplots_adjust(
        left=0.155, right=0.987, bottom=0.085, top=0.91,
        wspace=0.16, hspace=0.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={
        "Title": "Scalable unigram approximation versus simulations"})
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def maxima_over_displayed_points(cells: list[dict]) -> dict:
    tails = [
        diagnostic["profile_tail_diagnostics"]
        for cell in cells
        for diagnostic in cell["displayed_point_numerical_diagnostics"]
    ]
    clipping = [
        diagnostic["moment_clipping_diagnostics"]
        for cell in cells
        for diagnostic in cell["displayed_point_numerical_diagnostics"]
    ]
    return {
        "profile_tail_weighted_tolerance": tails[0]["weighted_tail_tolerance"],
        "maximum_omitted_symbol_mass": max(
            tail["omitted_symbol_mass"] for tail in tails),
        "maximum_omitted_count_mass": max(
            tail["omitted_count_mass"] for tail in tails),
        "maximum_omitted_factorial_weight": max(
            tail["omitted_factorial_weight"] for tail in tails),
        "negative_variance_clipping_detected": any(
            item["negative_variance_clipping_detected"] for item in clipping),
        "upper_probability_bound_clipping_detected": any(
            item["upper_probability_bound_clipping_detected"] for item in clipping),
        "maximum_codelength_change_from_tightening_profile_tail_tolerance_bits":
            max(
                cell["profile_tail_tolerance_sensitivity"]
                    ["absolute_codelength_change_bits"]
                for cell in cells),
        "profile_tail_tolerance_sensitivity_k": SELECTED_K[-1],
        "tighter_profile_tail_weighted_tolerance": 1e-11,
    }


def maximum_relative_difference_point(cells: list[dict]) -> dict:
    """Return the displayed point with the largest relative difference."""

    maximum = None
    for cell in cells:
        for index, relative_difference in enumerate(
                cell["absolute_relative_difference"]):
            if maximum is not None and relative_difference <= maximum[0]:
                continue
            measured = cell["measured_mean_bits"][index]
            predicted = cell["predicted_codelength_bits"][index]
            standard_error = cell["measured_standard_error_bits"][index]
            maximum = (relative_difference, {
                "d": cell["d"],
                "alpha": cell["alpha"],
                "source_depth": cell["source_depth"],
                "k": cell["tokens"][index],
                "simulation_mean_cumulative_bits": measured,
                "formula_cumulative_bits": predicted,
                "formula_minus_simulation_cumulative_bits": predicted - measured,
                "simulation_standard_error_cumulative_bits": standard_error,
                "absolute_relative_difference": relative_difference,
                "absolute_difference_in_simulation_standard_errors": (
                    abs(predicted - measured) / standard_error),
                "simulation_mean_bits_per_token": (
                    measured / cell["tokens"][index]),
                "formula_bits_per_token": (
                    predicted / cell["tokens"][index]),
            })
    if maximum is None:
        raise AssertionError("No displayed comparison points")
    return maximum[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--source-step", type=float, default=1 / 32)
    parser.add_argument("--cost-step", type=float, default=0.04)
    parser.add_argument(
        "--out-data", type=Path,
        default=ROOT / "results/mechanism/unigram_analytic_trajectory.json")
    parser.add_argument(
        "--out-figure", type=Path,
        default=ROOT / "figures/unigram_analytic_trajectory.pdf")
    args = parser.parse_args()

    started = time.monotonic()
    configure_store(args.store)
    from lsa.universal_tables import UniversalTables

    tables = UniversalTables(args.store, read_only=True)
    try:
        cells = []
        for source_depth in SOURCE_DEPTHS:
            for alpha in ALPHAS:
                cell = evaluate_cell(
                    alpha, source_depth, tables, source_step=args.source_step,
                    cost_step=args.cost_step)
                cells.append(cell)
                print(json.dumps({
                    "d": cell["d"],
                    "alpha": cell["alpha"],
                    "source_depth": cell["source_depth"],
                    "saved_repetitions": cell["saved_repetitions"],
                    "points": len(cell["tokens"]),
                    "maximum_absolute_relative_difference":
                        cell["maximum_absolute_relative_difference"],
                }), flush=True)
    finally:
        tables.close()

    plot(cells, args.out_figure)
    absolute_errors = np.concatenate([
        np.abs(np.asarray(cell["prediction_minus_measurement_bits"]))
        for cell in cells])
    relative_errors = np.concatenate([
        np.asarray(cell["absolute_relative_difference"]) for cell in cells])
    elapsed = time.monotonic() - started
    approximation_script = ROOT / "code/scripts/unigram_analytic_approximation.py"
    store_manifest = args.store / "anchors.json"
    result = {
        "schema_version": 3,
        "complete": True,
        "target": (
            "Expected cumulative codelength of the LSA unigram predictor as "
            "a function of sequence length k."),
        "figure_quantity": (
            "Expected cumulative codelength divided by k, in bits per token. "
            "The division changes neither the pointwise relative differences "
            "nor their ordering."),
        "grid": {
            "alphabet_size": ALPHABET_SIZE,
            "alpha_columns": list(ALPHAS),
            "source_depth_rows": list(SOURCE_DEPTHS),
            "sequence_length_range": [8, 2048],
        },
        "selected_checkpoints": {
            "values": list(SELECTED_K),
            "rule": (
                "A fixed, roughly logarithmically spaced subset of the 50 saved "
                "checkpoints common to all 144 simulation files on 8 <= k <= 2048."),
        },
        "method": (
            "The sampling-free scalable formula is compared at 13 common saved "
            "checkpoints with the mean of 16 saved source-table/path runs in each "
            "of nine parameter cells."),
        "sampling": (
            "The formula draws no samples. The comparison reuses 144 saved runs "
            "and generates no new source table or path."),
        "source_step": args.source_step,
        "cost_step": args.cost_step,
        "cells": cells,
        "summary": {
            "settings": len(cells),
            "saved_simulation_files": sum(
                len(cell["simulation_inputs"]) for cell in cells),
            "saved_repetitions_per_setting": REPETITIONS,
            "points_per_setting": len(SELECTED_K),
            "comparison_points": int(len(absolute_errors)),
            "median_absolute_difference_bits": float(np.median(absolute_errors)),
            "maximum_absolute_difference_bits": float(np.max(absolute_errors)),
            "median_absolute_relative_difference": float(
                np.median(relative_errors)),
            "maximum_absolute_relative_difference": float(np.max(relative_errors)),
            "maximum_absolute_relative_difference_point":
                maximum_relative_difference_point(cells),
            "displayed_point_numerical_maxima": maxima_over_displayed_points(cells),
        },
        "elapsed_seconds": elapsed,
        "provenance": {
            "trajectory_script": str(Path(__file__).relative_to(ROOT)),
            "trajectory_script_sha256": sha256(Path(__file__)),
            "approximation_script": str(approximation_script.relative_to(ROOT)),
            "approximation_script_sha256": sha256(approximation_script),
            "sealed_table_manifest": str(store_manifest),
            "sealed_table_manifest_sha256": sha256(store_manifest),
            "figure_pdf": str(args.out_figure.relative_to(ROOT)),
            "figure_pdf_sha256": sha256(args.out_figure),
            "figure_png": str(args.out_figure.with_suffix(".png").relative_to(ROOT)),
            "figure_png_sha256": sha256(args.out_figure.with_suffix(".png")),
        },
    }
    write_json(args.out_data, result)
    print(json.dumps({
        **result["summary"],
        "elapsed_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()

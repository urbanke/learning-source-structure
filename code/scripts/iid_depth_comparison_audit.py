"""Audit the saved iid depth comparison without running experiments.

python code/scripts/iid_depth_comparison_audit.py --out results/mechanism/iid_depth_comparison_audit.json
python code/scripts/iid_depth_comparison_audit.py --check
python code/scripts/iid_depth_comparison_audit.py --self-test

The full audit requires the existing ignored raw arrays. It independently
recomputes means and sample standard errors with accurately summed scalar
arithmetic; it never calls the experiment, prediction, or collector routines.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/mechanism/iid_depth_comparison_audit.json"
DEPTHS, DS, ALPHAS, KS = (1, 3, 6), (256, 4096), (.5, 1., 1.5, 2.), (256, 1024, 4096, 16384)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    values = [float(v) for v in values]
    n = len(values)
    assert n >= 2 and all(math.isfinite(v) for v in values)
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def close(actual, expected, tolerance=1e-9):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def self_test():
    values = [1., 2., 4., 8., 16.]
    mean, se = moments(values)
    close(mean, statistics.mean(values), 1e-14)
    close(se, statistics.stdev(values) / math.sqrt(len(values)), 1e-14)
    assert abs(se - statistics.pstdev(values) / math.sqrt(len(values))) > .1
    assert moments([7., 7., 7.]) == (7., 0.)


def audit():
    self_test()
    comparison_path = ROOT / "results/mechanism/iid_depth_comparison.json"
    prediction_path = ROOT / "results/mechanism/iid_depth_prediction.json"
    table_path = ROOT / "results/mechanism/iid_depth_comparison_table.tex"
    comparison, prediction = read(comparison_path), read(prediction_path)
    rows = comparison["rows"]
    expected = set(itertools.product(DEPTHS, DS, ALPHAS, KS))
    key = lambda r: (r["L"], r["d"], r["alpha"], r["k"])
    observed = {key(r): r for r in rows}
    assert len(rows) == len(observed) == 96 and set(observed) == expected
    assert comparison["sample_lengths"] == list(KS)
    prediction_rows = {key(r): r for r in prediction["rows"]}
    assert len(prediction_rows) == len(prediction["rows"])
    assert expected <= set(prediction_rows) and prediction["draws"] == 8192
    assert not prediction["smoke"]

    source_hashes = []
    for source in comparison["sources"]:
        path = ROOT / source["path"]
        assert digest(path) == source["sha256"], source["path"]
        source_hashes.append(source)

    # Independent arithmetic for prediction differences and uncertainty.
    for row in rows:
        pred = prediction_rows[key(row)]
        mean, se = row["measured_mean_bits"], row["measured_SE_bits"]
        pm, ps = pred["finite_random_scale"]["mean_bits"], pred["finite_random_scale"]["SE_bits"]
        close(row["prediction_mean_bits"], pm)
        close(row["prediction_SE_bits"], ps)
        close(row["original_sum_bits"], pred["original_sum_bits"])
        combined = math.sqrt(se * se + ps * ps)
        close(row["difference_bits"], mean - pm)
        close(row["difference_SE_bits"], combined)
        close(row["relative_difference"], mean / pm - 1., 1e-13)
        close(row["difference_in_combined_SE"], (mean - pm) / combined, 1e-12)
        assert row["prediction_draws"] == pred["draws"] == 8192

    cell_checks, raw_hashes = [], []
    for depth, alpha in itertools.product((1, 6), ALPHAS):
        path = ROOT / f"output/iid_depth_coverage_long/L{depth}_a{alpha:g}.json"
        source = read(path)
        assert source["complete"] and source["se_ddof"] == 1
        assert source["seed"] == 202609069 and source["repetition_start"] == 0
        npz_path = path.with_suffix(".npz")
        with np.load(npz_path) as arrays:
            assert arrays["ds"].tolist() == list(DS) and arrays["ks"].tolist() == list(KS)
            n = 512 if depth == 1 else 256
            assert arrays["redundancy"].shape == (n, 2, 4)
            residual = arrays["bayes_cost"] - arrays["known_source_cost"] - arrays["redundancy"]
            assert float(np.max(np.abs(residual))) < 1e-8
            for index, d in enumerate(DS):
                cell = next(c for c in source["cells"] if c["d"] == d)
                assert (cell["L"], cell["alpha"], cell["reps"], cell["k"]) == (depth, alpha, n, list(KS))
                max_mean_error, max_se_error = 0., 0.
                for j, k in enumerate(KS):
                    mean, se = moments(arrays["redundancy"][:, index, j])
                    row = observed[depth, d, alpha, k]
                    close(mean, cell["redundancy_mean"][j])
                    close(se, cell["redundancy_se"][j])
                    close(mean, row["measured_mean_bits"])
                    close(se, row["measured_SE_bits"])
                    assert row["repetitions"] == n and row["source"] == str(path.relative_to(ROOT))
                    max_mean_error = max(max_mean_error, abs(mean - row["measured_mean_bits"]))
                    max_se_error = max(max_se_error, abs(se - row["measured_SE_bits"]))
                    for array_name, field in (("known_source_cost", "known_source_cost_mean"), ("bayes_cost", "bayes_cost_mean"), ("distinct", "distinct_symbols_mean")):
                        close(moments(arrays[array_name][:, index, j])[0], cell[field][j])
                normalizer_mean, normalizer_se = moments(arrays["normalizers"][:, index])
                close(normalizer_mean, cell["normalizer_mean"])
                close(n * normalizer_se ** 2, cell["normalizer_variance"])
                refinement = float(np.max(arrays["numerical_error"][:, index]))
                close(refinement, cell["maximum_outer_refinement_difference_bits"], 1e-15)
                cell_checks.append(dict(L=depth, d=d, alpha=alpha, repetitions=n, checked_means=4,
                    maximum_mean_recomputation_difference_bits=max_mean_error,
                    maximum_SE_recomputation_difference_bits=max_se_error,
                    maximum_outer_refinement_difference_bits=refinement))
        raw_hashes.append(dict(path=str(npz_path.relative_to(ROOT)), sha256=digest(npz_path)))

    # Recheck the archived depth-three subset from its saved per-draw arrays.
    archive = read(ROOT / "results/mechanism/iid_layered_tilt_check.json")
    for d, alpha in itertools.product(DS, ALPHAS):
        cell = next(c for c in archive["cells"] if (c["L"], c["d"], c["alpha"]) == (3, d, alpha))
        assert cell["complete"] and cell["se_ddof"] == 1
        argv = cell["command_argv"]
        source_path = ROOT / argv[argv.index("--out") + 1]
        assert read(source_path) == cell
        npz_path = source_path.with_suffix(".npz")
        with np.load(npz_path) as arrays:
            n = 512 if d == 256 else 256
            assert arrays["redundancy"].shape == (n, len(cell["k"]))
            max_mean_error, max_se_error = 0., 0.
            for k in KS:
                j = cell["k"].index(k)
                mean, se = moments(arrays["redundancy"][:, j])
                row = observed[3, d, alpha, k]
                close(mean, row["measured_mean_bits"])
                close(se, row["measured_SE_bits"])
                assert row["repetitions"] == n
                max_mean_error = max(max_mean_error, abs(mean - row["measured_mean_bits"]))
                max_se_error = max(max_se_error, abs(se - row["measured_SE_bits"]))
            cell_checks.append(dict(L=3, d=d, alpha=alpha, repetitions=n, checked_means=4,
                maximum_mean_recomputation_difference_bits=max_mean_error,
                maximum_SE_recomputation_difference_bits=max_se_error,
                maximum_outer_refinement_difference_bits=cell["max_coarse_fine_difference_bits"]))
        raw_hashes.append(dict(path=str(npz_path.relative_to(ROOT)), sha256=digest(npz_path)))

    # Verify the percentage entries without importing the table generator.
    table_data_lines = [line for line in table_path.read_text().splitlines()
                        if line and line[0].isdigit()]
    assert len(table_data_lines) == 4
    for alpha, line in zip(ALPHAS, table_data_lines):
        entries = line.split(" & ")
        assert float(entries[0]) == alpha
        values = []
        for depth in DEPTHS:
            row = observed[depth, 4096, alpha, 4096]
            values += [100 * (row["measured_mean_bits"] / row["original_sum_bits"] - 1),
                       100 * row["relative_difference"]]
        for entry, value in zip(entries[1:], values):
            shown = float(entry.split("\\%")[0].strip("$"))
            assert abs(shown - value) <= .050000000001

    def summary(selected):
        worst = max(selected, key=lambda r: abs(r["relative_difference"]))
        return dict(points=len(selected),
            maximum_absolute_relative_difference=max(abs(r["relative_difference"]) for r in selected),
            maximum_absolute_difference_in_combined_SE=max(abs(r["difference_in_combined_SE"]) for r in selected),
            worst_relative_difference_setting={k: worst[k] for k in ("L", "d", "alpha", "k")})

    summaries = {"all": summary(rows),
        "k_4096_or_16384": summary([r for r in rows if r["k"] in (4096, 16384)]),
        "k_16384": summary([r for r in rows if r["k"] == 16384]),
        "d_equals_k_4096": summary([r for r in rows if r["d"] == r["k"] == 4096])}
    assert summaries["k_4096_or_16384"]["points"] == 48
    assert summaries["k_4096_or_16384"]["maximum_absolute_relative_difference"] < .04
    assert summaries["k_16384"]["points"] == 24
    assert summaries["k_16384"]["maximum_absolute_relative_difference"] < .033
    return dict(schema_version=1, status="passed", command="python code/scripts/iid_depth_comparison_audit.py --check",
        scope="Saved-data audit only: no source draws, Bayes integrals, or predictions generated.",
        method="Raw sample means and ddof1 standard errors recomputed with math.fsum scalar arithmetic, independently of the collector; source hashes, full grid, independent-prediction variance, relative-error denominator, and table entries verified.",
        uncertainty="Relative differences divide by the prediction. Difference SE is sqrt(measurement_SE^2+prediction_SE^2) because predictions use separate random draws. Checkpoints and some source settings share randomness; the maxima are descriptive, not simultaneous confidence guarantees.",
        comparison_sha256=digest(comparison_path), prediction_sha256=digest(prediction_path),
        table_sha256=digest(table_path), aggregate_source_hashes=source_hashes,
        raw_array_hashes=raw_hashes, checked_cells=len(cell_checks), checked_means=sum(c["checked_means"] for c in cell_checks),
        maximum_mean_recomputation_difference_bits=max(c["maximum_mean_recomputation_difference_bits"] for c in cell_checks),
        maximum_SE_recomputation_difference_bits=max(c["maximum_SE_recomputation_difference_bits"] for c in cell_checks),
        maximum_outer_refinement_difference_bits=max(c["maximum_outer_refinement_difference_bits"] for c in cell_checks),
        cell_checks=sorted(cell_checks, key=lambda c: (c["L"], c["d"], c["alpha"])), summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Recompute and compare with the saved audit JSON.")
    parser.add_argument("--self-test", action="store_true", help="Check independent moment arithmetic without raw files.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Independent mean/SE checks passed")
        return
    result = audit()
    if args.check:
        assert read(args.out) == result, "Saved audit differs; regenerate after reviewing the changed inputs."
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "checked_means": result["checked_means"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()

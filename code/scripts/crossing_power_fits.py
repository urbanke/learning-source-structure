"""Fit finite-range powers to the saved crossing observations, using stdlib.

    python3 code/scripts/crossing_power_fits.py
    python3 code/scripts/crossing_power_fits.py --check
    python3 code/scripts/crossing_power_fits.py --self-test

Every requested alphabet size must have an eligible observation. Refined
and checkpoint medians require all runs to have located brackets; mean
curve crossings require an uncensored configuration. Missing values are
never replaced by another statistic or dropped from a fitting window.
The refined search can miss later reversals: its fitted bracket endpoints
are not certified last crossings. These fits describe the measured range.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "results/mechanism/crossing_observations.json"
OUTPUT = REPO / "results/mechanism/crossing_power_fits.json"
ALPHAS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
DEPTHS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24)
WINDOWS = {
    "primary": (4096, 8192, 16384, 32768),
    "wider": (2048, 4096, 8192, 16384, 32768),
    "narrower": (8192, 16384, 32768),
}
METHODS = ("refined_median", "checkpoint_median", "mean_curve_crossing")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def fit_power(points):
    """Unweighted OLS of natural-log length against natural-log alphabet."""
    if len(points) < 2:
        raise ValueError("At least two observations are required")
    ds, values = zip(*points)
    if list(ds) != sorted(set(ds)) or not all(positive(d) for d in ds):
        raise ValueError("Alphabet sizes must be distinct, increasing and positive")
    if not all(positive(v) for v in values):
        raise ValueError("Crossing observations must be finite and positive")
    xs, ys = list(map(math.log, ds)), list(map(math.log, values))
    xbar, ybar = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    sxx = math.fsum((x - xbar) ** 2 for x in xs)
    slope = math.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    intercept = ybar - slope * xbar
    errors = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    residual_ss = math.fsum(e * e for e in errors)
    total_ss = math.fsum((y - ybar) ** 2 for y in ys)
    local = [dict(from_d=a, to_d=b, slope=math.log(vb / va) / math.log(b / a))
             for a, b, va, vb in zip(ds[:-1], ds[1:], values[:-1], values[1:])]
    endpoint_slope = math.log(values[-1] / values[0]) / math.log(ds[-1] / ds[0])

    # Independent identities: pairwise differences recover the OLS slope;
    # local slopes weighted by log interval length recover the endpoint slope.
    numerator = math.fsum(math.log(ds[j] / ds[i]) * math.log(values[j] / values[i])
                          for i in range(len(ds)) for j in range(i + 1, len(ds)))
    denominator = math.fsum(math.log(ds[j] / ds[i]) ** 2
                            for i in range(len(ds)) for j in range(i + 1, len(ds)))
    assert math.isclose(slope, numerator / denominator, rel_tol=1e-10, abs_tol=1e-12)
    local_endpoint = math.fsum(r["slope"] * math.log(r["to_d"] / r["from_d"])
                              for r in local) / math.log(ds[-1] / ds[0])
    assert math.isclose(endpoint_slope, local_endpoint, rel_tol=1e-10, abs_tol=1e-12)
    return dict(p=slope, prefactor=math.exp(intercept), log_intercept=intercept,
                log_R2=1 - residual_ss / total_ss if total_ss > 0 else None,
                max_multiplicative_residual=math.exp(max(map(abs, errors))),
                endpoint_slope=endpoint_slope, local_adjacent_slopes=local)


def index_records(data, instrument):
    indexed = {}
    for row in data[instrument]:
        key = tuple(row[k] for k in ("d", "alpha", "L"))
        if key in indexed:
            raise ValueError(f"Duplicate {instrument} configuration: {key}")
        if type(row["n_seeds"]) is not int or row["n_seeds"] < 1:
            raise ValueError(f"Invalid repetition count: {key}")
        if row["all_runs_located"] != (row["n_located"] == row["n_seeds"]):
            raise ValueError(f"Inconsistent located-run count: {key}")
        if row["all_runs_located"]:
            if row["n_censored"] or row["located"]["n"] != row["n_seeds"]:
                raise ValueError(f"Invalid complete-bracket flag: {key}")
        indexed[key] = row
    return indexed


def observation(index, method, d, alpha, depth):
    row = index.get((d, alpha, depth))
    if row is None:
        return dict(d=d, value=None, reason="configuration_missing")
    point = dict(d=d, value=None, n_seeds=row["n_seeds"])
    if method == "mean_curve_crossing":
        if row["n_censored"]:
            point["reason"] = "censored_configuration"
            return point
        value = row["mean_curve_crossing"]
        if value is None:
            point["reason"] = row["mean_curve_diagnostics"]["status"]
            return point
    else:
        if not row["all_runs_located"]:
            point.update(reason="not_all_runs_located", n_located=row["n_located"],
                         n_censored=row["n_censored"])
            return point
        value = row["located"]["median"]
    if not positive(value):
        raise ValueError(f"Invalid {method} observation: {(d, alpha, depth)}, {value}")
    point["value"] = value
    return point


def build(source):
    raw = source.read_bytes()
    data = json.loads(raw)
    refined, checkpoint = index_records(data, "refined"), index_records(data, "checkpoint")
    records = []
    for method in METHODS:
        index = refined if method == "refined_median" else checkpoint
        for window, ds in WINDOWS.items():
            for alpha in ALPHAS:
                for depth in DEPTHS:
                    inputs = [observation(index, method, d, alpha, depth) for d in ds]
                    eligible = all(p["value"] is not None for p in inputs)
                    records.append(dict(method=method, window=window, alpha=alpha, L=depth,
                                        eligible=eligible, inputs=inputs,
                                        fit=fit_power([(p["d"], p["value"]) for p in inputs])
                                        if eligible else None))
    counts = Counter((r["method"], r["window"]) for r in records if r["eligible"])
    return dict(schema_version=1, generator="code/scripts/crossing_power_fits.py",
                source=source.resolve().relative_to(REPO).as_posix(),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                model="log(n) = log(prefactor) + p log(d)", logarithm="natural",
                windows={k: list(v) for k, v in WINDOWS.items()},
                alphas=list(ALPHAS), source_depths=list(DEPTHS),
                notes=[
                    "Unweighted ordinary least squares on one summary observation per alphabet size.",
                    "Every requested window point must be eligible; points are never dropped or replaced.",
                    "Refined and checkpoint medians require all runs to have located brackets.",
                    "Mean-curve crossings require no censored runs; null means stay missing.",
                    "Refined locations are bracket upper endpoints; observed later reversals invalidate some as last crossings.",
                    "Three-, four-, and five-point fits describe the measured window, not an asymptotic law.",
                    "d is the nominal alphabet size; it is not replaced by occupied or distinct-symbol counts.",
                    "log_R2 measures straightness of these log summaries, not a goodness-of-fit test or sampling confidence.",
                    "max_multiplicative_residual = exp(max absolute log residual), always at least one.",
                    "Local slopes use adjacent observed alphabet doublings within the complete window.",
                    "No confidence intervals or uncertainty of estimated exponents are inferred from these few summaries.",
                    "Coding-depth cutoffs and run counts follow the source archive; L is the source depth.",
                    "Each fit is checked against pairwise-difference OLS and local-to-endpoint slope identities.",
                ],
                eligible_counts=[dict(method=m, window=w, eligible=counts[m, w], total=100)
                                 for m in METHODS for w in WINDOWS], records=records)


def json_text(data):
    meta = {k: v for k, v in data.items() if k != "records"}
    text = json.dumps(meta, indent=2, allow_nan=False).rstrip()[:-1].rstrip()
    text += ',\n  "records": [\n'
    text += ',\n'.join('    ' + json.dumps(r, separators=(',', ':'), allow_nan=False)
                       for r in data["records"])
    return text + '\n  ]\n}\n'


def tsv_text(data):
    fields = ["method", "window", "alpha", "L", "eligible", "p", "prefactor",
              "log_R2", "max_multiplicative_residual", "endpoint_slope",
              "inputs", "local_adjacent_slopes"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in data["records"]:
        fit = row["fit"] or {}
        out = {k: row[k] for k in fields[:5]}
        out.update({k: fit.get(k) for k in fields[5:10]})
        out["inputs"] = json.dumps(row["inputs"], separators=(',', ':'))
        out["local_adjacent_slopes"] = json.dumps(fit.get("local_adjacent_slopes"), separators=(',', ':'))
        writer.writerow(out)
    return stream.getvalue()


def self_test():
    for slope, prefactor in [(1.7, .03), (2., 1.), (-.4, 7.)]:
        for ds in WINDOWS.values():
            result = fit_power([(d, prefactor * d ** slope) for d in ds])
            assert math.isclose(result["p"], slope, abs_tol=1e-12)
            assert math.isclose(result["prefactor"], prefactor, rel_tol=1e-12)
            assert math.isclose(result["log_R2"], 1., abs_tol=1e-14)
            assert math.isclose(result["max_multiplicative_residual"], 1., abs_tol=1e-12)
            assert all(math.isclose(r["slope"], slope, abs_tol=1e-12)
                       for r in result["local_adjacent_slopes"])
    curved = fit_power([(1, 1), (2, 2), (4, 8), (8, 8)])
    assert math.isclose(curved["p"], 1.1, abs_tol=1e-12)
    assert math.isclose(curved["endpoint_slope"], 1., abs_tol=1e-12)
    assert not math.isclose(curved["p"], curved["endpoint_slope"], abs_tol=1e-4)
    assert fit_power([(1, 3), (2, 3), (4, 3)])["log_R2"] is None
    for bad in [[(1, 1), (2, None)], [(1, 1), (2, 0)], [(1, 1), (2, math.inf)], [(2, 1), (2, 3)]]:
        try:
            fit_power(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid observations accepted: {bad}")
    assert observation({}, "refined_median", 4096, .5, 3)["value"] is None
    partial = dict(n_seeds=16, n_located=15, n_censored=0, all_runs_located=False,
                   located=dict(n=15, median=100), mean_curve_crossing=None,
                   mean_curve_diagnostics=dict(status="nonpositive_at_common_end"))
    index = {(4096, .5, 3): partial}
    assert observation(index, "refined_median", 4096, .5, 3)["value"] is None
    assert observation(index, "checkpoint_median", 4096, .5, 3)["value"] is None
    assert observation(index, "mean_curve_crossing", 4096, .5, 3)["value"] is None
    partial["mean_curve_crossing"] = 99
    assert observation(index, "mean_curve_crossing", 4096, .5, 3)["value"] == 99
    partial["n_censored"] = 1
    assert observation(index, "mean_curve_crossing", 4096, .5, 3)["value"] is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Perfect-power, curved, endpoint, missing-value and invalid-input checks passed")
        return
    result = build(args.source)
    text_json, text_tsv = json_text(result), tsv_text(result)
    if args.check:
        assert args.out.read_text() == text_json, "JSON differs from regenerated fits"
        assert args.out.with_suffix(".tsv").read_text() == text_tsv, "TSV differs from regenerated fits"
        print("Source hash, all complete-window fits, omissions and TSV verified")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text_json)
        args.out.with_suffix(".tsv").write_text(text_tsv)
        print(json.dumps(result["eligible_counts"], indent=2))


if __name__ == "__main__":
    main()

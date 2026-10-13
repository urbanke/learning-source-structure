"""Independent replay audit of the extended LSA count/chain experiments.

OPENBLAS_NUM_THREADS=1 python code/scripts/crossing_lsa_extension_audit.py
python code/scripts/crossing_lsa_extension_audit.py --self-test

The default audit checks both curated extension and alpha=2 summaries.
Their ignored raw NPZ files and preserved producer snapshots must be present.
No chains, approximation samples, or moment tables are newly generated.
Existing seeded fixtures are regenerated solely to check the stored source
law and count sampling. Root interpolation is implemented independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def locate(ns, curves):
    """Last observed mean upcrossing; no value for an unlocated curve."""
    curves = np.atleast_2d(curves)
    count = len(ns)
    last = np.max(np.where(curves <= 0, np.arange(count)[None, :], -1), axis=1)
    values = np.full(len(curves), np.nan)
    valid = (last >= 0) & (last < count - 1)
    rows, left = np.flatnonzero(valid), last[valid]
    weight = -curves[rows, left] / (curves[rows, left + 1] - curves[rows, left])
    values[valid] = np.exp(np.log(ns[left]) + weight * np.log(ns[left + 1] / ns[left]))
    return values


def self_test():
    ns = np.array([2., 4., 8., 16.])
    values = locate(ns, [[1, 2, 3, 4], [-1, 1, -1, -2], [-1, 1, -1, 1]])
    assert np.isnan(values[:2]).all()
    assert abs(values[2] - math.sqrt(128)) < 1e-12
    return "Passed independent root fixtures: positive throughout, nonpositive endpoint, and last of multiple upcrossings."


def matching_driver(expected):
    candidates = [ROOT / "code/scripts/crossing_lsa_extension.py",
        ROOT / "output/crossing_lsa_extension/producer_generation_snapshot.py",
        ROOT / "output/crossing_lsa_extension/audited_driver_before_alpha2.py"]
    for path in candidates:
        if path.is_file() and sha(path) == expected:
            return str(path.relative_to(ROOT))
    raise AssertionError(f"No preserved producer matches recorded hash {expected}")


def audit_summary(path):
    summary = json.loads(path.read_text())
    assert summary["complete"]
    driver = matching_driver(summary["script_sha256"])
    if "collection" in summary:
        matching_driver(summary["collection"]["script_sha256"])
    evaluator = ROOT.parent / "layered-simplex-architecture/src/lsa/codelength.py"
    assert sha(evaluator) == summary["evaluator_sha256"]
    for name, expected in summary["dependency_sha256"].items():
        assert sha(ROOT / "code/scripts" / name) == expected
    methods = summary["methods"]
    checked = 0
    maximum_summary_error = maximum_root_error = maximum_occupation_error = 0.
    checked_cells = []
    for cell in summary["cells"]:
        raw_path = ROOT / cell["raw_path"]
        assert sha(raw_path) == cell["raw_sha256"]
        if cell["archive_path"] is not None:
            assert sha(ROOT / cell["archive_path"]) == cell["archive_sha256"]
        else:
            assert summary["no_archive"] and cell["archive_sha256"] is None
            assert cell["archived_mean_curve_crossing"] is None
            assert cell["window_rule"] == "predeclared_early_grid_without_archive"
            assert cell["tokens"] == sorted({2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96,
                128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4 * cell["d"]})
        with np.load(raw_path) as raw:
            d, reps = cell["d"], cell["repetitions"]
            ns, ts = raw["tokens"], raw["transitions"]
            count = len(ns)
            q, initial = raw["q"], raw["initial_states"]
            occupation, tables = raw["expected_occupation_laws"], raw["transition_matrices"]
            assert np.array_equal(ts, ns - 1)
            assert cell["tokens"] == ns.tolist() and cell["transitions"] == ts.tolist()
            q_check = np.arange(1, d + 1, dtype=float) ** (-cell["alpha"])
            q_check /= q_check.sum()
            assert np.array_equal(q, q_check)
            prefix = [cell["seed"], d, cell["L"], round(100 * cell["alpha"]), 0]
            weights = np.random.default_rng(prefix + [77]).exponential(size=(d, d, cell["L"])).prod(2)
            normalization = weights @ q
            assert np.array_equal(tables[0], weights * q[None, :] / normalization[:, None])
            assert np.array_equal(raw["normalized_row_sums"][0], normalization)
            assert int(np.random.default_rng(prefix + [78]).choice(d, p=q)) == initial[0]
            assert np.all(tables > 0) and np.max(abs(tables.sum(2) - 1)) < 1e-12
            j_check = np.flatnonzero(ts <= 31)[-1]
            state, total = q.copy(), np.zeros(d)
            for _ in range(int(ts[j_check])):
                total += state
                state = state @ tables[0]
            error = float(np.max(abs(total / ts[j_check] - occupation[0, j_check])))
            assert error < 1e-12
            maximum_occupation_error = max(maximum_occupation_error, error)
            paths = raw["chain_paths"]
            edges = {"chain": (paths[:, :-1], paths[:, 1:])}
            cdf = np.cumsum(tables[0], axis=1)
            cdf[:, -1] = 1.
            state = initial[0]
            for step, uniform in enumerate(np.random.default_rng(prefix + [79]).random(int(ts[-1]))):
                state = np.searchsorted(cdf[state], uniform, side="right")
                assert state == paths[0, step + 1]
            for method, size in [("block4", 4), ("block16", 16)]:
                if method not in methods:
                    continue
                origins, targets = raw[method + "_sources"], raw[method + "_targets"]
                assert np.array_equal(origins[:, 0], initial)
                inside = np.arange(1, origins.shape[1]) % size != 0
                assert np.array_equal(origins[:, 1:][:, inside], targets[:, :-1][:, inside])
                edges[method] = (origins, targets)
            draws = cell["bootstrap"]["chain"]["draws"]
            bootstrap_weights = np.random.default_rng([cell["seed"], d, cell["L"],
                round(100 * cell["alpha"]), 700]).multinomial(reps, np.full(reps, 1 / reps), size=draws) / reps
            root_draws = {}
            for method in methods:
                pointers = raw[method + "_count_indptr"]
                indices, values = raw[method + "_count_indices"], raw[method + "_count_values"]
                frequencies = raw[method + "_frequency_counts"]
                for rep in range(reps):
                    for j, t in enumerate(ts):
                        sl = slice(pointers[rep * count + j], pointers[rep * count + j + 1])
                        ix, value = indices[sl], values[sl]
                        assert value.sum() == t and np.all(value > 0) and np.all(np.diff(ix) > 0)
                        frequency = np.bincount(ix % d, weights=value, minlength=d).astype(int)
                        frequency[initial[rep]] += 1
                        assert np.array_equal(frequency, frequencies[rep, j])
                        if method in edges:
                            origins, targets = edges[method]
                            expected_ix, expected_value = np.unique(origins[rep, :t].astype(np.int64) * d
                                + targets[rep, :t], return_counts=True)
                            assert np.array_equal(expected_ix, ix) and np.array_equal(expected_value, value)
                        elif rep == 0 and j in [0, count // 2, count - 1]:
                            pair = (occupation[0, j, :, None] * tables[0]).ravel()
                            pair /= pair.sum()
                            expected = np.random.default_rng(prefix + [80, int(t)]).multinomial(int(t), pair)
                            assert np.array_equal(expected[ix], value) and expected.sum() == value.sum()
                        checked += 1
                f, c = raw[method + "_frequency_codelength"], raw[method + "_context_codelength"]
                gap = raw[method + "_gap"]
                assert np.array_equal(f - c, gap)
                for field, array in [("frequency_codelength", f), ("context_codelength", c), ("gap", gap)]:
                    for label, value in [("mean", array.mean(0)), ("SE", array.std(0, ddof=1) / np.sqrt(reps))]:
                        error = float(np.max(abs(value - cell["methods"][method][field][label])))
                        assert error < 1e-10
                        maximum_summary_error = max(maximum_summary_error, error)
                means = gap.mean(0)
                value = locate(ns, means)[0]
                root_record = cell["methods"][method]["mean_curve_crossing"]
                expected = root_record["value"]
                assert (expected is None and np.isnan(value)) or abs(value - expected) < 1e-10
                status = ("nonpositive_at_end" if means[-1] <= 0 else
                          "positive_at_every_check" if np.all(means > 0) else "located")
                assert root_record["status"] == status
                assert root_record["observed_upcrossings"] == np.sum((means[:-1] <= 0) & (means[1:] > 0))
                curves = bootstrap_weights @ gap
                roots = locate(ns, curves)
                recorded = raw[method + "_bootstrap_roots"]
                valid = np.isfinite(roots)
                assert np.array_equal(np.isnan(roots), np.isnan(recorded))
                assert valid.sum() == cell["bootstrap"][method]["located"]
                statuses = dict(located=int(valid.sum()),
                    nonpositive_at_end=int(np.sum(curves[:, -1] <= 0)),
                    positive_at_every_check=int(np.sum(np.all(curves > 0, axis=1))))
                assert {name: total for name, total in statuses.items() if total} == cell["bootstrap"][method]["status_counts"]
                if valid.any():
                    error = float(np.max(abs(roots[valid] - recorded[valid])))
                    assert error < 1e-10
                    maximum_root_error = max(maximum_root_error, error)
                root_draws[method] = roots
            for method in methods:
                if method == "chain":
                    continue
                valid = np.isfinite(root_draws[method]) & np.isfinite(root_draws["chain"])
                record = cell["bootstrap"][method + "_over_chain_minus_one"]
                assert valid.sum() == record["located"]
                if valid.any():
                    quantiles = np.quantile(root_draws[method][valid] / root_draws["chain"][valid] - 1,
                                            [.025, .5, .975])
                    assert np.allclose(quantiles, record["conditional_on_located_quantiles"], atol=1e-11)
            older = [old for campaign in summary.get("campaigns", []) for old in campaign["summary"]["cells"]
                     if (old["d"], old["L"], old["alpha"]) == (d, cell["L"], cell["alpha"])
                     and old["repetitions"] < reps]
            if older:
                previous = min(older, key=lambda item: item["repetitions"])
                size = previous["repetitions"]
                with np.load(ROOT / previous["raw_path"]) as pilot:
                    keys = ["transition_matrices", "stationary_laws", "normalized_row_sums",
                        "expected_occupation_laws", "initial_states", "chain_paths"]
                    keys += [method + suffix for method in ["block4", "block16"] if method in methods
                             for suffix in ["_sources", "_targets"]]
                    for key in keys:
                        assert np.array_equal(pilot[key], raw[key][:size]), key
                    for method in methods:
                        for suffix in ["frequency_counts", "frequency_codelength", "context_codelength", "gap"]:
                            assert np.array_equal(pilot[method + "_" + suffix], raw[method + "_" + suffix][:size])
                        limit = pilot[method + "_count_indptr"][-1]
                        assert np.array_equal(pilot[method + "_count_indptr"], raw[method + "_count_indptr"][:size * count + 1])
                        for suffix in ["count_indices", "count_values"]:
                            assert np.array_equal(pilot[method + "_" + suffix], raw[method + "_" + suffix][:limit])
            checked_cells.append(dict(d=d, L=cell["L"], alpha=cell["alpha"], repetitions=reps,
                raw_sha256=cell["raw_sha256"], previous_sample_prefix="exact" if older else "not_applicable"))
            print(f"Passed L={cell['L']}, alpha={cell['alpha']:g}, d={d}, repetitions={reps}", flush=True)
    return dict(summary_path=str(path.relative_to(ROOT)), summary_sha256=sha(path),
        matching_generation_driver=driver, cells=checked_cells, sparse_count_matrices_checked=checked,
        maximum_summary_error=maximum_summary_error, maximum_bootstrap_root_error=maximum_root_error,
        maximum_direct_occupation_error=maximum_occupation_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, default=[
        ROOT / "results/mechanism/crossing_lsa_extension.json",
        ROOT / "results/mechanism/crossing_lsa_alpha2.json"])
    parser.add_argument("--out", type=Path, default=ROOT / "results/mechanism/crossing_lsa_extension_audit.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    check = self_test()
    if args.self_test:
        print(check)
        return
    start = time.monotonic()
    summaries = [audit_summary(path.resolve()) for path in args.summaries]
    result = dict(status="passed", script_sha256=sha(Path(__file__)), command_argv=sys.argv,
        self_test=check, summaries=summaries, elapsed_seconds=time.monotonic() - start,
        scope="Checks source/count fixtures, all saved counts against paths/blocks, source hashes, mean/SE and gap identities, independent root/bootstrap replay, and exact retained pilot prefixes. It does not replace the original moment-evaluator validation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Passed {sum(len(s['cells']) for s in summaries)} cells and "
          f"{sum(s['sparse_count_matrices_checked'] for s in summaries)} count matrices.")


if __name__ == "__main__":
    main()

"""Audit the saved crossing searches against paired takeover checkpoints.

This reads existing runs; it neither simulates data nor repairs a search.
Only whole-cell exact files are used, so this paired audit concerns the
500-cell small-d set. Separate per-seed files cover larger alphabets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


def quantile(values: list[float], probability: float) -> float:
    """Linear interpolation, matching NumPy's default quantile convention."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[min(left + 1, len(ordered) - 1)] * fraction


def ratio_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "q25": quantile(values, 0.25),
        "median": median(values),
        "q75": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def key(record: dict) -> tuple[int, float, int]:
    return record["d"], record["alpha"], record["L"]


def audit(repo: Path, exact_dir: Path, takeover_dir: Path) -> dict:
    fingerprints: dict[str, str] = {}

    def read(path: Path) -> dict:
        raw = path.read_bytes()
        try:
            name = path.relative_to(repo).as_posix()
        except ValueError:
            name = str(path)
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    exact_paths = sorted(path for path in exact_dir.glob("exact_*.json")
                         if not re.search(r"_s\d+\.json$", path.name))
    if not exact_paths:
        raise ValueError(f"No whole-cell exact files in {exact_dir}")
    exact = {}
    for path in exact_paths:
        record = read(path)
        assert key(record) not in exact, f"Duplicate exact cell: {path}"
        exact[key(record)] = record
        assert sorted(row["seed"] for row in record["seeds"]) == list(range(256))

    takeover = {}
    for path in sorted(takeover_dir.rglob("summary_*.json")):
        record = read(path)
        assert key(record) not in takeover, f"Duplicate takeover cell: {path}"
        takeover[key(record)] = (path, record)
    assert set(exact) <= set(takeover), "A complete exact cell lacks a takeover summary"

    rows = [row for record in exact.values() for row in record["seeds"]]
    notes = Counter(row.get("note", "bracketed") for row in rows)
    table = Counter()
    examples = {"early_positive": [], "bracketed": []}
    cell_ratios, paired_cell_ratios, individual_ratios = [], [], []
    metadata = Counter()
    late_within_15 = Counter()
    late_anchor_store = Counter()
    eligible_flags = eligible_flags_late = 0
    selected_cells = []

    for cell, record in sorted(exact.items()):
        summary_path, summary = takeover[cell]
        assert summary["seeds"] == list(range(16))
        exact_by_seed = {row["seed"]: row for row in record["seeds"]}
        exact_median = median(row["first_pos"] for row in record["seeds"]
                              if row.get("first_pos") is not None)
        eligible = exact_median > 32 and bool(summary["n_star_last"])
        paired_exact, paired_takeover = [], []
        for seed in summary["seeds"]:
            matches = list(summary_path.parent.glob(f"*_seed{seed}.json"))
            assert len(matches) == 1, f"Missing or duplicate seed file: {cell}, {seed}"
            trajectory = read(matches[0])
            assert key(trajectory) == cell and trajectory["seed"] == seed
            assert len(trajectory["checkpoints"]) == len(trajectory["lam"])
            assert trajectory["checkpoints"] == sorted(set(trajectory["checkpoints"]))
            assert all(math.isclose(value, l_f - l_c, rel_tol=1e-11, abs_tol=1e-8)
                       for value, l_f, l_c in zip(trajectory["lam"], trajectory["l_F"], trajectory["l_C"]))
            all_nonpositive = [n for n, value in zip(trajectory["checkpoints"], trajectory["lam"])
                               if value <= 0]
            if not trajectory["censored"]:
                last_nonpositive = max(all_nonpositive, default=0)
                assert trajectory["n_star_last"] == min(n for n in trajectory["checkpoints"]
                                                        if n > last_nonpositive)
            row = exact_by_seed[seed]
            candidate = row["first_pos"]
            assert candidate is not None, "A paired exact result is censored"
            category = ("early_positive" if row.get("note") == "positive from the start"
                        else "bracketed")
            if category == "bracketed":
                assert candidate - row["last_neg"] == 1
            else:
                assert candidate == 8 and row["last_neg"] is None
            later = [(n, value) for n, value in zip(trajectory["checkpoints"], trajectory["lam"])
                     if n > candidate and value <= 0]
            table[(category, bool(later))] += 1
            if later and Path(trajectory.get("store", "")).name == "anchors_prod":
                late_anchor_store[category] += 1
            if later and any(n <= 1.5 * candidate for n, _ in later):
                late_within_15[category] += 1
            if eligible and category == "early_positive":
                eligible_flags += 1
                eligible_flags_late += bool(later)
            if later:
                first_n, first_value = later[0]
                last_n, last_value = later[-1]
                examples[category].append({
                    "d": cell[0], "alpha": cell[1], "L": cell[2], "seed": seed,
                    "exact_first_pos": candidate, "exact_last_neg": row["last_neg"],
                    "first_later_nonpositive_n": first_n,
                    "first_later_nonpositive_lambda_bits": first_value,
                    "last_later_nonpositive_n": last_n,
                    "last_later_nonpositive_lambda_bits": last_value,
                    "takeover_n_star_last": trajectory["n_star_last"],
                    "takeover_horizon": trajectory["horizon"],
                    "takeover_censored": trajectory["censored"],
                    "last_later_n_over_exact_candidate": last_n / candidate,
                })
            metadata[(cell[0], trajectory["l_max"], trajectory.get("sequence_estimator", ""),
                      trajectory.get("store", ""))] += 1
            if not trajectory["censored"]:
                paired_exact.append(candidate)
                paired_takeover.append(trajectory["n_star_last"])
                if candidate > 32:
                    individual_ratios.append(trajectory["n_star_last"] / candidate)
        if eligible:
            takeover_median = median(summary["n_star_last"])
            cell_ratios.append(takeover_median / exact_median)
            paired_cell_ratios.append(median(paired_takeover) / median(paired_exact))
            selected_cells.append({"d": cell[0], "alpha": cell[1], "L": cell[2],
                                   "exact_median": exact_median,
                                   "takeover_median": takeover_median})

    # Bind file names and content hashes without storing paths to every raw run.
    digest = hashlib.sha256()
    for name, content_hash in sorted(fingerprints.items()):
        digest.update(name.encode() + b"\0" + content_hash.encode() + b"\n")
    source_hashes = {}
    for name in ("mech_common.py", "exact_crossing.py", "takeover_experiment.py",
                 "aggregate_takeover.py", "exact_verdict.py"):
        source_hashes[name] = hashlib.sha256((repo / "code/scripts" / name).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "description": "Read-only comparison of saved, seed-paired crossing searches; no new simulation.",
        "pairing_basis": "Same (d, alpha, source L, seed), deterministic ChainSampler seed family; seeds 0..15.",
        "later_nonpositive_definition": "A saved takeover checkpoint n strictly greater than exact first_pos with Lambda(n) <= 0.",
        "interpretation": "A later nonpositive checkpoint contradicts interpreting the exact candidate as a permanent or last crossing. Its absence does not certify a permanent crossing.",
        "input_fingerprint": {"algorithm": "SHA256 of sorted relative filename, NUL, content SHA256, newline",
                              "files": len(fingerprints), "sha256": digest.hexdigest()},
        "source_sha256": source_hashes,
        "inventory": {
            "exact_complete_cells": len(exact), "exact_rows": len(rows),
            "exact_source_axes": {name: sorted({cell[i] for cell in exact})
                                  for i, name in enumerate(("d", "alpha", "L"))},
            "exact_notes": dict(notes),
            "exact_cell_median_equals_8": sum(median(row["first_pos"] for row in record["seeds"]) == 8
                                             for record in exact.values()),
            "takeover_summaries": len(takeover),
            "takeover_censored_seed_counts": dict(sorted(Counter(record["n_censored"]
                                                                for _, record in takeover.values()).items())),
            "takeover_mean_crossings": sum(record["mean_lambda_crossing"] is not None
                                           for _, record in takeover.values()),
            "paired_rows": sum(table.values()),
        },
        "paired_cross_table": {
            category: {"later_nonpositive": table[(category, True)],
                       "no_recorded_later_nonpositive": table[(category, False)],
                       "later_nonpositive_within_1_5_candidate": late_within_15[category],
                       "later_nonpositive_with_anchors_prod_metadata": late_anchor_store[category]}
            for category in ("early_positive", "bracketed")},
        "overlap_cell_median_ratio": {
            "selection": "Common cells with median(all 256 exact first_pos, including 8 flags) > 32 and an uncensored takeover median.",
            "numerator": "Median of 16 saved takeover n_star_last values",
            "denominator": "Median of 256 saved exact first_pos values",
            **ratio_summary(cell_ratios),
            "paired_early_positive_rows_in_selected_cells": eligible_flags,
            "of_those_with_later_nonpositive": eligible_flags_late,
        },
        "same_selected_cells_using_only_paired_16_seeds": ratio_summary(paired_cell_ratios),
        "paired_individual_ratios_with_exact_candidate_above_32": ratio_summary(individual_ratios),
        "examples": {
            category: sorted(values, key=lambda item: item["last_later_n_over_exact_candidate"], reverse=True)[:8]
            for category, values in examples.items()},
        "specific_small_example": next(item for item in examples["early_positive"]
                                       if (item["d"], item["alpha"], item["L"], item["seed"]) == (128, 0.6, 8, 0)),
        "takeover_evaluator_metadata": [
            {"d": d, "l_max": cutoff, "sequence_estimator": estimator, "store": store, "paired_rows": count}
            for (d, cutoff, estimator, store), count in sorted(metadata.items())],
        "limitations": [
            "The exact output does not save its complete probe history, code-version hash, evaluator settings or final verification horizon.",
            "The pairing checks metadata and the common deterministic generator implementation, not a stored token-sequence checksum.",
            "Some paired takeover paths have legacy universal_v2 metadata; the metadata counts identify these separately.",
            "Crossings of seed-averaged Lambda and medians of individual path crossings are different statistics.",
        ],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", type=Path, default=repo / "output/exact")
    parser.add_argument("--takeover-dir", type=Path, default=repo / "output/takeover")
    parser.add_argument("--out", type=Path, default=repo / "results/mechanism/crossing_search_audit.json")
    args = parser.parse_args()
    result = audit(repo, args.exact_dir, args.takeover_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": result["inventory"], "paired_cross_table": result["paired_cross_table"],
                      "overlap_cell_median_ratio": result["overlap_cell_median_ratio"]}, indent=2))


if __name__ == "__main__":
    main()

"""Archive selected mean-gap crossings from existing takeover summaries.

This reads saved results only. It does not simulate paths, replace missing
mean crossings by path medians, or estimate sampling uncertainty. Slopes
join consecutive available mean crossings at a fixed alpha; their alphabet
endpoints are recorded explicitly, including gaps over missing means.

    python code/scripts/crossing_above_one_audit.py
    python code/scripts/crossing_above_one_audit.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ALPHAS = (1.1, 1.25, 1.5)
ALPHABET_SIZES = tuple(2**power for power in range(6, 16))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def validate(summary: dict, d: int, alpha: float, path: Path) -> None:
    expected = (d, alpha, 3)
    actual = tuple(summary.get(key) for key in ("d", "alpha", "L"))
    if actual != expected:
        raise ValueError(f"{path}: configuration {actual} != {expected}")
    seeds = summary.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path}: missing seed identifiers")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"{path}: invalid seed identifier")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate seed identifier")
    censored = summary.get("n_censored")
    if type(censored) is not int or not 0 <= censored <= len(seeds):
        raise ValueError(f"{path}: invalid censoring count")
    if "mean_lambda_crossing" not in summary:
        raise ValueError(f"{path}: missing mean-gap crossing field")
    mean = summary["mean_lambda_crossing"]
    if mean is not None and (
        type(mean) not in (int, float) or not math.isfinite(mean) or mean <= 0
    ):
        raise ValueError(f"{path}: mean crossing must be finite and positive or null")
    cutoff = summary.get("l_max")
    if cutoff is not None and (type(cutoff) is not int or cutoff < 1):
        raise ValueError(f"{path}: invalid recorded coding-depth cutoff")


def build(source: Path) -> dict:
    records = []
    missing = []
    for alpha in ALPHAS:
        previous_mean = None
        for d in ALPHABET_SIZES:
            directory = source / f"d{d}_a{alpha}_L3"
            candidates = sorted(directory.glob("summary_*.json"))
            if not candidates:
                missing.append({"d": d, "alpha": alpha, "L": 3})
                continue
            if len(candidates) != 1:
                raise ValueError(f"{directory}: expected one summary, found {len(candidates)}")
            path = candidates[0]
            summary = json.loads(path.read_text())
            validate(summary, d, alpha, path)
            mean = summary["mean_lambda_crossing"]
            slope = None
            if mean is not None:
                if previous_mean is not None:
                    old_d, old_mean = previous_mean
                    log_ratio = math.log(d / old_d)
                    slope = {
                        "from_d": old_d,
                        "to_d": d,
                        "delta_d": d - old_d,
                        "alphabet_ratio": d / old_d,
                        "log_alphabet_ratio": log_ratio,
                        "exponent": math.log(mean / old_mean) / log_ratio,
                    }
                previous_mean = (d, mean)
            records.append({
                "d": d,
                "alpha": alpha,
                "L": 3,
                "seeds": summary["seeds"],
                "n_seeds": len(summary["seeds"]),
                "n_censored": summary["n_censored"],
                "n_cap": summary.get("n_cap"),
                "l_max": summary.get("l_max"),
                "mean_lambda_crossing": mean,
                "n_mean_over_d": mean / d if mean is not None else None,
                "slope_from_previous_available_mean": slope,
                "source": relative(path),
                "source_sha256": digest(path),
            })
    if not records:
        raise ValueError(f"{source}: no selected takeover summaries found")
    implementation = REPO / "code/scripts/takeover_experiment.py"
    return {
        "schema_version": 1,
        "generator": "code/scripts/crossing_above_one_audit.py",
        "source_root": relative(source),
        "selection": {"alpha": list(ALPHAS), "L": 3, "d": list(ALPHABET_SIZES)},
        "statistic": (
            "Last nonpositive-to-positive zero of the seed-averaged cumulative "
            "code gap on checkpoints common to all seeds, interpolated linearly "
            "in log token length; requires a positive final common-grid mean."
        ),
        "statistic_implementation": {
            "source": relative(implementation), "sha256": digest(implementation),
        },
        "notes": [
            "These are finite-horizon mean-gap crossings, not means or medians of path crossing times.",
            "Null means are preserved. A null does not identify why the mean crossing is unavailable.",
            "Slopes use consecutive non-null means at each alpha; endpoints record every gap in alphabet size.",
            "n_cap is the maximum budget, not necessarily the final common observation horizon.",
            "l_max is copied when recorded in a summary; null means absent, with no cutoff inferred.",
            "No uncertainty is inferred from these aggregate summaries.",
        ],
        "missing_configurations": missing,
        "records": records,
    }


def render(result: dict) -> str:
    metadata = {key: value for key, value in result.items() if key != "records"}
    prefix = json.dumps(metadata, indent=2, allow_nan=False)[:-2]
    rows = [json.dumps(row, separators=(",", ":"), allow_nan=False)
            for row in result["records"]]
    return prefix + ',\n  "records": [\n    ' + ',\n    '.join(rows) + '\n  ]\n}\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "output/takeover")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/mechanism/crossing_above_one_audit.json")
    parser.add_argument("--check", action="store_true",
                        help="verify that the saved artifact matches the current source summaries")
    args = parser.parse_args()
    result = build(args.source.resolve())
    if args.check:
        if json.loads(args.out.read_text()) != result:
            raise SystemExit("Saved audit does not match current inputs; regenerate it.")
        print(f"Verified {len(result['records'])} source records and their hashes.")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(result))
        print(f"Wrote {len(result['records'])} records to {args.out}")


if __name__ == "__main__":
    main()

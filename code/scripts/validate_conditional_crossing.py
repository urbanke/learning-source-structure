"""Reproduce bounded B/U/fixed-depth grid checks for conditional crossing.

Full mode constructs one common 1025-token source draw on three integration
grids. Smoke mode rechecks the exact cached files bound by an earlier report,
without evaluating new moment tables. Neither mode launches the main cohort.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
import eval_conditional_crossing as driver

GRIDS = {"production": (-30.0, 30.0), "wide": (-36.0, 36.0),
         "offset": (-36.0078125, 35.9921875)}
FIELDS = ("bayes_probs", "unigram_probs", "conditioned_bayes_probs", "bayes_regret_bits",
          "unigram_regret_bits", "instantaneous_gap_bits", "cumulative_gap_bits",
          "conditioned_to_bayes_kl_bits", "conditioned_to_unigram_kl_bits",
          "conditioned_gap_bits", "conditioned_cumulative_gap_bits", "bayes_loss_bits", "unigram_loss_bits")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("output/conditional_crossing/grid_validation"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/transformer/conditional_crossing/numerical_validation.json"))
    parser.add_argument("--replace", action="store_true", help="replace a derived validation receipt after rechecking")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--existing-report", type=Path, help="bound raw files to reuse with --smoke")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--source-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    args = parser.parse_args(argv)
    if args.smoke and args.existing_report is None:
        parser.error("--smoke requires --existing-report")
    previous = json.loads(args.existing_report.read_text()) if args.smoke else None
    if previous is not None and (previous.get("passed") is not True
                                 or previous["driver_sha256"] != driver.sha_file(Path(driver.__file__))):
        raise ValueError("prior numerical receipt is failed or belongs to a different driver")
    records, inputs = {}, {}
    for name, (lower, upper) in GRIDS.items():
        if previous is None:
            folder = args.work_dir / name
            driver.main(["--out", str(folder), "--d", str(args.d), "--n-ctx", str(args.n_ctx),
                         "--source-depth", str(args.source_depth), "--seed", str(args.seed),
                         "--start-index", str(args.index), "--n-seq", "1", "--store", str(args.store),
                         "--pmm-src", str(args.pmm_src), "--u-lo", str(lower), "--u-hi", str(upper)])
            path = folder / f"seq_{args.index:06d}.npz"
        else:
            path = Path(previous["inputs"][name]["path"])
            if driver.sha_file(path) != previous["inputs"][name]["sha256"]:
                raise ValueError(f"bound validation cache changed: {name}")
        identity = driver.make_identity(args.d, args.n_ctx, args.source_depth, args.seed, args.index,
                                        args.store, args.pmm_src, lower, upper)
        records[name] = driver.load_record(path, identity)
        inputs[name] = dict(path=str(path.resolve()), sha256=driver.sha_file(path))
    original, metadata = records["production"]
    comparisons = {}
    for name in ("wide", "offset"):
        arrays, other = records[name]
        same = (all(np.array_equal(original[key], arrays[key])
                    for key in ("tokens", "transition_probs", "initial_probs", "permutation"))
                and metadata["identity"]["source"] == other["identity"]["source"]
                and metadata["identity"]["code_sha256"] == other["identity"]["code_sha256"])
        errors = {key: float(np.max(np.abs(original[key] - arrays[key]))) for key in FIELDS}
        finite = all(np.isfinite(value) for value in errors.values())
        passed = (same and finite
                  and all(value < 1e-7 for key, value in errors.items() if key.endswith("_probs"))
                  and all(value < 1e-6 for key, value in errors.items() if not key.endswith("_probs")))
        comparisons[name] = dict(passed=passed, same_source=same, max_absolute_differences=errors,
                                 grid=other["identity"]["prediction"], raw_checks=other["checks"])
    checks = [dict(name=name, passed=value["passed"]) for name, value in comparisons.items()]
    report = dict(schema="conditional-crossing-numerical-check-v1", passed=all(c["passed"] for c in checks),
                  driver_sha256=driver.sha_file(Path(driver.__file__)),
                  validator_sha256=driver.sha_file(Path(__file__)), cached_only=args.smoke,
                  n_ctx=args.n_ctx, d=args.d, source_depth=args.source_depth,
                  master_seed=args.seed, sequence_index=args.index,
                  bayes_depths=metadata["identity"]["prediction"]["bayes_depths"],
                  unigram_depths=metadata["identity"]["prediction"]["unigram_depths"],
                  main_prediction_prior_not_conditioned=True, inputs=inputs, comparisons=comparisons,
                  checks=checks, production_checks=metadata["checks"],
                  production_sequence_seconds=metadata["elapsed_seconds"],
                  thresholds=dict(probabilities=1e-7, per_position_and_cumulative_bits=1e-6),
                  limitations=["One independent long path, not all possible count profiles.",
                               "Wider bounds and a half-cell phase shift test fixed-grid quadrature; the table approximation is shared."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if driver.canonical(json.loads(args.out.read_text())) != driver.canonical(report):
            if not args.replace:
                raise ValueError("an existing validation report differs; use a new --out path or --replace")
            temporary = args.out.with_name(args.out.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.out)
    else:
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(out=str(args.out), passed=report["passed"], cached_only=args.smoke)))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

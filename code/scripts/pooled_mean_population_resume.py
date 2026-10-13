"""Copy a validated calibration overlay, then run only the t=16 population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/transformer")]
import pooled_mean_population as population
import recover_pooled_mean_calibration as recovery


def verified_overlay(plan_path, raw, dependencies, recovery_out):
    """Replay the complete numerical gate; a receipt alone is insufficient."""
    recovery_out = Path(recovery_out).resolve()
    plan = recovery.load_plan(recovery_out, plan_path=plan_path, raw=raw,
                              dependencies=dependencies)
    summary = json.loads((recovery_out / "summary.json").read_text())
    population.require(summary["schema"] == recovery.SUMMARY_SCHEMA
                       and summary["plan_id"] == plan["plan_id"]
                       and summary["recovery_plan_sha256"] == population.sha(
                           recovery_out / "recovery_plan.json")
                       and summary["complete"] is True and summary["passed"] is True
                       and summary["population_eligible"] is True
                       and summary["recovered_repetitions"] == 141
                       and summary["population_submitted"] is False,
                       "Recovery is not a passed complete calibration")
    population.require(set(summary["recovery_result_sha256"]) ==
                       {str(i) for i in range(141)},
                       "Recovered repetition coverage changed")
    for item in plan["missing"]:
        index = item["recovery_index"]
        path = recovery.result_path(recovery_out, index)
        population.require(population.sha(path) ==
                           summary["recovery_result_sha256"][str(index)],
                           "Recovered repetition changed")
        recovery.check_result(json.loads(path.read_text()), plan, item)
    overlay = recovery_out / "overlay"
    binding = summary["overlay"]
    population.require(binding["path"] == str(overlay) and
        set(binding["task_sha256"]) == {
            f"calibration/tasks/task-{i:04d}.json" for i in range(52)},
        "Recovery overlay coverage changed")
    hashes = {"plan.json": binding["plan_sha256"],
              "prepared.json": binding["prepared_sha256"],
              "calibration/summary.json": binding["calibration_summary_sha256"],
              **binding["task_sha256"]}
    for name, expected in hashes.items():
        path = overlay / name
        population.require(path.is_file() and not path.is_symlink()
                           and population.sha(path) == expected,
                           "Recovery overlay changed: " + name)
    population.require(population.sha(Path(plan_path)) == binding["plan_sha256"],
                       "Overlay plan changed")
    saved = json.loads((overlay / "calibration/summary.json").read_text())
    replay = population.merge(plan_path, "calibration", raw, dependencies,
                              overlay, write=False)
    population.require(saved == replay and saved["complete"] is True
                       and saved["passed"] is True,
                       "Recovered calibration gate does not replay")
    population.calibration_gate(overlay, binding["plan_sha256"],
                                binding["prepared_sha256"])
    return overlay, hashes


def prepare(plan_path, raw, dependencies, recovery_out, out):
    out = Path(out).resolve()
    overlay, hashes = verified_overlay(plan_path, raw, dependencies, recovery_out)
    marker = out / "calibration_source.json"
    population.require(not marker.exists() and not (out / "prepared.json").exists(),
                       "Population preparation already attempted")
    for name in hashes:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        population.require(not destination.exists(), "Population file already exists: " + name)
        shutil.copyfile(overlay / name, destination)
        population.require(population.sha(destination) == hashes[name],
                           "Copied calibration differs: " + name)
    # The same checks used by every population worker must pass here.
    p, plan_hash, _ = population.read_plan(plan_path)
    prepared, prepared_hash = population.read_prepared(p, plan_hash, out,
                                                       dependencies, full=True)
    population.require(len(population.tasks(prepared, "population")) == 768,
                       "Population task coverage changed")
    population.calibration_gate(out, plan_hash, prepared_hash)
    replay = population.merge(plan_path, "calibration", raw, dependencies,
                              out, write=False)
    population.require(replay == json.loads((out / "calibration/summary.json").read_text()),
                       "Copied calibration does not replay")
    population.save(marker, dict(schema="pooled-mean-population-resume-source-v1",
                                 recovery=str(Path(recovery_out).resolve()),
                                 source_sha256=hashes))
    print(json.dumps(dict(event="population_prepared", tasks=768)), flush=True)


def merge(plan_path, raw, dependencies, recovery_out, out):
    out = Path(out).resolve()
    _, hashes = verified_overlay(plan_path, raw, dependencies, recovery_out)
    marker = json.loads((out / "calibration_source.json").read_text())
    population.require(marker == dict(schema="pooled-mean-population-resume-source-v1",
        recovery=str(Path(recovery_out).resolve()), source_sha256=hashes),
        "Population source binding changed")
    for name, expected in hashes.items():
        population.require(population.sha(out / name) == expected,
                           "Population calibration changed: " + name)
    population.require(not (out / "population/summary.json").exists(),
                       "Population merge already attempted")
    result = population.merge(plan_path, "population", raw, dependencies, out)
    return 0 if result["passed"] and result["scientific_reference_validated"] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "merge"))
    for name in ("plan", "raw", "dependencies", "recovery", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.plan, args.raw, args.dependencies, args.recovery, args.out)
        return 0
    return merge(args.plan, args.raw, args.dependencies, args.recovery, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

"""Immutable recovery of the missing pooled-mean calibration repetitions.

The failed parent campaign is read-only.  Recovery array entries each compute
one previously absent repetition with its original task specification and
seed.  A final job builds a separate, complete overlay and reruns the frozen
calibration merge.  This workflow never submits or runs population work.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/transformer")]
import pooled_mean_population as parent


SCHEMA = "pooled-mean-calibration-recovery-plan-v1"
RESULT_SCHEMA = "pooled-mean-calibration-recovery-repetition-v1"
SUMMARY_SCHEMA = "pooled-mean-calibration-recovery-summary-v1"
DEFAULT_PLAN = ROOT / "code/transformer/configs/pooled_mean_population_plan_2026-09-20.json"
DEFAULT_PARENT = ROOT / "output/pooled_mean_population_memory_recovery_2026-09-20"
DEFAULT_OUT = ROOT / "output/pooled_mean_calibration_recovery_2026-09-21"

# This is the verified state reported from Jed on 2026-09-21.  Recovery is
# deliberately specific to it; a changed parent requires a new review/plan.
EXPECTED_PREFIX_COUNTS = {
    7: 20, 10: 18, 11: 0, 14: 31, 15: 21, 19: 29, 22: 31, 23: 21,
    27: 28, 31: 15, 35: 28, 39: 22, 43: 23, 47: 24, 51: 28,
}
EXPECTED_MISSING_TASKS = sorted(EXPECTED_PREFIX_COUNTS)
EXPECTED_SAVED_REPETITIONS = 1523
EXPECTED_MISSING_REPETITIONS = 141
RECOVERY_MINUTES = 130
PREPARE_MINUTES = 60
MERGE_MINUTES = 60
CONCURRENCY = 16
CPUS_PER_TASK = 1
MEMORY_MIB = 6 * 1024
MAXIMUM_ATTEMPTS_PER_REPETITION = 1
CODE = (
    "code/scripts/recover_pooled_mean_calibration.py",
    "code/transformer/scitas/pooled_mean_calibration_recovery.sbatch",
    "code/transformer/scitas/submit_pooled_mean_calibration_recovery.sh",
)
require, sha, digest, save = parent.require, parent.sha, parent.digest, parent.save


def bound(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), "Frozen file missing or linked: " + str(path))
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def verify_bound(item):
    require(isinstance(item, dict) and {"path", "sha256", "size_bytes"} <= set(item),
            "Malformed frozen-file binding")
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
            "Frozen file changed: " + str(path))
    return path


def output_path(out, parent_out=None):
    out = Path(out).expanduser().resolve()
    parent_out = Path(parent_out or DEFAULT_PARENT).expanduser().resolve()
    require(out != parent_out and out not in parent_out.parents and parent_out not in out.parents,
            "Recovery output must be separate from the parent campaign")
    protected = [ROOT / "code", ROOT / "results", ROOT / "notes", ROOT / ".git"]
    require(out != ROOT and ROOT in out.parents
            and not any(out == path or out in path.parents or path in out.parents for path in protected),
            "Recovery output overlaps repository sources")
    return out


def result_path(out, recovery_index):
    return Path(out) / "repetitions" / f"repetition-{recovery_index:04d}.json"


def _load_parent(plan_path, raw, dependencies, parent_out):
    """Strictly replay the current incomplete parent calibration."""
    plan_path, raw, dependencies, parent_out = map(
        lambda value: Path(value).expanduser().resolve(),
        (plan_path, raw, dependencies, parent_out))
    p, plan_hash, _ = parent.read_plan(plan_path)
    prepared, prepared_hash = parent.read_prepared(p, plan_hash, parent_out, dependencies, full=True)
    specs = parent.tasks(prepared, "calibration")
    require(len(specs) == 52, "Parent calibration task coverage changed")
    reports, saved = {}, 0
    missing = []
    for spec in specs:
        path = parent.task_path(parent_out, "calibration", spec["task_index"])
        require(path.is_file() and not path.is_symlink(), "Parent task is missing: " + str(path))
        report = json.loads(path.read_text())
        parent.check_task(report, p, plan_hash, prepared_hash, spec)
        reports[spec["task_index"]] = report
        saved += len(report["records"])
        if not report["complete"]:
            missing.append(spec["task_index"])
    require(missing == EXPECTED_MISSING_TASKS, "Parent missing-task set changed")
    require(saved == EXPECTED_SAVED_REPETITIONS, "Parent saved-repetition count changed")
    for index, expected in EXPECTED_PREFIX_COUNTS.items():
        require(len(reports[index]["records"]) == expected,
                f"Parent task {index} prefix length changed")
    require(sum(p["repetitions"] - len(reports[i]["records"]) for i in missing)
            == EXPECTED_MISSING_REPETITIONS, "Recovery repetition count changed")
    summary_path = parent_out / "calibration/summary.json"
    require(summary_path.is_file() and not summary_path.is_symlink(),
            "Parent calibration summary is missing")
    summary = json.loads(summary_path.read_text())
    # The status command has one machine-readable response; suppress the parent
    # merger's ordinary progress line while retaining its full validation.
    with redirect_stdout(io.StringIO()):
        replay = parent.merge(plan_path, "calibration", raw, dependencies, parent_out, write=False)
    require(summary == replay and summary["complete"] is False and summary["passed"] is False
            and summary["missing_tasks"] == missing,
            "Parent incomplete calibration summary does not replay")
    return dict(plan=p, plan_hash=plan_hash, prepared=prepared,
                prepared_hash=prepared_hash, specs=specs, reports=reports,
                summary=summary, plan_path=plan_path, raw=raw,
                dependencies=dependencies, parent_out=parent_out)


def inspect(plan_path, raw, dependencies, parent_out):
    state = _load_parent(plan_path, raw, dependencies, parent_out)
    counts = {str(i): len(state["reports"][i]["records"]) for i in EXPECTED_MISSING_TASKS}
    return dict(schema="pooled-mean-calibration-recovery-status-v1", ready=True,
                parent_output=str(state["parent_out"]), saved_repetitions=sum(
                    len(r["records"]) for r in state["reports"].values()),
                missing_repetitions=EXPECTED_MISSING_REPETITIONS,
                missing_tasks=EXPECTED_MISSING_TASKS, saved_prefix_counts=counts,
                parent_complete=False, parent_passed=False,
                population_submitted=False, automatic_retries=False)


def _code_hashes():
    return {name: sha(ROOT / name) for name in CODE}


def _budget(rate):
    cpu_hours = CPUS_PER_TASK * (
        PREPARE_MINUTES + EXPECTED_MISSING_REPETITIONS * RECOVERY_MINUTES + MERGE_MINUTES) / 60
    return dict(maximum_cpu_hours=cpu_hours, cpu_hour_rate_chf=rate,
                maximum_cost_chf=cpu_hours * rate,
                cost_is_rate_based_projection=True)


def build_plan(plan_path, raw, dependencies, parent_out, out):
    state = _load_parent(plan_path, raw, dependencies, parent_out)
    out = output_path(out, parent_out)
    missing = []
    for spec in state["specs"]:
        start = len(state["reports"][spec["task_index"]]["records"])
        for repetition in range(start, state["plan"]["repetitions"]):
            missing.append(dict(recovery_index=len(missing), task_index=spec["task_index"],
                                repetition=repetition, task=spec))
    require(len(missing) == EXPECTED_MISSING_REPETITIONS, "Recovery scope changed")
    task_files = {}
    for spec in state["specs"]:
        relative = f"calibration/tasks/task-{spec['task_index']:04d}.json"
        task_files[relative] = bound(state["parent_out"] / relative)
    rate = state["plan"]["scheduler"]["cpu_hour_rate_chf"]
    plan = dict(schema=SCHEMA, output=str(out), parent_output=str(state["parent_out"]),
        original_plan=bound(state["plan_path"]), original_plan_sha256=state["plan_hash"],
        parent_prepared=bound(state["parent_out"] / "prepared.json"),
        parent_calibration_summary=bound(state["parent_out"] / "calibration/summary.json"),
        parent_task_files=task_files, prepared_sha256=state["prepared_hash"],
        missing=missing, expected_missing_tasks=EXPECTED_MISSING_TASKS,
        expected_saved_repetitions=EXPECTED_SAVED_REPETITIONS,
        expected_missing_repetitions=EXPECTED_MISSING_REPETITIONS,
        raw=str(state["raw"]), dependencies=str(state["dependencies"]),
        resources=dict(cpus_per_task=CPUS_PER_TASK, memory_mib=MEMORY_MIB,
            prepare_minutes=PREPARE_MINUTES, recovery_minutes=RECOVERY_MINUTES,
            merge_minutes=MERGE_MINUTES, concurrency=CONCURRENCY,
            seconds_per_repetition=state["plan"]["seconds_per_repetition"],
            worker_margin_seconds=state["plan"]["worker_margin_seconds"],
            maximum_attempts_per_repetition=MAXIMUM_ATTEMPTS_PER_REPETITION),
        budget=_budget(rate), code_sha256=_code_hashes(),
        automatic_retries=False, population_submitted=False,
        population_requires_passed_calibration_overlay=True)
    plan["plan_id"] = digest(plan)
    return plan


def prepare(plan_path, raw, dependencies, parent_out, out):
    out = output_path(out, parent_out)
    path = out / "recovery_plan.json"
    require(not path.exists(), "Recovery plan already exists; inspect it, do not replace it")
    plan = build_plan(plan_path, raw, dependencies, parent_out, out)
    save(path, plan)
    print(json.dumps(dict(event="recovery_prepared", plan_id=plan["plan_id"],
        missing_repetitions=len(plan["missing"]), maximum_cpu_hours=plan["budget"]["maximum_cpu_hours"],
        maximum_cost_chf=plan["budget"]["maximum_cost_chf"], population_submitted=False)), flush=True)
    return plan


def load_plan(out, *, plan_path=None, raw=None, dependencies=None, parent_out=None):
    out = Path(out).expanduser().resolve()
    plan = json.loads((out / "recovery_plan.json").read_text())
    identity = {k: v for k, v in plan.items() if k != "plan_id"}
    require(plan["schema"] == SCHEMA and plan["output"] == str(out)
            and plan["plan_id"] == digest(identity), "Recovery plan identity changed")
    expected_resources = dict(cpus_per_task=CPUS_PER_TASK, memory_mib=MEMORY_MIB,
        prepare_minutes=PREPARE_MINUTES, recovery_minutes=RECOVERY_MINUTES,
        merge_minutes=MERGE_MINUTES, concurrency=CONCURRENCY,
        seconds_per_repetition=7200,
        worker_margin_seconds=300,
        maximum_attempts_per_repetition=MAXIMUM_ATTEMPTS_PER_REPETITION)
    require(plan["automatic_retries"] is False and plan["population_submitted"] is False
            and plan["population_requires_passed_calibration_overlay"] is True
            and plan["resources"] == expected_resources
            and plan["expected_missing_tasks"] == EXPECTED_MISSING_TASKS
            and plan["expected_saved_repetitions"] == EXPECTED_SAVED_REPETITIONS
            and plan["expected_missing_repetitions"] == EXPECTED_MISSING_REPETITIONS
            and len(plan["missing"]) == EXPECTED_MISSING_REPETITIONS,
            "Recovery bounds changed")
    require(output_path(out, plan["parent_output"]) == out, "Recovery output path changed")
    expected_arguments = (("original_plan", plan_path), ("raw", raw),
                          ("dependencies", dependencies), ("parent_output", parent_out))
    for name, value in expected_arguments:
        if value is not None:
            actual = plan[name]["path"] if name == "original_plan" else plan[name]
            require(Path(actual).resolve() == Path(value).expanduser().resolve(),
                    "Recovery argument differs from its prepared plan: " + name)
    for name, expected in plan["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen recovery code changed: " + name)
    require(verify_bound(plan["original_plan"]) == Path(plan["original_plan"]["path"])
            and plan["original_plan"]["sha256"] == plan["original_plan_sha256"],
            "Original plan binding changed")
    parent_root = Path(plan["parent_output"])
    require(verify_bound(plan["parent_prepared"]) == (parent_root / "prepared.json").resolve()
            and verify_bound(plan["parent_calibration_summary"])
            == (parent_root / "calibration/summary.json").resolve(),
            "Parent receipt path changed")
    require(set(plan["parent_task_files"]) == {
        f"calibration/tasks/task-{i:04d}.json" for i in range(52)},
        "Parent task binding coverage changed")
    expected_missing = []
    saved = 0
    for index in range(52):
        relative = f"calibration/tasks/task-{index:04d}.json"
        path = verify_bound(plan["parent_task_files"][relative])
        require(path == (parent_root / relative).resolve(), "Parent task path changed")
        report = json.loads(path.read_text())
        count = len(report["records"]); saved += count
        require(report["task"]["task_index"] == index and 0 <= count <= 32,
                "Parent task scope changed")
        for repetition in range(count, 32):
            expected_missing.append(dict(recovery_index=len(expected_missing), task_index=index,
                repetition=repetition, task=report["task"]))
    require(saved == EXPECTED_SAVED_REPETITIONS and plan["missing"] == expected_missing,
            "Prepared missing-repetition map changed")
    p, _, _ = parent.read_plan(Path(plan["original_plan"]["path"]))
    require(plan["budget"] == _budget(p["scheduler"]["cpu_hour_rate_chf"]),
            "Recovery budget changed")
    return plan


def check_result(report, plan, item):
    require(report["schema"] == RESULT_SCHEMA and report["plan_id"] == plan["plan_id"]
            and report["recovery_index"] == item["recovery_index"]
            and report["task_index"] == item["task_index"]
            and report["repetition"] == item["repetition"] and report["task"] == item["task"],
            "Recovery repetition identity changed")
    require(report["complete"] == (report.get("record") is not None),
            "Recovery completion flag changed")
    if report["complete"]:
        require(report["status"] == "complete" and report["record_sha256"] == digest(report["record"]),
                "Recovery record binding changed")
        p, h, _ = parent.read_plan(Path(plan["original_plan"]["path"]))
        parent.check_record(report["record"], p, h, item["task"], item["repetition"])
    else:
        require(report["status"] in ("starting", "incomplete") and report.get("record") is None,
                "Invalid incomplete recovery record")


def recover(out, recovery_index, *, plan_path=None, raw=None, dependencies=None, parent_out=None):
    allocation_started = time.monotonic()
    plan = load_plan(out, plan_path=plan_path, raw=raw, dependencies=dependencies,
                     parent_out=parent_out)
    require(type(recovery_index) is int and 0 <= recovery_index < len(plan["missing"]),
            "Recovery index outside immutable plan")
    item = plan["missing"][recovery_index]
    path = result_path(out, recovery_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with parent.cal.writer_claim(path):
        require(not path.exists(), "Recovery repetition already attempted; never restart it")
        report = dict(schema=RESULT_SCHEMA, plan_id=plan["plan_id"],
            recovery_index=recovery_index, task_index=item["task_index"],
            repetition=item["repetition"], task=item["task"], complete=False,
            status="starting", record=None, record_sha256=None,
            scientific_reference_validated=False)
        save(path, report)
        p, plan_hash, _ = parent.read_plan(Path(plan["original_plan"]["path"]))
        prepared, prepared_hash = parent.read_prepared(
            p, plan_hash, plan["parent_output"], plan["dependencies"])
        require(prepared_hash == plan["prepared_sha256"]
                and parent.tasks(prepared, "calibration")[item["task_index"]] == item["task"],
                "Prepared recovery task changed")
        settings = parent.ARMS[item["task"]["arm"]]
        nodes, weights = parent.leggauss(settings["alpha_order"])
        dependency = Path(plan["dependencies"])
        kernel = parent.RankedLSARowEvidence(64, 1 + nodes / 2, depths=range(9),
            store=dependency / "tables/anchors_prod", src=dependency / "src",
            u_lo=settings["row_bounds"][0], u_hi=settings["row_bounds"][1],
            step=settings["row_step"], term_cache_size=p["term_cache_size"])
        sampler = parent.SharedBaseMCMC(64, kernel, weights / 2)
        began = time.monotonic()
        seconds = min(plan["resources"]["seconds_per_repetition"],
            60 * plan["resources"]["recovery_minutes"]
            - plan["resources"]["worker_margin_seconds"]
            - (began - allocation_started))
        if seconds <= 0:
            report.update(status="incomplete", error="Bounded recovery allocation exhausted",
                          elapsed_seconds=began - allocation_started)
            save(path, report)
            return 75
        def interrupted(sig, frame):
            raise InterruptedError("Signal " + signal.Signals(sig).name)
        handlers = {sig: signal.signal(sig, interrupted)
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
        try:
            with parent.old.heartbeat(path.with_suffix(".progress.json"), dict(
                    stage="calibration-recovery", recovery_index=recovery_index,
                    task=item["task_index"], repetition=item["repetition"])):
                record = parent.estimate(sampler, item["task"]["tokens"],
                    **{key: settings[key] for key in ("particles", "temperatures", "moves")},
                    seed=parent.seed_for(plan_hash, item["task"], item["repetition"]),
                    resample_fraction=.75,
                    seconds_limit=seconds)
            parent.check_record(record, p, plan_hash, item["task"], item["repetition"])
            report.update(complete=True, status="complete", record=record,
                          record_sha256=digest(record), elapsed_seconds=time.monotonic() - began)
            save(path, report)
            print(json.dumps(dict(event="repetition_recovered", recovery_index=recovery_index,
                task=item["task_index"], repetition=item["repetition"])), flush=True)
            return 0
        except (TimeoutError, InterruptedError) as error:
            report.update(status="incomplete", error=str(error),
                          elapsed_seconds=time.monotonic() - began)
            save(path, report)
            return 75
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _incomplete_summary(plan, missing, failed):
    return dict(schema=SUMMARY_SCHEMA, plan_id=plan["plan_id"],
        recovery_plan_sha256=sha(Path(plan["output"]) / "recovery_plan.json"), complete=False,
        passed=False, missing_recovery_indices=missing, failed_recovery_indices=failed,
        recovered_repetitions=EXPECTED_MISSING_REPETITIONS - len(missing) - len(failed),
        overlay=None, population_eligible=False, population_submitted=False,
        automatic_retries=False, scientific_reference_validated=False)


def merge(out, *, plan_path=None, raw=None, dependencies=None, parent_out=None):
    out = Path(out).expanduser().resolve()
    plan = load_plan(out, plan_path=plan_path, raw=raw, dependencies=dependencies,
                     parent_out=parent_out)
    missing, failed, reports = [], [], {}
    for item in plan["missing"]:
        path = result_path(out, item["recovery_index"])
        if not path.exists():
            missing.append(item["recovery_index"])
            continue
        report = json.loads(path.read_text())
        check_result(report, plan, item)
        if not report["complete"]:
            failed.append(item["recovery_index"])
        else:
            reports[item["recovery_index"]] = report
    summary_path = out / "summary.json"
    require(not summary_path.exists(), "Recovery summary already exists; never replace it")
    if missing or failed:
        result = _incomplete_summary(plan, missing, failed)
        save(summary_path, result)
        print(json.dumps(result), flush=True)
        return result

    overlay = out / "overlay"
    require(not overlay.exists(), "Recovery overlay already exists; never replace it")
    (overlay / "calibration/tasks").mkdir(parents=True)
    shutil.copyfile(verify_bound(plan["parent_prepared"]), overlay / "prepared.json")
    shutil.copyfile(verify_bound(plan["original_plan"]), overlay / "plan.json")
    p, plan_hash, _ = parent.read_plan(Path(plan["original_plan"]["path"]))
    prepared, prepared_hash = parent.read_prepared(
        p, plan_hash, plan["parent_output"], plan["dependencies"], full=True)
    require(prepared_hash == plan["prepared_sha256"], "Parent prepared binding changed")
    specs = parent.tasks(prepared, "calibration")
    recovered_by_task = {}
    for item in plan["missing"]:
        recovered_by_task.setdefault(item["task_index"], {})[item["repetition"]] = reports[
            item["recovery_index"]]
    overlay_hashes = {}
    for spec in specs:
        relative = f"calibration/tasks/task-{spec['task_index']:04d}.json"
        source = verify_bound(plan["parent_task_files"][relative])
        report = json.loads(source.read_text())
        parent.check_task(report, p, plan_hash, prepared_hash, spec)
        additions = recovered_by_task.get(spec["task_index"], {})
        require(sorted(additions) == list(range(len(report["records"]), p["repetitions"])),
                "Recovery overlay is not the exact missing suffix")
        completed = deepcopy(report)
        completed["records"] = report["records"] + [additions[j]["record"] for j in sorted(additions)]
        completed.update(records_sha256=digest(completed["records"]), complete=True, status="complete",
            scientific_reference_validated=False, recovery_overlay=dict(
                plan_id=plan["plan_id"], parent_task_sha256=plan["parent_task_files"][relative]["sha256"],
                recovery_result_sha256={str(j): sha(result_path(out, additions[j]["recovery_index"]))
                                        for j in sorted(additions)}))
        completed.pop("error", None)
        destination = overlay / relative
        save(destination, completed)
        parent.check_task(completed, p, plan_hash, prepared_hash, spec)
        overlay_hashes[relative] = sha(destination)
    calibration = parent.merge(Path(plan["original_plan"]["path"]), "calibration",
                               plan["raw"], plan["dependencies"], overlay)
    calibration_path = overlay / "calibration/summary.json"
    result = dict(schema=SUMMARY_SCHEMA, plan_id=plan["plan_id"],
        recovery_plan_sha256=sha(out / "recovery_plan.json"), complete=True,
        passed=bool(calibration["passed"]), missing_recovery_indices=[], failed_recovery_indices=[],
        recovered_repetitions=EXPECTED_MISSING_REPETITIONS,
        recovery_result_sha256={str(i): sha(result_path(out, i))
                                for i in range(EXPECTED_MISSING_REPETITIONS)},
        overlay=dict(path=str(overlay), plan_sha256=sha(overlay / "plan.json"),
            prepared_sha256=sha(overlay / "prepared.json"),
            task_sha256=overlay_hashes, calibration_summary_sha256=sha(calibration_path)),
        population_eligible=bool(calibration["complete"] and calibration["passed"]),
        population_submitted=False, automatic_retries=False,
        scientific_reference_validated=False)
    save(summary_path, result)
    print(json.dumps({key: result[key] for key in ("complete", "passed", "recovered_repetitions",
        "population_eligible", "population_submitted")}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("status", "prepare", "recover", "merge"))
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--dependencies", type=Path, required=True)
    ap.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--recovery-index", type=int)
    args = ap.parse_args()
    if args.command == "status":
        print(json.dumps(inspect(args.plan, args.raw, args.dependencies, args.parent),
                         indent=2, sort_keys=True)); return 0
    if args.command == "prepare":
        prepare(args.plan, args.raw, args.dependencies, args.parent, args.out); return 0
    if args.command == "recover":
        require(args.recovery_index is not None, "Recover needs --recovery-index")
        return recover(args.out, args.recovery_index, plan_path=args.plan, raw=args.raw,
                       dependencies=args.dependencies, parent_out=args.parent)
    require(args.recovery_index is None, "Merge does not take --recovery-index")
    result = merge(args.out, plan_path=args.plan, raw=args.raw,
                   dependencies=args.dependencies, parent_out=args.parent)
    return 0 if result["population_eligible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

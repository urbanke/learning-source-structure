"""Audit the complete t=8/order-32 checkpoint without new sampling.

The checkpoint was atomically saved at four times 262144 draws before its
allocation crossed the 5-GiB process ceiling during post-sampling work.  This
producer binds that failed allocation, verifies every checkpoint array and the
original 16384-draw prefixes, reconstructs the saved report without calling
``extend``, releases recomputable caches, and runs the unchanged raw and
fine-grid audits in a fresh bounded process.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

import copying_payoff_alpha_refinement as alpha
import copying_payoff_alpha_t7_t8_single_worker_recovery as failed


ROOT = alpha.ROOT
SCHEMA = "copying-payoff-alpha-t8-audit-only-salvage-v1"
OVERLAY_SCHEMA = alpha.OVERLAY_SCHEMA
DEFAULT_OUT = ROOT / "output/copying_payoff_alpha_t8_audit_only_salvage_2026-09-21"
SOURCE = ROOT / "output/copying_payoff_alpha_t7_t8_single_worker_recovery_2026-09-21"
BASELINE_DRAWS = alpha.BASELINE_DRAWS
TARGET_DRAWS = alpha.TARGET_DRAWS
PARENT_FILES = alpha.PARENT_FILES
TARGETS = (alpha.TARGETS[7],)
TASK_SECONDS = 900
RESERVED_BYTES = 5 * 2**30
EXPECTED_SOURCE_SHA256 = {
    "plan": "a996aed75f8da0122fc4d933d67089327385bdc475414c496bc8cb6a0a0fdaff",
    "allocation": "b8ad0fbf3f629d81609da3846badfa0dc8816c173557cf2f924ac8df75b1e060",
    "attempt": "0b3ca3472a8e64377f2c7b44a9778f89948c7c94fc9f816732c68827e4a236c2",
    "identity": "658e06fcd9d7d9be1e435744c71c6b689a9753789bf3b119bbde4d76a1906d0b",
    "prepared": "fcb914834acd7ed3840a8c38060b2c58b844a19791c83adda083f8b2581e2d66",
    "checkpoint": "18dcb5e87e1774548e1245a552d1927e75b5e04dccbccff2b419674c13b18e69",
    "failed_wrapper": "9b62c51a2663dc67ce81cc8d3ed219fb1fcc1cc3c3c14189a22e61c4dcc94787",
    "frozen_engine": "08c9980d3fbe482cd2b367778949038e61d9da6cf493a5eeae81dca5c008636a",
}
require, read, write, sha = alpha.require, alpha.read, alpha.write, alpha.sha


def bound(path):
    return alpha.bound(path)


def output_path(value):
    path = Path(value).absolute()
    require(path.parent.resolve() == (ROOT / "output").resolve()
            and path.resolve() != SOURCE.resolve() and not path.is_symlink(),
            "Audit-only output must be a separate direct child of project output")
    return path


def paths(out, task):
    return alpha.paths(Path(out), task)


def source_record():
    plan = failed.load_plan(SOURCE)
    spec = plan["tasks"][3]
    task = spec["task"]
    require(task["id"] == TARGETS[0][2] and task["order"] == 32 and task["time"] == 8,
            "Failed allocation task identity changed")
    p = failed.paths(SOURCE, task)
    allocation, attempt = read(SOURCE / "allocation.json"), read(p["attempt"])
    require(allocation["status"] == "failed" and "memory_limit" in allocation["reason"]
            and attempt["status"] == "memory_limit" and attempt["exit_code"] == 76
            and attempt["task"] == task,
            "Source did not stop at the declared post-sampling memory limit")
    files = dict(plan=bound(SOURCE / "plan.json"), allocation=bound(SOURCE / "allocation.json"),
        attempt=bound(p["attempt"]), identity=bound(p["identity"]),
        prepared=bound(p["prepared"]), checkpoint=bound(p["checkpoint"]),
        failed_wrapper=bound(ROOT / "code/scripts/copying_payoff_alpha_t7_t8_single_worker_recovery.py"),
        frozen_engine=bound(ROOT / "code/scripts/copying_payoff_alpha_refinement.py"))
    require({name: item["sha256"] for name, item in files.items()} == EXPECTED_SOURCE_SHA256,
            "Bound audit-only source or frozen code changed")
    require(not any(p[name].exists() for name in ("moments", "audit", "report", "receipt")),
            "Failed source unexpectedly contains completed audit artifacts")
    parent = failed.current_parent(spec)
    prefix = alpha.sample_prefix_proof(parent["paths"]["checkpoint"], p["checkpoint"], complete=True)
    from unseen_context_refinement import checkpoint_counts, checkpoint_metadata
    metadata = checkpoint_metadata(p["checkpoint"])
    require(checkpoint_counts(p["checkpoint"]) == [TARGET_DRAWS] * 4
            and metadata["config"] == dict(chains=4, burn=16384, thin=1, seed=task["seed"]),
            "Saved audit-only checkpoint is not the complete declared chain family")
    return dict(plan=plan, spec=spec, task=task, paths=p, allocation=allocation,
                attempt=attempt, files=files, parent=parent, prefix=prefix,
                metadata_sha256=sha(p["checkpoint"]))


def code_hashes():
    names = ("copying_payoff_alpha_t8_audit_only_salvage.py",
             "copying_payoff_alpha_t7_t8_single_worker_recovery.py",
             "copying_payoff_alpha_refinement.py",
             "copying_payoff_accuracy.py", "copying_payoff_dense.py",
             "copying_payoff_parallel.py")
    return {f"code/scripts/{name}": sha(ROOT / "code/scripts" / name) for name in names}


def build_plan(out):
    out = output_path(out)
    source = source_record()
    plan = dict(schema=SCHEMA, output=str(out.resolve()), tasks=[source["spec"]],
        source_files=source["files"], source_plan_id=source["plan"]["plan_id"],
        baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
        audit_only=True, sampling_performed=False, extend_forbidden=True,
        source_prefix=source["prefix"], source_checkpoint_sha256=source["metadata_sha256"],
        resources=dict(max_workers=1, task_seconds=TASK_SECONDS,
                       reserved_memory_bytes=RESERVED_BYTES, maximum_attempts=1),
        original_acceptance_criteria_unchanged=True, automatic_retries=False,
        code_sha256=code_hashes(), runtime=source["plan"]["runtime"],
        python=source["plan"]["python"], environment=source["plan"]["environment"],
        dependency_sha256=source["plan"]["dependency_sha256"], prior=source["plan"]["prior"])
    plan["plan_id"] = alpha.fingerprint(plan)
    return plan


def prepare(out):
    out = output_path(out)
    require(not out.exists(), "Use a new immutable audit-only output")
    plan = build_plan(out)
    out.mkdir()
    write(out / "plan.json", plan, immutable=True)
    return plan


def load_plan(out):
    out = output_path(out)
    saved = read(out / "plan.json")
    require(saved["schema"] == SCHEMA
            and saved["plan_id"] == alpha.fingerprint({k: v for k, v in saved.items() if k != "plan_id"})
            and saved == build_plan(out),
            "Audit-only plan, source, code, or limits changed")
    return saved


def release_caches(sampler):
    kernel = sampler.kernel
    before = kernel.cache_info()
    for name in ("_terms", "_evidence", "_predictive", "_baselines"):
        getattr(kernel, name).clear()
    gc.collect()
    after = kernel.cache_info()
    print(json.dumps(dict(event="audit_only_cache_release", before=before, after=after)), flush=True)


def worker(out, index):
    out = output_path(out); plan = load_plan(out)
    require(index == 0, "Audit-only producer has exactly one task")
    spec, task = plan["tasks"][0], plan["tasks"][0]["task"]
    p = paths(out, task); attempt = read(p["attempt"])
    require(attempt["status"] == "running" and attempt["plan_id"] == plan["plan_id"]
            and attempt["task"] == task and attempt["dispatcher_pid"] == os.getppid(),
            "Audit-only worker requires its live bounded supervisor")
    os.environ.update(alpha.dense.pilot.environment(plan))
    require(alpha.dense.pilot.runtime() == plan["runtime"], "Audit-only runtime differs")
    source = source_record(); began = time.monotonic()
    with alpha.lock(p["attempt"].with_suffix(".worker.lock")):
        require(not any(p[name].exists() for name in
            ("checkpoint", "identity", "moments", "audit", "report", "receipt", "prepared")),
            "Audit-only task cannot restart or overwrite")
        p["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["paths"]["checkpoint"], p["checkpoint"])
        require(sha(p["checkpoint"]) == EXPECTED_SOURCE_SHA256["checkpoint"],
                "Copied complete checkpoint changed")
        from ranked_lsa import RankedLSARowEvidence
        from shared_base_surrogate import SharedBaseSurrogateMCMC
        from shared_base_curve_runner import CurveChainRun
        from eval_shared_base_reference import selected_grid_checks
        from eval_shared_base_models import atomic_npz
        from unseen_context_refinement import checkpoint_metadata
        from validate_shared_base_mcmc import json_safe
        nodes, weights = np.polynomial.legendre.leggauss(task["order"])
        sampler = SharedBaseSurrogateMCMC(64,
            RankedLSARowEvidence(64, 1 + nodes / 2, term_cache_size=16384),
            weights / 2, surrogate_steps=64)
        tokens = np.asarray(task["tokens"], dtype=int)
        metadata = checkpoint_metadata(p["checkpoint"])
        run = CurveChainRun.load(sampler, tokens, p["checkpoint"])
        loaded = alpha.rng_proof(run, metadata)
        require(run.draw_counts.tolist() == [TARGET_DRAWS] * 4 and run.seed == task["seed"]
                and run.fingerprint == source["parent"]["identity"]["target"]
                and run.proposal_descriptor == source["parent"]["identity"]["proposal"],
                "Complete checkpoint target, proposal, seed, or draws changed")
        estimate = run.report()  # Deliberately no extend call.
        prefix = alpha.sample_prefix_proof(source["parent"]["paths"]["checkpoint"],
                                           p["checkpoint"], complete=True)
        source_prepared = read(source["paths"]["prepared"])
        prepared = dict(schema="copying-payoff-alpha-t8-audit-only-prepared-v1",
            plan_id=plan["plan_id"], task=task, source_files=plan["source_files"],
            sample_prefix=prefix, source_initial_rng_continuity=source_prepared["rng_continuity"],
            loaded_checkpoint_continuity=loaded, sampling_performed=False,
            audit_only_recovery=True)
        write(p["prepared"], prepared, immutable=True)
        identity = dict(plan_id=plan["plan_id"], source_plan_id=plan["source_plan_id"],
            task=task, target=run.fingerprint, proposal=run.proposal_descriptor,
            runtime=alpha.dense.pilot.runtime())
        write(p["identity"], identity, immutable=True)
        del run
        release_caches(sampler)
        moments, audit, fresh = alpha.dense.accuracy._audit_raw(
            task, identity, estimate, p["checkpoint"])
        atomic_npz(p["moments"], **moments)
        release_caches(sampler)
        grids = selected_grid_checks(sampler, tokens, estimate)
        audit["final_state_grid_checks"] = grids
        passed = bool(audit["passed"] and all(item["passed"] for item in grids))
        write(p["audit"], json_safe(audit), immutable=True)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin": peak *= 1024
        report = json_safe(dict(schema=SCHEMA, plan_id=plan["plan_id"],
            source_plan_id=plan["source_plan_id"], task=task, complete=True,
            audit_passed=passed, estimate=fresh, baseline_draws_per_chain=BASELINE_DRAWS,
            target_draws_per_chain=TARGET_DRAWS, parent_samples_preserved=True,
            exact_rng_restored_from_complete_checkpoint=True,
            sampling_performed=False, audit_only_recovery=True,
            elapsed_seconds=time.monotonic() - began, peak_rss_bytes=peak,
            **{name + "_sha256": sha(p[name]) for name in
               ("audit", "identity", "checkpoint", "moments")}))
        write(p["report"], report, immutable=True)
        receipt = dict(schema="copying-payoff-alpha-t8-audit-only-receipt-v1",
            plan_id=plan["plan_id"], source_plan_id=plan["source_plan_id"], task=task,
            status="ready" if passed else "unresolved", audit_passed=passed,
            sampling_complete=True, sampling_performed=False, audit_only_recovery=True,
            baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
            source_files=plan["source_files"], parent_plan=spec["parent_plan"],
            parent_files=spec["parent_files"],
            replacement_files={name: bound(p[name]) for name in PARENT_FILES},
            prepared_sha256=sha(p["prepared"]), sample_prefix=prefix,
            loaded_checkpoint_continuity=loaded,
            original_acceptance_criteria_unchanged=True, automatic_retry=False)
        write(p["receipt"], receipt, immutable=True)
    return 0 if passed else 2


def run(out):
    out = output_path(out); plan = load_plan(out); task = plan["tasks"][0]["task"]
    p = paths(out, task); allocation_path = out / "allocation.json"
    require(not allocation_path.exists() and not p["attempt"].exists(),
            "One audit-only attempt only")
    allocation = dict(schema="copying-payoff-alpha-t8-audit-only-allocation-v1",
        plan_id=plan["plan_id"], status="running", dispatcher_pid=os.getpid(),
        started_unix=time.time(), reserved_seconds=TASK_SECONDS,
        reserved_memory_bytes=RESERVED_BYTES, maximum_attempts=1)
    write(allocation_path, allocation, immutable=True)
    attempt = dict(plan_id=plan["plan_id"], task=task, status="running",
        dispatcher_pid=os.getpid(), reserved_seconds=TASK_SECONDS,
        reserved_memory_bytes=RESERVED_BYTES, started_unix=time.time(),
        observed_peak_rss_bytes=0)
    write(p["attempt"], attempt, immutable=True)
    stream = p["log"].open("x"); began = time.monotonic()
    process = subprocess.Popen([plan["python"], str(Path(__file__).resolve()), "worker",
        "--out", str(out), "--index", "0"], cwd=ROOT,
        env=alpha.dense.pilot.environment(plan), stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
    forced = None
    try:
        while process.poll() is None:
            current = alpha.rss_bytes(process.pid)
            attempt["observed_peak_rss_bytes"] = max(attempt["observed_peak_rss_bytes"], current)
            if current > RESERVED_BYTES:
                alpha.stop(process); forced = "memory_limit"; break
            if time.monotonic() - began >= TASK_SECONDS:
                alpha.stop(process); forced = "time_limit"; break
            time.sleep(.5)
        code = process.returncode if forced is None else (76 if forced == "memory_limit" else 75)
        status = forced or ("ready" if code == 0 else "unresolved" if code == 2 else "failed")
        attempt.update(status=status, exit_code=code, elapsed_seconds=time.monotonic() - began,
                       finished_unix=time.time())
        write(p["attempt"], attempt)
        allocation.update(status=status, attempt=attempt, finished_unix=time.time(),
                          elapsed_seconds=time.monotonic() - began)
        write(allocation_path, allocation)
        if forced or status == "failed":
            raise RuntimeError("Audit-only salvage stopped: " + status)
        if status == "ready": overlay(out, write_manifest=True)
        return 0 if status == "ready" else 2
    finally:
        stream.close()


def validate_replacement(plan, spec, *, require_pass):
    require(plan == load_plan(plan["output"]) and spec == plan["tasks"][0],
            "Audit-only replacement plan or task changed")
    source = source_record(); p = paths(plan["output"], spec["task"])
    report, identity, audit, receipt, prepared = (read(p[name]) for name in
        ("report", "identity", "audit", "receipt", "prepared"))
    require(report["schema"] == SCHEMA and report["plan_id"] == plan["plan_id"]
            and report["task"] == identity["task"] == receipt["task"] == spec["task"]
            and report["complete"] is True and report["sampling_performed"] is False
            and report["audit_only_recovery"] is True,
            "Audit-only report identity or mode changed")
    for name in ("audit", "identity", "checkpoint", "moments"):
        require(report[name + "_sha256"] == sha(p[name]),
                "Audit-only report binding changed: " + name)
    require(sha(p["checkpoint"]) == EXPECTED_SOURCE_SHA256["checkpoint"]
            and report["estimate"]["diagnostics"]["draws"] == TARGET_DRAWS,
            "Audit-only checkpoint or draw count changed")
    passed = bool(audit["passed"] and all(item["passed"] for item in audit["final_state_grid_checks"]))
    prefix = alpha.sample_prefix_proof(source["parent"]["paths"]["checkpoint"],
                                       p["checkpoint"], complete=True)
    require(report["audit_passed"] == receipt["audit_passed"] == passed
            and prepared["sample_prefix"] == receipt["sample_prefix"] == prefix
            and receipt["sampling_performed"] is False
            and receipt["source_files"] == plan["source_files"],
            "Audit-only receipt, proof, or acceptance changed")
    if require_pass: require(passed, "Audit-only replacement failed unchanged audits")
    return dict(parent=source["parent"], paths=p, report=report, audit=audit,
                receipt=receipt, prefix=prefix,
                rng=prepared["source_initial_rng_continuity"], passed=passed)


def overlay(out, *, write_manifest):
    out = output_path(out); plan = load_plan(out); spec = plan["tasks"][0]
    checked = validate_replacement(plan, spec, require_pass=True); p = checked["paths"]
    item = dict(time=spec["task"]["time"], order=spec["task"]["order"],
        task_id=spec["task"]["id"], task=spec["task"], producer_schema=SCHEMA,
        producer_plan=bound(out / "plan.json"), parent_plan=spec["parent_plan"],
        baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
        parent_files=spec["parent_files"],
        replacement_files={name: bound(p[name]) for name in (*PARENT_FILES, "receipt")},
        sample_prefix=checked["prefix"], rng_continuity=checked["rng"],
        audit_passed=True, sampling_performed=False, audit_only_recovery=True)
    result = dict(schema=OVERLAY_SCHEMA, complete=True, baseline_draws=BASELINE_DRAWS,
        target_draws=TARGET_DRAWS, original_acceptance_criteria_unchanged=True,
        recovery_plan=bound(out / "plan.json"), replacements=[item])
    if write_manifest: write(out / "overlay.json", result, immutable=True)
    else: require(read(out / "overlay.json") == result, "Audit-only overlay changed")
    return result


def check_plan(out):
    plan = load_plan(out)
    return dict(schema=SCHEMA, plan_id=plan["plan_id"], task_id=plan["tasks"][0]["task"]["id"],
                sampling_performed=False, target_draws=TARGET_DRAWS,
                reserved_memory_bytes=RESERVED_BYTES,
                sampling_started=(Path(out) / "allocation.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check-plan", "run", "worker", "check-overlay"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare": result, code = prepare(args.out), 0
    elif args.action == "check-plan": result, code = check_plan(args.out), 0
    elif args.action == "run": raise SystemExit(run(args.out))
    elif args.action == "worker": raise SystemExit(worker(args.out, args.index))
    else: result, code = overlay(args.out, write_manifest=False), 0
    print(json.dumps(result, indent=2)); raise SystemExit(code)


if __name__ == "__main__":
    main()

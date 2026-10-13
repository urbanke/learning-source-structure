"""Immutable two-reference recovery for the copying-payoff campaign.

Only the two declared t7/t8 numerical failures may be continued.  The
original campaign stays read-only; each checkpoint is copied to a separate
output, resumed from its saved RNG and terminal states, and extended to the
fixed total of 65,536 draws per chain.  One worker runs at a time, once, with
an independent 900-second and 6-GiB RSS bound for each task.

Passing individual audits produce an immutable overlay manifest.  This
script does not merge a cohort, assess aggregate precision, or approve a
payoff estimate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

import copying_payoff_dense as dense
import copying_payoff_parallel as parallel


ROOT = dense.ROOT
PARENT = ROOT / "output/copying_payoff_parallel_2026-09-20"
DEFAULT_OUT = ROOT / "output/copying_payoff_recovery_procinfo_2026-09-21"
SCHEMA = "copying-payoff-recovery-v1"
OVERLAY_SCHEMA = "copying-payoff-recovery-overlay-v1"
BASELINE_DRAWS = 16384
TARGET_DRAWS = 65536
TASK_WALL_SECONDS = 900
MAX_RSS_BYTES = 6 * 2**30
MINIMUM_FREE_DISK_BYTES = 5 * 2**30
TARGETS = (
    (7, "t0007_453a8f0bdcd20ca9731b_a32", dict(
        split_rhat_below_1_05=True,
        probability_ess_at_least_200=False,
        maximum_probability_mcse_at_most_0_005=True,
        raw_normalization=True)),
    (8, "t0008_fccb61c7bef875d43ac7_a16", dict(
        split_rhat_below_1_05=True,
        probability_ess_at_least_200=False,
        maximum_probability_mcse_at_most_0_005=False,
        raw_normalization=True)),
)
require, read, write = dense.require, dense.read, dense.write
sha, fingerprint, lock = dense.sha, dense.fingerprint, dense.lock


def bound(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Frozen file missing or linked: " + str(path))
    return dict(path=str(path.resolve()), sha256=sha(path), size_bytes=path.stat().st_size)


def verify_bound(item):
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == item["size_bytes"]
            and sha(path) == item["sha256"], "Frozen file changed: " + str(path))


def output_path(out):
    out = Path(out).absolute()
    require(out.parent.resolve() == (ROOT / "output").resolve()
            and out.resolve() != PARENT.resolve() and not out.is_symlink(),
            "Recovery output must be a separate direct child of project output")
    return out


def paths(out, task):
    result = dense.paths(out, task)
    folder = result["checkpoint"].parent
    result.update(prepared=folder / "prepared.json", receipt=folder / "receipt.json",
                  claim=folder / "worker.claim")
    return result


def code_hashes():
    names = ("copying_payoff_recovery.py", "copying_payoff_accuracy.py",
             "unseen_context_refinement.py")
    return {f"code/scripts/{name}": sha(ROOT / "code/scripts" / name) for name in names}


def validate_parent(plan, task, expected_gates):
    p = dense.paths(Path(plan["output"]), task)
    report, identity, audit, attempt = (read(p[name]) for name in
                                        ("report", "identity", "audit", "attempt"))
    require(task["id"] in {item[1] for item in TARGETS}
            and task["time"] in (7, 8) and task["target_draws"] == BASELINE_DRAWS,
            "Parent is not one of the two declared baseline tasks")
    require(report["schema"] == dense.SCHEMA and report["plan_id"] == plan["plan_id"]
            and report["task"] == identity["task"] == task and report["complete"] is True
            and report["audit_passed"] is False, "Parent report is not the declared completed failure")
    require(attempt["plan_id"] == plan["plan_id"] and attempt["task"] == task
            and attempt["status"] == "unresolved" and attempt["exit_code"] == 0,
            "Parent allocation is not the declared successful unresolved attempt")
    require(audit["passed"] is False and audit["classic_gates"] == expected_gates
            and audit["corrected_gates"] == expected_gates
            and all(row["passed"] for row in audit["final_state_grid_checks"]),
            "Parent failure gates differ from the declared numerical failures")
    diagnostics = report["estimate"]["diagnostics"]
    require({name: diagnostics[name] for name in ("chains", "burn", "draws", "thin", "seed")}
            == dict(chains=4, burn=16384, draws=BASELINE_DRAWS, thin=1, seed=task["seed"]),
            "Parent chain configuration or draw count changed")
    for name in ("report", "identity", "audit", "checkpoint", "moments"):
        if name != "report":
            require(report[name + "_sha256"] == sha(p[name]), "Parent report binding changed: " + name)
    require(audit["raw_inputs"]["canonical_accuracy:checkpoint"]["sha256"] == sha(p["checkpoint"]),
            "Parent audit checkpoint binding changed")
    return dict(plan=plan, task=task, report=report, identity=identity, audit=audit,
                paths=p, files={name: bound(p[name]) for name in
                                ("report", "identity", "audit", "checkpoint", "moments")})


def parent_specs():
    campaign = parallel.load_campaign(PARENT)
    allocation = read(PARENT / "allocation.json")
    require(allocation["campaign_id"] == campaign["campaign_id"]
            and allocation["status"] == "complete"
            and allocation["unresolved_task_ids"] == [item[1] for item in TARGETS],
            "Parent campaign completion or unresolved scope changed")
    specs = []
    for observed, task_id, gates in TARGETS:
        plan = dense.load_plan(PARENT / f"t{observed}")
        task = next((item for item in plan["tasks"] if item["id"] == task_id), None)
        require(task is not None and task["time"] == observed, "Declared parent task is missing")
        parent = validate_parent(plan, task, gates)
        specs.append(dict(time=observed, task=task, expected_parent_gates=gates,
                          parent_plan=bound(PARENT / f"t{observed}" / "plan.json"),
                          parent_files=parent["files"], parent_identity=parent["identity"]))
    return campaign, specs


def build_plan(out):
    out = output_path(out)
    campaign, specs = parent_specs()
    plan = dict(schema=SCHEMA, output=str(out.resolve()), parent_output=str(PARENT.resolve()),
        parent_campaign=bound(PARENT / "campaign.json"), campaign_id=campaign["campaign_id"],
        tasks=specs, baseline_draws_per_chain=BASELINE_DRAWS,
        target_draws_per_chain=TARGET_DRAWS,
        sampler=dict(chains=4, burn=16384, thin=1, surrogate_steps=64),
        resources=dict(max_workers=1, task_wall_seconds=TASK_WALL_SECONDS,
            maximum_attempts_per_task=1, max_rss_bytes=MAX_RSS_BYTES,
            minimum_free_disk_bytes=MINIMUM_FREE_DISK_BYTES),
        python=dense.load_plan(PARENT / "t7")["python"],
        runtime=dense.load_plan(PARENT / "t7")["runtime"],
        environment=dense.load_plan(PARENT / "t7")["environment"],
        code_sha256=code_hashes(), original_acceptance_criteria_unchanged=True,
        automatic_retries=False, population_reference_approved=False,
        payoff_forecast_approved=False)
    other = dense.load_plan(PARENT / "t8")
    for name in ("python", "runtime", "environment", "prior", "dependency_sha256"):
        left = plan[name] if name in plan else dense.load_plan(PARENT / "t7")[name]
        require(left == other[name], "Parent waves use different " + name)
    plan["prior"] = other["prior"]
    plan["dependency_sha256"] = other["dependency_sha256"]
    plan["plan_id"] = fingerprint(plan)
    return plan


def load_plan(out):
    out = output_path(out)
    saved = read(out / "plan.json")
    require(saved["schema"] == SCHEMA and saved["output"] == str(out.resolve())
            and saved["plan_id"] == fingerprint({k: v for k, v in saved.items() if k != "plan_id"}),
            "Recovery plan identity changed")
    require(saved == build_plan(out), "Recovery plan, parent inputs, code, or limits changed")
    for spec in saved["tasks"]:
        verify_bound(spec["parent_plan"])
        for item in spec["parent_files"].values():
            verify_bound(item)
    return saved


def checkpoint_helpers():
    from unseen_context_refinement import (checkpoint_counts, checkpoint_metadata,
                                           prefix_hash, verify_preserved_samples)
    return checkpoint_counts, checkpoint_metadata, prefix_hash, verify_preserved_samples


def sample_prefix_proof(parent_path, child_path, *, complete):
    counts, metadata, prefix_hash, verify = checkpoint_helpers()
    spec = dict(baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS)
    child_counts = verify(parent_path, child_path, spec, complete=complete)
    parent_meta = metadata(parent_path)
    names = [f"{kind}_{chain}" for chain in range(4)
             for kind in ("predictions", "depths", "alpha_indices", "scalars")]
    hashes = {name: prefix_hash(child_path, name, BASELINE_DRAWS) for name in names}
    require(all(hashes[name] == parent_meta["array_hashes"][name] for name in names),
            "Child retained-sample prefix differs from parent")
    return dict(verified=True, parent_draw_counts=counts(parent_path),
        child_draw_counts=child_counts, array_prefix_sha256=hashes,
        burn_done_sha256=parent_meta["array_hashes"]["burn_done"])


def rng_proof(run, parent_metadata):
    dense.accuracy.verify_loaded_parent(run, parent_metadata)
    loaded_rng = [rng.bit_generator.state for rng in run._rngs]
    loaded_states = [dict(alpha_index=state.alpha_index, ranks=state.ranks.tolist(),
        log_likelihood=state.log_likelihood, log_target=state.log_target) for state in run._states]
    result = dict(passed=True,
        parent_rng_states_sha256=fingerprint(parent_metadata["rng_states"]),
        loaded_rng_states_sha256=fingerprint(loaded_rng),
        parent_terminal_states_sha256=fingerprint(parent_metadata["states"]),
        loaded_terminal_states_sha256=fingerprint(loaded_states))
    require(result["parent_rng_states_sha256"] == result["loaded_rng_states_sha256"]
            and result["parent_terminal_states_sha256"] == result["loaded_terminal_states_sha256"],
            "Loaded RNG or terminal state differs from parent checkpoint")
    return result


def current_parent(plan, spec):
    verify_bound(spec["parent_plan"])
    for item in spec["parent_files"].values():
        verify_bound(item)
    parent_plan = dense.load_plan(Path(spec["parent_plan"]["path"]).parent)
    task = next(item for item in parent_plan["tasks"] if item["id"] == spec["task"]["id"])
    expected = next(item[2] for item in TARGETS if item[1] == task["id"])
    parent = validate_parent(parent_plan, task, expected)
    require(task == spec["task"] and parent["identity"] == spec["parent_identity"]
            and parent["files"] == spec["parent_files"], "Frozen parent specification changed")
    return parent


def worker(out, index):
    out = output_path(out)
    plan = load_plan(out)
    require(type(index) is int and 0 <= index < len(plan["tasks"]), "Invalid recovery task index")
    spec = plan["tasks"][index]
    task = spec["task"]
    p = paths(out, task)
    attempt = read(p["attempt"])
    require(attempt["plan_id"] == plan["plan_id"] and attempt["task"] == task
            and attempt["status"] == "running" and attempt["dispatcher_pid"] == os.getppid()
            and attempt["reserved_seconds"] == TASK_WALL_SECONDS
            and attempt["reserved_memory_bytes"] == MAX_RSS_BYTES,
            "Worker requires its live bounded supervisor allocation")
    p["claim"].parent.mkdir(parents=True, exist_ok=True)
    with p["claim"].open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    os.environ.update(dense.pilot.environment(plan))
    require(dense.pilot.runtime() == plan["runtime"], "Recovery numerical runtime differs")
    parent = current_parent(plan, spec)
    began = time.monotonic()
    with lock(ROOT / "output/.copying-payoff-recovery-worker.lock"):
        require(not any(p[name].exists() for name in
            ("checkpoint", "identity", "moments", "audit", "report", "receipt", "prepared")),
            "Recovery never overwrites or restarts a task")
        shutil.copyfile(parent["paths"]["checkpoint"], p["checkpoint"])
        require(sha(p["checkpoint"]) == spec["parent_files"]["checkpoint"]["sha256"],
                "Copied parent checkpoint differs")
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
        metadata = checkpoint_metadata(parent["paths"]["checkpoint"])
        run = CurveChainRun.load(sampler, tokens, p["checkpoint"])
        restored = rng_proof(run, metadata)
        require(run.draw_counts.tolist() == [BASELINE_DRAWS] * 4 and run.seed == task["seed"],
                "Loaded parent draw count or seed changed")
        require(run.fingerprint == parent["identity"]["target"]
                and run.proposal_descriptor == parent["identity"]["proposal"],
                "Loaded target or Markov kernel differs from parent")
        initial_prefix = sample_prefix_proof(parent["paths"]["checkpoint"], p["checkpoint"], complete=False)
        write(p["prepared"], dict(schema="copying-payoff-recovery-prepared-v1",
            plan_id=plan["plan_id"], task=task, parent_plan=spec["parent_plan"],
            parent_files=spec["parent_files"], sample_prefix=initial_prefix,
            rng_continuity=restored), immutable=True)
        identity = dict(plan_id=plan["plan_id"], parent_plan_id=parent["plan"]["plan_id"],
            task=task, target=run.fingerprint, proposal=run.proposal_descriptor,
            runtime=dense.pilot.runtime())
        require(identity["target"] == parent["identity"]["target"]
                and identity["proposal"] == parent["identity"]["proposal"]
                and identity["runtime"] == parent["identity"]["runtime"],
                "Continuation target, proposal, or runtime changed")
        write(p["identity"], identity, immutable=True)
        saved = time.monotonic()

        def progress(event):
            nonlocal saved
            now = time.monotonic()
            if event["draws"] == event["requested_draws"] or now - saved >= 120:
                run.save(p["checkpoint"])
                saved = now
            print(json.dumps(dict(event="copying_recovery_progress", task=task["id"], **event)), flush=True)

        estimate = run.extend(TARGET_DRAWS, block_draws=2048, progress=progress)
        run.save(p["checkpoint"])
        del run
        prefix = sample_prefix_proof(parent["paths"]["checkpoint"], p["checkpoint"], complete=True)
        moments, audit, fresh = dense.accuracy._audit_raw(task, identity, estimate, p["checkpoint"])
        atomic_npz(p["moments"], **moments)
        grids = selected_grid_checks(sampler, tokens, estimate)
        audit["final_state_grid_checks"] = grids
        passed = bool(audit["passed"] and all(item["passed"] for item in grids))
        write(p["audit"], json_safe(audit), immutable=True)
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_rss *= 1024
        report = json_safe(dict(schema=SCHEMA, plan_id=plan["plan_id"],
            parent_plan_id=parent["plan"]["plan_id"], task=task, complete=True,
            audit_passed=passed, estimate=fresh,
            baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
            parent_samples_preserved=True, exact_rng_restored_before_extension=True,
            elapsed_seconds=time.monotonic() - began, peak_rss_bytes=peak_rss,
            sampling_performed=True, automatic_retry=False,
            population_reference_approved=False, payoff_forecast_approved=False,
            **{name + "_sha256": sha(p[name]) for name in
               ("audit", "identity", "checkpoint", "moments")}))
        write(p["report"], report, immutable=True)
        replacement = {name: bound(p[name]) for name in
                       ("report", "identity", "audit", "checkpoint", "moments")}
        receipt = dict(schema="copying-payoff-recovery-receipt-v1", plan_id=plan["plan_id"],
            parent_plan_id=parent["plan"]["plan_id"], task=task,
            baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
            sampling_complete=True, audit_passed=passed,
            status="ready" if passed else "unresolved", parent_plan=spec["parent_plan"],
            parent_files=spec["parent_files"], replacement_files=replacement,
            prepared_sha256=sha(p["prepared"]), sample_prefix=prefix,
            rng_continuity=restored, original_acceptance_criteria_unchanged=True,
            automatic_retry=False)
        write(p["receipt"], receipt, immutable=True)
        current_parent(plan, spec)
    print(json.dumps(dict(event="copying_recovery_finished", task=task["id"],
        status=receipt["status"], draws=TARGET_DRAWS)), flush=True)
    return 0 if passed else 2


def expected_rng_proof(parent_checkpoint):
    _, metadata, _, _ = checkpoint_helpers()
    value = metadata(parent_checkpoint)
    return dict(passed=True,
        parent_rng_states_sha256=fingerprint(value["rng_states"]),
        loaded_rng_states_sha256=fingerprint(value["rng_states"]),
        parent_terminal_states_sha256=fingerprint(value["states"]),
        loaded_terminal_states_sha256=fingerprint(value["states"]))


def validate_replacement(plan, spec, *, require_pass):
    parent = current_parent(plan, spec)
    p = paths(plan["output"], spec["task"])
    report, identity, audit, receipt = (read(p[name]) for name in
                                        ("report", "identity", "audit", "receipt"))
    require(report["schema"] == SCHEMA and report["plan_id"] == plan["plan_id"]
            and report["parent_plan_id"] == parent["plan"]["plan_id"]
            and report["task"] == identity["task"] == receipt["task"] == spec["task"]
            and report["complete"] is True, "Replacement identity or completion changed")
    require(identity["target"] == parent["identity"]["target"]
            and identity["proposal"] == parent["identity"]["proposal"]
            and identity["runtime"] == parent["identity"]["runtime"],
            "Replacement target, proposal, or runtime changed")
    for name in ("audit", "identity", "checkpoint", "moments"):
        require(report[name + "_sha256"] == sha(p[name]), "Replacement report binding changed: " + name)
    require(report["estimate"]["diagnostics"]["draws"] == TARGET_DRAWS
            and report["estimate"]["diagnostics"]["seed"] == spec["task"]["seed"],
            "Replacement draw count or seed changed")
    passed = bool(audit["passed"] and all(item["passed"] for item in audit["final_state_grid_checks"]))
    require(report["audit_passed"] == receipt["audit_passed"] == passed,
            "Replacement audit acceptance changed")
    prefix = sample_prefix_proof(parent["paths"]["checkpoint"], p["checkpoint"], complete=True)
    restored = expected_rng_proof(parent["paths"]["checkpoint"])
    replacement = {name: bound(p[name]) for name in
                   ("report", "identity", "audit", "checkpoint", "moments")}
    expected_receipt = dict(schema="copying-payoff-recovery-receipt-v1", plan_id=plan["plan_id"],
        parent_plan_id=parent["plan"]["plan_id"], task=spec["task"],
        baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
        sampling_complete=True, audit_passed=passed, status="ready" if passed else "unresolved",
        parent_plan=spec["parent_plan"], parent_files=spec["parent_files"],
        replacement_files=replacement, prepared_sha256=sha(p["prepared"]),
        sample_prefix=prefix, rng_continuity=restored,
        original_acceptance_criteria_unchanged=True, automatic_retry=False)
    require(receipt == expected_receipt, "Replacement receipt or continuation proof changed")
    if require_pass:
        require(passed, "Both replacements must pass the unchanged individual audits")
    return dict(parent=parent, paths=p, report=report, audit=audit,
                receipt=receipt, prefix=prefix, rng=restored, passed=passed)


def overlay(out, *, write_manifest):
    out = output_path(out)
    plan = load_plan(out)
    replacements = []
    for spec in plan["tasks"]:
        checked = validate_replacement(plan, spec, require_pass=True)
        p = checked["paths"]
        replacements.append(dict(time=spec["task"]["time"], order=spec["task"]["order"],
            task_id=spec["task"]["id"], task=spec["task"], parent_plan=spec["parent_plan"],
            baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
            parent_files=spec["parent_files"], replacement_files={name: bound(p[name]) for name in
                ("report", "identity", "audit", "checkpoint", "moments", "receipt")},
            sample_prefix=checked["prefix"], rng_continuity=checked["rng"], audit_passed=True))
    result = dict(schema=OVERLAY_SCHEMA, campaign=plan["parent_campaign"],
        recovery_plan=bound(out / "plan.json"), parent_output=plan["parent_output"],
        baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
        original_acceptance_criteria_unchanged=True, complete=True,
        replacements=replacements)
    if write_manifest:
        write(out / "overlay.json", result, immutable=True)
    else:
        require(read(out / "overlay.json") == result, "Recovery overlay changed")
    return result


def rss_bytes(pid):
    """Read one worker's resident bytes without launching an unbounded helper."""
    if sys.platform == "darwin":
        import ctypes

        class TaskInfo(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64) for name in
                ("virtual_size", "resident_size", "total_user", "total_system",
                 "threads_user", "threads_system")
            ] + [
                (name, ctypes.c_int32) for name in
                ("policy", "faults", "pageins", "cow_faults", "messages_sent",
                 "messages_received", "syscalls_mach", "syscalls_unix", "csw",
                 "threadnum", "numrunning", "priority")
            ]

        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                         ctypes.c_void_p, ctypes.c_int]
        library.proc_pidinfo.restype = ctypes.c_int
        info = TaskInfo()
        copied = library.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        # The worker may finish between Popen.poll() and this query.  Zero then
        # means no live process; the next poll collects its exit status.
        if copied == 0:
            return 0
        require(copied == ctypes.sizeof(info), "Cannot inspect recovery worker memory")
        return int(info.resident_size)
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            require(len(fields) == 3 and fields[2] == "kB", "Malformed worker memory record")
            return int(fields[1]) * 1024
    raise ValueError("Cannot inspect recovery worker memory")


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def supervise(process, began):
    peak = 0
    while process.poll() is None:
        elapsed = time.monotonic() - began
        if elapsed >= TASK_WALL_SECONDS:
            stop(process)
            return 75, "time_limit", peak
        current = rss_bytes(process.pid)
        peak = max(peak, current)
        if current > MAX_RSS_BYTES:
            stop(process)
            return 76, "memory_limit", peak
        time.sleep(.5)
    return process.returncode, None, peak


def run(out):
    out = output_path(out)
    require(not out.exists(), "One immutable allocation only; choose a new output after inspection")
    require(shutil.disk_usage(out.parent).free >= MINIMUM_FREE_DISK_BYTES,
            "Below the fixed 5-GiB free-disk reserve")
    plan = build_plan(out)
    out.mkdir()
    write(out / "plan.json", plan, immutable=True)
    allocation = dict(schema="copying-payoff-recovery-allocation-v1", plan_id=plan["plan_id"],
        status="running", dispatcher_pid=os.getpid(), started_unix=time.time(),
        max_workers=1, task_wall_seconds=TASK_WALL_SECONDS,
        maximum_attempts_per_task=1, max_rss_bytes=MAX_RSS_BYTES, attempts=[])
    write(out / "allocation.json", allocation)
    active = None
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        stop(active)

    handlers = {number: signal.signal(number, on_signal) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, spec in enumerate(plan["tasks"]):
            require(not interrupted, "Recovery supervisor interrupted")
            task = spec["task"]
            p = paths(out, task)
            attempt = dict(plan_id=plan["plan_id"], task=task, status="running",
                dispatcher_pid=os.getpid(), reserved_seconds=TASK_WALL_SECONDS,
                reserved_memory_bytes=MAX_RSS_BYTES, started_unix=time.time())
            write(p["attempt"], attempt, immutable=True)
            began = time.monotonic()
            with p["log"].open("x") as stream:
                active = subprocess.Popen([plan["python"], str(Path(__file__).resolve()), "worker",
                    "--out", str(out), "--index", str(index)], cwd=ROOT,
                    env=dense.pilot.environment(plan), stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                code, forced, observed_peak = supervise(active, began)
            active = None
            status = forced or ("ready" if code == 0 else "unresolved" if code == 2 else "failed")
            attempt.update(status=status, exit_code=code, observed_peak_rss_bytes=observed_peak,
                           elapsed_seconds=time.monotonic() - began, finished_unix=time.time())
            write(p["attempt"], attempt)
            allocation["attempts"].append(attempt)
            write(out / "allocation.json", allocation)
            if forced or code not in (0, 2):
                allocation.update(status="failed", reason=status)
                break
        else:
            allocation["status"] = ("ready" if all(item["status"] == "ready" for item in allocation["attempts"])
                                    else "unresolved")
        if allocation["status"] == "ready":
            overlay(out, write_manifest=True)
    except BaseException as error:
        allocation.update(status="interrupted" if interrupted else "failed", reason=str(error))
        raise
    finally:
        stop(active)
        allocation.update(finished_unix=time.time())
        write(out / "allocation.json", allocation)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return {"ready": 0, "unresolved": 2}.get(allocation["status"], 3)


def check_parent():
    campaign, specs = parent_specs()
    return dict(schema=SCHEMA, campaign_id=campaign["campaign_id"],
        tasks=[dict(time=spec["time"], task_id=spec["task"]["id"],
                    order=spec["task"]["order"]) for spec in specs],
        baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
        max_workers=1, task_wall_seconds=TASK_WALL_SECONDS, max_rss_bytes=MAX_RSS_BYTES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-parent", "run", "worker", "check-overlay"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "check-parent":
        result, code = check_parent(), 0
    elif args.action == "run":
        raise SystemExit(run(args.out))
    elif args.action == "worker":
        require(args.index is not None, "Worker needs a frozen task index")
        raise SystemExit(worker(args.out, args.index))
    else:
        result, code = overlay(args.out, write_manifest=False), 0
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

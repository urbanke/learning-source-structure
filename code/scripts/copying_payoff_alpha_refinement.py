"""Targeted continuation of the dominant copying-payoff 16/32 references.

The all-distinct canonical prefix carries most of the aggregate exponent-rule
comparison variance at each of times 5--8.  This immutable campaign continues
exactly those eight already passing chains, preserving their retained samples,
terminal states and RNG states.  It changes neither target nor proposal and
does not approve a payoff crossing.  Four workers are admitted under the same
10-GiB aggregate reservation used by the completed laptop campaign.
"""
from __future__ import annotations

import argparse
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
SCHEMA = "copying-payoff-alpha-refinement-v1"
OVERLAY_SCHEMA = "copying-payoff-alpha-refinement-overlay-v1"
DEFAULT_OUT = ROOT / "output/copying_payoff_alpha_refinement_2026-09-21"
T5_PARENT = ROOT / "output/copying_payoff_dense_2026-09-20/t5"
CAMPAIGN_PARENT = ROOT / "output/copying_payoff_parallel_2026-09-20"
BASELINE_DRAWS = 16384
TARGET_DRAWS = 262144
MAX_WORKERS = 4
TASK_WALL_SECONDS = 1800
TOTAL_WALL_SECONDS = 3700
RESERVED_BYTES_PER_WORKER = 5 * 2**29
TOTAL_RESERVED_BYTES = 10 * 2**30
MINIMUM_FREE_DISK_BYTES = 5 * 2**30
TARGETS = (
    (5, 74, "t0005_7f2c5f390b0c7e30e5a6_a16", 16),
    (5, 75, "t0005_7f2c5f390b0c7e30e5a6_a32", 32),
    (6, 160, "t0006_4e098021cf9ec15699ef_a16", 16),
    (6, 161, "t0006_4e098021cf9ec15699ef_a32", 32),
    (7, 316, "t0007_d71a595b6e83e7180caf_a16", 16),
    (7, 317, "t0007_d71a595b6e83e7180caf_a32", 32),
    (8, 494, "t0008_29101ebf1ead68502950_a16", 16),
    (8, 495, "t0008_29101ebf1ead68502950_a32", 32),
)
PARENT_FILES = ("report", "identity", "audit", "checkpoint", "moments")
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


def output_path(value):
    path = Path(value).absolute()
    require(path.parent.resolve() == (ROOT / "output").resolve()
            and path not in (T5_PARENT.resolve(), CAMPAIGN_PARENT.resolve())
            and not path.is_symlink(),
            "Refinement output must be a separate direct child of project output")
    return path


def paths(out, task):
    result = dense.paths(Path(out), task)
    folder = result["checkpoint"].parent
    result.update(prepared=folder / "prepared.json", receipt=folder / "receipt.json",
                  claim=folder / "worker.claim")
    return result


def parent_folder(observed):
    return T5_PARENT if observed == 5 else CAMPAIGN_PARENT / f"t{observed}"


def code_hashes():
    names = ("copying_payoff_alpha_refinement.py", "copying_payoff_accuracy.py",
             "copying_payoff_dense.py", "copying_payoff_parallel.py")
    return {f"code/scripts/{name}": sha(ROOT / "code/scripts" / name) for name in names}


def validate_parent(plan, task):
    p = dense.paths(Path(plan["output"]), task)
    report, identity, audit = (read(p[name]) for name in ("report", "identity", "audit"))
    require(report["schema"] == dense.SCHEMA and report["plan_id"] == plan["plan_id"]
            and identity["plan_id"] == plan["plan_id"]
            and report["task"] == identity["task"] == task
            and report["complete"] is True and report["audit_passed"] is True,
            "Refinement parent is not a completed passing record")
    passed = audit["passed"] and all(item["passed"] for item in audit["final_state_grid_checks"])
    require(passed is True, "Refinement parent audit no longer passes")
    for name in ("identity", "audit", "checkpoint", "moments"):
        require(report[name + "_sha256"] == sha(p[name]),
                "Parent report binding changed: " + name)
    diagnostics = report["estimate"]["diagnostics"]
    require({name: diagnostics[name] for name in ("chains", "burn", "draws", "thin", "seed")}
            == dict(chains=4, burn=16384, draws=BASELINE_DRAWS, thin=1, seed=task["seed"]),
            "Parent chain configuration changed")
    return dict(plan=plan, task=task, report=report, identity=identity, audit=audit,
                paths=p, files={name: bound(p[name]) for name in PARENT_FILES})


def parent_specs():
    campaign = parallel.load_campaign(CAMPAIGN_PARENT)
    specs = []
    common = None
    for observed, task_index, task_id, order in TARGETS:
        folder = parent_folder(observed)
        plan = dense.load_plan(folder)
        require(plan["observed"] == observed, "Parent wave time changed")
        task = next((item for item in plan["tasks"] if item["id"] == task_id), None)
        require(task is not None and task["task_index"] == task_index
                and task["order"] == order and task["time"] == observed
                and task["tokens"] == list(range(observed))
                and task["target_draws"] == BASELINE_DRAWS,
                "Fixed all-distinct refinement task changed")
        parent = validate_parent(plan, task)
        identity = {name: plan[name] for name in
                    ("runtime", "python", "environment", "dependency_sha256", "prior")}
        if common is None:
            common = identity
        require(identity == common, "Parent waves use different numerical environments")
        specs.append(dict(time=observed, task=task, parent_plan=bound(folder / "plan.json"),
                          parent_files=parent["files"], parent_identity=parent["identity"]))
    require(len(specs) == 8 and len({spec["task"]["id"] for spec in specs}) == 8,
            "Refinement task family is incomplete")
    return campaign, specs, common


def build_plan(out):
    out = output_path(out)
    campaign, specs, common = parent_specs()
    plan = dict(schema=SCHEMA, output=str(out.resolve()),
        t5_parent_plan=bound(T5_PARENT / "plan.json"),
        campaign_parent=bound(CAMPAIGN_PARENT / "campaign.json"),
        campaign_id=campaign["campaign_id"], tasks=specs,
        baseline_draws_per_chain=BASELINE_DRAWS,
        target_draws_per_chain=TARGET_DRAWS,
        sampler=dict(chains=4, burn=16384, thin=1, surrogate_steps=64),
        resources=dict(max_workers=MAX_WORKERS, task_wall_seconds=TASK_WALL_SECONDS,
            total_wall_seconds=TOTAL_WALL_SECONDS,
            reserved_memory_bytes_per_worker=RESERVED_BYTES_PER_WORKER,
            total_reserved_memory_bytes=TOTAL_RESERVED_BYTES,
            maximum_attempts_per_task=1,
            minimum_free_disk_bytes=MINIMUM_FREE_DISK_BYTES),
        code_sha256=code_hashes(), original_acceptance_criteria_unchanged=True,
        exact_rng_and_retained_prefix_required=True, automatic_retries=False,
        population_reference_approved=False, payoff_forecast_approved=False,
        selection="Both 16/32 rules for the all-distinct canonical prefix at t=5,6,7,8; selected before continuation from aggregate variance contributions.",
        **common)
    plan["plan_id"] = fingerprint(plan)
    return plan


def prepare(out):
    out = output_path(out)
    require(not out.exists(), "Use a new output directory; refinement preparation is immutable")
    plan = build_plan(out)
    out.mkdir()
    write(out / "plan.json", plan, immutable=True)
    return plan


def load_plan(out):
    out = output_path(out)
    saved = read(out / "plan.json")
    require(saved["schema"] == SCHEMA and saved["output"] == str(out.resolve())
            and saved["plan_id"] == fingerprint({k: v for k, v in saved.items() if k != "plan_id"}),
            "Refinement plan identity changed")
    require(saved == build_plan(out), "Refinement plan, parent inputs, code, or limits changed")
    for spec in saved["tasks"]:
        verify_bound(spec["parent_plan"])
        for item in spec["parent_files"].values():
            verify_bound(item)
    return saved


def current_parent(spec):
    verify_bound(spec["parent_plan"])
    for item in spec["parent_files"].values():
        verify_bound(item)
    plan = dense.load_plan(Path(spec["parent_plan"]["path"]).parent)
    task = next(item for item in plan["tasks"] if item["id"] == spec["task"]["id"])
    parent = validate_parent(plan, task)
    require(task == spec["task"] and parent["identity"] == spec["parent_identity"]
            and parent["files"] == spec["parent_files"], "Frozen parent specification changed")
    return parent


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
            "Loaded RNG or terminal state differs from parent")
    return result


def worker(out, index):
    out = output_path(out)
    plan = load_plan(out)
    require(type(index) is int and 0 <= index < len(plan["tasks"]), "Invalid refinement task index")
    spec, task = plan["tasks"][index], plan["tasks"][index]["task"]
    p = paths(out, task)
    attempt = read(p["attempt"])
    require(attempt["plan_id"] == plan["plan_id"] and attempt["task"] == task
            and attempt["status"] == "running" and attempt["dispatcher_pid"] == os.getppid()
            and attempt["reserved_seconds"] == TASK_WALL_SECONDS
            and attempt["reserved_memory_bytes"] == RESERVED_BYTES_PER_WORKER,
            "Worker requires its live bounded supervisor allocation")
    p["claim"].parent.mkdir(parents=True, exist_ok=True)
    with p["claim"].open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    os.environ.update(dense.pilot.environment(plan))
    require(dense.pilot.runtime() == plan["runtime"], "Refinement numerical runtime differs")
    parent = current_parent(spec)
    began = time.monotonic()
    with lock(p["attempt"].with_suffix(".worker.lock")):
        require(not any(p[name].exists() for name in
            ("checkpoint", "identity", "moments", "audit", "report", "receipt", "prepared")),
            "Refinement never overwrites or restarts a task")
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
        write(p["prepared"], dict(schema="copying-payoff-alpha-refinement-prepared-v1",
            plan_id=plan["plan_id"], task=task, parent_plan=spec["parent_plan"],
            parent_files=spec["parent_files"], sample_prefix=initial_prefix,
            rng_continuity=restored), immutable=True)
        identity = dict(plan_id=plan["plan_id"], parent_plan_id=parent["plan"]["plan_id"],
            task=task, target=run.fingerprint, proposal=run.proposal_descriptor,
            runtime=dense.pilot.runtime())
        write(p["identity"], identity, immutable=True)
        saved = time.monotonic()

        def progress(event):
            nonlocal saved
            now = time.monotonic()
            if event["draws"] == event["requested_draws"] or now - saved >= 120:
                run.save(p["checkpoint"])
                saved = now
            print(json.dumps(dict(event="copying_alpha_progress", task=task["id"], **event)), flush=True)

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
        replacement = {name: bound(p[name]) for name in PARENT_FILES}
        receipt = dict(schema="copying-payoff-alpha-refinement-receipt-v1",
            plan_id=plan["plan_id"], parent_plan_id=parent["plan"]["plan_id"], task=task,
            baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
            sampling_complete=True, audit_passed=passed,
            status="ready" if passed else "unresolved", parent_plan=spec["parent_plan"],
            parent_files=spec["parent_files"], replacement_files=replacement,
            prepared_sha256=sha(p["prepared"]), sample_prefix=prefix,
            rng_continuity=restored, original_acceptance_criteria_unchanged=True,
            automatic_retry=False)
        write(p["receipt"], receipt, immutable=True)
        current_parent(spec)
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
    parent = current_parent(spec)
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
        require(report[name + "_sha256"] == sha(p[name]),
                "Replacement report binding changed: " + name)
    require(report["estimate"]["diagnostics"]["draws"] == TARGET_DRAWS
            and report["estimate"]["diagnostics"]["seed"] == spec["task"]["seed"],
            "Replacement draw count or seed changed")
    passed = bool(audit["passed"] and all(item["passed"] for item in audit["final_state_grid_checks"]))
    require(report["audit_passed"] == receipt["audit_passed"] == passed,
            "Replacement audit acceptance changed")
    prefix = sample_prefix_proof(parent["paths"]["checkpoint"], p["checkpoint"], complete=True)
    restored = expected_rng_proof(parent["paths"]["checkpoint"])
    replacement = {name: bound(p[name]) for name in PARENT_FILES}
    expected = dict(schema="copying-payoff-alpha-refinement-receipt-v1",
        plan_id=plan["plan_id"], parent_plan_id=parent["plan"]["plan_id"], task=spec["task"],
        baseline_draws_per_chain=BASELINE_DRAWS, target_draws_per_chain=TARGET_DRAWS,
        sampling_complete=True, audit_passed=passed, status="ready" if passed else "unresolved",
        parent_plan=spec["parent_plan"], parent_files=spec["parent_files"],
        replacement_files=replacement, prepared_sha256=sha(p["prepared"]),
        sample_prefix=prefix, rng_continuity=restored,
        original_acceptance_criteria_unchanged=True, automatic_retry=False)
    require(receipt == expected, "Replacement receipt or continuation proof changed")
    if require_pass:
        require(passed, "Every alpha refinement must pass the unchanged individual audits")
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
                (*PARENT_FILES, "receipt")}, sample_prefix=checked["prefix"],
            rng_continuity=checked["rng"], audit_passed=True))
    result = dict(schema=OVERLAY_SCHEMA, t5_parent_plan=plan["t5_parent_plan"],
        campaign_parent=plan["campaign_parent"], recovery_plan=bound(out / "plan.json"),
        baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
        original_acceptance_criteria_unchanged=True, complete=True,
        replacements=replacements)
    if write_manifest:
        write(out / "overlay.json", result, immutable=True)
    else:
        require(read(out / "overlay.json") == result, "Alpha refinement overlay changed")
    return result


def rss_bytes(pid):
    if sys.platform == "darwin":
        import ctypes

        class TaskInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                ("virtual_size", "resident_size", "total_user", "total_system",
                 "threads_user", "threads_system")] + [(name, ctypes.c_int32) for name in
                ("policy", "faults", "pageins", "cow_faults", "messages_sent",
                 "messages_received", "syscalls_mach", "syscalls_unix", "csw",
                 "threadnum", "numrunning", "priority")]

        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                         ctypes.c_void_p, ctypes.c_int]
        library.proc_pidinfo.restype = ctypes.c_int
        info = TaskInfo()
        copied = library.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if copied == 0:
            return 0
        require(copied == ctypes.sizeof(info), "Cannot inspect refinement worker memory")
        return int(info.resident_size)
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            require(len(fields) == 3 and fields[2] == "kB", "Malformed worker memory record")
            return int(fields[1]) * 1024
    raise ValueError("Cannot inspect refinement worker memory")


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


def run(out):
    out = output_path(out)
    plan = load_plan(out)
    allocation_path = out / "allocation.json"
    require(not allocation_path.exists(), "One immutable allocation only; inspect rather than restart")
    require(not any(paths(out, spec["task"])["attempt"].exists() for spec in plan["tasks"]),
            "Refinement has prior task attempts; inspect rather than restart")
    require(shutil.disk_usage(out).free >= MINIMUM_FREE_DISK_BYTES,
            "Below the fixed 5-GiB free-disk reserve")
    allocation = dict(schema="copying-payoff-alpha-refinement-allocation-v1",
        plan_id=plan["plan_id"], status="running", dispatcher_pid=os.getpid(),
        started_unix=time.time(), max_workers=MAX_WORKERS,
        task_wall_seconds=TASK_WALL_SECONDS, total_wall_seconds=TOTAL_WALL_SECONDS,
        maximum_attempts_per_task=1, attempts=[])
    write(allocation_path, allocation, immutable=True)
    active, pending = {}, list(range(len(plan["tasks"])))
    started = time.monotonic()
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    handlers = {number: signal.signal(number, on_signal) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with lock(ROOT / "output/.copying-payoff-alpha-refinement-dispatcher.lock"):
            while pending or active:
                require(not interrupted, "Refinement supervisor interrupted")
                require(time.monotonic() - started < TOTAL_WALL_SECONDS,
                        "Refinement total wall limit reached")
                now = time.monotonic()
                for index, item in list(active.items()):
                    process, attempt, p, began, stream = item
                    current = rss_bytes(process.pid) if process.poll() is None else 0
                    attempt["observed_peak_rss_bytes"] = max(
                        attempt.get("observed_peak_rss_bytes", 0), current)
                    if current > RESERVED_BYTES_PER_WORKER:
                        stop(process); code, forced = 76, "memory_limit"
                    elif now - began >= TASK_WALL_SECONDS and process.poll() is None:
                        stop(process); code, forced = 75, "time_limit"
                    elif process.poll() is None:
                        continue
                    else:
                        code, forced = process.returncode, None
                    stream.close()
                    status = forced or ("ready" if code == 0 else "unresolved" if code == 2 else "failed")
                    attempt.update(status=status, exit_code=code,
                        elapsed_seconds=time.monotonic() - began, finished_unix=time.time())
                    write(p["attempt"], attempt)
                    allocation["attempts"].append(attempt)
                    del active[index]
                    if status == "failed" or forced:
                        raise RuntimeError(f"Refinement task {index} stopped: {status}")
                total_rss = sum(rss_bytes(item[0].pid) for item in active.values()
                                if item[0].poll() is None)
                require(total_rss <= TOTAL_RESERVED_BYTES,
                        "Active refinement workers exceed the 10-GiB aggregate bound")
                while pending and len(active) < MAX_WORKERS:
                    require(TOTAL_WALL_SECONDS - (time.monotonic() - started) > TASK_WALL_SECONDS,
                            "No complete per-task allocation remains")
                    require(shutil.disk_usage(out).free >= MINIMUM_FREE_DISK_BYTES,
                            "Below the fixed 5-GiB free-disk reserve")
                    index = pending.pop(0); spec = plan["tasks"][index]; task = spec["task"]
                    p = paths(out, task)
                    attempt = dict(plan_id=plan["plan_id"], task=task, status="running",
                        dispatcher_pid=os.getpid(), reserved_seconds=TASK_WALL_SECONDS,
                        reserved_memory_bytes=RESERVED_BYTES_PER_WORKER,
                        started_unix=time.time(), observed_peak_rss_bytes=0)
                    write(p["attempt"], attempt, immutable=True)
                    stream = p["log"].open("x")
                    process = subprocess.Popen([plan["python"], str(Path(__file__).resolve()),
                        "worker", "--out", str(out), "--index", str(index)], cwd=ROOT,
                        env=dense.pilot.environment(plan), stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    active[index] = (process, attempt, p, time.monotonic(), stream)
                allocation.update(active_task_indices=sorted(active), pending_task_indices=pending,
                                  updated_unix=time.time())
                write(allocation_path, allocation)
                if active:
                    time.sleep(.5)
            allocation["status"] = ("ready" if all(item["status"] == "ready"
                                      for item in allocation["attempts"]) else "unresolved")
            if allocation["status"] == "ready":
                overlay(out, write_manifest=True)
    except BaseException as error:
        allocation.update(status="interrupted" if interrupted else "failed", reason=str(error))
        raise
    finally:
        for process, attempt, p, began, stream in active.values():
            stop(process); stream.close()
            attempt.update(status="interrupted", exit_code=process.returncode,
                elapsed_seconds=time.monotonic() - began, finished_unix=time.time())
            write(p["attempt"], attempt)
            allocation["attempts"].append(attempt)
        allocation.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started,
                          active_task_indices=[], pending_task_indices=pending)
        write(allocation_path, allocation)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return {"ready": 0, "unresolved": 2}.get(allocation["status"], 3)


def check_plan(out):
    plan = load_plan(out)
    return dict(schema=SCHEMA, plan_id=plan["plan_id"], task_ids=[spec["task"]["id"] for spec in plan["tasks"]],
        baseline_draws=BASELINE_DRAWS, target_draws=TARGET_DRAWS,
        max_workers=MAX_WORKERS, task_wall_seconds=TASK_WALL_SECONDS,
        reserved_memory_bytes_per_worker=RESERVED_BYTES_PER_WORKER,
        total_reserved_memory_bytes=TOTAL_RESERVED_BYTES,
        sampling_started=(Path(out) / "allocation.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check-plan", "run", "worker", "check-overlay"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        result, code = prepare(args.out), 0
    elif args.action == "check-plan":
        result, code = check_plan(args.out), 0
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

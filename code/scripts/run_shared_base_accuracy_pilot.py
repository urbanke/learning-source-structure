"""Bounded computational pilot for shared-base reference cost and precision.

The fixed plan contains sources0..7 and times16,64,256,1024. An explicit
--prefixes argument selects which planned times to run, initially --prefixes 16.
This never dispatches the subsequent scientific cohort. Failed calculations
remain in their original files, and all designated cases remain in status.json.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER = ROOT / "code/transformer"
sys.path.insert(0, str(TRANSFORMER))
from eval_shared_base_models import atomic_json, fingerprint, sha_file
from eval_shared_base_reference import load_inputs
from eval_shared_base_surrogate import CODE_FILES, SEED_TAG
from shared_base_full_context_summary import comparison_metrics, metric_refinement, probability_refinement
from source import seed_of


INDICES = tuple(range(8))
TIMES = (16, 64, 256, 1024)
SETTINGS = dict(chains=4, burn=16384, thin=1, seed=20260926, surrogate_steps=64,
                initial_draws=256, max_draws=4096, block_draws=1024, term_cache_size=16384)
INITIAL_ORDERS = (16, 32)
FALLBACK_ORDERS = (64, 128)
DEFAULT_INPUT = ROOT / "output/shared_base_context_accuracy_2026-09-15"
DEFAULT_OUT = ROOT / "output/shared_base_accuracy_cost_pilot_2026-09-15"


def now():
    return datetime.now(timezone.utc).isoformat()


def case_key(index, prefix):
    return f"seq_{index:06d}_t{prefix:04d}"


def job_key(index, prefix, order):
    return f"{case_key(index, prefix)}_alpha{order}"


def requires_reuse_audit(index, prefix):
    return prefix == 1024 and index < 4


def make_protocol(folder):
    cohort = json.loads((folder / "cohort.json").read_text())
    if cohort["n_sequences"] != 1024 or cohort["prefixes"] != list(TIMES):
        raise ValueError("input must be the fixed complete1024-source context cohort")
    inputs = {}
    for prefix in TIMES:
        for index in INDICES:
            _, _, _, _, identity = load_inputs(folder, index, prefix)
            inputs[case_key(index, prefix)] = identity
    code = {name: sha_file(TRANSFORMER / name) for name in CODE_FILES}
    code["code/scripts/run_shared_base_accuracy_pilot.py"] = sha_file(__file__)
    code["code/scripts/shared_base_full_context_summary.py"] = sha_file(Path(__file__).with_name("shared_base_full_context_summary.py"))
    store = os.environ.get("LSA_STORE")
    store_files = {name: sha_file(Path(store) / name) for name in ("manifest.json", "anchors.json")
                   if store and (Path(store) / name).exists()}
    return dict(schema="shared-base-accuracy-cost-pilot-v1", input_root=str(folder.resolve()),
                purpose="Computational cost/precision pilot, not the final eight-sequence paper experiment.",
                source_indices=list(INDICES), observed_lengths=list(TIMES), settings=SETTINGS,
                initial_orders=list(INITIAL_ORDERS), fallback_orders=list(FALLBACK_ORDERS),
                fallback="Use64/128 only if locally valid16/32 predictions fail exponent-refinement checks; sampling failures remain unresolved at the cost cap.",
                endpoint_reuse="Indices0..3 at1024 are skipped and labelled reuse_pending_audit; existing references are not automatically accepted.",
                scheduling="Times outermost, source indices ascending, at most two independent processes.",
                inputs=inputs, code_sha256=code, python=sys.executable, python_version=sys.version,
                numerical_environment=dict(LSA_STORE=store, PMM_SRC=os.environ.get("PMM_SRC"), store_sha256=store_files))


def job_command(folder, out, index, prefix, order):
    command = [sys.executable, str(TRANSFORMER / "eval_shared_base_surrogate.py"),
               "--input", str(folder.resolve()), "--out", str(out.resolve()),
               "--index", str(index), "--prefix", str(prefix), "--alpha-order", str(order)]
    for name, value in SETTINGS.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def reference_path(out, index, prefix, order):
    return out / "refs" / f"alpha{order}" / f"{case_key(index, prefix)}.json"


def inspect_reference(out, protocol, index, prefix, order):
    path = reference_path(out, index, prefix, order)
    if not path.exists():
        return dict(state="missing", terminal=False)
    report = json.loads(path.read_text())
    settings = report["settings"]
    expected_seed = seed_of(SEED_TAG, SETTINGS["seed"], index, prefix, order)
    if (report["identity"]["inputs"] != protocol["inputs"][case_key(index, prefix)]
            or report["reference_id"] != fingerprint(report["identity"])
            or report["sequence_index"] != index or report["prefix_length"] != prefix
            or settings["alpha_order"] != order or settings["seed"] != expected_seed
            or any(settings[name] != SETTINGS[name] for name in ("chains", "burn", "thin", "surrogate_steps"))
            or report["identity"]["code_sha256"] != {name: protocol["code_sha256"][name] for name in CODE_FILES}):
        raise ValueError(f"reference provenance differs from this pilot: {path}")
    moments = path.with_name(report["moments_file"])
    if not moments.exists() or sha_file(moments) != report["moments_sha256"]:
        raise ValueError(f"reference moment checksum failed: {moments}")
    draws = report["estimate"]["diagnostics"]["draws"]
    if draws > SETTINGS["max_draws"]:
        raise ValueError("a cost-pilot job must not contain more than4096 retained draws per chain")
    valid = bool(report["local_validation_passed"])
    terminal = valid or draws == SETTINGS["max_draws"]
    return dict(state="locally_valid" if valid else ("unresolved_at_cost_cap" if terminal else "partial"),
                terminal=terminal, report_path=str(path.resolve()), report_sha256=sha_file(path),
                draws_per_chain=draws, elapsed_seconds=report["estimate"]["elapsed_seconds"],
                local_validation_passed=valid)


def assess_pair(folder, out, protocol, index, prefix, orders):
    jobs = [inspect_reference(out, protocol, index, prefix, order) for order in orders]
    if not all(job.get("local_validation_passed", False) for job in jobs):
        return dict(state="sampling_or_job_unresolved", orders=list(orders), jobs=jobs,
                    reason="The cost cap does not certify the reference; retain this source and its reports.")
    _, _, oracle, models, _ = load_inputs(folder, index, prefix)
    reports = [json.loads(Path(job["report_path"]).read_text()) for job in jobs]
    if reports[0]["settings"]["seed"] == reports[1]["settings"]["seed"]:
        raise ValueError("refinement runs must have independent sampling seeds")
    estimates = [report["estimate"] for report in reports]
    vector = probability_refinement(estimates[0]["predictive_probs"], estimates[1]["predictive_probs"],
                                    estimates[0]["mcse_probs"], estimates[1]["mcse_probs"])
    metrics = {}
    for name, model in models.items():
        candidates = []
        for job, report, estimate in zip(jobs, reports, estimates):
            with np.load(Path(job["report_path"]).with_name(report["moments_file"]), allow_pickle=False) as moments:
                candidates.append(comparison_metrics(estimate["predictive_probs"], model["prediction"], oracle,
                                                       moments["cov_batch"], moments["cov_chain"]))
        metrics[name] = metric_refinement(*candidates)
    passed = vector["passed"] and all(item["passed"] for item in metrics.values())
    return dict(state="pair_validated" if passed else "exponent_refinement_failed", orders=list(orders),
                jobs=jobs, probability_refinement=vector, model_metric_refinement=metrics)


@contextmanager
def exclusive_dispatch(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".dispatcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another dispatcher is already using this output folder") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_jobs(jobs, folder, out, protocol, status, workers, save):
    pending = list(jobs)
    active = {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    last_update = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < workers:
                index, prefix, order = pending.pop(0)
                key = job_key(index, prefix, order)
                entry = status["jobs"].setdefault(key, dict(attempts=[]))
                existing = inspect_reference(out, protocol, index, prefix, order)
                if existing["terminal"]:
                    entry.update(existing, reused=True)
                    save()
                    continue
                attempt = len(entry["attempts"]) + 1
                log_path = out / "logs" / f"{key}_attempt{attempt:03d}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("x")
                command = job_command(folder, out, index, prefix, order)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                entry["attempts"].append(dict(started_at=now(), pid=process.pid, command=command,
                                              log_path=str(log_path.resolve())))
                entry.update(state="running", terminal=False, reused=False)
                active[key] = (process, log, index, prefix, order)
                save()
            for key, (process, log, index, prefix, order) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                entry = status["jobs"][key]
                entry["attempts"][-1].update(finished_at=now(), returncode=code)
                result = inspect_reference(out, protocol, index, prefix, order)
                entry.update(result)
                if not result["terminal"]:
                    entry["state"] = "interrupted_or_failed"
                del active[key]
                save()
            if time.monotonic() - last_update > 30:
                save()
                last_update = time.monotonic()
            if pending or active:
                time.sleep(0.25)
    finally:
        for process, _, _, _, _ in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for key, (process, log, _, _, _) in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            log.close()
            status["jobs"][key].update(state="interrupted", terminal=False)
            status["jobs"][key]["attempts"][-1].update(finished_at=now(), returncode=process.returncode)
        if active:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prefixes", type=int, nargs="+", choices=TIMES, required=True,
                        help="explicit stage selection; start with --prefixes 16")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.input.resolve():
        parser.error("pilot output must differ from frozen input")
    prefixes = [t for t in TIMES if t in args.prefixes]
    with exclusive_dispatch(args.out):
        protocol = make_protocol(args.input)
        protocol_path = args.out / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("pilot inputs, code or settings changed; use a new output folder")
        else:
            atomic_json(protocol_path, protocol)
        status_path = args.out / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["protocol_id"] != fingerprint(protocol):
                raise ValueError("status protocol identity changed")
        else:
            status = dict(schema="shared-base-accuracy-cost-pilot-status-v1", protocol_id=fingerprint(protocol),
                          created_at=now(), jobs={}, cases={case_key(i, t): dict(
                              sequence_index=i, observed_length=t,
                              state="reuse_pending_audit" if requires_reuse_audit(i, t) else "not_started")
                              for t in TIMES for i in INDICES})
        status.update(selected_prefixes=prefixes, workers=args.workers, dispatcher_pid=os.getpid(),
                      stage="planned" if args.plan_only else "running")

        def save():
            status["updated_at"] = now()
            atomic_json(status_path, status)

        save()
        if args.plan_only:
            print(json.dumps(dict(stage="planned", prefixes=prefixes, sources=list(INDICES),
                                  protocol=str(protocol_path), max_workers=args.workers)), flush=True)
            return
        try:
            for prefix in prefixes:
                indices = [i for i in INDICES if not requires_reuse_audit(i, prefix)]
                initial = [(i, prefix, order) for i in indices for order in INITIAL_ORDERS]
                run_jobs(initial, args.input, args.out, protocol, status, args.workers, save)
                fallback_indices = []
                for index in indices:
                    outcome = assess_pair(args.input, args.out, protocol, index, prefix, INITIAL_ORDERS)
                    case = status["cases"][case_key(index, prefix)]
                    case.update(state=outcome["state"], initial_pair=outcome)
                    if outcome["state"] == "exponent_refinement_failed":
                        fallback_indices.append(index)
                    save()
                fallback = [(i, prefix, order) for i in fallback_indices for order in FALLBACK_ORDERS]
                run_jobs(fallback, args.input, args.out, protocol, status, args.workers, save)
                for index in fallback_indices:
                    outcome = assess_pair(args.input, args.out, protocol, index, prefix, FALLBACK_ORDERS)
                    status["cases"][case_key(index, prefix)].update(state=outcome["state"], fallback_pair=outcome)
                    save()
                print(json.dumps(dict(prefix=prefix, cases={str(i): status["cases"][case_key(i, prefix)]["state"]
                                                            for i in INDICES})), flush=True)
            status["stage"] = "selected_stage_finished"
        except BaseException:
            status["stage"] = "interrupted_or_failed"
            raise
        finally:
            save()


if __name__ == "__main__":
    main()

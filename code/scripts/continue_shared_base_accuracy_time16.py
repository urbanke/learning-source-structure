"""Continue the first eight time16 references without repeating warmup.

Closed cost-pilot checkpoints, reports and moments are copied byte-for-byte
to a distinct folder. Only the draw budget changes. Original outputs are
never overwritten; all source indices remain in the resulting status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

import run_shared_base_accuracy_pilot as cost
from shared_base_surrogate_summary import run_audit, _json_safe

ROOT = cost.ROOT
INDICES = tuple(range(8))
PREFIX = 16
SETTINGS = dict(cost.SETTINGS, initial_draws=8192, max_draws=65536)
DEFAULT_ORIGINAL = ROOT / "output/shared_base_accuracy_cost_pilot_v2_2026-09-15"
DEFAULT_OUT = ROOT / "output/shared_base_accuracy_time16_extension_2026-09-15"


def command(folder, out, index, order):
    result = [sys.executable, str(cost.TRANSFORMER / "eval_shared_base_surrogate.py"),
              "--input", str(folder.resolve()), "--out", str(out.resolve()),
              "--index", str(index), "--prefix", str(PREFIX), "--alpha-order", str(order)]
    for name, value in SETTINGS.items():
        result.extend(["--" + name.replace("_", "-"), str(value)])
    return result


def verify_checkpoint_load(folder, original, index, order):
    """Reconstruct target states and restore RNG state, taking no MCMC steps."""
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from surrogate_chain_runner import SurrogateChainRun
    _, tokens, _, _, _ = cost.load_inputs(folder, index, PREFIX)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    kernel = RankedLSARowEvidence(64, 1 + nodes / 2, term_cache_size=SETTINGS["term_cache_size"])
    sampler = SharedBaseSurrogateMCMC(64, kernel, weights / 2, surrogate_steps=SETTINGS["surrogate_steps"])
    checkpoint = original / "chains" / f"alpha{order}" / f"{cost.case_key(index, PREFIX)}.npz"
    run = SurrogateChainRun.load(sampler, tokens, checkpoint)
    if (run.seed != cost.seed_of(cost.SEED_TAG, SETTINGS["seed"], index, PREFIX, order)
            or run.burn != SETTINGS["burn"] or run.thin != SETTINGS["thin"]
            or run.chains != SETTINGS["chains"] or not np.all(run.draw_counts == 4096)
            or not np.all(run._burn_done == SETTINGS["burn"])):
        raise ValueError("copied checkpoint does not contain the expected completed warmup and4096 draws")
    return dict(sequence_index=index, order=order, passed=True,
                restored_draws_per_chain=run.draw_counts.tolist(), warmup_already_completed=True)


def prepare(folder, original, out, *, verify_load=False):
    initial = cost.make_protocol(folder)
    old_protocol_path = original / "protocol.json"
    old_protocol = json.loads(old_protocol_path.read_text())
    old_status = json.loads((original / "status.json").read_text())
    if old_protocol["inputs"] != initial["inputs"] or old_protocol["settings"] != cost.SETTINGS:
        raise ValueError("original cost pilot has different source inputs or sampling settings")
    copies, validations = [], []
    for index in INDICES:
        for order in cost.INITIAL_ORDERS:
            key = cost.job_key(index, PREFIX, order)
            entry = old_status["jobs"][key]
            if (entry["state"] == "running" or not entry.get("terminal")
                    or entry["attempts"][-1].get("returncode") is None):
                raise ValueError(f"cost-pilot job is not closed: {key}")
            old = cost.inspect_reference(original, old_protocol, index, PREFIX, order)
            if not old["terminal"] or old["draws_per_chain"] != 4096:
                raise ValueError("continuation requires the closed4096-draw cost checkpoint")
            report_path = Path(old["report_path"])
            report = json.loads(report_path.read_text())
            paths = (report_path, report_path.with_name(report["moments_file"]), Path(report["chain_checkpoint"]))
            for path in paths:
                relative = path.resolve().relative_to(original.resolve())
                copies.append(dict(source=str(path.resolve()), relative_path=str(relative), sha256=cost.sha_file(path)))
            if verify_load:
                validations.append(verify_checkpoint_load(folder, original, index, order))
    protocol = dict(schema="shared-base-time16-continuation-v1", source_indices=list(INDICES),
                    observed_length=PREFIX, settings=SETTINGS, original_protocol_sha256=cost.sha_file(old_protocol_path),
                    source_protocol=initial, copied_inputs=copies,
                    method="Resume copied states and RNG without new warmup; no cohort expansion.",
                    fallback="Use64/128 only after local gates pass and the independent raw-draw audit isolates exponent-refinement failure.",
                    code_sha256={name: cost.sha_file(Path(__file__).with_name(name)) for name in
                                 ("continue_shared_base_accuracy_time16.py", "shared_base_surrogate_summary.py")})
    protocol_path = out / "protocol.json"
    resumed = protocol_path.exists()
    if resumed and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("continuation inputs, source snapshots or code changed; use another output folder")
    for item in copies:
        target = out / item["relative_path"]
        if not target.exists():
            if resumed:
                raise ValueError(f"continuation file disappeared; refusing to reset it: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
            shutil.copyfile(item["source"], temporary)
            if cost.sha_file(temporary) != item["sha256"] or cost.sha_file(item["source"]) != item["sha256"]:
                temporary.unlink()
                raise ValueError("original closed file changed while copying")
            temporary.replace(target)
        elif not resumed and cost.sha_file(target) != item["sha256"]:
            raise ValueError("unregistered output file differs; refusing to overwrite it")
    if not resumed:
        cost.atomic_json(protocol_path, protocol)
    if validations:
        cost.atomic_json(out / "checkpoint_load_validation.json", dict(passed=True, checks=validations,
                                                                     mcmc_steps_taken=0))
    return protocol


def inspect(out, protocol, index, order):
    path = cost.reference_path(out, index, PREFIX, order)
    if not path.exists():
        return dict(state="missing", terminal=False)
    report = json.loads(path.read_text())
    expected = protocol["source_protocol"]
    settings = report["settings"]
    if (report["identity"]["inputs"] != expected["inputs"][cost.case_key(index, PREFIX)]
            or report["reference_id"] != cost.fingerprint(report["identity"])
            or report["identity"]["code_sha256"] != {name: expected["code_sha256"][name] for name in cost.CODE_FILES}
            or settings["alpha_order"] != order
            or settings["seed"] != cost.seed_of(cost.SEED_TAG, SETTINGS["seed"], index, PREFIX, order)
            or any(settings[name] != SETTINGS[name] for name in ("chains", "burn", "thin", "surrogate_steps"))):
        raise ValueError("continued reference identity changed")
    moments = path.with_name(report["moments_file"])
    if cost.sha_file(moments) != report["moments_sha256"]:
        raise ValueError("continued moment checksum failed")
    draws = report["estimate"]["diagnostics"]["draws"]
    if draws > SETTINGS["max_draws"]:
        raise ValueError("continued report exceeds the declared draw cap")
    progressed = draws >= SETTINGS["initial_draws"]
    valid = bool(report["local_validation_passed"]) and progressed
    if progressed and Path(report["chain_checkpoint"]).resolve() != (out / "chains" / f"alpha{order}" / f"{cost.case_key(index, PREFIX)}.npz").resolve():
        raise ValueError("continued report still points to an external checkpoint")
    return dict(state="locally_valid" if valid else ("unresolved_at_cap" if draws == SETTINGS["max_draws"] else "awaiting_extension"),
                terminal=valid or draws == SETTINGS["max_draws"], local_validation_passed=valid,
                draws_per_chain=draws, report_sha256=cost.sha_file(path), report_path=str(path.resolve()),
                elapsed_seconds=report["estimate"]["elapsed_seconds"])


def run_jobs(jobs, folder, out, protocol, status, workers, save):
    pending, active = list(jobs), {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    last_update = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < workers:
                index, order = pending.pop(0)
                key = cost.job_key(index, PREFIX, order)
                entry = status["jobs"].setdefault(key, dict(attempts=[]))
                existing = inspect(out, protocol, index, order)
                if existing["terminal"]:
                    entry.update(existing, reused=True)
                    save()
                    continue
                log_path = out / "logs" / f"{key}_attempt{len(entry['attempts']) + 1:03d}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("x")
                args = command(folder, out, index, order)
                try:
                    process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                entry["attempts"].append(dict(started_at=cost.now(), pid=process.pid, command=args, log_path=str(log_path)))
                entry.update(state="running", terminal=False)
                active[key] = (process, log, index, order)
                save()
            for key, (process, log, index, order) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                entry = status["jobs"][key]
                entry["attempts"][-1].update(finished_at=cost.now(), returncode=code)
                entry.update(inspect(out, protocol, index, order))
                if not entry["terminal"]:
                    entry["state"] = "interrupted_or_failed"
                del active[key]
                save()
            if time.monotonic() - last_update > 30:
                save()
                last_update = time.monotonic()
            if pending or active:
                time.sleep(.25)
    finally:
        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for key, (process, log, _, _) in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            log.close()
            status["jobs"][key].update(state="interrupted", terminal=False)
        if active:
            save()


def audit_pair(folder, out, index, orders):
    args = SimpleNamespace(input=folder.resolve(), references=out.resolve(), index=index, prefix=PREFIX,
                           coarse_order=orders[0], refined_order=orders[1],
                           store=Path(os.environ["LSA_STORE"]), pmm_src=Path(os.environ["PMM_SRC"]))
    result = run_audit(args)
    result["exporter_sha256"] = cost.sha_file(Path(__file__).with_name("shared_base_surrogate_summary.py"))
    path = out / "audits" / f"seq_{index:06d}_t0016_alpha{orders[0]}_vs{orders[1]}.json"
    cost.atomic_json(path, _json_safe(result))
    failed = [entry["name"] for entry in result["checks"] if not entry["passed"]]
    refinement_only = bool(failed) and all(name.startswith(("probability_refinement:", "metric_refinement:"))
                                          and "numerical_precision" not in name for name in failed)
    return dict(passed=result["passed"], failed_checks=failed, refinement_only_failure=refinement_only,
                audit_path=str(path.resolve()), audit_sha256=cost.sha_file(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=cost.DEFAULT_INPUT)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-checkpoint-load", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() in (args.input.resolve(), args.original.resolve()):
        parser.error("continuation output must be a new folder")
    with cost.exclusive_dispatch(args.out):
        protocol = prepare(args.input, args.original, args.out, verify_load=args.verify_checkpoint_load)
        status_path = args.out / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else dict(
            schema="shared-base-time16-continuation-status-v1", protocol_id=cost.fingerprint(protocol), jobs={},
            cases={str(index): dict(state="awaiting_extension") for index in INDICES})
        if status["protocol_id"] != cost.fingerprint(protocol):
            raise ValueError("continuation status identity differs")
        status.update(stage="planned" if args.plan_only else "running", workers=args.workers, dispatcher_pid=os.getpid())

        def save():
            status["updated_at"] = cost.now()
            cost.atomic_json(status_path, status)

        save()
        if args.plan_only:
            print(json.dumps(dict(stage="planned", sources=list(INDICES), prefix=PREFIX,
                                  copied_files=len(protocol["copied_inputs"]), no_new_warmup=True)), flush=True)
            return
        try:
            run_jobs([(i, a) for i in INDICES for a in cost.INITIAL_ORDERS], args.input, args.out, protocol, status, args.workers, save)
            fallback = []
            for index in INDICES:
                result = audit_pair(args.input, args.out, index, cost.INITIAL_ORDERS)
                local = all(inspect(args.out, protocol, index, order).get("local_validation_passed", False) for order in cost.INITIAL_ORDERS)
                status["cases"][str(index)].update(state="validated" if result["passed"] else "unresolved", initial_audit=result)
                if local and result["refinement_only_failure"]:
                    fallback.append(index)
                save()
            run_jobs([(i, a) for i in fallback for a in cost.FALLBACK_ORDERS], args.input, args.out, protocol, status, args.workers, save)
            for index in fallback:
                result = audit_pair(args.input, args.out, index, cost.FALLBACK_ORDERS)
                status["cases"][str(index)].update(state="validated" if result["passed"] else "unresolved", fallback_audit=result)
                save()
            status["stage"] = "bounded_stage_finished"
        except BaseException:
            status["stage"] = "interrupted_or_failed"
            raise
        finally:
            save()


if __name__ == "__main__":
    main()

"""Continue the fixed 36 stopped time-64 references, preserving their originals.

Prepare once with an explicit UTC cutoff preceding the new pending-work run.
Run at most two unchanged workers; saved chains and RNG states resume at
65,536 draws and extend to exactly 131,072 draws per chain. Failed extensions
remain recorded and are never automatically retried or used to advance stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_shared_base_curve_campaign as campaign

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "output/shared_base_curve_campaign_2026-09-15"
DEFAULT_OUT = ROOT / "output/shared_base_curve_continuation_2026-09-16"
SCHEMA = "shared-base-curve-continuation-v1"
TOTAL_DRAWS = 131072


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "cutoff must include a timezone")
    return result


def filenames(original, job):
    report, lock = campaign.job_paths(original, job)
    return [report, report.with_name(report.stem + "_chains.npz"),
            report.with_name(report.stem + "_moments.npz")], lock


def validate_original(plan):
    require(campaign.digest(plan["original_plan"]) == plan["original_plan_sha256"], "original plan changed")
    require(campaign.digest(__file__) == plan["continuation_code_sha256"], "continuation code changed")
    original = json.loads(Path(plan["original_plan"]).read_text())
    campaign.verify_assets(original, original["global_paths"])
    for case in plan["cases"]:
        for name, sha in case["original_files"].items():
            require(campaign.digest(name) == sha, "original reference changed: " + name)
    return original


def prepare(campaign_root, out, before):
    out = Path(out).resolve()
    plan_path = Path(campaign_root).resolve() / "plan.json"
    original = json.loads(plan_path.read_text())
    refs = Path(original["references"]).resolve()
    require(out != refs and out not in refs.parents and refs not in out.parents,
            "continuation output must be separate from the original references")
    existing = out / "plan.json"
    if existing.exists():
        plan = json.loads(existing.read_text())
        require(plan.get("plan_id") == campaign.fingerprint({k: v for k, v in plan.items() if k != "plan_id"}),
                "continuation plan checksum failed")
        require(plan["before"] == before and plan["original_plan"] == str(plan_path), "continuation plan changed")
        validate_original(plan)
        if (out / "status.json").exists():
            return plan
    else:
        cutoff = stamp(before)
        status = json.loads((Path(campaign_root) / "status.json").read_text())
        selected = [j for j in original["jobs"] if j["prefix"] == 64
                    and status["jobs"][j["key"]]["state"] == "unresolved"
                    and stamp(status["jobs"][j["key"]]["attempts"][-1]["finished"]) <= cutoff]
        require(len(selected) == 36, "the fixed initial continuation must contain exactly 36 old failures")
        campaign.verify_assets(original, original["global_paths"])
        cases = []
        for job in selected:
            files, lock = filenames(original, job)
            with campaign.campaign_lock(lock):
                require(campaign.report_state(original, job) == "unresolved", "case is no longer unresolved")
                report = json.loads(files[0].read_text())
                require(report["history"][-1]["draws"] == 65536, "unexpected original draw count")
                require(campaign.digest(files[1]) == report["checkpoint_sha256"]
                        and campaign.digest(files[2]) == report["moments_sha256"], "original reference checksum failed")
                cases.append(dict(job=job, reference_id=report["reference_id"],
                                  original_files={str(p): campaign.digest(p) for p in files}))
        plan = dict(schema=SCHEMA, before=before, original_plan=str(plan_path),
                    original_plan_sha256=campaign.digest(plan_path), continuation_code_sha256=campaign.digest(__file__),
                    references=str(out / "references"), total_draws_per_chain=TOTAL_DRAWS,
                    maximum_workers=2, maximum_global_workers=12, cases=cases,
                    no_auto_retry=True, no_auto_stage_advance=True,
                    interpretation="Same saved chains, seeds, warmup and numerical target. Original reports remain unchanged. "
                    "A copied report retains its original checkpoint path until the continuation worker finishes; "
                    "only completed new reports may enter the later complete-cohort audit.")
        plan["plan_id"] = campaign.fingerprint(plan)
        campaign.atomic_json(existing, plan)
    for case in plan["cases"]:
        files, lock = filenames(original, case["job"])
        with campaign.campaign_lock(lock):
            for source in files:
                expected = case["original_files"][str(source)]
                require(campaign.digest(source) == expected, "original changed during preparation")
                target = Path(plan["references"]) / source.relative_to(refs)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    require(campaign.digest(target) == expected, "partially prepared copy differs")
                else:
                    temporary = target.with_name(target.name + ".copying")
                    shutil.copy2(source, temporary)
                    require(campaign.digest(temporary) == expected, "copied reference checksum failed")
                    os.replace(temporary, target)
    state = dict(schema=SCHEMA, plan_id=plan["plan_id"], jobs={c["job"]["key"]:
                 dict(state="pending", attempts=[]) for c in plan["cases"]})
    campaign.atomic_json(out / "status.json", state)
    return plan


def worker_command(original, plan, job):
    adapted = dict(original, references=plan["references"], worker=dict(original["worker"],
                   initial_draws=TOTAL_DRAWS, max_draws=TOTAL_DRAWS))
    return campaign.command(adapted, job)


def run(out, max_hours=12., poll_seconds=5.):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    require(plan.get("schema") == SCHEMA and plan.get("plan_id") == campaign.fingerprint(
        {k: v for k, v in plan.items() if k != "plan_id"}), "continuation plan checksum failed")
    original = validate_original(plan)
    adapted = dict(original, references=plan["references"])
    state = json.loads((out / "status.json").read_text())
    cases = {c["job"]["key"]: c for c in plan["cases"]}
    require(state["plan_id"] == plan["plan_id"] and set(state["jobs"]) == set(cases), "status differs from fixed plan")
    # The outer dispatcher lock prevents two owners; live orphan workers keep their per-case locks.
    for key, item in state["jobs"].items():
        _, lock = campaign.job_paths(adapted, cases[key]["job"])
        require(not campaign.lock_busy(lock), "a previous continuation worker is still running")
        if item["state"] == "running":
            item["state"] = "pending"
            item["attempts"][-1]["interrupted"] = True
    active, stop = {}, []
    started = time.monotonic()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda signum, frame: stop.append("signal; active workers allowed to finish"))
    def save():
        state.update(updated=campaign.utc(), counts=dict(Counter(x["state"] for x in state["jobs"].values())))
        campaign.atomic_json(out / "status.json", state)
    try:
        while True:
            for key, (process, stream, began) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                item = state["jobs"][key]
                item["attempts"][-1].update(exit_code=code, finished=campaign.utc(), wall_seconds=time.monotonic()-began)
                try:
                    outcome = campaign.report_state(adapted, cases[key]["job"])
                    report_path, _ = campaign.job_paths(adapted, cases[key]["job"])
                    report = json.loads(report_path.read_text())
                    require(report["reference_id"] == cases[key]["reference_id"]
                            and report["history"][-1]["draws"] == TOTAL_DRAWS, "continued reference identity or budget changed")
                    checkpoint = report_path.with_name(report_path.stem + "_chains.npz")
                    moments = report_path.with_name(report_path.stem + "_moments.npz")
                    require(report["checkpoint"] == str(checkpoint.resolve())
                            and campaign.digest(checkpoint) == report["checkpoint_sha256"]
                            and campaign.digest(moments) == report["moments_sha256"], "continued checkpoint or moments changed")
                    item["state"] = "ready" if code == 0 and outcome == "ready" else "unresolved" if code == 2 else "failed"
                except (ValueError, OSError, KeyError) as error:
                    item.update(state="failed", error=str(error))
                del active[key]
            if time.monotonic() - started >= max_hours * 3600:
                stop.append("dispatch time budget reached")
            if shutil.disk_usage(out).free < 50e9:
                stop.append("50 GB disk reserve reached")
            old_running = sum(campaign.lock_busy(p) for p in Path(original["references"]).glob("a1_L*/alpha*/seq*_t*.lock"))
            state.update(original_active_workers=old_running, active_workers=len(active),
                         stop_reason=stop[0] if stop else None)
            if not stop:
                for key, case in cases.items():
                    if len(active) >= 2 or len(active) + old_running >= 12:
                        break
                    item = state["jobs"][key]
                    if item["state"] != "pending":
                        continue
                    job = case["job"]
                    campaign.verify_assets(original, original["global_paths"] + original["cases"][f"a1_L{job['depth']}/{job['index']}"]["files"])
                    path = out / "logs" / (key.replace("/", "_") + f"_attempt{len(item['attempts'])+1:03d}.log")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = path.open("x")
                    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                               **{k: v for k, v in original["worker"]["environment"].items() if v is not None})
                    process = subprocess.Popen(worker_command(original, plan, job), stdout=stream,
                        stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
                    item.update(state="running", pid=process.pid)
                    item["attempts"].append(dict(started=campaign.utc(), pid=process.pid, log=str(path.resolve())))
                    active[key] = process, stream, time.monotonic()
                    save()
            save()
            if not active and (stop or all(x["state"] != "pending" for x in state["jobs"].values())):
                break
            time.sleep(poll_seconds)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return state["counts"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--before", help="explicit timestamp freezing the original failed-case set")
    parser.add_argument("--max-hours", type=float, default=12.)
    args = parser.parse_args(argv)
    require(0 < args.max_hours <= 12, "dispatch budget must be in (0,12] hours")
    require(not args.prepare or args.before is not None, "--prepare requires --before")
    with campaign.campaign_lock(args.out / "dispatcher.lock"):
        if args.prepare:
            plan = prepare(args.campaign, args.out, args.before)
            print(json.dumps(dict(prepared=len(plan["cases"]), plan_id=plan["plan_id"], launched=0)))
        else:
            print(json.dumps(run(args.out, args.max_hours)))


if __name__ == "__main__":
    main()

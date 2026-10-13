"""Bounded, population-only Jed submission after the t=16 calibration recovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "code/transformer/configs/pooled_mean_population_plan_2026-09-20.json"
OUT = ROOT / "output/pooled_mean_population_resume_2026-09-22"
CODE = ("code/scripts/submit_pooled_mean_population_resume.py",
        "code/scripts/pooled_mean_population_resume.py",
        "code/transformer/scitas/pooled_mean_population_resume.sbatch",
        "code/scripts/pooled_mean_population.py",
        "code/transformer/scitas/pooled_mean_population.sbatch")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_paths(args):
    for name in ("plan", "raw", "dependencies", "recovery", "out"):
        path = getattr(args, name).expanduser().resolve()
        require(not any(c in str(path) for c in "\n\r\t"), "Path contains control characters")
        setattr(args, name, path)
    protected = [args.raw, args.dependencies, args.recovery,
                 *[ROOT / name for name in ("code", "results", "notes", "figures", ".git")]]
    require(args.out != ROOT and ROOT in args.out.parents
            and all(args.out != path and args.out not in path.parents
                    and path not in args.out.parents for path in protected),
            "Output overlaps frozen input or repository sources")


def preflight(args):
    """Small, read-only checks suitable for a login node; full replay is in prepare."""
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "pooled-mean-population-plan-v1"
            and plan["observed"] == 16 and plan["repetitions"] == 32
            and plan["scheduler"]["population"]["tasks"] == 768
            and plan["scheduler"]["population"]["allocation_minutes"] == 240
            and plan["scheduler"]["concurrency"] == 64,
            "Population scope or bounds changed")
    require(plan["automatic_retries"] is False, "Automatic retries forbidden")
    for name, expected in plan["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen numerical code changed: " + name)
    receipt = json.loads((args.recovery / "summary.json").read_text())
    prepared = args.recovery / "recovery_plan.json"
    require(receipt["schema"] == "pooled-mean-calibration-recovery-summary-v1"
            and receipt["complete"] is True and receipt["passed"] is True
            and receipt["population_eligible"] is True
            and receipt["recovered_repetitions"] == 141
            and receipt["population_submitted"] is False
            and sha(prepared) == receipt["recovery_plan_sha256"],
            "Recovery is not a complete, passed calibration")
    rplan = json.loads(prepared.read_text())
    require(rplan["plan_id"] == receipt["plan_id"]
            and rplan["original_plan_sha256"] == sha(args.plan)
            and rplan["raw"] == str(args.raw)
            and rplan["dependencies"] == str(args.dependencies)
            and rplan["output"] == str(args.recovery)
            and [item["recovery_index"] for item in rplan["missing"]]
                == list(range(141)),
            "Recovery plan differs from selected inputs")
    require(set(receipt["recovery_result_sha256"]) ==
            {str(i) for i in range(141)}, "Recovered repetition coverage changed")
    for index, expected in receipt["recovery_result_sha256"].items():
        path = args.recovery / "repetitions" / f"repetition-{int(index):04d}.json"
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "Recovered repetition changed: " + index)
    binding = receipt["overlay"]
    overlay = args.recovery / "overlay"
    require(binding["path"] == str(overlay) and
            set(binding["task_sha256"]) == {
                f"calibration/tasks/task-{i:04d}.json" for i in range(52)},
            "Calibration task coverage changed")
    hashes = {"plan.json": binding["plan_sha256"],
              "prepared.json": binding["prepared_sha256"],
              "calibration/summary.json": binding["calibration_summary_sha256"],
              **binding["task_sha256"]}
    for name, expected in hashes.items():
        path = overlay / name
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "Calibration overlay changed: " + name)
    require(binding["plan_sha256"] == sha(args.plan), "Overlay plan changed")
    calibration = json.loads((overlay / "calibration/summary.json").read_text())
    require(calibration["schema"] == "pooled-mean-stage-v1"
            and calibration["stage"] == "calibration"
            and calibration["plan_sha256"] == binding["plan_sha256"]
            and calibration["prepared_sha256"] == binding["prepared_sha256"]
            and calibration["complete"] is True and calibration["passed"] is True
            and calibration["missing_tasks"] == []
            and set(calibration["task_sha256"]) == set(binding["task_sha256"])
            and calibration["task_sha256"] == binding["task_sha256"],
            "Complete calibration gate absent")
    for script in ("pooled_mean_population_resume.sbatch", "pooled_mean_population.sbatch"):
        source = (ROOT / "code/transformer/scitas" / script).read_text()
        directives = dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)\s*$", source, re.M))
        require(all(directives.get(k) == "1" for k in ("nodes", "ntasks", "cpus-per-task"))
                and directives.get("mem") == "6G", "Resource directives changed")
    hours = 1 + 768 * 4 + 1
    cost = hours * plan["scheduler"]["cpu_hour_rate_chf"]
    require(math.isfinite(cost) and 0 < cost <= args.max_cost_chf
            and hours <= args.max_cpu_hours, "Population exceeds reviewed cap")
    return hours, cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--raw", type=Path, default=Path.home() /
                        "Projects/layered_architectures_multifamily/output/multifamily_2026-09-16")
    parser.add_argument("--dependencies", type=Path, default=Path.home() /
                        "Projects/shared_base_dependencies_2026-09-16")
    parser.add_argument("--recovery", type=Path, required=True,
                        help="Successful recovery output; never the failed default attempt")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-cpu-hours", type=float, required=True)
    parser.add_argument("--max-cost-chf", type=float, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    require(all(math.isfinite(x) and x > 0 for x in
                (args.max_cpu_hours, args.max_cost_chf)), "Caps must be positive and finite")
    checked_paths(args)
    hours, cost = preflight(args)
    identity = dict(plan_sha256=sha(args.plan), plan=str(args.plan), raw=str(args.raw),
                    dependencies=str(args.dependencies), recovery=str(args.recovery),
                    recovery_summary_sha256=sha(args.recovery / "summary.json"),
                    recovery_plan_sha256=sha(args.recovery / "recovery_plan.json"),
                    overlay_calibration_sha256=sha(
                        args.recovery / "overlay/calibration/summary.json"),
                    out=str(args.out), maximum_cpu_hours=hours,
                    maximum_cost_chf=cost, approved_cpu_hour_cap=args.max_cpu_hours,
                    approved_cost_cap_chf=args.max_cost_chf,
                    code_sha256={name: sha(ROOT / name) for name in CODE})
    print(json.dumps(dict(stage="planned", population_tasks=768,
        maximum_simultaneous_workers=64, maximum_cpu_hours=hours,
        maximum_cost_chf=cost, automatic_retries=False,
        calibration_resubmitted=False)), flush=True)
    wrapper = ROOT / "code/transformer/scitas/pooled_mean_population_resume.sbatch"
    original = ROOT / "code/transformer/scitas/pooled_mean_population.sbatch"

    def command(stage, dependency=None):
        cmd = ["sbatch", "--parsable", "--chdir=" + str(ROOT),
               "--job-name=pmean16-resume-" + stage,
               "--time=" + ("04:00:00" if stage == "population" else "01:00:00"),
               "--output=" + str(args.out / "logs" / (stage + "-%A_%a.out"))]
        if dependency:
            cmd += ["--dependency=" + ("afterany:" if stage == "merge" else "afterok:")
                    + dependency, "--kill-on-invalid-dep=yes"]
        if stage == "population":
            cmd += ["--array=0-767%64"]
            cmd += [str(original), str(args.plan), str(args.raw),
                    str(args.dependencies), str(args.out), "population", "-", "0"]
        else:
            cmd += [str(wrapper), str(args.plan), str(args.raw),
                    str(args.dependencies), str(args.recovery), str(args.out), stage]
        return cmd

    if args.dry_run:
        for stage, dependency in (("prepare", None), ("population", "DRY_prepare"),
                                  ("merge", "DRY_population")):
            print(shlex.join(command(stage, dependency)), flush=True)
        return 0
    # The recovery, not the chosen output name, owns the one-use spending claim.
    source_lock = args.recovery / ".population_submission.lock"
    source_claim = args.recovery / "population_submission.json"
    require(not source_lock.is_symlink() and not source_claim.is_symlink(),
            "Recovery submission claim is linked")
    with source_lock.open("a+") as claim_lock:
        fcntl.flock(claim_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if source_claim.exists():
            old = json.loads(source_claim.read_text())
            require(old["identity"] == identity, "Recovery already claimed by another output")
            path = args.out / "submission.json"
            require(path.is_file() and not path.is_symlink(),
                    "Recovery claim lacks matching output receipt")
            copy = json.loads(path.read_text())
            require(old == copy and old["status"] == "submitted"
                    and old["submitting"] is None,
                    "Prior population submission uncertain; inspect scheduler before any retry")
            print(json.dumps(dict(action="already_submitted", jobs=old["jobs"])), flush=True)
            return 0
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / ".submission.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = args.out / "submission.json"
            require(set(args.out.iterdir()) == {args.out / ".submission.lock"},
                    "Nonempty output has no submission receipt")
            (args.out / "logs").mkdir()
            receipt = dict(schema="pooled-mean-population-resume-submission-v1",
                           identity=identity, status="preparing", submitting=None,
                           jobs=[], automatic_retries=False, calibration_resubmitted=False)

            def save():
                receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
                for destination in (source_claim, path):
                    temp = destination.with_name("." + destination.name + ".tmp." + str(os.getpid()))
                    with temp.open("w") as stream:
                        json.dump(receipt, stream, indent=2, sort_keys=True)
                        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temp, destination)

            def dispatch(stage, dependency=None):
                cmd = command(stage, dependency)
                receipt["submitting"] = dict(stage=stage, command=cmd)
                receipt["status"] = "submitting"; save()
                print(shlex.join(cmd), flush=True)
                result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
                job = result.stdout.strip().split(";")[0]
                require(job.isdigit(), "Uncertain scheduler response; inspect before retry")
                receipt["jobs"].append(dict(stage=stage, job_id=job))
                receipt["submitting"] = None; save()
                return job

            save()  # Freeze the recovery-side claim before the first sbatch.
            prepare = dispatch("prepare")
            population = dispatch("population", prepare)
            dispatch("merge", population)
            receipt["status"] = "submitted"; save()
            print(json.dumps(dict(stage="submitted", jobs=receipt["jobs"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

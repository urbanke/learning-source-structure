"""Focused tests for the immutable pooled-mean calibration recovery."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def claim(path):
    marker = Path(path).with_suffix(".writer")
    marker.mkdir()
    try:
        yield
    finally:
        marker.rmdir()


fake = types.ModuleType("pooled_mean_population")
fake.require, fake.sha, fake.digest, fake.save = require, sha, digest, save
fake.cal = types.SimpleNamespace(writer_claim=claim)
fake.old = types.SimpleNamespace(heartbeat=lambda *args, **kwargs: claim(Path(args[0]).with_suffix(".heartbeat")))
fake.ARMS = {"A": dict(particles=1, temperatures=1, moves=1, alpha_order=1,
                        row_bounds=[-1., 1.], row_step=.5)}
fake.leggauss = lambda order: (np.asarray([0.]), np.asarray([2.]))
fake.RankedLSARowEvidence = lambda *args, **kwargs: object()
fake.SharedBaseMCMC = lambda *args, **kwargs: object()

previous = sys.modules.get("pooled_mean_population")
sys.modules["pooled_mean_population"] = fake
spec = importlib.util.spec_from_file_location(
    "pooled_mean_calibration_recovery_under_test",
    SOURCE_ROOT / "code/scripts/recover_pooled_mean_calibration.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
if previous is None:
    del sys.modules["pooled_mean_population"]
else:
    sys.modules["pooled_mean_population"] = previous


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    for relative in recovery.CODE:
        destination = root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative, destination)
    monkeypatch.setattr(recovery, "ROOT", root)
    original_plan = root / "original-plan.json"
    scientific_plan = dict(repetitions=32, seconds_per_repetition=7200, worker_margin_seconds=300,
        term_cache_size=4096,
        scheduler=dict(cpu_hour_rate_chf=.0055))
    save(original_plan, scientific_plan)
    plan_hash = sha(original_plan)
    specs = [dict(task_index=i, stage="calibration", observed=4, reference_key=f"r{i}",
                  arm="A", tokens=[i % 2]) for i in range(52)]
    prepared = dict(specs=specs)
    parent_out = root / "output/parent"; save(parent_out / "prepared.json", prepared)
    prepared_hash = sha(parent_out / "prepared.json")
    reports = {}
    for item in specs:
        index = item["task_index"]
        count = recovery.EXPECTED_PREFIX_COUNTS.get(index, 32)
        records = [dict(seed=100000 + index * 100 + j, repetition=j) for j in range(count)]
        report = dict(schema="pooled-mean-task-v1", plan_sha256=plan_hash,
            prepared_sha256=prepared_hash, task=item, calibration_summary_sha256=None,
            records=records, records_sha256=digest(records), complete=count == 32,
            status="complete" if count == 32 else "incomplete",
            scientific_reference_validated=False)
        path = parent_out / f"calibration/tasks/task-{index:04d}.json"
        save(path, report); reports[index] = report
    summary = dict(schema="pooled-mean-stage-v1", stage="calibration",
        plan_sha256=plan_hash, prepared_sha256=prepared_hash, complete=False, passed=False,
        missing_tasks=recovery.EXPECTED_MISSING_TASKS, task_sha256={},
        scientific_reference_validated=False, by_context={})
    save(parent_out / "calibration/summary.json", summary)
    raw = root / "raw"; raw.mkdir()
    dependencies = root / "dependencies"; dependencies.mkdir()

    fake.read_plan = lambda path: (scientific_plan, sha(path), None)
    fake.read_prepared = lambda p, h, out, deps, full=False: (prepared, prepared_hash)
    fake.tasks = lambda value, stage: value["specs"]
    fake.task_path = lambda out, stage, index: Path(out) / stage / "tasks" / f"task-{index:04d}.json"
    def check_task(report, p, h, ph, item):
        require(report["plan_sha256"] == h and report["prepared_sha256"] == ph
                and report["task"] == item and report["records_sha256"] == digest(report["records"])
                and report["complete"] == (len(report["records"]) == 32), "bad task")
        for j, record in enumerate(report["records"]):
            require(record["seed"] == 100000 + item["task_index"] * 100 + j, "bad record")
    fake.check_task = check_task
    fake.seed_for = lambda h, item, j: 100000 + item["task_index"] * 100 + j
    fake.check_record = lambda record, p, h, item, j: require(
        record == dict(seed=100000 + item["task_index"] * 100 + j, repetition=j), "bad recovered record")
    def merge(plan_path, stage, raw_path, deps, out, write=True):
        out = Path(out)
        if out.resolve() == parent_out.resolve():
            return summary
        for item in specs:
            check_task(json.loads((out / f"calibration/tasks/task-{item['task_index']:04d}.json").read_text()),
                       scientific_plan, plan_hash, prepared_hash, item)
        result = dict(schema="pooled-mean-stage-v1", stage="calibration", complete=True,
                      passed=True, missing_tasks=[], task_sha256={}, scientific_reference_validated=False)
        if write:
            save(out / "calibration/summary.json", result)
        return result
    fake.merge = merge
    fake.estimate = lambda sampler, tokens, **kwargs: dict(seed=kwargs["seed"],
        repetition=(kwargs["seed"] - 100000) % 100)
    out = root / "output/recovery"
    return types.SimpleNamespace(root=root, original_plan=original_plan, parent=parent_out,
        raw=raw, dependencies=dependencies, out=out, reports=reports, specs=specs,
        prepared=prepared, plan_hash=plan_hash)


def prepare(campaign):
    return recovery.prepare(campaign.original_plan, campaign.raw, campaign.dependencies,
                            campaign.parent, campaign.out)


def test_prepare_freezes_exact_141_repetition_scope_and_budget(campaign):
    plan = prepare(campaign)
    assert len(plan["missing"]) == 141
    assert plan["missing"][0]["task_index"] == 7
    assert plan["missing"][0]["repetition"] == 20
    assert plan["missing"][-1]["task_index"] == 51
    assert plan["missing"][-1]["repetition"] == 31
    assert plan["budget"]["maximum_cpu_hours"] == pytest.approx(307.5)
    assert plan["budget"]["maximum_cost_chf"] == pytest.approx(1.69125)
    assert plan["automatic_retries"] is False and plan["population_submitted"] is False
    with pytest.raises(ValueError, match="already exists"):
        prepare(campaign)


def test_changed_parent_prefix_is_rejected(campaign):
    path = campaign.parent / "calibration/tasks/task-0007.json"
    report = json.loads(path.read_text()); report["records"].pop()
    report["records_sha256"] = digest(report["records"]); save(path, report)
    with pytest.raises(ValueError, match="prefix length changed|saved-repetition count changed"):
        prepare(campaign)


def test_worker_uses_exact_original_seed_once(campaign):
    plan = prepare(campaign)
    item = plan["missing"][0]
    assert recovery.recover(campaign.out, 0) == 0
    report = json.loads(recovery.result_path(campaign.out, 0).read_text())
    assert report["complete"] and report["record"]["repetition"] == item["repetition"]
    assert report["record"]["seed"] == 100000 + item["task_index"] * 100 + item["repetition"]
    with pytest.raises(ValueError, match="already attempted"):
        recovery.recover(campaign.out, 0)


def save_recovery_results(campaign, plan, *, omit=None, fail=None):
    for item in plan["missing"]:
        index = item["recovery_index"]
        if index == omit:
            continue
        record = None if index == fail else dict(
            seed=100000 + item["task_index"] * 100 + item["repetition"],
            repetition=item["repetition"])
        report = dict(schema=recovery.RESULT_SCHEMA, plan_id=plan["plan_id"],
            recovery_index=index, task_index=item["task_index"], repetition=item["repetition"],
            task=item["task"], complete=record is not None,
            status="complete" if record is not None else "incomplete", record=record,
            record_sha256=digest(record) if record is not None else None,
            scientific_reference_validated=False)
        save(recovery.result_path(campaign.out, index), report)


def test_merge_creates_separate_complete_overlay_and_only_then_releases_gate(campaign):
    plan = prepare(campaign); save_recovery_results(campaign, plan)
    parent_hashes = {i: sha(campaign.parent / f"calibration/tasks/task-{i:04d}.json")
                     for i in range(52)}
    result = recovery.merge(campaign.out)
    assert result["complete"] and result["passed"] and result["population_eligible"]
    assert result["population_submitted"] is False
    for i in range(52):
        overlay = json.loads((campaign.out / f"overlay/calibration/tasks/task-{i:04d}.json").read_text())
        assert len(overlay["records"]) == 32 and overlay["complete"]
        assert sha(campaign.parent / f"calibration/tasks/task-{i:04d}.json") == parent_hashes[i]


def test_incomplete_recovery_never_builds_overlay_or_releases_population(campaign):
    plan = prepare(campaign); save_recovery_results(campaign, plan, omit=3, fail=4)
    result = recovery.merge(campaign.out)
    assert not result["complete"] and not result["passed"]
    assert result["missing_recovery_indices"] == [3]
    assert result["failed_recovery_indices"] == [4]
    assert not result["population_eligible"] and not result["population_submitted"]
    assert not (campaign.out / "overlay").exists()


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "repo"; scripts = root / "code/transformer/scitas"; scripts.mkdir(parents=True)
    for relative in recovery.CODE:
        destination = root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative, destination)
    plan = root / "plan.json"; save(plan, dict(schema="pooled-mean-population-plan-v1",
        observed=16, scheduler=dict(cpu_hour_rate_chf=.0055)))
    raw = root / "raw"; (raw / "heldout").mkdir(parents=True); save(raw / "heldout/manifest.json", {})
    dependencies = root / "dependencies"; (dependencies / "tables/anchors_prod").mkdir(parents=True)
    save(dependencies / "tables/anchors_prod/anchors.json", {})
    parent_out = root / "output/parent"; save(parent_out / "prepared.json", {})
    save(parent_out / "calibration/summary.json", {})
    out = root / "output/recovery"
    binary = tmp_path / "bin"; binary.mkdir(); calls = tmp_path / "calls.jsonl"
    (binary / "sbatch").write_text("""#!/usr/bin/env python3
import json,os,pathlib,sys
p=pathlib.Path(os.environ['CALLS']); n=len(p.read_text().splitlines()) if p.exists() else 0
with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL_CALL') == str(n+1): sys.exit(1)
print(7000+n)
"""); (binary / "sbatch").chmod(0o755)
    (binary / "squeue").write_text("#!/bin/sh\nexit 0\n"); (binary / "squeue").chmod(0o755)
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"], CALLS=str(calls))
    base = ["bash", str(scripts / "submit_pooled_mean_calibration_recovery.sh"),
        "--plan", str(plan), "--raw", str(raw), "--dependencies", str(dependencies),
        "--parent", str(parent_out), "--out", str(out)]
    return types.SimpleNamespace(base=base, env=env, out=out, calls=calls)


def test_launcher_is_bounded_and_submits_no_population(launcher):
    dry = subprocess.run(launcher.base + ["--dry-run"], text=True, capture_output=True, env=launcher.env)
    assert dry.returncode == 0, dry.stderr
    assert '"maximum_cpu_hours": 307.5' in dry.stdout
    assert '"maximum_cost_chf": 1.69125' in dry.stdout
    assert not launcher.out.exists() and not launcher.calls.exists()
    run = subprocess.run(launcher.base + ["--submit", "--max-cpu-hours", "307.5",
        "--max-cost-chf", "1.69125"], text=True, capture_output=True, env=launcher.env)
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in launcher.calls.read_text().splitlines()]
    assert len(calls) == 3
    assert "--array=0-140%16" in calls[1]
    assert "--dependency=afterok:7000" in calls[1]
    assert "--dependency=afterany:7001" in calls[2]
    assert all("population" not in " ".join(call) for call in calls)
    receipt = json.loads((launcher.out / "submission.json").read_text())
    assert receipt["population_submitted"] is False and receipt["automatic_retries"] is False
    duplicate = subprocess.run(launcher.base + ["--submit", "--max-cpu-hours", "307.5",
        "--max-cost-chf", "1.69125"], text=True, capture_output=True, env=launcher.env)
    assert duplicate.returncode == 0 and "already_submitted" in duplicate.stdout
    assert len(launcher.calls.read_text().splitlines()) == 3


def test_launcher_requires_reviewed_caps(launcher):
    result = subprocess.run(launcher.base + ["--submit", "--max-cpu-hours", "295",
        "--max-cost-chf", "2"], text=True, capture_output=True, env=launcher.env)
    assert result.returncode != 0
    assert not launcher.out.exists() and not launcher.calls.exists()


def test_uncertain_partial_submission_cannot_be_repeated(launcher):
    launcher.env["FAIL_CALL"] = "2"
    first = subprocess.run(launcher.base + ["--submit", "--max-cpu-hours", "307.5",
        "--max-cost-chf", "1.69125"], text=True, capture_output=True, env=launcher.env)
    assert first.returncode != 0
    receipt = json.loads((launcher.out / "submission.json").read_text())
    assert receipt["status"] == "submitting" and receipt["submitting"]["stage"] == "recover"
    assert [job["job_id"] for job in receipt["jobs"]] == ["7000"]
    launcher.env.pop("FAIL_CALL")
    second = subprocess.run(launcher.base + ["--submit", "--max-cpu-hours", "307.5",
        "--max-cost-chf", "1.69125"], text=True, capture_output=True, env=launcher.env)
    assert second.returncode != 0 and "uncertain" in second.stderr
    assert len(launcher.calls.read_text().splitlines()) == 2

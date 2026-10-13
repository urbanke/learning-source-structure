"""The population continuation must preserve its gate and bounded scope."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/transformer")]
import pooled_mean_population_resume as worker

spec = importlib.util.spec_from_file_location(
    "population_resume_launcher", ROOT / "code/scripts/submit_pooled_mean_population_resume.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argv_for(campaign):
    return ["launch", "--plan", str(campaign.plan), "--raw", str(campaign.raw),
        "--dependencies", str(campaign.dependencies), "--recovery", str(campaign.recovery),
        "--out", str(campaign.out), "--submit", "--max-cpu-hours", "3074",
        "--max-cost-chf", "16.907"]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    scripts = {**{name: name for name in launcher.CODE},
        "code/scripts/pooled_mean_validation.py": "validation",
        "code/scripts/pooled_predictive_population.py": "old",
        "code/transformer/scitas/submit_pooled_mean_population.sh": "old-launcher"}
    for name, content in scripts.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".sbatch"):
            content = "\n".join(f"#SBATCH --{key}={value}" for key, value in
                (("nodes", 1), ("ntasks", 1), ("cpus-per-task", 1), ("mem", "6G"))) + "\n"
        path.write_text(content)
    plan_path = root / "plan.json"
    write(plan_path, dict(schema="pooled-mean-population-plan-v1", observed=16,
        repetitions=32, automatic_retries=False,
        scheduler=dict(population=dict(tasks=768, allocation_minutes=240),
                       concurrency=64, cpu_hour_rate_chf=.0055),
        code_sha256={name: sha(root / name) for name in scripts}))
    raw = root / "raw"; raw.mkdir()
    deps = root / "dependencies"; deps.mkdir()
    recovery = root / "output/recovery"; overlay = recovery / "overlay"
    rplan = dict(plan_id="plan", original_plan_sha256=sha(plan_path),
                 raw=str(raw), dependencies=str(deps), output=str(recovery),
                 missing=[dict(recovery_index=i) for i in range(141)])
    write(recovery / "recovery_plan.json", rplan)
    result_hashes = {}
    for i in range(141):
        path = recovery / "repetitions" / f"repetition-{i:04d}.json"
        write(path, dict(recovery_index=i, complete=True))
        result_hashes[str(i)] = sha(path)
    task_hashes = {}
    for i in range(52):
        name = f"calibration/tasks/task-{i:04d}.json"
        write(overlay / name, dict(task_index=i, complete=True,
                                   records=list(range(32))))
        task_hashes[name] = sha(overlay / name)
    shutil.copyfile(plan_path, overlay / "plan.json")
    write(overlay / "prepared.json", dict(dummy=True))
    write(overlay / "calibration/summary.json",
          dict(schema="pooled-mean-stage-v1", stage="calibration",
               plan_sha256=sha(plan_path),
               prepared_sha256=sha(overlay / "prepared.json"),
               complete=True, passed=True, missing_tasks=[], task_sha256=task_hashes))
    binding = dict(path=str(overlay), plan_sha256=sha(overlay / "plan.json"),
                   prepared_sha256=sha(overlay / "prepared.json"),
                   calibration_summary_sha256=sha(overlay / "calibration/summary.json"),
                   task_sha256=task_hashes)
    write(recovery / "summary.json", dict(
        schema="pooled-mean-calibration-recovery-summary-v1", plan_id="plan",
        recovery_plan_sha256=sha(recovery / "recovery_plan.json"),
        complete=True, passed=True, population_eligible=True,
        recovered_repetitions=141, population_submitted=False,
        recovery_result_sha256=result_hashes, overlay=binding))
    args = argparse.Namespace(plan=plan_path, raw=raw, dependencies=deps,
        recovery=recovery, out=root / "output/population",
        max_cpu_hours=3074, max_cost_chf=16.907, submit=True, dry_run=False)
    return args


def test_preflight_requires_exact_complete_overlay(campaign):
    assert launcher.preflight(campaign) == (3074, pytest.approx(16.907))
    path = campaign.recovery / "summary.json"
    receipt = json.loads(path.read_text()); receipt["passed"] = False
    write(path, receipt)
    with pytest.raises(ValueError, match="complete, passed"):
        launcher.preflight(campaign)


def test_preflight_rejects_changed_task_and_missing_coverage(campaign):
    task = campaign.recovery / "overlay/calibration/tasks/task-0013.json"
    task.write_text("changed\n")
    with pytest.raises(ValueError, match="overlay changed"):
        launcher.preflight(campaign)
    receipt_path = campaign.recovery / "summary.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["overlay"]["task_sha256"]["calibration/tasks/task-0013.json"]
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="coverage"):
        launcher.preflight(campaign)


def test_preflight_enforces_budget_and_memory(campaign):
    campaign.max_cost_chf = 16.90
    with pytest.raises(ValueError, match="reviewed cap"):
        launcher.preflight(campaign)
    campaign.max_cost_chf = 16.907
    script = launcher.ROOT / "code/transformer/scitas/pooled_mean_population.sbatch"
    script.write_text(script.read_text().replace("6G", "8G"))
    with pytest.raises(ValueError, match="Frozen numerical code changed"):
        launcher.preflight(campaign)


def test_submission_is_population_only_and_idempotent(campaign, monkeypatch):
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout=str(7000 + len(calls)))
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    monkeypatch.setattr(sys, "argv", argv_for(campaign))
    assert launcher.main() == 0
    assert len(calls) == 3
    assert "--array=0-767%64" in calls[1]
    assert "--dependency=afterok:7001" in calls[1]
    assert "--dependency=afterany:7002" in calls[2]
    assert not any("calibration" in " ".join(call) for call in calls)
    assert json.loads((campaign.recovery / "population_submission.json").read_text()) == json.loads(
        (campaign.out / "submission.json").read_text())
    assert launcher.main() == 0 and len(calls) == 3


def test_second_output_cannot_duplicate_population(campaign, monkeypatch):
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout=str(7100 + len(calls)))
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    monkeypatch.setattr(sys, "argv", argv_for(campaign))
    assert launcher.main() == 0 and len(calls) == 3
    campaign.out = campaign.out.with_name("second-population")
    monkeypatch.setattr(sys, "argv", argv_for(campaign))
    with pytest.raises(ValueError, match="already claimed"):
        launcher.main()
    assert len(calls) == 3 and not campaign.out.exists()


def test_partial_submission_blocks_automatic_retry(campaign, monkeypatch):
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return types.SimpleNamespace(stdout=str(7200 + len(calls)))
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    monkeypatch.setattr(sys, "argv", argv_for(campaign))
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
    claim = json.loads((campaign.recovery / "population_submission.json").read_text())
    assert claim["status"] == "submitting" and claim["submitting"]["stage"] == "population"
    with pytest.raises(ValueError, match="uncertain"):
        launcher.main()
    assert len(calls) == 2


def test_same_output_rejects_changed_recovery_source(campaign, monkeypatch):
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout=str(7300 + len(calls)))
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    monkeypatch.setattr(sys, "argv", argv_for(campaign))
    assert launcher.main() == 0
    path = campaign.recovery / "summary.json"
    receipt = json.loads(path.read_text()); receipt["new_note"] = "changed"
    write(path, receipt)
    with pytest.raises(ValueError, match="already claimed"):
        launcher.main()
    assert len(calls) == 3


def test_population_merge_rechecks_copied_source(campaign, monkeypatch):
    overlay = campaign.recovery / "overlay"
    hashes = {"prepared.json": sha(overlay / "prepared.json")}
    monkeypatch.setattr(worker, "verified_overlay", lambda *a: (overlay, hashes))
    out = campaign.out; out.mkdir(parents=True)
    write(out / "calibration_source.json", dict(
        schema="pooled-mean-population-resume-source-v1",
        recovery=str(campaign.recovery), source_sha256=hashes))
    (out / "prepared.json").write_text("changed\n")
    monkeypatch.setattr(worker.population, "merge", lambda *a, **kw:
                        pytest.fail("population merger reached with changed calibration"))
    with pytest.raises(ValueError, match="calibration changed"):
        worker.merge(campaign.plan, campaign.raw, campaign.dependencies,
                     campaign.recovery, out)


def test_worker_replays_numerical_calibration_before_population(campaign, monkeypatch):
    rplan = json.loads((campaign.recovery / "recovery_plan.json").read_text())
    monkeypatch.setattr(worker.recovery, "load_plan", lambda *a, **k: rplan)
    monkeypatch.setattr(worker.recovery, "check_result", lambda report, plan, item:
                        pytest.fail("bad recovery identity") if
                        report["recovery_index"] != item["recovery_index"] else None)
    saved = json.loads((campaign.recovery / "overlay/calibration/summary.json").read_text())
    monkeypatch.setattr(worker.population, "merge", lambda *a, **k: {**saved, "passed": False})
    with pytest.raises(ValueError, match="does not replay"):
        worker.verified_overlay(campaign.plan, campaign.raw, campaign.dependencies,
                                campaign.recovery)
    monkeypatch.setattr(worker.population, "merge", lambda *a, **k: saved)
    monkeypatch.setattr(worker.population, "calibration_gate", lambda *a: "gate")
    overlay, hashes = worker.verified_overlay(campaign.plan, campaign.raw,
                                               campaign.dependencies, campaign.recovery)
    assert overlay == campaign.recovery / "overlay" and len(hashes) == 55

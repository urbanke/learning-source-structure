"""Continuation orchestration tests with tiny synthetic files; no sampling."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_shared_base_curve_failures as continuation

CUTOFF = "2026-09-16T19:30:49+00:00"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    campaign, refs, out = (tmp_path / name for name in ("campaign", "original", "continued"))
    campaign.mkdir()
    jobs = [dict(key=f"a1_L4/seq_{i:06d}/t0064/alpha16", depth=4, index=i, prefix=64, order=16)
            for i in range(37)]
    plan = dict(references=str(refs), source_root=str(tmp_path / "sources"), jobs=jobs,
                global_paths=[], assets={}, cases={f"a1_L4/{i}": dict(files=[]) for i in range(37)},
                worker=dict(python="original-python", script="unchanged-worker", initial_draws=4096,
                            max_draws=65536, seed=20261006, burn=16384, environment={}))
    states = {}
    for job in jobs:
        files, _ = continuation.filenames(plan, job)
        files[0].parent.mkdir(parents=True, exist_ok=True)
        files[1].write_bytes(b"saved chains and RNG")
        files[2].write_bytes(b"saved moments")
        report = dict(history=[dict(draws=65536)], reference_id=f"fixed-{job['index']}",
                      checkpoint=str(files[1]), checkpoint_sha256=continuation.campaign.digest(files[1]),
                      moments_sha256=continuation.campaign.digest(files[2]), ready_for_cohort_check=False)
        files[0].write_text(json.dumps(report))
        states[job["key"]] = dict(state="unresolved", attempts=[dict(finished=
            "2026-09-16T20:00:00+00:00" if job["index"] == 36 else "2026-09-16T18:00:00+00:00")])
    (campaign / "plan.json").write_text(json.dumps(plan))
    (campaign / "status.json").write_text(json.dumps(dict(jobs=states)))
    monkeypatch.setattr(continuation.campaign, "report_state", lambda p, j:
        "ready" if json.loads(continuation.campaign.job_paths(p, j)[0].read_text())["ready_for_cohort_check"] else "unresolved")
    return campaign, refs, out, plan


def test_prepares_exact_old_failure_set_and_preserves_originals(setup):
    campaign, refs, out, original = setup
    before = {p: p.read_bytes() for p in refs.rglob("*") if p.is_file()}
    plan = continuation.prepare(campaign, out, CUTOFF)
    assert len(plan["cases"]) == 36
    assert all(c["job"]["index"] < 36 for c in plan["cases"])
    assert all(p.read_bytes() == value for p, value in before.items())
    for case in plan["cases"]:
        for source, sha in case["original_files"].items():
            target = Path(plan["references"]) / Path(source).relative_to(refs)
            assert continuation.campaign.digest(target) == sha
    assert continuation.prepare(campaign, out, CUTOFF) == plan
    command = continuation.worker_command(original, plan, plan["cases"][0]["job"])
    assert command[:2] == ["original-python", "unchanged-worker"]
    for flag, expected in (("--initial-draws", "131072"), ("--max-draws", "131072"),
                           ("--burn", "16384"), ("--seed", "20261006"), ("--proposal", "iid"), ("--steps", "64")):
        assert command[command.index(flag)+1] == expected


def test_rejects_changed_original_and_unsafe_or_drifting_selection(setup):
    campaign, refs, out, _ = setup
    with pytest.raises(ValueError, match="separate"):
        continuation.prepare(campaign, refs / "bad", CUTOFF)
    with pytest.raises(ValueError, match="exactly 36"):
        continuation.prepare(campaign, out, "2026-09-16T21:00:00+00:00")
    plan = continuation.prepare(campaign, out, CUTOFF)
    source = Path(next(iter(plan["cases"][0]["original_files"])))
    source.write_text("changed original")
    with pytest.raises(ValueError, match="original reference changed"):
        continuation.validate_original(plan)


def test_dispatch_reuses_worker_keeps_failures_and_does_not_retry(setup, monkeypatch):
    campaign, refs, out, original = setup
    plan = continuation.prepare(campaign, out, CUTOFF)
    launched, live = [], []
    def fake_popen(command, **kwargs):
        assert len(live) < 2
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == kwargs["env"]["OMP_NUM_THREADS"] == "1"
        index = int(command[command.index("--index")+1])
        launched.append(index)
        live.append(index)
        job = original["jobs"][index]
        path, _ = continuation.campaign.job_paths(dict(original, references=plan["references"]), job)
        report = json.loads(path.read_text())
        report["checkpoint"] = str(path.with_name(path.stem + "_chains.npz"))
        report["history"].append(dict(draws=131072))
        report["ready_for_cohort_check"] = index != 0
        path.write_text(json.dumps(report))
        def poll():
            live.remove(index)
            return 2 if index == 0 else 0
        return SimpleNamespace(pid=1000+index, poll=poll)
    monkeypatch.setattr(continuation.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(continuation.time, "sleep", lambda _: None)
    monkeypatch.setattr(continuation.shutil, "disk_usage", lambda _: SimpleNamespace(free=100e9))
    old_locks = {continuation.campaign.job_paths(original, j)[1] for j in original["jobs"][:10]}
    monkeypatch.setattr(continuation.campaign, "lock_busy", lambda p: p in old_locks)
    result = continuation.run(out)
    assert result == {"ready": 35, "unresolved": 1}
    assert sorted(launched) == list(range(36))
    assert continuation.run(out) == result
    assert len(launched) == 36
    original_failed = json.loads(continuation.campaign.job_paths(original, original["jobs"][0])[0].read_text())
    assert original_failed["history"] == [dict(draws=65536)]
    state = json.loads((out / "status.json").read_text())
    assert state["original_active_workers"] == 10


def test_refuses_orphan_worker_and_changed_plan(setup, monkeypatch):
    campaign, refs, out, original = setup
    plan = continuation.prepare(campaign, out, CUTOFF)
    monkeypatch.setattr(continuation.campaign, "lock_busy", lambda _: True)
    with pytest.raises(ValueError, match="still running"):
        continuation.run(out)
    plan["total_draws_per_chain"] = 999
    (out / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksum"):
        continuation.run(out)

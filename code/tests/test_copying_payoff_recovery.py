"""Lifecycle and provenance tests; no production sampler is launched."""
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import copying_payoff_recovery as recovery


def task(observed=7, name="t0007_453a8f0bdcd20ca9731b_a32", order=32, index=87):
    return dict(task_index=index, id=name, reference_key=name.rsplit("_a", 1)[0],
        tokens=[0] + [1] * (observed - 1), time=observed, order=order,
        target_draws=recovery.BASELINE_DRAWS, kind="fresh", seed=1234 + observed)


def test_fixed_scope_and_resource_contract():
    assert [(time, name) for time, name, _ in recovery.TARGETS] == [
        (7, "t0007_453a8f0bdcd20ca9731b_a32"),
        (8, "t0008_fccb61c7bef875d43ac7_a16")]
    assert recovery.BASELINE_DRAWS == 16384 and recovery.TARGET_DRAWS == 65536
    assert recovery.TASK_WALL_SECONDS == 900
    assert recovery.MAX_RSS_BYTES == 6 * 2**30
    assert recovery.DEFAULT_OUT.name == "copying_payoff_recovery_procinfo_2026-09-21"


@pytest.fixture
def parent_record(tmp_path, monkeypatch):
    item = task()
    folder = tmp_path / "tasks" / item["id"]
    folder.mkdir(parents=True)
    files = {name: folder / filename for name, filename in dict(
        report="report.json", identity="identity.json", audit="raw_audit.json",
        attempt="attempt.json", checkpoint="chains.npz", moments="moments.npz").items()}
    files["checkpoint"].write_bytes(b"checkpoint")
    files["moments"].write_bytes(b"moments")
    identity = dict(plan_id="parent-plan", task=item, target={"same": True},
                    proposal={"same": True}, runtime={"same": True})
    recovery.write(files["identity"], identity)
    gates = deepcopy(recovery.TARGETS[0][2])
    audit = dict(passed=False, classic_gates=gates, corrected_gates=gates,
        final_state_grid_checks=[dict(passed=True)],
        raw_inputs={"canonical_accuracy:checkpoint": {"sha256": recovery.sha(files["checkpoint"])}})
    recovery.write(files["audit"], audit)
    report = dict(schema=recovery.dense.SCHEMA, plan_id="parent-plan", task=item,
        complete=True, audit_passed=False,
        estimate={"diagnostics": dict(chains=4, burn=16384, draws=16384,
                                      thin=1, seed=item["seed"])},
        **{name + "_sha256": recovery.sha(files[name]) for name in
           ("identity", "audit", "checkpoint", "moments")})
    recovery.write(files["report"], report)
    recovery.write(files["attempt"], dict(plan_id="parent-plan", task=item,
        status="unresolved", exit_code=0))
    monkeypatch.setattr(recovery.dense, "paths", lambda out, candidate: files)
    return dict(plan=dict(output=str(tmp_path), plan_id="parent-plan"), task=item,
                gates=gates, files=files)


def test_parent_must_be_exact_declared_completed_failure(parent_record):
    p = parent_record
    result = recovery.validate_parent(p["plan"], p["task"], p["gates"])
    assert result["task"] == p["task"] and result["report"]["audit_passed"] is False
    audit = recovery.read(p["files"]["audit"])
    audit["corrected_gates"]["raw_normalization"] = False
    recovery.write(p["files"]["audit"], audit)
    with pytest.raises(ValueError, match="failure gates"):
        recovery.validate_parent(p["plan"], p["task"], p["gates"])


def test_prefix_proof_covers_every_saved_array(monkeypatch):
    parent_hashes = {f"{kind}_{chain}": f"hash-{kind}-{chain}" for chain in range(4)
                     for kind in ("predictions", "depths", "alpha_indices", "scalars")}
    parent_hashes["burn_done"] = "burn-hash"
    calls = []

    def verify(parent, child, spec, complete):
        calls.append((parent, child, spec, complete))
        return [65536] * 4

    monkeypatch.setattr(recovery, "checkpoint_helpers", lambda: (
        lambda path: [16384] * 4 if str(path) == "parent" else [65536] * 4,
        lambda path: {"array_hashes": parent_hashes},
        lambda path, name, retained: parent_hashes[name], verify))
    proof = recovery.sample_prefix_proof("parent", "child", complete=True)
    assert proof["verified"] and proof["parent_draw_counts"] == [16384] * 4
    assert proof["child_draw_counts"] == [65536] * 4
    assert len(proof["array_prefix_sha256"]) == 16
    assert proof["burn_done_sha256"] == "burn-hash"
    assert calls[0][2] == dict(baseline_draws=16384, target_draws=65536)


def test_rng_proof_attests_loaded_saved_states(monkeypatch):
    rng_states = [{"state": i} for i in range(4)]
    terminal = [dict(alpha_index=i, ranks=[i], log_likelihood=float(i),
                     log_target=float(i + 1)) for i in range(4)]
    run = SimpleNamespace(
        _rngs=[SimpleNamespace(bit_generator=SimpleNamespace(state=value)) for value in rng_states],
        _states=[SimpleNamespace(**dict(value, ranks=recovery.np.asarray(value["ranks"])))
                 for value in terminal])
    seen = []
    monkeypatch.setattr(recovery.dense.accuracy, "verify_loaded_parent",
                        lambda actual, metadata: seen.append((actual, metadata)))
    proof = recovery.rng_proof(run, dict(rng_states=rng_states, states=terminal))
    assert proof["passed"] and proof["parent_rng_states_sha256"] == proof["loaded_rng_states_sha256"]
    assert proof["parent_terminal_states_sha256"] == proof["loaded_terminal_states_sha256"]
    assert seen


def test_worker_refuses_missing_live_supervisor(tmp_path, monkeypatch):
    out = tmp_path / "recovery"
    item = task()
    plan = dict(plan_id="recovery-plan", tasks=[dict(task=item)])
    monkeypatch.setattr(recovery, "output_path", lambda value: Path(value))
    monkeypatch.setattr(recovery, "load_plan", lambda value: plan)
    p = recovery.paths(out, item)
    recovery.write(p["attempt"], dict(plan_id="recovery-plan", task=item,
        status="running", dispatcher_pid=-1, reserved_seconds=900,
        reserved_memory_bytes=6 * 2**30))
    with pytest.raises(ValueError, match="bounded supervisor"):
        recovery.worker(out, 0)
    assert not p["claim"].exists()


class FakeProcess:
    next_pid = 1000

    def __init__(self):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode


def test_supervisor_enforces_each_time_and_memory_cap(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(recovery, "rss_bytes", lambda pid: 0)
    monkeypatch.setattr(recovery, "TASK_WALL_SECONDS", 1)
    first = FakeProcess()
    monkeypatch.setattr(recovery, "stop", lambda process: setattr(process, "returncode", -15))
    assert recovery.supervise(first, 0)[:2] == (75, "time_limit")
    second = FakeProcess()
    monkeypatch.setattr(recovery, "rss_bytes", lambda pid: recovery.MAX_RSS_BYTES + 1)
    assert recovery.supervise(second, clock[0])[:2] == (76, "memory_limit")


@pytest.mark.skipif(sys.platform != "darwin", reason="direct macOS process-memory API")
def test_mac_memory_guard_needs_no_ps_subprocess(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("memory guard launched ps"))
    resident = recovery.rss_bytes(os.getpid())
    assert 0 < resident < recovery.MAX_RSS_BYTES
    assert recovery.rss_bytes(2**30 - 1) == 0


def test_run_is_sequential_one_attempt_and_cannot_restart(tmp_path, monkeypatch):
    out = tmp_path / "recovery"
    tasks = [task(), task(8, "t0008_fccb61c7bef875d43ac7_a16", 16, 104)]
    plan = dict(plan_id="recovery-plan", python=sys.executable, environment={},
                tasks=[dict(task=value) for value in tasks])
    monkeypatch.setattr(recovery, "output_path", lambda value: Path(value))
    monkeypatch.setattr(recovery, "build_plan", lambda value: plan)
    monkeypatch.setattr(recovery.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 2**30))
    live = []

    def popen(*args, **kwargs):
        process = FakeProcess()
        live.append(process)
        return process

    def supervise(process, began):
        assert all(item.returncode is not None for item in live[:-1])
        process.returncode = 0
        return 0, None, 100

    emitted = []
    monkeypatch.setattr(recovery.subprocess, "Popen", popen)
    monkeypatch.setattr(recovery, "supervise", supervise)
    monkeypatch.setattr(recovery, "overlay", lambda folder, write_manifest: emitted.append(write_manifest))
    assert recovery.run(out) == 0
    allocation = recovery.read(out / "allocation.json")
    assert allocation["status"] == "ready" and len(allocation["attempts"]) == 2
    assert all(item["reserved_seconds"] == 900 for item in allocation["attempts"])
    assert emitted == [True]
    with pytest.raises(ValueError, match="One immutable allocation"):
        recovery.run(out)


def test_overlay_is_deterministic_and_binds_receipts(tmp_path, monkeypatch):
    out = tmp_path / "recovery"
    out.mkdir()
    (out / "plan.json").write_text("{}")
    (out / "campaign.json").write_text("campaign")
    specs = []
    checked = {}
    for value in (task(), task(8, "t0008_fccb61c7bef875d43ac7_a16", 16, 104)):
        p = recovery.paths(out, value)
        for name in ("report", "identity", "audit", "checkpoint", "moments", "receipt"):
            p[name].parent.mkdir(parents=True, exist_ok=True)
            p[name].write_text(name)
        spec = dict(task=value, parent_plan={"path": "parent-plan", "sha256": "p"},
                    parent_files={"report": {"path": "parent-report", "sha256": "r"}})
        specs.append(spec)
        checked[value["id"]] = dict(paths=p, prefix={"verified": True},
                                    rng={"passed": True})
    plan = dict(parent_campaign={"path": str(out / "campaign.json"), "sha256": "campaign-hash"},
                parent_output="parent", tasks=specs)
    monkeypatch.setattr(recovery, "output_path", lambda value: Path(value))
    monkeypatch.setattr(recovery, "load_plan", lambda value: plan)
    monkeypatch.setattr(recovery, "validate_replacement",
        lambda plan, spec, require_pass: checked[spec["task"]["id"]])
    first = recovery.overlay(out, write_manifest=True)
    second = recovery.overlay(out, write_manifest=False)
    assert first == second and first["complete"] is True
    assert [item["task_id"] for item in first["replacements"]] == [value["id"] for value in
                                                                    (spec["task"] for spec in specs)]
    assert all("receipt" in item["replacement_files"] for item in first["replacements"])
    assert all(item["baseline_draws"] == 16384 and item["target_draws"] == 65536
               for item in first["replacements"])
    assert first["baseline_draws"] == 16384 and first["target_draws"] == 65536

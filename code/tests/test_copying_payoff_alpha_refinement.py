"""Focused lifecycle tests; no refinement sampler is launched."""
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import copying_payoff_alpha_refinement as refinement


def fake_task(index=0):
    observed, task_index, task_id, order = refinement.TARGETS[index]
    return dict(task_index=task_index, id=task_id, reference_key=task_id.rsplit("_a", 1)[0],
        tokens=list(range(observed)), time=observed, order=order,
        target_draws=refinement.BASELINE_DRAWS, kind="fresh", seed=1000 + index)


def test_fixed_eight_task_resource_contract():
    assert len(refinement.TARGETS) == 8
    assert [(item[0], item[3]) for item in refinement.TARGETS] == [
        (5, 16), (5, 32), (6, 16), (6, 32),
        (7, 16), (7, 32), (8, 16), (8, 32)]
    assert [item[1] for item in refinement.TARGETS] == [74, 75, 160, 161, 316, 317, 494, 495]
    assert refinement.BASELINE_DRAWS == 16384
    assert refinement.TARGET_DRAWS == 262144
    assert refinement.MAX_WORKERS == 4
    assert refinement.TASK_WALL_SECONDS == 1800
    assert refinement.RESERVED_BYTES_PER_WORKER == 5 * 2**29
    assert refinement.TOTAL_RESERVED_BYTES == 10 * 2**30
    assert refinement.DEFAULT_OUT.name == "copying_payoff_alpha_refinement_2026-09-21"


def test_prefix_proof_preserves_every_parent_array(monkeypatch):
    hashes = {f"{kind}_{chain}": f"hash-{kind}-{chain}" for chain in range(4)
              for kind in ("predictions", "depths", "alpha_indices", "scalars")}
    hashes["burn_done"] = "burn"
    calls = []

    def verify(parent, child, spec, complete):
        calls.append((spec, complete))
        return [refinement.TARGET_DRAWS] * 4

    monkeypatch.setattr(refinement, "checkpoint_helpers", lambda: (
        lambda path: [refinement.BASELINE_DRAWS] * 4,
        lambda path: {"array_hashes": hashes},
        lambda path, name, retained: hashes[name], verify))
    proof = refinement.sample_prefix_proof("parent", "child", complete=True)
    assert proof["verified"] and len(proof["array_prefix_sha256"]) == 16
    assert proof["parent_draw_counts"] == [16384] * 4
    assert proof["child_draw_counts"] == [262144] * 4
    assert calls == [(dict(baseline_draws=16384, target_draws=262144), True)]


def test_worker_requires_live_bounded_dispatcher(tmp_path, monkeypatch):
    out = tmp_path / "alpha"
    task = fake_task()
    plan = dict(plan_id="plan", tasks=[dict(task=task)])
    monkeypatch.setattr(refinement, "output_path", lambda value: Path(value))
    monkeypatch.setattr(refinement, "load_plan", lambda value: plan)
    p = refinement.paths(out, task)
    refinement.write(p["attempt"], dict(plan_id="plan", task=task, status="running",
        dispatcher_pid=-1, reserved_seconds=1800,
        reserved_memory_bytes=refinement.RESERVED_BYTES_PER_WORKER))
    with pytest.raises(ValueError, match="bounded supervisor"):
        refinement.worker(out, 0)
    assert not p["claim"].exists()


def test_prepare_is_plan_only_and_refuses_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "alpha"
    plan = dict(schema=refinement.SCHEMA, output=str(out), plan_id="fixed")
    monkeypatch.setattr(refinement, "output_path", lambda value: Path(value))
    monkeypatch.setattr(refinement, "build_plan", lambda value: plan)
    assert refinement.prepare(out) == plan
    assert refinement.read(out / "plan.json") == plan
    assert not (out / "allocation.json").exists()
    with pytest.raises(ValueError, match="new output"):
        refinement.prepare(out)


def test_overlay_contains_exactly_the_eight_bound_replacements(tmp_path, monkeypatch):
    out = tmp_path / "alpha"; out.mkdir()
    (out / "plan.json").write_text("{}")
    specs, checked = [], {}
    for index in range(8):
        task = fake_task(index); p = refinement.paths(out, task)
        for name in (*refinement.PARENT_FILES, "receipt"):
            p[name].parent.mkdir(parents=True, exist_ok=True)
            p[name].write_text(name)
        spec = dict(task=task, parent_plan={"path": "p", "sha256": "p"},
                    parent_files={"report": {"path": "r", "sha256": "r"}})
        specs.append(spec)
        checked[task["id"]] = dict(paths=p, prefix={"verified": True}, rng={"passed": True})
    plan = dict(t5_parent_plan={"path": "t5", "sha256": "t5"},
                campaign_parent={"path": "campaign", "sha256": "campaign"}, tasks=specs)
    monkeypatch.setattr(refinement, "output_path", lambda value: Path(value))
    monkeypatch.setattr(refinement, "load_plan", lambda value: plan)
    monkeypatch.setattr(refinement, "validate_replacement",
        lambda plan, spec, require_pass: checked[spec["task"]["id"]])
    first = refinement.overlay(out, write_manifest=True)
    second = refinement.overlay(out, write_manifest=False)
    assert first == second and first["complete"] is True
    assert len(first["replacements"]) == 8
    assert {item["task_id"] for item in first["replacements"]} == {
        fake_task(index)["id"] for index in range(8)}
    assert all(item["target_draws"] == 262144 for item in first["replacements"])

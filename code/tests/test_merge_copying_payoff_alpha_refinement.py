"""Scope and reconstruction tests for the targeted alpha overlay merger."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_copying_payoff_alpha_refinement as merger
import combine_copying_payoff_alpha_audit_recovery as combined


def moments():
    mean = np.full(64, 1 / 64)
    zero = np.zeros((64, 64))
    return dict(mean=mean, chain_means=np.tile(mean, (4, 1)), cov_batch=zero,
                cov_chain=zero, mcse_probs=np.zeros(64), chains=np.array(4),
                draws=np.array(16384))


def test_final_combiner_partitions_all_tasks_across_six_producers():
    parts = combined.COMPONENT_IDS
    assert len(parts) == 6
    assert sum(map(len, parts)) == len(set().union(*parts)) == 8
    assert set().union(*parts) == {item[2] for item in merger.alpha.TARGETS}
    assert combined.FIRST_IDS == {merger.alpha.TARGETS[0][2]}
    assert combined.T5_AUDIT_IDS == {merger.alpha.TARGETS[1][2]}
    assert combined.SAMPLING_IDS == {merger.alpha.TARGETS[2][2]}
    assert combined.T6_AUDIT_IDS == {merger.alpha.TARGETS[3][2]}
    assert combined.T7_T8_IDS == {item[2] for item in merger.alpha.TARGETS[4:7]}
    assert combined.T8_AUDIT_IDS == {merger.alpha.TARGETS[7][2]}
    assert combined.BASELINE_DRAWS == merger.alpha.BASELINE_DRAWS
    assert combined.TARGET_DRAWS == merger.alpha.TARGET_DRAWS


def test_merger_recognizes_new_direct_producers_and_binds_their_code(tmp_path):
    for producer in (merger.alpha_t6_audit_only, merger.alpha_t7_t8_recovery,
                     merger.alpha_t8_audit_only):
        folder = tmp_path / producer.SCHEMA
        folder.mkdir()
        (folder / "plan.json").write_text('{"schema":"' + producer.SCHEMA + '"}')
        assert merger.producer_for_overlay(folder / "overlay.json") is producer
    names = {path.name for path in merger.producer_code_paths(
        {merger.alpha_t6_audit_only, merger.alpha_t7_t8_recovery,
         merger.alpha_t8_audit_only, combined})}
    assert "copying_payoff_alpha_t6_audit_only_salvage.py" in names
    assert "copying_payoff_alpha_t7_t8_single_worker_recovery.py" in names
    assert "copying_payoff_alpha_t8_audit_only_salvage.py" in names
    assert "combine_copying_payoff_alpha_audit_recovery.py" in names


def test_final_combiner_requires_the_declared_allocation_history(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in
             ("first", "t5_audit", "sampling", "t6_audit", "t7_t8",
              "t8_audit", "unused_tail", "unused_t7_t8")}
    for root in roots.values():
        root.mkdir()
    tasks = {task_id: dict(id=task_id, time=time, order=order)
             for time, _, task_id, order in merger.alpha.TARGETS}

    def save(root, value):
        (root / "allocation.json").write_text(json.dumps(value))

    first, t5_order32, t6_order16, t6_order32 = [item[2] for item in merger.alpha.TARGETS[:4]]
    t7_t8 = [item[2] for item in merger.alpha.TARGETS[4:]]
    serial_ids, t8_order32 = t7_t8[:3], t7_t8[3]
    save(roots["first"], dict(status="failed", reason="memory_limit",
        attempts=[dict(task=tasks[first], status="ready"),
                  dict(task=tasks[t5_order32], status="memory_limit")]))
    save(roots["t5_audit"], dict(status="ready",
        attempt=dict(task=tasks[t5_order32], status="ready")))
    save(roots["sampling"], dict(status="failed", reason="memory_limit",
        attempts=[dict(task=tasks[t6_order16], status="ready"),
                  dict(task=tasks[t6_order32], status="memory_limit")],
        pending_task_indices=[2, 3, 4, 5]))
    save(roots["t6_audit"], dict(status="ready",
        attempt=dict(task=tasks[t6_order32], status="ready")))
    save(roots["t7_t8"], dict(status="failed", reason="memory_limit",
        pending_task_indices=[],
        attempts=[dict(task=tasks[task_id], status="ready") for task_id in serial_ids]
                 + [dict(task=tasks[t8_order32], status="memory_limit")]))
    save(roots["t8_audit"], dict(status="ready",
        attempt=dict(task=tasks[t8_order32], status="ready")))
    for name in ("unused_tail", "unused_t7_t8"):
        (roots[name] / "plan.json").write_text("{}")
    monkeypatch.setattr(combined.unused_tail, "load_plan", lambda root: dict(
        tasks=[dict(task=tasks[task_id]) for task_id in t7_t8[2:]]))
    monkeypatch.setattr(combined.unused_t7_t8, "load_plan", lambda root: dict(
        tasks=[dict(task=tasks[task_id]) for task_id in t7_t8]))

    def component(producer, root, expected_ids):
        items = [dict(task_id=task_id, **tasks[task_id]) for task_id in expected_ids]
        records = {task_id: dict(producer=producer) for task_id in expected_ids}
        return dict(producer_schema=producer.SCHEMA,
                    task_ids=sorted(expected_ids)), items, records

    monkeypatch.setattr(combined, "component", component)
    result, records = combined.collect(
        first=roots["first"], t5_audit=roots["t5_audit"],
        sampling=roots["sampling"], t6_audit=roots["t6_audit"],
        t7_t8=roots["t7_t8"], t8_audit=roots["t8_audit"],
        unused_tail_root=roots["unused_tail"],
        unused_t7_t8_root=roots["unused_t7_t8"])
    assert len(result["components"]) == 6
    assert len(result["unused_plans"]) == 2
    assert set(records) == combined.EXPECTED_IDS
    assert [item["task_id"] for item in result["replacements"]] == [
        item[2] for item in merger.alpha.TARGETS]

    save(roots["sampling"], dict(status="ready", attempts=[], pending_task_indices=[]))
    with pytest.raises(ValueError, match="Sampling-memory recovery"):
        combined.collect(
            first=roots["first"], t5_audit=roots["t5_audit"],
            sampling=roots["sampling"], t6_audit=roots["t6_audit"],
            t7_t8=roots["t7_t8"], t8_audit=roots["t8_audit"],
            unused_tail_root=roots["unused_tail"],
            unused_t7_t8_root=roots["unused_t7_t8"])


def test_apply_replacements_requires_one_pair_per_time(monkeypatch):
    waves, records = {}, {}
    monkeypatch.setattr(merger, "paired_check", lambda *args: {"passed": True})
    for observed in merger.TIMES:
        key = f"key-{observed}"
        refs = {16: {key: moments()}, 32: {key: moments()}}
        waves[observed] = dict(refs=refs, result=dict())
        for order in (16, 32):
            task_id = f"task-{observed}-{order}"
            task = dict(id=task_id, time=observed, order=order, reference_key=key)
            records[task_id] = dict(item=dict(task=task), moments=moments())
    old_counts = merger.dense.COUNTS
    monkeypatch.setattr(merger.dense, "COUNTS", {time: 1 for time in merger.TIMES})
    monkeypatch.setattr(merger, "EXPECTED_PAIR_COUNT", 4)
    try:
        pairs = merger.apply_replacements(waves, records)
        assert len(pairs) == 4
        assert all(len(waves[t]["result"]["alpha_refined_task_ids"]) == 2 for t in merger.TIMES)
        records.pop(next(iter(records)))
        with pytest.raises(ValueError, match="Each time"):
            merger.apply_replacements(waves, records)
    finally:
        monkeypatch.setattr(merger.dense, "COUNTS", old_counts)


class Archive(dict):
    def __enter__(self): return self
    def __exit__(self, *args): return False


def use_for(tokens):
    pattern, labels = merger.old.canonicalize(tokens)
    return dict(reference_key=merger.old.reference_key(pattern),
        canonical_to_original_seen=list(labels), queried_original_label=int(tokens[-1]),
        original_prefix_sha256=hashlib.sha256(
            np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()), dict(tokens=list(pattern))


@pytest.mark.parametrize("observed", [5, 6])
def test_diagnostic_wave_supports_t5_and_later_mapping(observed, tmp_path, monkeypatch):
    prefix = np.arange(observed) % 4
    use, canonical = use_for(prefix); key = use["reference_key"]
    names = ("a1_L0", "a1_L2", "a1_L3")
    token_row = np.zeros(1025, dtype=int); token_row[:observed] = prefix
    tokens = np.tile(token_row, (256, 1))
    transition = np.full((256, 64, 64), 1 / 64)
    stationary = np.tile(np.arange(1, 9, dtype=float), (256, 1))
    plan = dict(observed=observed, tasks=[dict(reference_key=key, **canonical)],
        uses=[dict(use, condition=name, index=index) for name in names for index in range(256)],
        source_bindings={name: dict(path=str(tmp_path / (name + ".npz")), sha256="h")
                         for name in names})
    refs = {16: {key: moments()}, 32: {key: moments()}}
    result = dict()
    monkeypatch.setattr(merger.old, "frozen_inputs", lambda plan: ({}, {}))
    monkeypatch.setattr(merger, "sha", lambda path: "h")

    def archive(path, **kwargs):
        if Path(path).parent == tmp_path:
            return Archive(tokens=tokens, transition=transition)
        return Archive(stationary_loss=stationary,
            stationary_leave_one_bank_out_loss=np.tile(stationary, (16, 1, 1)))

    monkeypatch.setattr(merger.np, "load", archive)
    monkeypatch.setattr(merger.old, "model_loss",
        lambda name, seed, role, source, bindings: np.full(256, observed + (role == "shuffled")))
    monkeypatch.setattr(merger.parent, "model_loss",
        lambda name, seed, role, source, bindings, time: np.full(256, time + (role == "shuffled")))
    arrays = merger.diagnostic_wave(plan, result, refs, {})
    assert result["diagnostic_arrays_validated"] and len(result["diagnostic_groups"]) == 3
    assert all(arrays[name + "_bayes_loss_a32"].shape == (256,) for name in names)
    plan["uses"].pop()
    with pytest.raises(ValueError, match="histories"):
        merger.diagnostic_wave(plan, result, refs, {})

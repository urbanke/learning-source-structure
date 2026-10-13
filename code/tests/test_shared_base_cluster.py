"""Portable CPU shards: exact coverage, immutable identities and evidence arithmetic."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import shared_base_cluster as cluster


READY = dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True)
NUMERICAL = dict(compiled_kernel_available=True, code_sha256={"kernel.py": "frozen"})


def write_plan(tmp_path, repetitions=16):
    source = tmp_path / "src" / "product_model_with_memory"
    source.mkdir(parents=True, exist_ok=True)
    (source / "kernel.py").write_text("fixed kernel\n")
    manifest = tmp_path / "dependencies.json"
    manifest.write_text(json.dumps(dict(schema="shared-base-dependency-manifest-v1", files={
        "src/product_model_with_memory/kernel.py": dict(sha256=cluster.file_hash(source / "kernel.py"),
                                                       size_bytes=(source / "kernel.py").stat().st_size)})))
    config = dict(purpose="exact_short", requires=[], case_ids=["a", "b"], n_tokens=5,
                  particles=8, temperatures=4, moves=2, replicates=repetitions, alpha_order=16,
                  seed=37, grid_bounds=[-32., 32.], grid_step=1/32, sampler="original")
    plan = dict(schema=cluster.PLAN_SCHEMA, prior=cluster.SUPPORTED_PRIOR,
                acceptance=dict(exact_short=cluster.SHORT_GATE, long_precision_bits=.01, production_reference_approved=False),
                dependency_manifest=str(manifest), cases=[
                    dict(id="a", name="fixed a", kind="control", tokens=[0, 1, 0, 1, 0], known_log_evidence=-10.),
                    dict(id="b", name="fixed b", kind="control", tokens=[0, 0, 0, 0, 0], known_log_evidence=-9.)],
                stages=dict(stage=config, next=dict(config, purpose="confirmation", requires=["stage"], seed=53)))
    plan["dependency_manifest_sha256"] = cluster.file_hash(manifest)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def estimate(context, case_id, replica, logz=None):
    config = context["config"]
    return dict(log_evidence=float(cluster.known_evidence(context, case_id) if logz is None else logz),
                seed=cluster.expected_seed(context, case_id, replica), sampler_class="SharedBaseMCMC",
                particles=config["particles"], temperatures=config["temperatures"], moves=config["moves"],
                n_tokens=config["n_tokens"], elapsed_seconds=1.)


def make_shard(context, case_id, start=0, stop=None, logs=None, complete=True):
    stop = context["config"]["replicates"] if stop is None else stop
    shard = cluster.new_shard(context, case_id, copy.deepcopy(NUMERICAL), start, stop, READY)
    for i in range(start, stop):
        cluster.append_record(shard, context, i, estimate(context, case_id, i, None if logs is None else logs[i-start]))
    shard.update(complete=complete, status="completed" if complete else "time_limit")
    return shard


def rehash_record(shard, index=0):
    record = shard["records"][index]
    record["record_sha256"] = cluster.fingerprint({k: v for k, v in record.items() if k != "record_sha256"})
    shard["records_sha256"] = cluster.fingerprint(shard["records"])


def test_global_seeds_and_evidence_are_independent_of_partition(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    full = [make_shard(context, cid) for cid in ("a", "b")]
    shards = [make_shard(context, cid, start, stop) for cid in ("a", "b") for start, stop in ((0, 3), (3, 11), (11, 16))]
    whole, code = cluster.merge_reports(context, full)
    partitioned, partitioned_code = cluster.merge_reports(context, list(reversed(shards)))
    assert code == partitioned_code == 0
    assert whole == partitioned
    assert whole["gate"]["status"] == "passed_exact_short"
    assert whole["gate"]["production_approved"] is False
    changed_budget = copy.deepcopy(context)
    changed_budget["config"]["particles"] *= 2
    changed_budget["stage"] = "a new stage name"
    assert cluster.expected_seed(context, "a", 10) == cluster.expected_seed(changed_budget, "a", 10)
    assert len({cluster.expected_seed(context, cid, i) for cid in ("a", "b") for i in range(16)}) == 32


def test_strict_merge_rejects_duplicates_and_marks_missing_incomplete(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    a, b = make_shard(context, "a"), make_shard(context, "b")
    with pytest.raises(ValueError, match="duplicate"):
        cluster.merge_reports(context, [a, b, a])
    partial, code = cluster.merge_reports(context, [a])
    assert code == 75 and partial["complete"] is False
    assert len(partial["missing"]) == 16 and "case_summaries" not in partial
    assert partial["gate"]["status"] == "incomplete"
    b.update(complete=False, status="time_limit")
    partial, code = cluster.merge_reports(context, [a, b])
    assert code == 75 and len(partial["unfinished_shards"]) == 1


@pytest.mark.parametrize("field", ["outer_seed", "inner_seed", "raw_hash", "array_hash", "particle_config"])
def test_corrupted_record_is_rejected_even_if_rehashed(tmp_path, field):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    shard = make_shard(context, "a")
    record = shard["records"][0]
    if field == "outer_seed":
        record["seed"] += 1
        rehash_record(shard)
    elif field == "inner_seed":
        record["estimate"]["seed"] += 1
        rehash_record(shard)
    elif field == "particle_config":
        record["estimate"]["particles"] *= 2
        rehash_record(shard)
    elif field == "raw_hash":
        record["estimate"]["log_evidence"] += 1
        shard["records_sha256"] = cluster.fingerprint(shard["records"])
    else:
        shard["records_sha256"] = "bad"
    with pytest.raises(ValueError):
        cluster.validate_shard(shard, context)


def test_resume_rejects_changed_range_case_plan_and_numerics(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    shard = make_shard(context, "a", 0, 4)
    assert cluster.validate_shard(shard, context, NUMERICAL, (0, 4)) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="boundaries"):
        cluster.validate_shard(shard, context, NUMERICAL, (0, 8))
    for changed in (dict(context, plan_sha256="new"), dict(context, config=dict(context["config"], particles=99))):
        with pytest.raises(ValueError, match="identity mismatch"):
            cluster.validate_shard(shard, changed)
    changed = copy.deepcopy(context)
    changed["cases"]["a"]["tokens"][1] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        cluster.validate_shard(shard, changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        cluster.validate_shard(shard, context, dict(NUMERICAL, kernel="changed"))
    with pytest.raises(ValueError, match="outside"):
        cluster.append_record(shard, context, 4, estimate(context, "a", 4))


def test_merge_averages_likelihood_before_log(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path, 2), "stage")
    logs = np.log([.1, .4])
    result, _ = cluster.merge_reports(context, [make_shard(context, cid, logs=logs) for cid in ("a", "b")])
    summary = result["case_summaries"]["a"]
    assert summary["loss_of_mean_evidence_bits"] == pytest.approx(2.)
    assert summary["arithmetic_mean_replicate_loss_bits_diagnostic_only"] != pytest.approx(2.)
    assert result["gate"]["status"] == "failed_exact_short"


def test_gate_scope_and_bias_are_explicit(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    reports = [make_shard(context, "a", logs=[-11.]*16), make_shard(context, "b")]
    receipt, _ = cluster.merge_reports(context, reports)
    assert receipt["gate"]["status"] == "failed_exact_short"
    assert receipt["gate"]["passed_by_case"] == dict(a=False, b=True)
    for purpose, expected in (("preflight", "preflight_only"), ("repeatability", "repeatability_only")):
        context["config"]["purpose"] = purpose
        reports = [make_shard(context, cid) for cid in ("a", "b")]
        receipt, _ = cluster.merge_reports(context, reports)
        assert receipt["gate"]["status"] == expected
        assert receipt["gate"]["accuracy_approved"] is False
        assert receipt["gate"]["production_approved"] is False
        if purpose == "repeatability":
            assert receipt["gate"]["precision_passed"] is True
        else:
            assert receipt["gate"]["runtime_ready"] is True


def test_dependency_manifest_checked_before_use(tmp_path):
    context = cluster.read_plan(write_plan(tmp_path), "stage")
    verified = cluster.verify_dependencies(context, tmp_path / "store", tmp_path / "src")
    assert len(verified) == 1
    (tmp_path / "src/product_model_with_memory/kernel.py").write_text("changed code\n")
    with pytest.raises(ValueError, match="dependency"):
        cluster.verify_dependencies(context, tmp_path / "store", tmp_path / "src")


def test_plan_pins_dependency_manifest_bytes(tmp_path):
    plan = write_plan(tmp_path)
    manifest = tmp_path / "dependencies.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash"):
        cluster.read_plan(plan, "stage")


def test_prerequisite_recomputes_gate_and_coverage(tmp_path):
    plan = write_plan(tmp_path)
    context = cluster.read_plan(plan, "stage")
    receipt, _ = cluster.merge_reports(context, [make_shard(context, cid) for cid in ("a", "b")])
    next_context = cluster.read_plan(plan, "next")
    assert cluster.validate_prerequisite_receipt(next_context, "stage", receipt)
    assert cluster.validate_merge_receipt(context, receipt)
    changed = copy.deepcopy(receipt)
    changed["case_summaries"]["a"]["log_mean_evidence_nats"] += 1
    with pytest.raises(ValueError, match="raw repetitions"):
        cluster.validate_prerequisite_receipt(next_context, "stage", changed)
    changed = copy.deepcopy(receipt)
    changed["cases"]["a"]["records"].pop()
    with pytest.raises(ValueError, match="complete"):
        cluster.validate_prerequisite_receipt(next_context, "stage", changed)
    failed, _ = cluster.merge_reports(context, [make_shard(context, cid, logs=[-20.]*16) for cid in ("a", "b")])
    assert cluster.validate_merge_receipt(context, failed)
    with pytest.raises(ValueError, match="calibration gate failed"):
        cluster.validate_prerequisite_receipt(next_context, "stage", failed)


def test_atomic_resume_reuses_global_seed_and_never_repeats_completed_records(tmp_path, monkeypatch):
    plan = write_plan(tmp_path, 4)
    context = cluster.read_plan(plan, "stage")
    numerical = dict(NUMERICAL, code_sha256={})
    sampler = SimpleNamespace(kernel=SimpleNamespace(cache_info=lambda: {}))
    monkeypatch.setattr(cluster, "numerical_setup", lambda *args: (sampler, numerical, READY))
    seen = []
    def fake_estimate(sampler, tokens, **kwargs):
        index = next(i for i in range(4) if cluster.expected_seed(context, "a", i) == kwargs["seed"])
        seen.append(index)
        if index == 2 and seen == [0, 1, 2]:
            raise TimeoutError("bounded test interruption")
        return estimate(context, "a", index)
    monkeypatch.setattr(cluster, "estimate_log_evidence", fake_estimate)
    out = tmp_path / "shard.json"
    report, code = cluster.run_shard(plan, "stage", "a", 0, 4, out, "unused", "unused", 600)
    assert code == 75 and [r["replica"] for r in report["records"]] == [0, 1]
    report, code = cluster.run_shard(plan, "stage", "a", 0, 4, out, "unused", "unused", 600)
    assert code == 0 and seen == [0, 1, 2, 2, 3]
    assert report["complete"] and cluster.validate_shard(report, context) == {0, 1, 2, 3}
    cluster.run_shard(plan, "stage", "a", 0, 4, out, "unused", "unused", 600)
    assert seen == [0, 1, 2, 2, 3]


def test_direct_run_cannot_skip_prerequisites(tmp_path, monkeypatch):
    plan = write_plan(tmp_path)
    monkeypatch.setattr(cluster, "numerical_setup", lambda *args: pytest.fail("must reject before numerical setup"))
    with pytest.raises(ValueError, match="requires --receipts"):
        cluster.run_shard(plan, "next", "a", 0, 1, tmp_path / "out.json", "unused", "unused")


def test_disk_merge_hashes_exact_input_bytes(tmp_path):
    plan = write_plan(tmp_path)
    context = cluster.read_plan(plan, "stage")
    directory = tmp_path / "shards"
    directory.mkdir()
    for cid in ("a", "b"):
        cluster.atomic_json(directory / f"{cid}.json", make_shard(context, cid))
    receipt, code = cluster.merge_shards(plan, "stage", directory, tmp_path / "merged.json")
    assert code == 0
    for path, digest in receipt["shard_sha256"].items():
        assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_concurrent_duplicate_output_lock_rejected(tmp_path):
    path = tmp_path / "shard.json"
    with cluster.output_lock(path):
        with pytest.raises(RuntimeError, match="another process"):
            with cluster.output_lock(path):
                pass


def test_committed_plan_loads_all_stages():
    path = cluster.ROOT / "code/transformer/configs/shared_base_cluster_plan_2026-09-16.json"
    plan = json.loads(path.read_text())
    for stage in plan["stages"]:
        context = cluster.read_plan(path, stage)
        assert context["config"] == plan["stages"][stage]

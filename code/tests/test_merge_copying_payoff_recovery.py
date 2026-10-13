"""Strict provenance and complete-cohort tests for the copying recovery merge."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_copying_payoff_recovery as recovery


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=recovery.sha(path))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True))


def task(time):
    task_id = recovery.EXPECTED_REPLACEMENTS[time]
    return dict(task_index=0, id=task_id, reference_key=task_id.rsplit("_a", 1)[0],
                tokens=list(range(time)), time=time,
                order=32 if time == 7 else 16, target_draws=16384,
                kind="fresh", seed=time)


def overlay_fixture(tmp_path):
    campaign = tmp_path / "campaign"
    write_json(campaign / "campaign.json", dict(schema="campaign"))
    recovery_root = tmp_path / "recovery"
    recovery_plan = dict(plan_id="recovery", output=str(recovery_root),
        parent_output=str(campaign), parent_campaign=bound(campaign / "campaign.json"),
        baseline_draws_per_chain=16384, target_draws_per_chain=65536,
        original_acceptance_criteria_unchanged=True)
    write_json(recovery_root / "plan.json", recovery_plan)
    replacements, plans = [], {}
    for time in (7, 8):
        frozen = task(time)
        wave = campaign / f"t{time}"
        write_json(wave / "plan.json", dict(plan_id=f"parent-{time}"))
        plans[time] = dict(plan_id=f"parent-{time}", tasks=[frozen])
        original = wave / "tasks" / frozen["id"]
        identity = dict(plan_id=f"parent-{time}", task=frozen,
                        target={"time": time}, proposal={"id": "same"}, runtime={"python": 1})
        audit = dict(passed=False, final_state_grid_checks=[dict(passed=True)])
        write_json(original / "identity.json", identity)
        write_json(original / "raw_audit.json", audit)
        (original / "chains.npz").write_bytes(b"parent checkpoint")
        (original / "moments.npz").write_bytes(b"parent moments")
        report = dict(plan_id=f"parent-{time}", task=frozen, complete=True,
                      audit_passed=False,
                      identity_sha256=recovery.sha(original / "identity.json"),
                      audit_sha256=recovery.sha(original / "raw_audit.json"),
                      checkpoint_sha256=recovery.sha(original / "chains.npz"),
                      moments_sha256=recovery.sha(original / "moments.npz"))
        write_json(original / "report.json", report)

        replacement = recovery_root / "tasks" / frozen["id"]
        for filename in ("identity.json", "raw_audit.json", "report.json"):
            write_json(replacement / filename, {})
        (replacement / "chains.npz").write_bytes(b"child checkpoint")
        (replacement / "moments.npz").write_bytes(b"child moments")
        write_json(replacement / "prepared.json", dict(prepared=True))
        prefix = dict(verified=True, parent_draw_counts=[16384] * 4,
                      child_draw_counts=[65536] * 4,
                      array_prefix_sha256={f"{kind}_{chain}": "a" * 64
                          for chain in range(4) for kind in
                          ("predictions", "depths", "alpha_indices", "scalars")},
                      burn_done_sha256="b" * 64)
        continuity = dict(passed=True, parent_rng_states_sha256="c" * 64,
                          loaded_rng_states_sha256="c" * 64,
                          parent_terminal_states_sha256="d" * 64,
                          loaded_terminal_states_sha256="d" * 64)
        parent_plan = bound(wave / "plan.json")
        parent_files = {name: bound(original / filename) for name, filename in dict(
                report="report.json", identity="identity.json", audit="raw_audit.json",
                checkpoint="chains.npz", moments="moments.npz").items()}
        replacement_files = {name: bound(replacement / filename) for name, filename in dict(
                report="report.json", identity="identity.json", audit="raw_audit.json",
                checkpoint="chains.npz", moments="moments.npz").items()}
        receipt = dict(task=frozen, sample_prefix=prefix, rng_continuity=continuity,
            plan_id="recovery", parent_plan_id=f"parent-{time}",
            baseline_draws_per_chain=16384, target_draws_per_chain=65536,
            sampling_complete=True, audit_passed=True, status="ready",
            original_acceptance_criteria_unchanged=True, parent_plan=parent_plan,
            parent_files=parent_files, replacement_files=replacement_files,
            prepared_sha256=recovery.sha(replacement / "prepared.json"))
        write_json(replacement / "receipt.json", receipt)
        replacements.append(dict(time=time, order=frozen["order"], task_id=frozen["id"],
            task=frozen, baseline_draws=16384, target_draws=65536,
            parent_plan=parent_plan, parent_files=parent_files,
            replacement_files={**replacement_files, "receipt": bound(replacement / "receipt.json")},
            sample_prefix=prefix, rng_continuity=continuity, audit_passed=True))
    overlay = dict(schema="copying-payoff-recovery-overlay-v1", complete=True,
        original_acceptance_criteria_unchanged=True, baseline_draws=16384, target_draws=65536,
        campaign=bound(campaign / "campaign.json"), recovery_plan=bound(recovery_root / "plan.json"),
        parent_output=str(campaign), replacements=replacements)
    overlay_path = recovery_root / "overlay.json"
    write_json(overlay_path, overlay)
    return campaign, overlay_path, plans, overlay


def test_overlay_accepts_exact_two_bound_replacements(tmp_path, monkeypatch):
    campaign, overlay_path, plans, _ = overlay_fixture(tmp_path)
    monkeypatch.setattr(recovery.dense, "load_plan",
                        lambda wave: plans[int(Path(wave).name[1:])])
    monkeypatch.setattr(recovery.parent, "campaign_binding", lambda *args: ({}, {}))
    monkeypatch.setattr(recovery, "validate_recovery_plan", lambda path: recovery.read(path))
    monkeypatch.setattr(recovery, "_check_prefix_provenance", lambda *args: None)
    overlay, by_time, bindings = recovery.load_overlay(overlay_path, campaign)
    assert overlay["complete"] and set(by_time) == {7, 8}
    assert {by_time[t]["task"]["id"] for t in by_time} == set(recovery.EXPECTED_REPLACEMENTS.values())
    assert str(overlay_path.resolve()) in bindings


@pytest.mark.parametrize("mutation", ["drop", "wrong_task", "wrong_draws", "extra"])
def test_overlay_rejects_selected_or_extra_replacements(tmp_path, monkeypatch, mutation):
    campaign, overlay_path, plans, overlay = overlay_fixture(tmp_path)
    if mutation == "drop":
        overlay["replacements"].pop()
    elif mutation == "wrong_task":
        overlay["replacements"][0]["task_id"] = "different"
    elif mutation == "wrong_draws":
        overlay["replacements"][0]["target_draws"] = 32768
    else:
        overlay["replacements"].append(deepcopy(overlay["replacements"][0]))
    write_json(overlay_path, overlay)
    monkeypatch.setattr(recovery.dense, "load_plan",
                        lambda wave: plans[int(Path(wave).name[1:])])
    monkeypatch.setattr(recovery.parent, "campaign_binding", lambda *args: ({}, {}))
    monkeypatch.setattr(recovery, "validate_recovery_plan", lambda path: recovery.read(path))
    monkeypatch.setattr(recovery, "_check_prefix_provenance", lambda *args: None)
    with pytest.raises(ValueError, match="two declared|task set"):
        recovery.load_overlay(overlay_path, campaign)


def moments(draws=65536):
    mean = np.full(64, 1 / 64)
    zero = np.zeros((64, 64))
    return dict(mean=mean, chain_means=np.tile(mean, (4, 1)), cov_batch=zero,
                cov_chain=zero, mcse_probs=np.zeros(64), chains=np.array(4),
                draws=np.array(draws), batch_length=np.array(128),
                batches_per_chain=np.array(draws // 128))


def test_continued_moments_require_exact_65536_draws():
    value = moments()
    recovery.moments_checked(value, dict(predictive_probs=value["mean"].tolist()), 8, 65536)
    value["draws"] = np.array(65535)
    with pytest.raises(ValueError, match="draw count"):
        recovery.moments_checked(value, dict(predictive_probs=value["mean"].tolist()), 8, 65536)


class Archive(dict):
    def __enter__(self): return self
    def __exit__(self, *args): return False


def reference():
    value = moments(16384)
    return {key: value[key] for key in ("mean", "chain_means", "cov_batch", "cov_chain", "mcse_probs")}


def use_for(tokens):
    pattern, labels = recovery.old.canonicalize(tokens)
    return dict(reference_key=recovery.old.reference_key(pattern),
        canonical_to_original_seen=list(labels), queried_original_label=int(tokens[-1]),
        original_prefix_sha256=hashlib.sha256(
            np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()), dict(tokens=list(pattern))


@pytest.mark.parametrize("observed", [7, 8])
def test_diagnostic_wave_keeps_all_histories_and_shared_reference_covariance(
        observed, tmp_path, monkeypatch):
    prefix = np.array([4, 2, 4, 7, 8, 2, 3, 4])[:observed]
    use, canonical = use_for(prefix)
    key = use["reference_key"]
    names = ("a1_L0", "a1_L2", "a1_L3")
    token_row = np.zeros(1025, dtype=int); token_row[:observed] = prefix
    tokens = np.tile(token_row, (256, 1))
    transition = np.full((256, 64, 64), 1 / 64)
    stationary = np.tile(np.arange(1, 9, dtype=float), (256, 1))
    plan = dict(observed=observed,
        tasks=[dict(reference_key=key, **canonical)],
        uses=[dict(use, condition=name, index=index) for name in names for index in range(256)],
        source_bindings={name: dict(path=str(tmp_path / (name + ".npz")), sha256="h")
                         for name in names})
    refs = {16: {key: reference()}, 32: {key: reference()}}
    result = dict(aggregate_precision_checked=False, population_reference_approved=False,
                  payoff_forecast_approved=False)
    monkeypatch.setattr(recovery.old, "frozen_inputs", lambda plan: ({}, {}))
    monkeypatch.setattr(recovery, "sha", lambda path: "h")
    def archive(path, **kwargs):
        if Path(path).parent == tmp_path:
            return Archive(tokens=tokens, transition=transition)
        return Archive(stationary_loss=stationary,
                       stationary_leave_one_bank_out_loss=np.tile(stationary, (16, 1, 1)))
    monkeypatch.setattr(recovery.np, "load", archive)
    monkeypatch.setattr(recovery.parent, "model_loss",
        lambda name, seed, role, source, bindings, time:
            np.full(256, time + (role == "shuffled")))
    calls = []
    original_shared_error = recovery.old.shared_error
    def checked_shared_error(oracles, keys, references):
        calls.append((oracles.shape, len(keys), len(set(keys))))
        return original_shared_error(oracles, keys, references)
    monkeypatch.setattr(recovery.old, "shared_error", checked_shared_error)
    arrays = recovery.diagnostic_wave(plan, result, refs, {})
    assert result["diagnostic_arrays_validated"] and len(result["diagnostic_groups"]) == 3
    assert calls == [((256, 64), 256, 1)] * 6
    assert all(group["n_sequences"] == 256 for group in result["diagnostic_groups"])
    assert all(arrays[name + "_bayes_loss_a32"].shape == (256,) for name in names)
    plan["uses"].pop()
    with pytest.raises(ValueError, match="histories"):
        recovery.diagnostic_wave(plan, result, refs, {})


def test_failed_global_family_never_builds_recovery_arrays(monkeypatch, tmp_path):
    waves = {time: dict(result=dict(individual_audits_passed=True,
                                    retained_parent_records=2 * recovery.dense.COUNTS[time]),
                             plan={}, refs={})
             for time in (6, 7, 8)}
    overlay = dict(campaign=dict(sha256="c"))
    monkeypatch.setattr(recovery, "load_campaign_references",
                        lambda *args: (overlay, waves, {}, False))
    monkeypatch.setattr(recovery, "diagnostic_wave",
                        lambda *args: pytest.fail("Built arrays from a failed family"))
    monkeypatch.setattr(recovery, "sha", lambda path: "o")
    result, arrays = recovery.merge(tmp_path, tmp_path / "overlay.json")
    assert arrays == {} and not result["paired_orders_passed"]
    assert not result["diagnostic_arrays_validated"]
    assert recovery.exit_status(result) == 2

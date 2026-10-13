"""Independent fixed-budget follow-up review; synthetic SMC receipts only."""
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_shared_base_abaca as review
import shared_base_cluster as cluster

READY = dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True)
NUMERICAL = dict(compiled_kernel_available=True, code_sha256={"kernel.py": "unchanged"})


def write(path, value):
    path.write_text(json.dumps(value))


def receipt(context, errors=None, numerical=None):
    context = dict(context, prerequisite_receipts={name: "frozen-prerequisite"
                   for name in context["config"].get("requires", [])})
    config, shards = context["config"], []
    for cid in config["case_ids"]:
        shard = cluster.new_shard(context, cid, numerical or NUMERICAL, 0, config["replicates"], READY)
        for replica in range(config["replicates"]):
            err = (errors or {}).get(cid, 0.)
            err = err[replica % len(err)] if isinstance(err, list) else err
            estimate = dict(log_evidence=context["cases"][cid]["known_log_evidence"] - err * math.log(2),
                seed=cluster.expected_seed(context, cid, replica), sampler_class="SharedBaseMCMC",
                particles=config["particles"], temperatures=config["temperatures"], moves=config["moves"],
                n_tokens=config["n_tokens"], elapsed_seconds=1.)
            cluster.append_record(shard, context, replica, estimate)
        shard.update(complete=True, status="completed")
        shards.append(shard)
    return cluster.merge_reports(context, shards)[0]


@pytest.fixture
def trial(tmp_path):
    old = json.loads(review.DEFAULT_ORIGINAL.read_text())
    new = json.loads(review.DEFAULT_FOLLOWUP.read_text())
    paths = {key: tmp_path / f"{key}.json" for key in
             ("original_plan", "followup_plan", "original_receipt", "followup_receipt")}
    write(paths["original_plan"], old)
    new["followup_of"].update(original_plan=str(paths["original_plan"]),
                            original_plan_sha256=cluster.file_hash(paths["original_plan"]))
    write(paths["followup_plan"], new)
    a = cluster.read_plan(paths["original_plan"], "calibrate")
    b = cluster.read_plan(paths["followup_plan"], "abaca_check")
    write(paths["original_receipt"], receipt(a, {review.ABACA: .02}))
    write(paths["followup_receipt"], receipt(b, {review.ABACA: [-.001, .009]}))
    return paths


def test_preserves_original_failure_and_separate_likelihood_means(trial):
    result = review.review(**trial)
    assert result["all_six_controls_meet_threshold_with_followup"] is True
    assert result["original_calibration_gate_preserved"] == "failed_exact_short"
    assert result["production_approved"] is False
    assert result["no_auto_promotion"] is True and result["pooled_estimates"] is False
    assert result["original_abaca"]["repetitions"] == 64
    assert result["followup_abaca"]["repetitions"] == 128
    assert result["original_abaca"]["estimated_minus_catalog_loss_bits"] == pytest.approx(.02)
    expected = -math.log2((2**.001 + 2**-.009)/2)
    assert result["followup_abaca"]["estimated_minus_catalog_loss_bits"] == pytest.approx(expected)
    assert expected != pytest.approx(.004, abs=1e-7)
    assert set(result["evidence_by_control"]) == set(review.CONTROLS)
    for cid, evidence in result["evidence_by_control"].items():
        assert evidence["source"] == ("followup" if cid == review.ABACA else "original")


def test_failed_followup_is_reported_without_promotion(trial):
    context = cluster.read_plan(trial["followup_plan"], "abaca_check")
    write(trial["followup_receipt"], receipt(context, {review.ABACA: .03}))
    result = review.review(**trial)
    assert result["all_six_controls_meet_threshold_with_followup"] is False
    assert result["sources"]["original"]["gate"]["status"] == "failed_exact_short"
    assert result["sources"]["followup"]["gate"]["status"] == "failed_exact_short"
    assert result["production_approved"] is False


@pytest.mark.parametrize("change", ["particles", "replicates", "seed", "confirmation_seed", "tokens", "reference",
                                   "threshold", "prior", "binding", "alpha_order", "extra_case"])
def test_rejects_changed_fixed_protocol(trial, change):
    path = trial["followup_plan"]
    plan = json.loads(path.read_text())
    config = plan["stages"]["abaca_check"]
    if change in {"particles", "replicates", "alpha_order"}:
        config[change] *= 2
    elif change in {"seed", "confirmation_seed"}:
        original_stage = "calibrate" if change == "seed" else "confirm"
        config["seed"] = json.loads(trial["original_plan"].read_text())["stages"][original_stage]["seed"]
    elif change == "tokens":
        plan["cases"][0]["tokens"][3] = 1
    elif change == "reference":
        plan["cases"][0]["known_log_evidence"] += .00001
    elif change == "threshold":
        plan["acceptance"]["exact_short"]["tolerance_bits"] = .0101
    elif change == "prior":
        plan["prior"]["alpha_uniform"] = [.4, 1.6]
    elif change == "binding":
        plan["followup_of"]["original_plan_sha256"] = "other"
    else:
        other = dict(plan["cases"][0], id="extra")
        plan["cases"].append(other)
        config["case_ids"].append("extra")
    write(path, plan)
    with pytest.raises(ValueError):
        review.review(**trial)


@pytest.mark.parametrize("errors", [{review.ABACA: .02, "cal_aaaaa": .02}, {}])
def test_requires_five_original_passes_and_original_abaca_failure(trial, errors):
    context = cluster.read_plan(trial["original_plan"], "calibrate")
    write(trial["original_receipt"], receipt(context, errors))
    with pytest.raises(ValueError, match="five other passing"):
        review.review(**trial)


@pytest.mark.parametrize("change", ["incomplete", "gate", "nested_stats", "summary", "seed", "numerical"])
def test_rejects_corrupt_or_incompatible_receipt(trial, change):
    path = trial["followup_receipt"]
    saved = json.loads(path.read_text())
    if change == "incomplete":
        saved["complete"] = False
    elif change == "gate":
        saved["gate"]["criteria"]["tolerance_bits"] = .02
    elif change == "nested_stats":
        saved["cases"][review.ABACA]["statistics"]["repetitions"] -= 1
    elif change == "summary":
        saved["case_summaries"][review.ABACA]["estimated_minus_catalog_loss_bits"] = 0.
    elif change == "seed":
        r = saved["cases"][review.ABACA]["records"][0]
        r["seed"] = r["estimate"]["seed"] = 4
        r["record_sha256"] = cluster.fingerprint({k: v for k, v in r.items() if k != "record_sha256"})
    else:
        context = cluster.read_plan(trial["followup_plan"], "abaca_check")
        saved = receipt(context, numerical=dict(NUMERICAL, code_sha256={"kernel.py": "changed"}))
    write(path, saved)
    with pytest.raises(ValueError):
        review.review(**trial)


def test_cli_writes_compact_summary_and_preserves_inputs(trial, tmp_path, capsys):
    before = {key: path.read_bytes() for key, path in trial.items()}
    out = tmp_path / "review.json"
    argv = [part for key, path in trial.items() for part in ("--"+key.replace("_", "-"), str(path))]
    assert review.main(argv + ["--out", str(out)]) == 0
    compact = json.loads(capsys.readouterr().out)
    assert compact["production_approved"] is False
    assert len(compact) == 6
    assert json.loads(out.read_text())["schema"] == "shared-base-abaca-review-v1"
    assert all(path.read_bytes() == before[key] for key, path in trial.items())
    with pytest.raises(SystemExit) as exc:
        review.main(argv + ["--out", str(trial["original_receipt"])])
    assert exc.value.code == 2
    assert trial["original_receipt"].read_bytes() == before["original_receipt"]


def test_rejects_cross_run_seed_collision_even_with_valid_hashes(trial, monkeypatch):
    context = cluster.read_plan(trial["followup_plan"], "abaca_check")
    old = json.loads(trial["original_receipt"].read_text())
    old_seed = old["cases"][review.ABACA]["records"][0]["seed"]
    expected_seed = cluster.expected_seed
    monkeypatch.setattr(cluster, "expected_seed", lambda ctx, cid, i:
        old_seed if ctx["stage"] == "abaca_check" and i == 0 else expected_seed(ctx, cid, i))
    write(trial["followup_receipt"], receipt(context))
    with pytest.raises(ValueError, match="reused a repetition seed"):
        review.review(**trial)

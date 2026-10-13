"""Independent generator, oracle, alignment and frozen-cohort integration checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))

from eval_mixture import (check_causality, cohort_arrays, condition_plan, create_sources,
                         draw_source, evaluate_checkpoint, heldout_seed, load_manifest, load_model,
                         scores_from_logits, summarize, validate_oracle)
from families import LSAFamily, ProductFamily


def test_declared_conditions_and_independent_case_seeds():
    plan = condition_plan()
    assert len(plan) == 23
    assert sum(item["n_seq"] for item in plan) == 8192
    assert len({item["id"] for item in plan}) == 23
    seeds = [tuple(heldout_seed(2026091602, item["id"], index))
             for item in plan for index in range(item["n_seq"])]
    assert len(set(seeds)) == 8192
    assert heldout_seed(1, "shared_base_prior", 0) != heldout_seed(2, "shared_base_prior", 0)


@pytest.mark.parametrize("family", ["shared_base", "uniform_base", "product"])
def test_source_matches_existing_generator_and_oracle(family):
    condition = next(item for item in condition_plan(2, 2) if item["id"] == family + "_prior")
    seed = heldout_seed(17, condition["id"], 0)
    record = draw_source(condition, 33, seed)
    source = (ProductFamily(8, 8, 33, ls=(1, 8)) if family == "product" else
              LSAFamily(64, 33, alpha=(.5, 1.5) if family == "shared_base" else 0., ls=(0, 8)))
    tokens, metadata = source.batch(1, np.random.default_rng(seed))
    np.testing.assert_array_equal(record["tokens"], tokens[0])
    for key, value in record["parameters"].items():
        np.testing.assert_array_equal(value, metadata[key][0])
    validate_oracle(record["tokens"], record["initial_probs"], record["transition_probs"])
    entropy = -(record["transition_probs"] * np.log2(record["transition_probs"])).sum(axis=1)
    np.testing.assert_allclose(entropy[record["tokens"][:-1]], metadata["h_true"][0], atol=6e-7, rtol=0)
    if family != "shared_base":
        np.testing.assert_allclose(record["initial_probs"], np.full(64, 1 / 64), atol=1e-15)


def test_conditional_draws_and_realized_loss_alignment():
    condition = dict(id="shared_base_a1_L0", family="shared_base", kind="conditional", alpha=1., ls=0, n_seq=2)
    arrays, params = cohort_arrays(condition, 8, [1, 4, 8], 73)
    assert all(item["alpha"] == 1. and item["ls"] == 0 for item in params)
    for i, x in enumerate(arrays["tokens"]):
        q, matrix = arrays["initial_probs"][i], arrays["transition_probs"][i]
        np.testing.assert_allclose(matrix, np.broadcast_to(q, (64, 64)), atol=1e-15)
        direct_losses = -np.log2(q[x[1:]])
        np.testing.assert_allclose(arrays["oracle_next_loss_bits"][i], direct_losses[[0, 3, 7]])
        np.testing.assert_allclose(arrays["oracle_cumulative_loss_bits"][i], direct_losses.cumsum()[[0, 3, 7]])
        replay = draw_source(condition, 9, arrays["sequence_seed"][i])
        np.testing.assert_array_equal(replay["tokens"], x)
        np.testing.assert_array_equal(replay["transition_probs"], matrix)


def test_sources_freeze_replay_and_detect_corruption(tmp_path):
    conditions = condition_plan(2, 2)[:3]
    first = create_sources(tmp_path, conditions, 8, [1, 4, 8], 99)
    second = create_sources(tmp_path, conditions, 8, [1, 4, 8], 99)
    assert first == second == load_manifest(tmp_path)
    with pytest.raises(ValueError, match="source plan changed"):
        create_sources(tmp_path, conditions, 8, [1, 4, 8], 100)
    path = tmp_path / first["cohorts"][0]["data"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manifest(tmp_path)


def test_uniform_predictions_have_exact_six_bit_losses_and_correct_kl():
    condition = condition_plan(2, 2)[0]
    arrays, _ = cohort_arrays(condition, 8, [1, 4, 8], 44)
    scores = scores_from_logits(np.zeros((2, 8, 64)), arrays["tokens"], arrays["oracle_rows"], [1, 4, 8])
    np.testing.assert_allclose(scores["next_loss_bits"], 6., atol=1e-14)
    np.testing.assert_allclose(scores["cumulative_loss_bits"], np.broadcast_to([6., 24., 48.], (2, 3)), atol=1e-13)
    np.testing.assert_allclose(scores["next_kl_bits"], 6. - arrays["source_entropy_bits"], atol=1e-14)
    selected_logits = np.zeros((2, 8, 64))
    selected_logits[:, [0, 3, 7]] = np.log(arrays["oracle_rows"])
    oracle = scores_from_logits(selected_logits, arrays["tokens"], arrays["oracle_rows"], [1, 4, 8])
    np.testing.assert_allclose(oracle["next_kl_bits"], 0., atol=2e-14)
    np.testing.assert_allclose(oracle["next_loss_bits"], arrays["oracle_next_loss_bits"], atol=2e-14)


def test_actual_gpt_causality_and_future_token_exclusion():
    torch = pytest.importorskip("torch")
    from model import GPT
    torch.set_num_threads(1)
    torch.manual_seed(719)
    model = GPT(64, 8, n_layer=1, n_head=2, d_model=8).eval()
    tokens = torch.randint(64, (2, 9))
    assert check_causality(model, tokens, [1, 4, 8])["passed"]
    with torch.inference_mode():
        original = model(tokens[:, :-1])
        altered = tokens.clone()
        altered[:, 4:] = (altered[:, 4:] + 7) % 64
        other = model(altered[:, :-1])
    np.testing.assert_allclose(original[:, :4].numpy(), other[:, :4].numpy(), atol=1e-7, rtol=0)


def test_pooled_receipt_config_and_final_checkpoint_requirement(tmp_path):
    torch = pytest.importorskip("torch")
    from model import GPT
    model = GPT(64, 8, n_layer=1, n_head=2, d_model=8)
    path = tmp_path / "model.pt"
    receipt = dict(model=model.cfg, config=dict(steps=7, seed=1))
    torch.save(dict(model=model.state_dict(), cfg=receipt, step=6), path)
    with pytest.raises(ValueError, match="predeclared final checkpoint"):
        load_model(path)
    torch.save(dict(model=model.state_dict(), cfg=receipt, step=7), path)
    loaded, metadata = load_model(path)
    assert loaded.cfg == model.cfg
    assert metadata["predeclared_final_step"] == 7
    assert metadata["config"]["config"]["seed"] == 1


def test_end_to_end_cpu_checkpoint_eval_paired_summary(tmp_path):
    torch = pytest.importorskip("torch")
    from model import GPT
    torch.set_num_threads(1)
    torch.manual_seed(112)
    source_dir, eval_dir = tmp_path / "sources", tmp_path / "eval"
    create_sources(source_dir, condition_plan(2, 2)[:3], 8, [1, 4, 8], 91)
    model = GPT(64, 8, n_layer=1, n_head=2, d_model=8)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict(), cfg=dict(model=model.cfg, family="pooled3", steps=2), step=2), checkpoint)
    first = evaluate_checkpoint(source_dir, checkpoint, "pooled_seed0", eval_dir, "cpu", 2)
    resumed = evaluate_checkpoint(source_dir, checkpoint, "pooled_seed0", eval_dir, "cpu", 2)
    assert first == resumed
    evaluate_checkpoint(source_dir, checkpoint, "specialist_shared", eval_dir, "cpu", 2, ["shared_base"])
    result = summarize(source_dir, eval_dir, tmp_path / "summary.json", [("pooled_seed0", "specialist_shared")])
    assert len(result["models"]["pooled_seed0"]["cohorts"]) == 3
    assert len(result["models"]["specialist_shared"]["cohorts"]) == 1
    difference = result["paired_differences"]["pooled_seed0_minus_specialist_shared"]["shared_base_prior"]
    for metric in difference.values():
        assert metric["mean"] == [0., 0., 0.]
        assert metric["se"] == [0., 0., 0.]
        assert metric["n_seq"] == 2
    assert result["bayesian_references"]["mixture"] == "pending"
    assert result["bayesian_references"]["family_aware"] == "pending"
    # A checkpoint replacement is never silently combined with old cohort scores.
    torch.save(dict(model=model.state_dict(), cfg=dict(model=model.cfg, steps=3), step=3), checkpoint)
    with pytest.raises(ValueError, match="different inputs"):
        evaluate_checkpoint(source_dir, checkpoint, "pooled_seed0", eval_dir, "cpu", 2)

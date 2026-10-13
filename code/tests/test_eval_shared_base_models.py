"""Independent source/oracle checks and actual-transformer alignment tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))

from eval_shared_base_models import (check_model_alignment, draw_source, model_case,
                                     selected_predictions, sources, validate_checkpoint,
                                     validate_source_record)
from source import Source


def test_source_matches_actual_generator_and_oracle_permutation():
    d, n, seed = 7, 37, 20260922
    actual = draw_source(d, n, seed)
    source = Source(d, n, alpha=(0.5, 1.5), ls=(0, 8), permute=True)
    tokens, parameters = source.batch(1, np.random.default_rng(seed))
    np.testing.assert_array_equal(actual["tokens"], tokens[0])
    assert actual["parameters"] == {key: value[0].tolist() for key, value in parameters.items()}
    rng = np.random.default_rng(seed)
    alpha, depth = source.draw_params(1, rng)
    q, transitions = source.matrices(alpha, depth, rng)
    permutation = parameters["perm"][0]
    np.testing.assert_array_equal(actual["initial_probs"][permutation], q[0])
    np.testing.assert_array_equal(actual["transition_probs"][permutation][:, permutation], transitions[0])
    # Check every observed current/next pair on the realized sequence.
    original = np.argsort(permutation)[tokens[0]]
    np.testing.assert_array_equal(
        actual["transition_probs"][tokens[0, :-1], tokens[0, 1:]],
        transitions[0, original[:-1], original[1:]])


def test_source_cohort_expansion_preserves_draws_and_rejects_changed_seed(tmp_path):
    initial = sources(tmp_path, 2, 20260922, create=True)
    expanded = sources(tmp_path, 3, 20260922, create=True)
    assert [item["file_sha256"] for item in initial] == [item["file_sha256"] for item in expanded[:2]]
    assert len({item["record"]["identity"]["sequence_seed"] for item in expanded}) == 3
    assert all(len(item["record"]["tokens"]) == 1025 for item in expanded)
    with pytest.raises(ValueError, match="inputs changed"):
        sources(tmp_path, 2, 20260923, create=False)
    with pytest.raises(FileNotFoundError, match="missing frozen source"):
        sources(tmp_path, 4, 20260922, create=False)


def test_source_checksums_detect_token_and_oracle_changes(tmp_path):
    item = sources(tmp_path, 1, 20260922, create=True)[0]
    value = json.loads(item["path"].read_text())
    value["tokens"][-1] = (value["tokens"][-1] + 1) % 64
    with pytest.raises(ValueError, match="token checksum"):
        validate_source_record(value)
    value = json.loads(item["path"].read_text())
    # Preserve the row sum to ensure the checksum, not normalization, catches it.
    row = value["oracle"]["transition_probs"][0]
    change = min(row[0], row[1]) / 10
    row[0] += change
    row[1] -= change
    with pytest.raises(ValueError, match="oracle probability checksum"):
        validate_source_record(value)


def test_selected_logits_use_prefix_minus_one_and_next_unseen_token():
    logits = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10
    tokens = np.array([[0, 2, 1, 0, 2], [2, 1, 0, 2, 1]])
    result = selected_predictions(logits, tokens, [1, 4])
    np.testing.assert_array_equal(result["logits"], logits[:, [0, 3]])
    np.testing.assert_array_equal(result["targets"], [[2, 2], [1, 1]])
    for b in range(2):
        for j, t in enumerate((1, 4)):
            weights = np.exp(logits[b, t - 1].astype(float))
            expected = weights / weights.sum()
            np.testing.assert_allclose(result["predictive_probs"][b, j], expected, atol=1e-15)
            assert result["selected_loss_bits"][b, j] == pytest.approx(-np.log2(expected[tokens[b, t]]))


def test_actual_gpt_batch_prefix_causality_and_target_exclusion():
    torch = pytest.importorskip("torch")
    from model import GPT
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    model = GPT(vocab=5, n_ctx=16, n_layer=2, n_head=2, d_model=16).eval()
    tokens = torch.randint(5, (3, 17))
    with torch.no_grad():
        full = model(tokens[:, :-1])
    check = check_model_alignment(model, tokens[:, :-1], full, tolerance=1e-6)
    assert check["passed"]
    selected = selected_predictions(full.numpy(), tokens.numpy(), [1, 7, 16])
    with torch.no_grad():
        for j, t in enumerate((1, 7, 16)):
            for sequence in range(3):
                separate = model(tokens[sequence:sequence + 1, :t])[0, -1].numpy()
                np.testing.assert_allclose(selected["logits"][sequence, j], separate, atol=1e-6, rtol=0)
    # Changing the unobserved final target must change loss selection only.
    altered = tokens.numpy().copy()
    altered[:, 16] = (altered[:, 16] + 1) % 5
    second = selected_predictions(full.numpy(), altered, [16])
    np.testing.assert_array_equal(second["predictive_probs"][:, 0], selected["predictive_probs"][:, 2])
    np.testing.assert_array_equal(second["targets"][:, 0], altered[:, 16])


def test_configuration_and_checkpoint_source_prior_mismatch_rejected(tmp_path):
    config = dict(d=64, n_ctx=1024, alpha=[0.5, 1.5], ls=[0, 8], no_permute=False, seed=0,
                  model=dict(vocab=64, n_ctx=1024, n_layer=2, n_head=4, d_model=128, mlp=True, dropout=0.0))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model_100000.pt").write_bytes(b"not loaded by configuration check")
    case = model_case(tmp_path)
    checkpoint = dict(step=100000, cfg=dict(config))
    validate_checkpoint(checkpoint, case)
    checkpoint["cfg"]["alpha"] = (0.5, 1.5)
    checkpoint["cfg"]["ls"] = (0, 8)
    validate_checkpoint(checkpoint, case)
    checkpoint["cfg"]["alpha"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="mismatch: alpha"):
        validate_checkpoint(checkpoint, case)
    config["no_permute"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not match"):
        model_case(tmp_path)

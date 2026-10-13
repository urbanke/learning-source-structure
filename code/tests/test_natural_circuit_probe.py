"""Independent alignment, attention-mask, intervention and loading checks."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from circuit_probe import run_probe
from model import GPT
from natural_circuit_probe import CONDITION_SCALES, NaturalCircuitProbe, load_model


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(918)
        result = GPT(vocab=5, n_ctx=16, n_layer=2, n_head=2, d_model=8).double().eval()
    return result


@pytest.fixture
def sequence():
    return np.array([0, 0, 1, 0, 1, 2, 2, 0, 4], dtype=np.int64)


def test_every_prefix_probability_matches_original_network_and_projection_ablation(model, sequence):
    result = NaturalCircuitProbe(model).run(sequence)
    for condition, deleted_head in (("intact", None), ("copier_deleted", 1), ("control_deleted", 0)):
        handle = None
        if deleted_head is not None:
            def remove_slice(module, inputs):
                joined = inputs[0].clone()
                joined[..., 4 * deleted_head:4 * (deleted_head + 1)] = 0
                return (joined,)
            handle = model.blocks[0].attn.proj.register_forward_pre_hook(remove_slice)
        try:
            with torch.inference_mode():
                expected = np.stack([model(torch.from_numpy(sequence[:length])[None])[0, -1].softmax(-1).numpy()
                                     for length in range(1, len(sequence))])
        finally:
            if handle is not None:
                handle.remove()
        np.testing.assert_allclose(result["conditions"][condition]["probs"], expected, atol=2e-14, rtol=2e-13)
    np.testing.assert_array_equal(result["prefix_lengths"], np.arange(1, len(sequence)))


def test_matching_sites_include_completed_self_repeat_and_exclude_future_target(model, sequence):
    result = NaturalCircuitProbe(model).run(sequence)
    # At model index1 the completed0->0 transition contributes its successor
    # at index1 itself. The same applies to2->2 at index6. No future transition
    # contributes: current contexts1 and2 are still unseen at indices2 and5.
    expected_counts = [0, 1, 0, 2, 1, 0, 1, 3]
    np.testing.assert_array_equal(result["matching_successor_count"], expected_counts)
    np.testing.assert_array_equal(result["matching_successor_defined"], np.array(expected_counts) > 0)
    np.testing.assert_allclose(result["matching_successor_uniform_attention"], np.array(expected_counts) / np.arange(1, 9))
    np.testing.assert_array_equal(result["previous_token_defined"], [False] + [True] * 7)
    idx = torch.from_numpy(sequence[:-1])[None]
    for condition, scales in CONDITION_SCALES.items():
        cache = run_probe(model, idx, head_scales=scales)
        expected_successor = np.zeros((8, 2, 2))
        expected_previous = np.zeros_like(expected_successor)
        for query in range(8):
            sites = [site for site in range(1, query + 1) if sequence[site - 1] == sequence[query]]
            for layer in range(2):
                for head in range(2):
                    values = cache["attention"][layer][0, head, query]
                    expected_successor[query, layer, head] = sum(float(values[site]) for site in sites)
                    if query:
                        expected_previous[query, layer, head] = float(values[query - 1])
        np.testing.assert_allclose(result["conditions"][condition]["matching_successor_attention"], expected_successor, atol=2e-15)
        np.testing.assert_allclose(result["conditions"][condition]["previous_token_attention"], expected_previous, atol=0, rtol=0)
    changed = sequence.copy()
    changed[-1] = 3  # Only the held-out final target changes.
    another = NaturalCircuitProbe(model).run(changed)
    for condition in CONDITION_SCALES:
        for key, array in result["conditions"][condition].items():
            np.testing.assert_array_equal(array, another["conditions"][condition][key])


def test_original_check_only_first_call_but_causal_checks_each_call_and_no_shared_buffers(model, sequence):
    probe = NaturalCircuitProbe(model)
    first = probe.run(sequence)
    second = probe.run(sequence[:4])
    assert first["validation"]["original_checked_this_call"]
    assert not second["validation"]["original_checked_this_call"]
    assert first["validation"]["original_validation"] == second["validation"]["original_validation"]
    for condition in CONDITION_SCALES:
        assert set(first["validation"]["conditions"][condition]["causal_prefix_replay"]) == {"1", "7", "8"}
        assert set(second["validation"]["conditions"][condition]["causal_prefix_replay"]) == {"1", "3"}
        for array in first["conditions"][condition].values():
            assert array.flags.owndata
            assert array.base is None
    # The returned receipt cannot overwrite the validation cached for later sequences.
    first["validation"]["original_validation"]["passed"] = False
    assert probe.run(sequence[:3])["validation"]["original_validation"]["passed"]


def test_single_prediction_and_empty_successor_context(model):
    result = NaturalCircuitProbe(model).run(np.array([0, 1]))
    np.testing.assert_array_equal(result["matching_successor_count"], [0])
    assert not result["matching_successor_defined"].any()
    for condition in CONDITION_SCALES:
        assert result["conditions"][condition]["probs"].shape == (1, 5)
        assert not result["conditions"][condition]["previous_token_attention"].any()
        assert not result["conditions"][condition]["matching_successor_attention"].any()


def test_sequence_values_and_model_parameters_modes_flags_are_unchanged(model, sequence):
    before_sequence = sequence.copy()
    before_state = {key: value.clone() for key, value in model.state_dict().items()}
    before_modes = [module.training for module in model.modules()]
    before_manual = [block.attn.manual for block in model.blocks]
    NaturalCircuitProbe(model).run(sequence)
    np.testing.assert_array_equal(sequence, before_sequence)
    assert [module.training for module in model.modules()] == before_modes
    assert [block.attn.manual for block in model.blocks] == before_manual
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[key], atol=0, rtol=0)


@pytest.mark.parametrize("sequence", [[], [0], [0., 1.], [[0, 1]], [-1, 0], [0, 5], [False, True], [0] * 18])
def test_invalid_paths_rejected(model, sequence):
    with pytest.raises(ValueError):
        NaturalCircuitProbe(model).run(sequence)


def test_invalid_model_mode_or_precision_rejected(model):
    with pytest.raises(ValueError, match="evaluation"):
        NaturalCircuitProbe(model.train())
    model.eval().float()
    with pytest.raises(ValueError, match="float64"):
        NaturalCircuitProbe(model)


@pytest.fixture
def saved_checkpoint(tmp_path):
    config = dict(family="lsa", alpha=[0., 0.], ls=[0, 8], seed=0, d=64, n_ctx=1024, no_permute=False,
                  model=dict(vocab=64, n_ctx=1024, n_layer=2, n_head=4, d_model=128, mlp=True, dropout=0.0))
    with torch.random.fork_rng():
        torch.manual_seed(44)
        original = GPT(**config["model"]).float().eval()
    saved = dict(step=100000, cfg=copy.deepcopy(config), model=original.state_dict())
    checkpoint, config_path = tmp_path / "model.pt", tmp_path / "config.json"
    torch.save(saved, checkpoint)
    config_path.write_text(json.dumps(config))
    return checkpoint, config_path, saved


def test_loader_preserves_every_float32_weight_exactly_and_returns_stable_metadata(saved_checkpoint):
    checkpoint, config_path, saved = saved_checkpoint
    model, metadata = load_model(checkpoint, config_path)
    second, second_metadata = load_model(checkpoint, config_path)
    assert metadata == second_metadata
    assert metadata["parameter_values_preserved"]
    assert metadata["step"] == 100000
    json.dumps(metadata, allow_nan=False)
    assert not model.training
    for key, value in model.state_dict().items():
        assert value.dtype == torch.float64
        torch.testing.assert_close(value, saved["model"][key].double(), atol=0, rtol=0)


@pytest.mark.parametrize("location,key,value", [("config", "alpha", [.5, 1.5]), ("config", "ls", [1, 8]),
                                                ("config", "d", 32), ("saved", "step", 95000),
                                                ("saved_cfg", "family", "product:8x8")])
def test_loader_rejects_wrong_family_prior_size_or_checkpoint_step(saved_checkpoint, location, key, value):
    checkpoint, config_path, saved = saved_checkpoint
    if location == "config":
        config = json.loads(config_path.read_text())
        config[key] = value
        config_path.write_text(json.dumps(config))
    else:
        (saved if location == "saved" else saved["cfg"])[key] = value
        torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        load_model(checkpoint, config_path)

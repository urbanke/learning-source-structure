"""Independent checks of the local two-site intervention and six-query indexing."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"transformer"))
from circuit_probe import run_probe
from copier_probes import make_probes
from model import GPT
from trace_copier_key_path import batched_sites, site_patch, run_controlled


def test_swap_is_simultaneous_local_and_does_not_mutate_cache():
    original = torch.arange(3*7*4, dtype=torch.float64).reshape(3, 7, 4)
    saved = original.clone()
    a, b = np.array([1, 5, 0]), np.array([4, 2, 6])
    patch = site_patch(original, a, b, mode="swap")
    for i in range(3):
        for j in range(7):
            expected_j = b[i] if j == a[i] else a[i] if j == b[i] else j
            torch.testing.assert_close(patch["values"][i, j], original[i, expected_j], rtol=0, atol=0)
            assert bool(patch["mask"][i, j]) == (j in (a[i], b[i]))
    torch.testing.assert_close(original, saved, rtol=0, atol=0)


def test_restoration_zeros_the_whole_head_except_the_two_sites():
    x = torch.arange(2*8*3, dtype=torch.float64).reshape(2, 8, 3)+1
    patch = site_patch(x, [2, 4], [6, 1], mode="restore")
    assert "mask" not in patch
    for i, sites in enumerate(((2, 6), (4, 1))):
        for j in range(8):
            torch.testing.assert_close(patch["values"][i, j], x[i, j] if j in sites else torch.zeros(3, dtype=x.dtype))
    torch.testing.assert_close(site_patch(x, [2, 4], [6, 1], mode="sham")["values"], x, rtol=0, atol=0)


def test_batched_sites_follow_group_major_slices_not_pair_major_repetition():
    probes = make_probes(11, seed=794)
    first, second = batched_sites(probes, 3, 8)
    for group in range(6):
        for local, sequence in enumerate(range(3, 8)):
            row = group*5+local
            assert first[row] == probes["source_b_pos"][sequence]
            assert second[row] == probes["source_c_pos"][sequence]
            assert probes["clean"][sequence, first[row]-1] == probes["context_a"][sequence]
            assert probes["clean"][sequence, second[row]-1] == probes["context_d"][sequence]


@pytest.mark.parametrize("a,b", [([1., 2.], [3, 4]), ([1, 2], [1, 4]), ([-1, 2], [3, 4]), ([1, 2], [3, 7]), ([1], [3, 4])])
def test_invalid_sites_rejected(a, b):
    with pytest.raises(ValueError):
        site_patch(torch.zeros(2, 7, 3), a, b, mode="swap")


def test_live_restore_uses_patching_without_zero_scale_and_preserves_causality():
    torch.manual_seed(431)
    model = GPT(vocab=9, n_ctx=8, n_layer=2, n_head=2, d_model=12, mlp=True).double().eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1], [2, 4, 0, 8, 6, 7, 5, 2]])
    intact = run_probe(model, tokens)
    old = intact["head_outputs"][0][:, 1]
    a, b = [3, 4], [5, 6]
    restored = site_patch(old, a, b, mode="restore")
    result = run_probe(model, tokens, head_patches={(0, 1): restored})
    torch.testing.assert_close(result["head_outputs"][0][:, 1], restored["values"], rtol=0, atol=0)
    # This would fail if the production rescue incorrectly combined patching
    # with a zero scale: the probe intentionally scales after patching.
    assert result["head_outputs"][0][:, 1].abs().max() > 0
    swapped = run_probe(model, tokens, head_patches={(0, 1): site_patch(old, a, b, mode="swap")})
    torch.testing.assert_close(swapped["logits"][:, :3], intact["logits"][:, :3], rtol=0, atol=0)
    sham = run_probe(model, tokens, head_patches={(0, 1): site_patch(old, a, b, mode="sham")})
    torch.testing.assert_close(sham["logits"], intact["logits"], rtol=0, atol=0)


def test_batched_driver_last_partial_batch_and_attention_match_single_prompt():
    torch.manual_seed(753)
    model = GPT(vocab=64, n_ctx=64, n_layer=2, n_head=4, d_model=16, mlp=True).double().eval()
    probes = make_probes(3, seed=634)
    original_tokens = probes["clean"].copy()
    outputs, attention, full, checks = run_controlled(model, probes, (0, 1), batch=2)
    # The final counterfactual E query lies in a partial batch; construct it
    # directly instead of using the production group's reshape helper.
    i, g = 2, 5
    tokens = torch.from_numpy(probes["counterfactual"][i:i+1].copy())
    tokens[0, -1] = int(probes["control_e"][i])
    untouched = run_probe(model, tokens)
    b, c = int(probes["source_b_pos"][i]), int(probes["source_c_pos"][i])
    values = untouched["head_outputs"][0][:, 1].clone()
    original = values.clone()
    values[0, b], values[0, c] = original[0, c], original[0, b]
    # A full patch with unchanged remaining sites is algebraically equivalent
    # to the production mask, giving an independent site-mask comparison.
    swapped = run_probe(model, tokens, head_patches={(0, 1): {"values": values}})
    for name, expected in (("intact", untouched), ("swap_successors", swapped)):
        np.testing.assert_allclose(outputs[name][i, g], expected["logits"][0, -1].numpy(), atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(full[name][i, g], expected["attention"][-1][0, :, -1].numpy(), atol=2e-15, rtol=2e-15)
        for h in range(4):
            for j, pos in enumerate((b, c)):
                np.testing.assert_allclose(attention[name][i, g, h, j], expected["attention"][-1][0, h, -1, pos].item(), atol=2e-15, rtol=2e-15)
    assert checks["sham_max_logit_error"] == 0
    assert np.array_equal(probes["clean"], original_tokens)

"""Independent forward and intervention checks for attention-head probing."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from circuit_probe import run_probe
from model import GPT


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(412)
        network = GPT(vocab=13, n_ctx=12, n_layer=2, n_head=3, d_model=12).double().eval()
    return network


@pytest.fixture
def tokens():
    return torch.tensor([[0, 3, 8, 3, 1, 2, 7, 4], [6, 5, 6, 11, 8, 0, 2, 1]])


def independent_forward(network, idx):
    """Loop over heads; project each slice separately and add bias once.

    This uses direct functional operations rather than run_probe's forward
    path, to expose layer-norm, head-layout, residual, MLP and bias mistakes.
    """
    batch, length = idx.shape
    x = network.tok.weight[idx] + network.pos.weight[:length][None]
    for block in network.blocks:
        normalized = F.layer_norm(x, block.ln1.normalized_shape, block.ln1.weight,
                                  block.ln1.bias, block.ln1.eps)
        c, heads = block.attn.d_model, block.attn.n_head
        width = c // heads
        q, k, v = F.linear(normalized, block.attn.qkv.weight, block.attn.qkv.bias).chunk(3, dim=-1)
        summed = torch.zeros_like(x)
        for head in range(heads):
            sl = slice(head * width, (head + 1) * width)
            score = torch.einsum("btd,bsd->bts", q[..., sl], k[..., sl]) / np.sqrt(width)
            score = score.masked_fill(torch.arange(length)[None, :] > torch.arange(length)[:, None], -torch.inf)
            output = torch.einsum("bts,bsd->btd", score.softmax(-1), v[..., sl])
            summed += F.linear(output, block.attn.proj.weight[:, sl], bias=None)
        x = x + summed + block.attn.proj.bias
        if block.mlp is not None:
            normalized = F.layer_norm(x, block.ln2.normalized_shape, block.ln2.weight,
                                      block.ln2.bias, block.ln2.eps)
            hidden = F.gelu(F.linear(normalized, block.mlp[0].weight, block.mlp[0].bias))
            x = x + F.linear(hidden, block.mlp[2].weight, block.mlp[2].bias)
    x = F.layer_norm(x, network.ln_f.normalized_shape, network.ln_f.weight,
                     network.ln_f.bias, network.ln_f.eps)
    return F.linear(x, network.head.weight, bias=None)


def test_float64_forward_matches_independent_headwise_reference(model, tokens):
    with torch.no_grad():
        # Nonzero biases make accidental per-head projection biases visible.
        for block in model.blocks:
            block.attn.proj.bias.copy_(torch.linspace(-0.07, 0.09, 12))
        expected = independent_forward(model, tokens)
    result = run_probe(model, tokens)
    torch.testing.assert_close(result["logits"], expected, atol=2e-14, rtol=2e-13)
    for attention, heads in zip(result["attention"], result["head_outputs"]):
        assert attention.shape == (2, 3, 8, 8)
        assert heads.shape == (2, 3, 8, 4)
        torch.testing.assert_close(attention.sum(-1), torch.ones((2, 3, 8), dtype=torch.float64))
        assert torch.count_nonzero(torch.triu(attention, diagonal=1)) == 0
    assert not result["logits"].requires_grad
    assert all(not value.requires_grad for value in result["attention"] + result["head_outputs"])


@pytest.mark.parametrize("manual", [False, True])
def test_matches_original_fused_and_manual_forward_without_mutation(model, tokens, manual):
    for block in model.blocks:
        block.attn.manual = manual
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_modes = [module.training for module in model.modules()]
    original_flags = [block.attn.manual for block in model.blocks]
    with torch.no_grad():
        expected = model(tokens)
    result = run_probe(model, tokens)
    torch.testing.assert_close(result["logits"], expected, atol=2e-14, rtol=2e-13)
    assert original_modes == [module.training for module in model.modules()]
    assert original_flags == [block.attn.manual for block in model.blocks]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)


@pytest.mark.parametrize("mlp", [False, True])
def test_matches_single_head_and_attention_only_configurations(mlp):
    with torch.random.fork_rng():
        torch.manual_seed(75)
        network = GPT(vocab=5, n_ctx=4, n_layer=1, n_head=1, d_model=8, mlp=mlp).double().eval()
    idx = torch.tensor([[1, 0, 2, 3]])
    torch.testing.assert_close(run_probe(network, idx)["logits"], independent_forward(network, idx),
                               atol=2e-14, rtol=2e-13)


def test_zero_scale_matches_independent_projection_input_hook(model, tokens):
    layer, head, width = 0, 1, 4
    for block in model.blocks:
        block.attn.manual = True

    def zero_slice(module, args):
        joined = args[0].clone()
        joined[..., head * width:(head + 1) * width] = 0
        return (joined,)

    handle = model.blocks[layer].attn.proj.register_forward_pre_hook(zero_slice)
    try:
        with torch.no_grad():
            expected = model(tokens)
    finally:
        handle.remove()
    result = run_probe(model, tokens, head_scales={(layer, head): 0})
    torch.testing.assert_close(result["logits"], expected, atol=2e-14, rtol=2e-13)
    assert torch.count_nonzero(result["head_outputs"][layer][:, head]) == 0


def test_ablating_every_head_adds_the_projection_bias_once(model, tokens):
    with torch.no_grad():
        model.blocks[0].attn.proj.bias.copy_(torch.linspace(-.2, .3, 12))
    handle = model.blocks[0].attn.proj.register_forward_pre_hook(lambda module, args: (torch.zeros_like(args[0]),))
    try:
        with torch.no_grad():
            expected = model(tokens)
    finally:
        handle.remove()
    result = run_probe(model, tokens, head_scales={(0, head): 0 for head in range(3)})
    torch.testing.assert_close(result["logits"], expected, atol=2e-14, rtol=2e-13)


def test_masked_donor_patch_matches_projection_hook_and_preserves_unselected_sites(model, tokens):
    donor_tokens = tokens.flip(1)
    base, donor = run_probe(model, tokens), run_probe(model, donor_tokens)
    values = donor["head_outputs"][0][:, 2].clone()
    original_values = values.clone()
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[0, 4] = True
    original_mask = mask.clone()

    def patch_slice(module, args):
        joined = args[0].clone()
        joined[..., 8:12] = torch.where(mask[..., None], values, joined[..., 8:12])
        return (joined,)

    handle = model.blocks[0].attn.proj.register_forward_pre_hook(patch_slice)
    try:
        with torch.no_grad():
            expected = model(tokens)
    finally:
        handle.remove()
    patched = run_probe(model, tokens, head_patches={(0, 2): {"values": values, "mask": mask}})
    torch.testing.assert_close(patched["logits"], expected, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(patched["head_outputs"][0][0, 2, 4], values[0, 4], atol=0, rtol=0)
    torch.testing.assert_close(patched["head_outputs"][0][:, 2][~mask],
                               base["head_outputs"][0][:, 2][~mask], atol=0, rtol=0)
    torch.testing.assert_close(patched["logits"][0, :4], base["logits"][0, :4], atol=0, rtol=0)
    torch.testing.assert_close(patched["logits"][1], base["logits"][1], atol=0, rtol=0)
    assert torch.max(torch.abs(patched["logits"][0, 4:] - base["logits"][0, 4:])) > 1e-9
    torch.testing.assert_close(values, original_values, atol=0, rtol=0)
    torch.testing.assert_close(mask, original_mask, atol=0, rtol=0)


def test_patch_precedes_scale_and_default_mask_covers_every_site(model, tokens):
    values = torch.full((2, 8, 4), 0.3, dtype=torch.float64)
    result = run_probe(model, tokens, head_patches={(1, 0): {"values": values}}, head_scales={(1, 0): 2.5})
    torch.testing.assert_close(result["head_outputs"][1][:, 0], values * 2.5, atol=0, rtol=0)
    zeroed = run_probe(model, tokens, head_patches={(1, 0): {"values": values}}, head_scales={(1, 0): 0})
    ablated = run_probe(model, tokens, head_scales={(1, 0): 0})
    torch.testing.assert_close(zeroed["logits"], ablated["logits"], atol=0, rtol=0)


def test_future_tokens_and_future_patches_do_not_change_earlier_predictions(model, tokens):
    base = run_probe(model, tokens)
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 4) % 13
    future_tokens = run_probe(model, changed)
    torch.testing.assert_close(base["logits"][:, :5], future_tokens["logits"][:, :5], atol=0, rtol=0)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 5:] = True
    patched = run_probe(model, tokens, head_patches={(0, 0): {
        "values": torch.full((2, 8, 4), 17., dtype=torch.float64), "mask": mask}})
    torch.testing.assert_close(base["logits"][:, :5], patched["logits"][:, :5], atol=0, rtol=0)
    for length in (1, 3, 5):
        truncated = run_probe(model, tokens[:, :length])
        torch.testing.assert_close(base["logits"][:, :length], truncated["logits"], atol=2e-14, rtol=2e-13)


def test_return_cache_false_retains_only_logits(model, tokens):
    result = run_probe(model, tokens, return_cache=False)
    assert result["attention"] == [] and result["head_outputs"] == []
    torch.testing.assert_close(result["logits"], run_probe(model, tokens)["logits"], atol=0, rtol=0)


@pytest.mark.parametrize("idx", [torch.tensor([1, 2]), torch.zeros((1, 4), dtype=torch.float64),
                                 torch.zeros((0, 4), dtype=torch.int64), torch.zeros((1, 0), dtype=torch.int64),
                                 torch.zeros((1, 13), dtype=torch.int64), torch.tensor([[13]]), torch.tensor([[-1]])])
def test_bad_token_inputs_rejected(model, idx):
    with pytest.raises(ValueError):
        run_probe(model, idx)


@pytest.mark.parametrize("key", [0, (0,), (0, 1, 2), (True, 0), (0.0, 0), (-1, 0), (2, 0), (0, 3)])
def test_bad_head_indices_rejected(model, tokens, key):
    with pytest.raises(ValueError):
        run_probe(model, tokens, head_scales={key: 0})


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), True, "zero", torch.tensor(0.)])
def test_bad_head_scales_rejected(model, tokens, scale):
    with pytest.raises(ValueError):
        run_probe(model, tokens, head_scales={(0, 0): scale})


def test_bad_patch_description_shape_dtype_mask_and_nonfinite_values_rejected(model, tokens):
    valid = torch.zeros((2, 8, 4), dtype=torch.float64)
    bad_patches = [{}, {"values": valid, "unexpected": 3}, {"values": valid[:1]},
                   {"values": valid.float()}, {"values": torch.full_like(valid, torch.nan)},
                   {"values": valid, "mask": torch.zeros((2, 8))},
                   {"values": valid, "mask": torch.zeros((1, 8), dtype=torch.bool)},
                   {"values": valid.tolist()}, {"values": valid, "mask": [[True] * 8] * 2}]
    for patch in bad_patches:
        with pytest.raises(ValueError):
            run_probe(model, tokens, head_patches={(0, 0): patch})


def test_training_mode_and_nonmapping_interventions_rejected(model, tokens):
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        run_probe(model, tokens)
    model.eval()
    model.blocks[0].train()
    with pytest.raises(ValueError, match="evaluation"):
        run_probe(model, tokens)
    model.eval()
    for kwargs in ({"head_scales": []}, {"head_patches": []}, {"return_cache": 1}):
        with pytest.raises(ValueError):
            run_probe(model, tokens, **kwargs)

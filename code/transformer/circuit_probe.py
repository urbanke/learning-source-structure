"""Read and intervene on attention heads of the existing evaluation-mode GPT.

The explicit causal attention computation mirrors ``model.py`` without
changing that producer, its ``manual`` flags, parameters, or module modes.
Interventions act on head outputs before the shared output projection.
Patches are applied first and scales second, so a zero scale also ablates a
patched head. The projection, including its bias, is applied exactly once.

Positions, layers and heads are zero indexed. Logits at input position t
predict the next token. An intervention at position t can affect that
prediction and later ones, never a prediction at an earlier position.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch


def _head_key(key, model):
    if (not isinstance(key, tuple) or len(key) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in key)):
        raise ValueError("Head intervention keys must be (layer, head) integer pairs")
    layer, head = map(int, key)
    if not 0 <= layer < len(model.blocks):
        raise ValueError(f"Layer index {layer} is outside the model")
    if not 0 <= head < model.blocks[layer].attn.n_head:
        raise ValueError(f"Head index {head} is outside layer {layer}")
    return layer, head


def _validate(model, idx, head_scales, head_patches, return_cache):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("run_probe requires the model and every submodule in evaluation mode")
    if not isinstance(return_cache, bool):
        raise ValueError("return_cache must be a boolean")
    if (not isinstance(idx, torch.Tensor) or idx.ndim != 2
            or idx.dtype not in (torch.int32, torch.int64)):
        raise ValueError("idx must be a two-dimensional int32 or int64 token tensor")
    batch, length = idx.shape
    if batch < 1 or not 1 <= length <= model.cfg["n_ctx"]:
        raise ValueError("idx must contain a nonempty batch within the model context length")
    if idx.device != model.tok.weight.device:
        raise ValueError("idx must be on the same device as the model")
    if bool(torch.any(idx < 0)) or bool(torch.any(idx >= model.tok.num_embeddings)):
        raise ValueError("idx contains a token outside the model vocabulary")
    scales, patches = {}, {}
    if head_scales is not None:
        if not isinstance(head_scales, Mapping):
            raise ValueError("head_scales must map (layer, head) pairs to finite real scalars")
        for key, scale in head_scales.items():
            key = _head_key(key, model)
            if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
                raise ValueError("Every head scale must be a finite real scalar")
            scales[key] = float(scale)
    if head_patches is not None:
        if not isinstance(head_patches, Mapping):
            raise ValueError("head_patches must map (layer, head) pairs to patch descriptions")
        for key, patch in head_patches.items():
            key = _head_key(key, model)
            if not isinstance(patch, Mapping) or "values" not in patch or set(patch) - {"values", "mask"}:
                raise ValueError("Each patch must contain values and an optional mask only")
            attention = model.blocks[key[0]].attn
            expected = (batch, length, attention.d_model // attention.n_head)
            values = patch["values"]
            if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
                raise ValueError(f"Patch values for {key} must have shape {expected}")
            if values.device != model.tok.weight.device or values.dtype != model.tok.weight.dtype:
                raise ValueError("Patch values must have the same device and dtype as the model")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Patch values must be finite")
            mask = patch.get("mask")
            if mask is not None and (not isinstance(mask, torch.Tensor)
                                     or tuple(mask.shape) != (batch, length)
                                     or mask.dtype != torch.bool or mask.device != idx.device):
                raise ValueError("Patch mask must be a bool tensor [batch, length] on the model device")
            patches[key] = (values, mask)
    return scales, patches


@torch.no_grad()
def run_probe(model, idx, *, head_scales=None, head_patches=None, return_cache=True):
    """Return logits and optional per-layer attention/head-output caches.

    ``head_scales[(layer,head)]`` is a finite real scalar. A patch is
    ``{"values": tensor[B,T,D], "mask": optional_bool_tensor[B,T]}``, where
    D is the head width. With no mask, every site in that head is patched.
    Values and masks must already have the correct device/dtype; no implicit
    conversions or broadcasting occur. Neither patch tensors nor the model
    are modified. The whole call runs without gradient recording.

    The result has ``logits`` [B,T,vocab], ``attention`` (one [B,H,T,T]
    tensor per layer), and ``head_outputs`` (one [B,H,T,D] tensor per layer).
    Cached head outputs include interventions and are the values actually
    sent to the output projection. If return_cache=False, both cache lists
    are empty. Attention in downstream layers includes intervention effects.
    """
    scales, patches = _validate(model, idx, head_scales, head_patches, return_cache)
    batch, length = idx.shape
    positions = torch.arange(length, device=idx.device)
    x = model.tok(idx) + model.pos(positions)[None]
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=idx.device).tril()
    attention_cache, head_cache = [], []
    for layer, block in enumerate(model.blocks):
        attention = block.attn
        width, heads = attention.d_model, attention.n_head
        head_width = width // heads
        q, k, v = attention.qkv(block.ln1(x)).split(width, dim=2)
        q = q.view(batch, length, heads, head_width).transpose(1, 2)
        k = k.view(batch, length, heads, head_width).transpose(1, 2)
        v = v.view(batch, length, heads, head_width).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)
        weights = weights.masked_fill(~causal_mask, float("-inf")).softmax(dim=-1)
        head_outputs = weights @ v
        if any(key[0] == layer for key in patches) or any(key[0] == layer for key in scales):
            head_outputs = head_outputs.clone()
            for head in range(heads):
                key = (layer, head)
                if key in patches:
                    values, mask = patches[key]
                    head_outputs[:, head] = (values if mask is None else
                                            torch.where(mask[..., None], values, head_outputs[:, head]))
                if key in scales:
                    head_outputs[:, head] *= scales[key]
        if return_cache:
            attention_cache.append(weights)
            head_cache.append(head_outputs)
        joined = head_outputs.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + attention.proj(joined)
        if block.mlp is not None:
            x = x + block.mlp(block.ln2(x))
    return dict(logits=model.head(model.ln_f(x)), attention=attention_cache, head_outputs=head_cache)

"""Compact, faithful attention-head measurements along a natural sequence.

The frozen circuit_probe forward is reused unchanged. One whole-sequence
causal pass gives every prefix prediction for each intervention. Deleting
a head deletes it at all observed positions, exactly as if the same
intervened network were run separately on each prefix.

A matching-successor site s satisfies 1 <= s <= q and x[s-1] == x[q],
where q is the current input-token index. Site s=q is included when the
last observed transition is a self-repeat: it is already in the prefix.
Neither the future target x[q+1] nor a future attention site is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from circuit_probe import run_probe
from model import GPT

ROOT = Path(__file__).resolve().parents[2]

CONDITION_SCALES = {
    "intact": {},
    "copier_deleted": {(0, 1): 0.0},
    "control_deleted": {(0, 0): 0.0},
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint=None, config_path=None, *, torch_threads=1):
    """Load the fixed uniform-base checkpoint, retaining its exact weights.

    A separate float64 model receives the float32-trained tensor values.
    No checkpoint, model producer, parameter value, or saved configuration
    is changed. Returns (evaluation-mode model, provenance metadata).
    """
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, Integral) or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    checkpoint = (ROOT / "results/transformer/runs/lsa_a0_d64/model_100000.pt"
                  if checkpoint is None else Path(checkpoint))
    config_path = checkpoint.parent / "config.json" if config_path is None else Path(config_path)
    config = json.loads(config_path.read_text())
    required = dict(family="lsa", alpha=[0., 0.], ls=[0, 8], seed=0, d=64, n_ctx=1024, no_permute=False)
    if any(config.get(key) != expected for key, expected in required.items()):
        raise ValueError("Configuration is not the fixed uniform-base d64 source")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["step"] != 100000 or any(saved["cfg"].get(key) != expected for key, expected in required.items()):
        raise ValueError("Checkpoint is not the fixed uniform-base step100000 model")
    expected_model = dict(vocab=64, n_ctx=1024, n_layer=2, n_head=4, d_model=128, mlp=True, dropout=0.0)
    if config["model"] != expected_model or saved["cfg"]["model"] != config["model"]:
        raise ValueError("Checkpoint and configuration disagree with the recorded architecture")
    if any(value.dtype != torch.float32 for value in saved["model"].values() if value.is_floating_point()):
        raise ValueError("Expected unchanged float32-trained checkpoint tensors")
    torch.set_num_threads(int(torch_threads))
    with torch.random.fork_rng():
        model = GPT(**config["model"]).double().eval()
    model.load_state_dict(saved["model"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, saved["model"][name].double()):
            raise ValueError("Loaded parameters differ from the saved values")
    metadata = dict(checkpoint=dict(path=str(checkpoint.resolve()), sha256=_sha(checkpoint)),
                    config=dict(path=str(config_path.resolve()), sha256=_sha(config_path)),
                    step=100000, architecture=config["model"], source={key: config[key] for key in required},
                    precision="float64 computation of unchanged float32-trained parameter values",
                    torch_threads=int(torch_threads), parameter_values_preserved=True)
    return model, metadata


class NaturalCircuitProbe:
    """Measure independent sequences without retaining full attention caches.

    ``run(sequence)`` takes the complete one-dimensional path, including
    its final target. It feeds sequence[:-1] and returns probabilities for
    sequence[1:] at prefix lengths1..len(sequence)-1. Cache reductions use
    axes [prefix,layer,head]; probability arrays use [prefix,vocabulary].
    Undefined first-position previous-token attention is zero with a
    separate boolean mask. Successor attention is zero when no eligible
    predecessor occurrence exists, also marked with an explicit mask.
    """

    def __init__(self, model, *, replay_tolerance=1e-10):
        if not np.isfinite(replay_tolerance) or replay_tolerance <= 0:
            raise ValueError("replay_tolerance must be finite and positive")
        if model.training or any(module.training for module in model.modules()):
            raise ValueError("NaturalCircuitProbe requires evaluation mode")
        parameters = list(model.parameters())
        if any(parameter.dtype != torch.float64 or parameter.device.type != "cpu" for parameter in parameters):
            raise ValueError("NaturalCircuitProbe requires a float64 CPU model")
        if not len(model.blocks) or model.blocks[0].attn.n_head < 2:
            raise ValueError("The model must contain the candidate and control heads")
        heads = [block.attn.n_head for block in model.blocks]
        if len(set(heads)) != 1:
            raise ValueError("Compact attention arrays require equal head counts across layers")
        self.model = model
        self.replay_tolerance = float(replay_tolerance)
        self._original_receipt = None

    def _check_original(self, tokens, logits):
        """Compare separately instantiated original fused and manual forwards."""
        original = copy.deepcopy(self.model)
        results = {}
        with torch.inference_mode():
            for manual in (False, True):
                for block in original.blocks:
                    block.attn.manual = manual
                expected = original(tokens)
                logit_error = float(torch.max(torch.abs(expected - logits)))
                probability_error = float(torch.max(torch.abs(expected.softmax(-1) - logits.softmax(-1))))
                if not np.isfinite([logit_error, probability_error]).all() or max(logit_error, probability_error) > self.replay_tolerance:
                    raise ValueError("Probe differs from the original evaluation forward")
                results["manual" if manual else "fused"] = dict(logit_max_abs=logit_error, probability_max_abs=probability_error)
        return dict(passed=True, scored_prefixes=tokens.shape[1],
                    input_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                    tolerance=self.replay_tolerance, comparisons=results)

    @torch.no_grad()
    def run(self, sequence):
        sequence = np.asarray(sequence)
        if (sequence.ndim != 1 or sequence.dtype.kind not in "iu" or len(sequence) < 2
                or len(sequence) > self.model.cfg["n_ctx"] + 1):
            raise ValueError("Expected a nonempty integer path plus its final target, within context length")
        if np.any(sequence < 0) or np.any(sequence >= self.model.cfg["vocab"]):
            raise ValueError("A sequence token is outside the vocabulary")
        # Copy so even an unusual downstream in-place operation could not
        # modify the caller's complete path. The final target is never fed.
        tokens = torch.from_numpy(np.array(sequence[:-1], dtype=np.int64, copy=True))[None]
        length = tokens.shape[1]
        prefix_lengths = np.arange(1, length + 1, dtype=np.int64)
        match = torch.zeros((1, length, length), dtype=torch.bool)
        if length > 1:
            match[:, :, 1:] = tokens[:, :, None] == tokens[:, None, :-1]
        match &= torch.ones((length, length), dtype=torch.bool).tril()[None]
        counts = match.sum(dim=-1)[0].numpy().copy()
        replay_prefixes = sorted({min(length, value) for value in (1, 7, 64, 512)})
        conditions, checks = {}, {}
        original_checked = self._original_receipt is None
        original_receipt = self._original_receipt
        for name, scales in CONDITION_SCALES.items():
            cache = run_probe(self.model, tokens, head_scales=scales, return_cache=True)
            logits = cache["logits"]
            probabilities = logits.softmax(-1)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(probabilities).all()):
                raise ValueError("Nonfinite natural-sequence predictions")
            probability_sum_error = float(torch.max(torch.abs(probabilities.sum(-1) - 1)))
            if probability_sum_error > 1e-12:
                raise ValueError("Natural-sequence probabilities do not normalize")
            if name == "intact" and original_checked:
                original_receipt = self._check_original(tokens, logits)
            n_layers, n_heads = len(cache["attention"]), self.model.blocks[0].attn.n_head
            previous = np.zeros((length, n_layers, n_heads), dtype=np.float64)
            if length > 1:
                diagonal = torch.stack([value[:, :, torch.arange(1, length), torch.arange(length - 1)]
                                        for value in cache["attention"]], dim=1)
                previous[1:] = diagonal[0].permute(2, 0, 1).numpy()
            matched = torch.stack([(value * match[:, None]).sum(dim=-1)
                                   for value in cache["attention"]], dim=1)
            successor = matched[0].permute(2, 0, 1).numpy().copy()
            attention_sum_error = max(float(torch.max(torch.abs(value.sum(-1) - 1))) for value in cache["attention"])
            if attention_sum_error > 1e-12 or np.any(successor < 0) or np.any(successor > 1 + 1e-12):
                raise ValueError("Invalid attention reduction")
            replay = {}
            for prefix in replay_prefixes:
                truncated = run_probe(self.model, tokens[:, :prefix], head_scales=scales, return_cache=False)["logits"][:, -1]
                logit_error = float(torch.max(torch.abs(truncated - logits[:, prefix - 1])))
                probability_error = float(torch.max(torch.abs(truncated.softmax(-1) - probabilities[:, prefix - 1])))
                if not np.isfinite([logit_error, probability_error]).all() or max(logit_error, probability_error) > self.replay_tolerance:
                    raise ValueError(f"Causal prefix replay failed for {name} at length{prefix}")
                replay[str(prefix)] = dict(logit_max_abs=logit_error, probability_max_abs=probability_error)
            # Explicit copies sever ownership of any large torch attention
            # buffers. Only compact probabilities/reductions leave the call.
            conditions[name] = dict(probs=probabilities[0].numpy().copy(),
                                    previous_token_attention=previous,
                                    matching_successor_attention=successor)
            checks[name] = dict(probability_sum_error=probability_sum_error,
                                attention_sum_error=attention_sum_error, causal_prefix_replay=replay)
            del cache, logits, probabilities, matched
        self._original_receipt = copy.deepcopy(original_receipt)
        return dict(prefix_lengths=prefix_lengths, conditions=conditions,
                    previous_token_defined=np.arange(length) > 0,
                    matching_successor_count=counts, matching_successor_defined=counts > 0,
                    matching_successor_uniform_attention=counts / prefix_lengths,
                    validation=dict(passed=True, original_checked_this_call=original_checked,
                                    original_validation=copy.deepcopy(original_receipt),
                                    replay_tolerance=self.replay_tolerance, conditions=checks),
                    attention_semantics="At input indexq, matching successor sites are1<=s<=q with input[s-1]==input[q]. A completed self-repeat at s=q is included; the future target is excluded.")

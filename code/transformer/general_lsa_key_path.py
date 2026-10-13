"""Measure the copying candidate's mediated effect through second-layer keys.

The donor deletes one first-layer head. Only its second-layer key vectors
are inserted into a recipient forward. Recipient queries, values, residual
states, and parameters retain their intact computation. This is a path
intervention, not a claim that its remainder is a pure unigram predictor.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from circuit_probe import run_probe
from copier_probes import make_probes
from identify_copier import (contrasts, probability_summaries, query_inputs,
                             sha, statistics, validate_forward)
from model import GPT


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


@torch.no_grad()
def run_key_path(model, idx, *, donor_head, return_cache=False):
    """Return the tested probe output with donor second-layer keys only.

    ``donor_head=None`` is a sham using intact donor keys. Otherwise the
    donor deletes the specified zero-based first-layer head. Temporary
    hooks are always removed, including if a forward raises an exception.
    No tensors in the model, input, or existing hook registries are edited.
    """
    if len(model.blocks) != 2:
        raise ValueError("This path test requires the fixed two-layer architecture")
    if donor_head is not None:
        donor_head = tuple(donor_head)
        if (len(donor_head) != 2 or donor_head[0] != 0
                or not 0 <= donor_head[1] < model.blocks[0].attn.n_head):
            raise ValueError("Donor ablation must identify a first-layer head")
    target = model.blocks[1].attn.qkv
    width = model.blocks[1].attn.d_model
    captured = {}

    def capture(_module, _inputs, output):
        if "keys" in captured or output.shape[-1] != 3 * width:
            raise ValueError("Unexpected donor QKV call or shape")
        captured["keys"] = output[..., width:2 * width].detach().clone()

    hook = target.register_forward_hook(capture)
    try:
        run_probe(model, idx, head_scales={} if donor_head is None else {donor_head: 0.0},
                  return_cache=False)
    finally:
        hook.remove()
    if "keys" not in captured:
        raise ValueError("Donor forward did not produce second-layer keys")

    def replace_keys(_module, _inputs, output):
        keys = captured["keys"]
        if output[..., width:2 * width].shape != keys.shape:
            raise ValueError("Donor and recipient key shapes differ")
        return torch.cat((output[..., :width], keys, output[..., 2 * width:]), dim=-1)

    hook = target.register_forward_hook(replace_keys)
    try:
        return run_probe(model, idx, return_cache=return_cache)
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-probes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026091722)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if (selection.get("stage") != "discovery" or not selection.get("positive_candidate")
            or args.seed in (selection["seed"], 2026091721)):
        raise ValueError("Require frozen discovery and fresh controlled probes")
    if args.n_probes < 2 or args.batch < 1 or not 1 <= args.threads <= 4:
        raise ValueError("Invalid probe count, batch size, or thread count")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; existing records are immutable")
    for path, expected in selection["provenance"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Discovery provenance changed: " + path)
    candidate = tuple(selection["candidate_head"])
    control = tuple(selection["control_head"])
    checkpoint = ROOT / "results/transformer/runs/d64/model_100000.pt"
    config_path = checkpoint.parent / "config.json"
    if str(checkpoint.relative_to(ROOT)) not in selection["provenance"]:
        raise ValueError("Selection does not identify the declared checkpoint")
    config = json.loads(config_path.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.set_num_threads(args.threads)
    with torch.random.fork_rng():
        float_model = GPT(**config["model"]).eval()
    float_model.load_state_dict(saved["model"], strict=True)
    model = copy.deepcopy(float_model).double()
    probes = make_probes(args.n_probes, seed=args.seed)
    fidelity = validate_forward(model, float_model, probes["clean"])
    initial_hooks = dict(model.blocks[1].attn.qkv._forward_hooks)
    provenance = dict(selection["provenance"])
    provenance.update({str(path.resolve().relative_to(ROOT)): sha(path)
                       for path in (Path(__file__), args.selection)})
    protocol = dict(schema="general-lsa-key-path-controlled-v1", n_probes=args.n_probes,
                    seed=args.seed, controlled_length=64, candidate_head=list(candidate),
                    control_head=list(control), source={key: config[key] for key in
                    ("d", "alpha", "ls", "no_permute", "seed")},
                    checkpoint=str(checkpoint.relative_to(ROOT)), architecture=config["model"],
                    provenance=provenance, fidelity=fidelity,
                    intervention="Intact recipient second-layer Q,V,residual; replace all second-layer K by keys from same-input donor with one first-layer head deleted.",
                    interpretation="Mediated effect through retrieval keys; not an isolated unigram or pure copying contribution.")
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    variants = dict(sham=None, candidate_keys_off=candidate, control_keys_off=control)
    outputs = {name: np.empty((args.n_probes, 6, 64))
               for name in ("intact", *variants)}
    checks = dict(sham_max_logit_error=0.0, first_prediction_max_logit_error=0.0,
                  causal_prefix_max_logit_error=0.0)
    started = time.monotonic()
    for left in range(0, args.n_probes, args.batch):
        right = min(left + args.batch, args.n_probes)
        size = right - left
        tokens = torch.from_numpy(query_inputs(probes, left, right))
        intact = run_probe(model, tokens, return_cache=False)["logits"]
        outputs["intact"][left:right] = intact[:, -1].numpy().reshape(6, size, 64).transpose(1, 0, 2)
        for name, head in variants.items():
            logits = run_key_path(model, tokens, donor_head=head)["logits"]
            first_error = float((logits[:, 0] - intact[:, 0]).abs().max())
            checks["first_prediction_max_logit_error"] = max(checks["first_prediction_max_logit_error"], first_error)
            if name == "sham":
                checks["sham_max_logit_error"] = max(checks["sham_max_logit_error"], float((logits - intact).abs().max()))
            if left == 0:
                for prefix in (1, 7, 32):
                    partial = run_key_path(model, tokens[:4, :prefix], donor_head=head)["logits"][:, -1]
                    error = float((partial - logits[:4, prefix - 1]).abs().max())
                    checks["causal_prefix_max_logit_error"] = max(checks["causal_prefix_max_logit_error"], error)
            outputs[name][left:right] = logits[:, -1].numpy().reshape(6, size, 64).transpose(1, 0, 2)
        if left % (8 * args.batch) == 0 or right == args.n_probes:
            print(json.dumps(dict(completed=right, total=args.n_probes,
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
    if max(checks.values()) > 1e-10:
        raise ValueError("Key-path sham, first-prediction, or causal-prefix check failed")
    if dict(model.blocks[1].attn.qkv._forward_hooks) != initial_hooks:
        raise ValueError("Temporary forward hook was not removed")
    if any(not torch.equal(value, saved["model"][name].double())
           for name, value in model.state_dict().items()):
        raise ValueError("Intervention changed model parameters")
    checks.update(hooks_restored=True, parameters_preserved=True)
    metrics = {name: contrasts(logits, probes) for name, logits in outputs.items()}
    conditions = {name: {**{key: statistics(value) for key, value in values.items()},
                         "reduction_from_intact": statistics(metrics["intact"]["query_specific"] - values["query_specific"]),
                         "probabilities": probability_summaries(outputs[name], probes)}
                  for name, values in metrics.items()}
    arrays = {"probe_" + key: value for key, value in probes.items()}
    arrays.update({"logits_" + name: values for name, values in outputs.items()})
    np.savez_compressed(args.out / "records.npz", **arrays)
    summary = dict(protocol=protocol, checks=checks, conditions=conditions,
                   elapsed_seconds=time.monotonic() - started,
                   records_sha256=sha(args.out / "records.npz"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: values["query_specific"] for name, values in conditions.items()}), flush=True)


if __name__ == "__main__":
    main()

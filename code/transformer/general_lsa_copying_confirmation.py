"""Held-out causal tests of the frozen shared-base LSA copying candidate."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from copier_probes import make_probes
from identify_copier import controlled, sha, summarize, validate_forward
from model import GPT
from trace_copier_key_path import run_controlled, summarize as summarize_routing


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-probes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026091721)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if (selection.get("stage") != "discovery" or not selection.get("positive_candidate")
            or args.seed == selection["seed"]):
        raise ValueError("Require frozen positive discovery and an independent probe seed")
    if args.n_probes < 2 or args.batch < 1 or not 1 <= args.threads <= 4:
        raise ValueError("Invalid probe count, batch size, or thread count")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; previous results are immutable")
    for path, expected in selection["provenance"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Discovery provenance changed: " + path)
    candidate = tuple(selection["candidate_head"])
    control = tuple(selection["control_head"])
    if candidate[0] != 0:
        raise ValueError("The existing site-routing confirmation tests first-layer candidates")
    checkpoint = ROOT / "results/transformer/runs/d64/model_100000.pt"
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if str(checkpoint.relative_to(ROOT)) not in selection["provenance"]:
        raise ValueError("Discovery did not use the declared shared-base checkpoint")
    torch.set_num_threads(args.threads)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["step"] != 100000 or saved["cfg"]["model"] != config["model"]:
        raise ValueError("Checkpoint/configuration mismatch")
    with torch.random.fork_rng():
        float_model = GPT(**config["model"]).eval()
    float_model.load_state_dict(saved["model"], strict=True)
    model = copy.deepcopy(float_model).double()
    if any(not torch.equal(value, saved["model"][name].double())
           for name, value in model.state_dict().items()):
        raise ValueError("Loaded weights differ from checkpoint")
    probes = make_probes(args.n_probes, seed=args.seed)
    fidelity = validate_forward(model, float_model, probes["clean"])
    provenance = dict(selection["provenance"])
    provenance.update({str(path.resolve().relative_to(ROOT)): sha(path) for path in
                       (Path(__file__), HERE / "trace_copier_key_path.py", args.selection)})
    protocol = dict(schema="general-lsa-copying-confirmation-v1", stage="confirmation",
                    source={key: config[key] for key in ("d", "alpha", "ls", "no_permute", "seed")},
                    architecture=config["model"], step=100000,
                    n_probes=args.n_probes, seed=args.seed, controlled_length=64,
                    candidate_head=list(candidate), control_head=list(control),
                    discovery_selection=selection, provenance=provenance,
                    torch_threads=args.threads, batch=args.batch, fidelity=fidelity,
                    precision="float64 with unchanged float32-trained weights",
                    scope="Fresh controlled prompts; causal removal and successor-site exchange/restoration. Natural-source base preservation and payoff timing are separate tests.",
                    frozen_hypothesis="Removing the candidate reduces the paired query-specific successor response; swapping its two successor-site outputs reverses the response; other first-layer head swaps preserve it.")
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    outputs, attention, upstream = controlled(model, probes, args.batch)
    all_heads = summarize(outputs, attention, probes, upstream)
    routing_outputs, site_attention, full_attention, checks = run_controlled(
        model, probes, candidate, batch=args.batch)
    # Both independently reused evaluators must agree on unchanged input.
    intact_error = float(np.max(np.abs(outputs["intact"] - routing_outputs["intact"])))
    if intact_error > 1e-10:
        raise ValueError("Independent held-out evaluators disagree on intact logits")
    routing = summarize_routing(routing_outputs, site_attention, probes)
    arrays = {"probe_" + key: value for key, value in probes.items()}
    arrays.update({"all_heads_logits_" + key: value for key, value in outputs.items()})
    arrays["all_heads_attention_final"] = attention
    arrays.update({"routing_logits_" + key: value for key, value in routing_outputs.items()})
    arrays.update({"routing_attention_sites_" + key: value for key, value in site_attention.items()})
    arrays.update({"routing_attention_final_" + key: value for key, value in full_attention.items()})
    np.savez_compressed(args.out / "records.npz", **arrays)
    summary = dict(protocol=protocol, all_heads=all_heads, routing=routing,
                   checks={**checks, "independent_intact_logit_max_abs": intact_error},
                   records_sha256=sha(args.out / "records.npz"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["query_specific"]
                      for name, value in routing["conditions"].items()}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

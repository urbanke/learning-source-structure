"""Screen attention heads for copying in the existing shared-base LSA model.

This discovery reuses the frozen, count-matched controlled-prompt test.
It nominates candidates; it does not validate a copier or an isolated
unigram remainder. Natural-source preservation and held-out intervention
tests are separate follow-ups.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from copier_probes import make_probes
from identify_copier import (HEADS, controlled, label, sha, summarize,
                             validate_forward)
from model import GPT


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-probes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026091720)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.n_probes < 2 or args.batch < 1 or not 1 <= args.threads <= 4:
        raise ValueError("Require >=2 probes, positive batch, and 1..4 threads")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; previous records are immutable")
    checkpoint = ROOT / "results/transformer/runs/d64/model_100000.pt"
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    source = dict(d=64, n_ctx=1024, alpha=[0.5, 1.5], ls=[0, 8],
                  no_permute=False, seed=0)
    if any(config.get(key) != value for key, value in source.items()):
        raise ValueError("Checkpoint configuration is not the shared-base d64 model")
    expected_model = dict(vocab=64, n_ctx=1024, n_layer=2, n_head=4,
                          d_model=128, mlp=True, dropout=0.0)
    if config["model"] != expected_model:
        raise ValueError("Architecture does not match the controlled probe interface")
    torch.set_num_threads(args.threads)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["step"] != 100000 or saved["cfg"]["model"] != expected_model:
        raise ValueError("Wrong checkpoint step or architecture")
    checkpoint_source = {key: saved["cfg"].get(key) for key in source}
    # JSON config arrays and checkpoint tuples encode the same source range.
    checkpoint_source = json.loads(json.dumps(checkpoint_source))
    if checkpoint_source != source:
        raise ValueError("Checkpoint and configuration source identities disagree")
    with torch.random.fork_rng():
        float_model = GPT(**expected_model).eval()
    float_model.load_state_dict(saved["model"], strict=True)
    model = copy.deepcopy(float_model).double()
    for name, value in model.state_dict().items():
        if not torch.equal(value, saved["model"][name].double()):
            raise ValueError("Loaded weights differ from checkpoint")
    probes = make_probes(args.n_probes, seed=args.seed)
    fidelity = validate_forward(model, float_model, probes["clean"])
    provenance_paths = (checkpoint, config_path, Path(__file__), HERE / "model.py",
                        HERE / "source.py", HERE / "circuit_probe.py",
                        HERE / "identify_copier.py", HERE / "copier_probes.py")
    provenance = {str(path.relative_to(ROOT)): sha(path) for path in provenance_paths}
    protocol = dict(schema="general-lsa-copying-discovery-v1", stage="discovery",
                    source=source, architecture=expected_model, step=100000,
                    training_sequences=config["batch"] * saved["step"],
                    checkpoint=str(checkpoint.relative_to(ROOT)),
                    n_probes=args.n_probes, seed=args.seed, controlled_length=64,
                    batch=args.batch, torch_threads=args.threads,
                    precision="float64 with unchanged float32-trained parameter values",
                    observations="Controlled prompts, not natural source sequences",
                    heads_zero_indexed=[list(key) for key in HEADS],
                    provenance=provenance, fidelity=fidelity)
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.monotonic()
    outputs, attention, upstream = controlled(model, probes, args.batch)
    report = summarize(outputs, attention, probes, upstream)
    ordered = sorted(HEADS, key=lambda key: report["conditions"]["ablate_" + label(key)]
                     ["reduction_from_intact"]["mean"], reverse=True)
    candidate, control = ordered[0], ordered[-1]
    candidate_reduction = report["conditions"]["ablate_" + label(candidate)]["reduction_from_intact"]
    selection = dict(stage="discovery", candidate_head=list(candidate),
                     control_head=list(control), candidate_label=label(candidate),
                     control_label=label(control),
                     candidate_reduction_bits=candidate_reduction,
                     positive_candidate=candidate_reduction["mean"] > 0,
                     validated=False,
                     rule="Maximum mean reduction in A-minus-unseen-E successor-swap log-odds response across all eight single-head ablations; control is minimum reduction.",
                     seed=args.seed, n_probes=args.n_probes, provenance=provenance)
    arrays = {"probe_" + key: value for key, value in probes.items()}
    arrays.update({"logits_" + key: value for key, value in outputs.items()})
    arrays["attention_final"] = attention
    np.savez_compressed(args.out / "records.npz", **arrays)
    summary = dict(protocol=protocol, **report, selection=selection,
                   elapsed_seconds=time.monotonic() - start,
                   records_sha256=sha(args.out / "records.npz"))
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    compact = dict(status="discovery_completed", validated=False,
                   intact_response=report["conditions"]["intact"]["query_specific"],
                   candidate_head=selection["candidate_head"],
                   control_head=selection["control_head"],
                   candidate_reduction_bits=candidate_reduction,
                   elapsed_seconds=summary["elapsed_seconds"])
    print(json.dumps(compact, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

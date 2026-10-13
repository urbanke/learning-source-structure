"""Test the discovered first-layer copier candidate at two fixed input sites.

This follow-up changes cached head outputs within each unchanged prompt. It
tests a local routing hypothesis, not sequence-time emergence or an isolated
second-layer key projection: the downstream residual/MLP can change both K
and V. All indices in the raw records are zero based.
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
from copier_probes import make_probes, validate_probes
from identify_copier import (ROOT, HERE, contrasts, label, probability_summaries,
                             query_inputs, sha, statistics)
from model import GPT

SEED = 20261003
GROUPS = ("clean_A", "counterfactual_A", "clean_D", "counterfactual_D",
          "clean_E", "counterfactual_E")


def site_patch(values, first, second, *, mode):
    """Return a patch which exchanges or restores exactly two sites per row.

    ``swap`` exchanges the two original vectors simultaneously; ``restore``
    zeros every site except these two, and ``sham`` reinstalls their own
    unchanged vectors. No input/cache tensor is mutated. Restore uses a
    whole-head patch with no scale so it survives the probe's patch-then-scale
    ordering.
    """
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("Expected cached head outputs [batch, length, width]")
    batch, length, _ = values.shape
    positions = []
    for sites in (first, second):
        sites = torch.as_tensor(sites, device=values.device)
        if sites.shape != (batch,) or sites.dtype not in (torch.int32, torch.int64):
            raise ValueError("Each site vector must contain one integer per row")
        if bool(((sites < 0) | (sites >= length)).any()):
            raise ValueError("Patch site outside the context")
        positions.append(sites.long())
    first, second = positions
    if bool((first == second).any()):
        raise ValueError("Require two distinct sites in every prompt")
    rows = torch.arange(batch, device=values.device)
    mask = torch.zeros((batch, length), dtype=torch.bool, device=values.device)
    mask[rows, first] = mask[rows, second] = True
    if mode == "restore":
        changed = torch.zeros_like(values)
        changed[rows, first] = values[rows, first]
        changed[rows, second] = values[rows, second]
        return dict(values=changed)
    changed = values.clone()
    if mode == "swap":
        changed[rows, first] = values[rows, second]
        changed[rows, second] = values[rows, first]
    elif mode != "sham":
        raise ValueError("Unknown site intervention")
    return dict(values=changed, mask=mask)


def batched_sites(probes, left, right):
    """Match the six group-major batches made by query_inputs exactly."""
    return tuple(np.tile(probes[key][left:right], len(GROUPS))
                 for key in ("source_b_pos", "source_c_pos"))


def run_controlled(model, probes, candidate, *, batch=8):
    validate_probes(probes, vocab=64)
    if candidate[0] != 0 or not 0 <= candidate[1] < 4 or batch < 1:
        raise ValueError("Require a first-layer candidate and positive batch")
    n, length = probes["clean"].shape
    controls = [(0, h) for h in range(4) if h != candidate[1]]
    variants = ["intact", "zero_candidate", "swap_successors", "restore_successors",
                "restore_contexts", "swap_contexts", "sham"]
    variants += ["swap_"+label(key) for key in controls]
    outputs = {name: np.empty((n, 6, 64)) for name in variants}
    site_attention = {name: np.empty((n, 6, 4, 2)) for name in variants}
    full_attention = {name: np.empty((n, 6, 4, length))
                      for name in ("intact", "swap_successors")}
    started = time.monotonic()
    sham_error = 0.0
    for left in range(0, n, batch):
        right = min(left+batch, n)
        size = right-left
        tokens = torch.from_numpy(query_inputs(probes, left, right))
        before = tokens.clone()
        cache = run_probe(model, tokens)
        first, second = batched_sites(probes, left, right)
        rows = torch.arange(6*size)

        def store(name, result):
            outputs[name][left:right] = result["logits"][:, -1].numpy().reshape(6, size, 64).transpose(1, 0, 2)
            attn = result["attention"][-1][:, :, -1]
            chosen = torch.stack((attn[rows, :, torch.from_numpy(first)],
                                  attn[rows, :, torch.from_numpy(second)]), dim=-1)
            site_attention[name][left:right] = chosen.numpy().reshape(6, size, 4, 2).transpose(1, 0, 2, 3)
            if name in full_attention:
                full_attention[name][left:right] = attn.numpy().reshape(6, size, 4, length).transpose(1, 0, 2, 3)

        store("intact", cache)
        store("zero_candidate", run_probe(model, tokens, head_scales={candidate: 0.0}))
        values = cache["head_outputs"][candidate[0]][:, candidate[1]]
        for name, mode, offset in (("swap_successors", "swap", 0),
                                   ("restore_successors", "restore", 0),
                                   ("restore_contexts", "restore", -1),
                                   ("swap_contexts", "swap", -1),
                                   ("sham", "sham", 0)):
            patch = site_patch(values, first+offset, second+offset, mode=mode)
            result = run_probe(model, tokens, head_patches={candidate: patch})
            if name == "sham":
                sham_error = max(sham_error, float((result["logits"]-cache["logits"]).abs().max()))
            store(name, result)
        for key in controls:
            values = cache["head_outputs"][key[0]][:, key[1]]
            patch = site_patch(values, first, second, mode="swap")
            store("swap_"+label(key), run_probe(model, tokens, head_patches={key: patch}))
        if not torch.equal(tokens, before):
            raise ValueError("An intervention changed input tokens")
        if left % (8*batch) == 0 or right == n:
            print(json.dumps(dict(completed=right, total=n,
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
    if sham_error > 1e-10:
        raise ValueError("Self patch changed the faithful forward computation")
    if not all(np.isfinite(x).all() for x in outputs.values()):
        raise ValueError("Nonfinite intervention logits")
    if not all(np.isfinite(x).all() and np.min(x) >= 0 and np.max(x) <= 1
               for x in site_attention.values()):
        raise ValueError("Invalid recorded attention probabilities")
    attention_sum_error = max(float(np.max(np.abs(x.sum(axis=-1)-1)))
                              for x in full_attention.values())
    if attention_sum_error > 1e-10:
        raise ValueError("Final-query attention does not normalize")
    return outputs, site_attention, full_attention, dict(sham_max_logit_error=sham_error,
                                                        full_attention_sum_error=attention_sum_error,
                                                        elapsed_seconds=time.monotonic()-started)


def odds(logits, probes):
    rows, groups = np.arange(len(logits))[:, None], np.arange(6)[None, :]
    return (logits[rows, groups, probes["target_b"][:, None]]
            - logits[rows, groups, probes["target_c"][:, None]]) / np.log(2.0)


def summarize(outputs, site_attention, probes):
    intact = contrasts(outputs["intact"], probes)
    zero = contrasts(outputs["zero_candidate"], probes)
    intact_odds = odds(outputs["intact"], probes)
    metrics = {}
    for name, logits in outputs.items():
        values = contrasts(logits, probes)
        changed_odds = odds(logits, probes)-intact_odds
        entry = {key: statistics(value) for key, value in values.items()}
        entry["reduction_from_intact"] = statistics(intact["query_specific"]-values["query_specific"])
        entry["recovery_from_zero"] = statistics(values["query_specific"]-zero["query_specific"])
        entry["probabilities"] = probability_summaries(logits, probes)
        entry["log_odds_B_over_C"] = {group: statistics(odds(logits, probes)[:, g])
                                      for g, group in enumerate(GROUPS)}
        entry["log_odds_change_from_intact"] = {group: statistics(changed_odds[:, g])
                                                for g, group in enumerate(GROUPS)}
        metrics[name] = entry
    attention = {}
    for name, values in site_attention.items():
        attention[name] = {}
        for g, group in enumerate(GROUPS):
            attention[name][group] = {}
            for head in range(4):
                difference = values[:, g, head, 0]-values[:, g, head, 1]
                original = site_attention["intact"][:, g, head, 0]-site_attention["intact"][:, g, head, 1]
                attention[name][group][label((1, head))] = {
                    "earlier_A_successor_site": statistics(values[:, g, head, 0]),
                    "earlier_D_successor_site": statistics(values[:, g, head, 1]),
                    "site_attention_difference": statistics(difference),
                    "difference_change_from_intact": statistics(difference-original)}
    return dict(conditions=metrics, final_layer_attention=attention,
                units="Log-odds and their differences are bits; attention and probabilities are dimensionless.",
                uncertainty="One standard error across independent probe families. All six queries/conditions and every intervention remain paired.",
                interpretation="These are fixed-input upstream-head interventions. Attention redirection supports routing through the tested sites; the interventions can affect both downstream keys and values and do not establish an isolated key circuit, natural-source prevalence, or emergence timing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"output/copier_circuit/key_path_2026-09-15")
    parser.add_argument("--selection", type=Path, default=ROOT/"output/copier_circuit/discovery_2026-09-15/selection.json")
    parser.add_argument("--n-probes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.n_probes < 2 or args.batch < 1 or args.seed in (20260930, 20261001, 20261002):
        raise ValueError("Require independent follow-up seed, at least two probes and positive batch")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; completed and partial runs are immutable")
    selection = json.loads(args.selection.read_text())
    if (selection.get("stage") != "discovery" or not selection.get("selection_valid")
            or selection.get("n_probes") != 256 or selection.get("discovery_seed") != 20260930):
        raise ValueError("Require the frozen 256-probe discovery selection")
    for path, expected in selection["provenance"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("Stale discovery input: "+path)
    candidates = [tuple(k) for k in selection["selected_heads"] if k[0] == 0]
    if len(candidates) != 1 or candidates[0] != (0, 1):
        raise ValueError("This predeclared follow-up targets the discovered L1H2")
    candidate = candidates[0]
    checkpoint = ROOT/"results/transformer/runs/lsa_a0_d64/model_100000.pt"
    config_path = checkpoint.parent/"config.json"
    config = json.loads(config_path.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = dict(family="lsa", alpha=[0., 0.], ls=[0, 8], seed=0, d=64, n_ctx=1024)
    if saved["step"] != 100000 or any(saved["cfg"].get(k) != v or config.get(k) != v for k, v in required.items()):
        raise ValueError("Wrong checkpoint/source configuration")
    if saved["cfg"]["model"] != config["model"] or config.get("no_permute") is not False or saved["cfg"].get("no_permute") is not False:
        raise ValueError("Checkpoint/configuration mismatch")
    torch.set_num_threads(1)
    float_model = GPT(**config["model"]).eval()
    float_model.load_state_dict(saved["model"])
    model = copy.deepcopy(float_model).double()
    probes = make_probes(args.n_probes, seed=args.seed)
    provenance = dict(selection["provenance"])
    provenance[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    provenance[str(args.selection.resolve().relative_to(ROOT))] = sha(args.selection)
    protocol = dict(schema="copier-key-path-v1", n_probes=args.n_probes, seed=args.seed,
                    controlled_length=64, candidate=list(candidate), group_order=GROUPS,
                    selection=selection, provenance=provenance,
                    intervention="First-layer pre-projection head output exchange/restoration within each unchanged input; restore zeros the head everywhere else and uses no scale.",
                    predicted_swap="Swap A-successor and D-successor site outputs: A clean should move B/C odds downward, A counterfactual upward; D should reverse these directions. E is a neutral query control.",
                    predicted_restoration="Restoring only the two successor-site outputs after head deletion should recover part of the query-specific response; restoring context sites is a control.",
                    controls="Context-site swap and restoration, same-site self patch, and successor-site swaps in the other three first-layer heads. Context-site controls are off the hypothesized path, not asserted to be universally irrelevant.",
                    precision="Float64 computation of unchanged float32-trained parameters; original forward independently certified in the frozen discovery/confirmation runner.",
                    scope="Fresh controlled follow-up after discovery and initial confirmation. No natural-source or sequence-time experiment; not an isolated downstream K-only intervention.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    outputs, site_attention, full_attention, checks = run_controlled(model, probes, candidate, batch=args.batch)
    arrays = {"probe_"+k: v for k, v in probes.items()}
    arrays.update({"logits_"+k: v for k, v in outputs.items()})
    arrays.update({"attention_sites_"+k: v for k, v in site_attention.items()})
    arrays.update({"attention_final_"+k: v for k, v in full_attention.items()})
    np.savez_compressed(args.out/"records.npz", **arrays)
    summary = dict(protocol=protocol, checks=checks, **summarize(outputs, site_attention, probes),
                   records_sha256=sha(args.out/"records.npz"))
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: value["query_specific"] for name, value in summary["conditions"].items()}), flush=True)


if __name__ == "__main__":
    main()

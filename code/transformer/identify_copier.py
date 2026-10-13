"""Identify and causally test copying in the existing uniform-base transformer.

Discovery and confirmation are separate runs. These controlled prompts test
an internal computation; they are not the planned sequence-time experiment.
All original model/source/reference files remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from circuit_probe import run_probe
from copier_probes import make_probes, DISCOVERY_SEED, HELDOUT_SEED
from model import GPT
from source import Source

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HEADS = [(layer, head) for layer in range(2) for head in range(4)]
LN2 = np.log(2.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(key):
    return f"L{key[0]+1}H{key[1]+1}"


def statistics(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Statistics require finite independent-probe values")
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1)/np.sqrt(len(x))), n=len(x))


def query_inputs(probes, left, right):
    clean, changed = (probes[key][left:right] for key in ("clean", "counterfactual"))
    dclean, dchanged = clean.copy(), changed.copy()
    dclean[:, -1] = dchanged[:, -1] = probes["context_d"][left:right]
    return np.concatenate((clean, changed, dclean, dchanged,
                           probes["neutral_clean"][left:right],
                           probes["neutral_counterfactual"][left:right]))


def contrasts(logits, probes):
    """Output per-probe log-odds responses: A query, D query, unseen E query."""
    n = len(probes["target_b"])
    if logits.shape != (n, 6, 64):
        raise ValueError("Expected six query/condition distributions per probe")
    rows, groups = np.arange(n)[:, None], np.arange(6)[None, :]
    odds = (logits[rows, groups, probes["target_b"][:, None]]
            - logits[rows, groups, probes["target_c"][:, None]]) / LN2
    delta = odds[:, ::2] - odds[:, 1::2]
    return dict(A=delta[:, 0], D=delta[:, 1], E=delta[:, 2],
                query_specific=delta[:, 0]-delta[:, 2])


def probability_summaries(logits, probes):
    p = np.exp(logits-logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    rows = np.arange(len(p))
    b, c = probes["target_b"], probes["target_c"]
    return dict(demonstrated_successor=statistics((p[rows, 0, b]+p[rows, 1, c])/2),
                other_successor=statistics((p[rows, 0, c]+p[rows, 1, b])/2))


def validate_forward(model, float_model, inputs):
    tokens = torch.from_numpy(inputs[:4])
    manual = copy.deepcopy(model)
    for block in manual.blocks:
        block.attn.manual = True
    with torch.inference_mode():
        observed = run_probe(model, tokens)["logits"]
        reference = manual(tokens)
        original = float_model(tokens).double()
        manual_error = float((reference-observed).abs().max())
        probability_error = float((observed.softmax(-1)-original.softmax(-1)).abs().max())
        causal_error = 0.0
        for t in (1, 7, 32):
            truncated = run_probe(model, tokens[:, :t])["logits"][:, -1]
            causal_error = max(causal_error, float((truncated-observed[:, t-1]).abs().max()))
    if manual_error > 1e-10 or causal_error > 1e-10 or probability_error > 1e-5:
        raise ValueError(f"Instrument fidelity failed: {manual_error}, {causal_error}, {probability_error}")
    return dict(double_manual_logit_max_abs=manual_error,
                double_causal_logit_max_abs=causal_error,
                original_float32_probability_max_abs=probability_error,
                tolerances=dict(double_logits=1e-10, float32_probability=1e-5))


def controlled(model, probes, batch, selection=None):
    n, length = probes["clean"].shape
    variants = ["intact"]+["ablate_"+label(k) for k in HEADS]
    selected = [tuple(k) for k in selection["selected_heads"]] if selection else []
    controls = [tuple(k) for k in selection["control_heads"]] if selection else []
    if selection:
        variants += ["ablate_selected", "ablate_controls"]
        variants += ["patch_"+label(k) for k in selected+controls]
        variants += ["patch_selected", "patch_sham", "patch_irrelevant_sites"]
    outputs = {name: np.empty((n, 6, 64)) for name in variants}
    attention = np.empty((n, 6, 2, 4, length))
    upstream_attention = np.empty_like(attention) if selected else None
    started = time.monotonic()
    for left in range(0, n, batch):
        right = min(left+batch, n)
        size = right-left
        tokens = torch.from_numpy(query_inputs(probes, left, right))
        cache = run_probe(model, tokens)
        attention[left:right] = torch.stack([a[:, :, -1] for a in cache["attention"]], dim=1).numpy().reshape(6, size, 2, 4, length).transpose(1, 0, 2, 3, 4)
        donor = torch.cat([torch.arange(g*size, (g+1)*size) for g in (1, 0, 3, 2, 5, 4)])

        def save(name, result):
            outputs[name][left:right] = result["logits"][:, -1].numpy().reshape(6, size, 64).transpose(1, 0, 2)

        def patches(keys, *, sham=False, irrelevant=False):
            result = {}
            for layer, head in keys:
                values = cache["head_outputs"][layer][:, head]
                mask = torch.zeros((6*size, length), dtype=torch.bool)
                if irrelevant:
                    pos = np.tile(probes["source_b_pos"][left:right]+1, 6)
                    mask[torch.arange(6*size), torch.from_numpy(pos)] = True
                elif layer == 0:
                    mask[:] = True
                else:
                    mask[:, -1] = True
                result[(layer, head)] = dict(values=values if sham else values[donor], mask=mask)
            return result

        save("intact", cache)
        for key in HEADS:
            result = run_probe(model, tokens, head_scales={key: 0.0}, return_cache=bool(selected and key == selected[0]))
            save("ablate_"+label(key), result)
            if selected and key == selected[0]:
                upstream_attention[left:right] = torch.stack([a[:, :, -1] for a in result["attention"]], dim=1).numpy().reshape(6, size, 2, 4, length).transpose(1, 0, 2, 3, 4)
        if selection:
            for name, keys in (("ablate_selected", selected), ("ablate_controls", controls)):
                save(name, run_probe(model, tokens, head_scales={k: 0.0 for k in keys}, return_cache=False))
            for key in selected+controls:
                save("patch_"+label(key), run_probe(model, tokens, head_patches=patches([key]), return_cache=False))
            for name, kwargs in (("patch_selected", {}), ("patch_sham", {"sham": True}),
                                 ("patch_irrelevant_sites", {"irrelevant": True})):
                save(name, run_probe(model, tokens, head_patches=patches(selected, **kwargs), return_cache=False))
        if left % (batch*8) == 0 or right == n:
            print(json.dumps(dict(completed=right, total=n, elapsed_seconds=time.monotonic()-started)), flush=True)
    if selection and np.max(np.abs(outputs["patch_sham"]-outputs["intact"])) > 1e-10:
        raise ValueError("Self-patching changed the logits")
    return outputs, attention, upstream_attention


def summarize(outputs, attention, probes, upstream_attention=None):
    metrics = {name: contrasts(logits, probes) for name, logits in outputs.items()}
    report = {}
    for name, values in metrics.items():
        report[name] = {key: statistics(v) for key, v in values.items()}
        report[name]["reduction_from_intact"] = statistics(metrics["intact"]["query_specific"]-values["query_specific"])
        report[name]["probabilities"] = probability_summaries(outputs[name], probes)
    positions = probes["source_b_pos"]
    rows = np.arange(len(positions))
    def attention_stats(a):
        return {label(k): dict(successor=statistics(a[rows, 0, k[0], k[1], positions]),
                              context=statistics(a[rows, 0, k[0], k[1], positions-1])) for k in HEADS}
    return dict(conditions=report, head_attention=attention_stats(attention),
                head_attention_after_upstream_ablation=(attention_stats(upstream_attention) if upstream_attention is not None else None),
                units="log-odds differences in bits; attention and probabilities are dimensionless",
                primary_metric="(log2 odds B/C in AB...DC...A minus AC...DB...A) minus the same difference with final unseen E",
                D_query_control="Replacing final A by D should reverse the signed response to the B/C swap.",
                patch_direction="Both directions: each sequence receives donor head outputs from its paired successor-swap sequence.",
                patch_sites="All positions for first-layer heads; final query only for second-layer heads. Irrelevant-site control patches the position immediately after the earlier A-successor.",
                uncertainty="One standard error across independent controlled probe pairs; each pair's six query/condition outputs remain together.")


def natural_discovery(model, n=256, batch=8):
    tokens, meta = Source(64, 129, alpha=(0., 0.), ls=(0, 8), permute=True).batch(n, np.random.default_rng(20261002))
    previous, induction, baseline = [], [], []
    for left in range(0, n, batch):
        x = torch.from_numpy(tokens[left:left+batch, :-1])
        cache = run_probe(model, x)
        size, length = x.shape
        match = torch.zeros((size, length, length), dtype=torch.bool)
        match[:, :, 1:] = x[:, :, None] == x[:, None, :-1]
        match &= torch.ones((length, length), dtype=torch.bool).tril()[None]
        valid = match.any(dim=-1)
        denom = valid.sum(dim=-1).clamp_min(1)
        prev_all, induction_all = [], []
        for a in cache["attention"]:
            prev_all.append(a[:, :, torch.arange(1, length), torch.arange(length-1)].mean(dim=-1).numpy())
            score = (a*match[:, None]).sum(dim=-1)
            induction_all.append(((score*valid[:, None]).sum(-1)/denom[:, None]).numpy())
        previous.append(np.stack(prev_all, axis=1))
        induction.append(np.stack(induction_all, axis=1))
        baseline.append(((match.sum(-1)/(torch.arange(length)+1))*valid).sum(-1).numpy()/denom.numpy())
    previous, induction, baseline = map(np.concatenate, (previous, induction, baseline))
    return dict(tokens=tokens, depths=meta["ls"], previous=previous, induction=induction, chance=baseline), {
        label(k): dict(previous_token_attention=statistics(previous[:, k[0], k[1]]),
                       matching_successor_attention=statistics(induction[:, k[0], k[1]]),
                       matching_successor_uniform_attention=statistics(baseline)) for k in HEADS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("discovery", "confirm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-probes", type=int)
    parser.add_argument("--natural-n", type=int, default=256)
    args = parser.parse_args()
    n = args.n_probes or (256 if args.stage == "discovery" else 1024)
    if n < 2 or args.batch < 1 or args.natural_n < 2:
        raise ValueError("Require at least two independent examples and positive batch size")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory; existing records are immutable")
    torch.set_num_threads(1)
    checkpoint = ROOT/"results/transformer/runs/lsa_a0_d64/model_100000.pt"
    config_path = checkpoint.parent/"config.json"
    config = json.loads(config_path.read_text())
    if (config["family"], config["alpha"], config["ls"], config["seed"]) != ("lsa", [0., 0.], [0, 8], 0):
        raise ValueError("Wrong uniform-base model configuration")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["step"] != 100000 or saved["cfg"]["model"] != config["model"]:
        raise ValueError("Checkpoint/configuration mismatch")
    float_model = GPT(**config["model"]).eval()
    float_model.load_state_dict(saved["model"])
    model = copy.deepcopy(float_model).double()
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in
                  (checkpoint, config_path, HERE/"model.py", HERE/"source.py", HERE/"circuit_probe.py", HERE/"copier_probes.py", Path(__file__))}
    selection = None
    if args.stage == "confirm":
        if not args.selection:
            raise ValueError("Confirmation requires a frozen discovery selection")
        selection = json.loads(args.selection.read_text())
        if (selection["provenance"] != provenance or selection["stage"] != "discovery"
                or not selection.get("selection_valid", False)):
            raise ValueError("Selection belongs to different model/code inputs")
    seed = DISCOVERY_SEED if args.stage == "discovery" else HELDOUT_SEED
    probes = make_probes(n, seed=seed)
    fidelity = validate_forward(model, float_model, probes["clean"])
    args.out.mkdir(parents=True)
    protocol = dict(stage=args.stage, n_probes=n, controlled_length=64, seed=seed,
                    natural_discovery_n=args.natural_n if args.stage == "discovery" else 0,
                    precision="float64 forward with unchanged float32-trained parameter values",
                    provenance=provenance, selection=selection, fidelity=fidelity)
    (args.out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    outputs, attention, upstream = controlled(model, probes, args.batch, selection)
    arrays = {"probe_"+k: v for k, v in probes.items()}
    arrays.update({"logits_"+k: v for k, v in outputs.items()})
    arrays["attention_final"] = attention
    if upstream is not None:
        arrays["attention_after_upstream_ablation"] = upstream
    summary = dict(protocol=protocol, **summarize(outputs, attention, probes, upstream))
    if args.stage == "discovery":
        selected, controls, unresolved_layers = [], [], []
        for layer in range(2):
            ordered = sorted([k for k in HEADS if k[0] == layer],
                             key=lambda k: summary["conditions"]["ablate_"+label(k)]["reduction_from_intact"]["mean"], reverse=True)
            if summary["conditions"]["ablate_"+label(ordered[0])]["reduction_from_intact"]["mean"] <= 0:
                unresolved_layers.append(layer)
                continue
            selected.append(ordered[0]); controls.append(ordered[-1])
        selection = dict(stage="discovery", selected_heads=selected, control_heads=controls,
                         selection_valid=not unresolved_layers, unresolved_layers=unresolved_layers,
                         rule="Per layer, maximal positive discovery reduction in A-versus-E paired log-odds response; controls are minima. This selects candidates, not a validated circuit.",
                         provenance=provenance, discovery_seed=seed, n_probes=n)
        (args.out/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
        natural, summary["natural_discovery"] = natural_discovery(model, args.natural_n, args.batch)
        arrays.update({"natural_"+k: v for k, v in natural.items()})
        summary["selection"] = selection
    np.savez_compressed(args.out/"records.npz", **arrays)
    summary["records_sha256"] = sha(args.out/"records.npz")
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(stage=args.stage, n_probes=n, selected=selection["selected_heads"],
                         intact=summary["conditions"]["intact"]["query_specific"], fidelity=fidelity)), flush=True)
    if not selection.get("selection_valid", False):
        raise SystemExit("No positive candidate in at least one layer; complete records retained for review")


if __name__ == "__main__":
    main()

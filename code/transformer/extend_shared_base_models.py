"""Stream a reproducible extension of the original shared-base test cohort.

Only source draws and existing-transformer predictions are computed. The frozen
pilot evaluator supplies source construction, checksums, checkpoint checks and
prediction alignment. Each sequence has 1025 tokens; at time t the network sees
tokens[:t] and predicts tokens[t]. Full transition matrices are evaluation oracles
and are never passed to the network. Bayesian inference is a separate operation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import eval_shared_base_models as frozen


ROOT = frozen.ROOT
PREFIXES = [16, 64, 256, 1024]
DEFAULT_OUT = ROOT / "output/shared_base_context_accuracy_2026-09-15"
DEFAULT_PILOT = ROOT / "output/shared_base_full_context"


def read_source(path, index, master_seed):
    identity, _ = frozen.source_identity(index, master_seed)
    record = frozen.validate_source_record(json.loads(path.read_text()), identity)
    return dict(path=path, record=record, file_sha256=frozen.sha_file(path))


def prepare_source(out, index, master_seed, pilot):
    """Preserve the pilot bytes and per-index RNG; never write to the pilot."""
    path = Path(out) / "data" / f"seq_{index:06d}.json"
    original = Path(pilot) / "data" / path.name
    if path.exists():
        item = read_source(path, index, master_seed)
    elif original.exists():
        original_item = read_source(original, index, master_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(original.read_bytes())
        temporary.replace(path)
        item = read_source(path, index, master_seed)
        if item["file_sha256"] != original_item["file_sha256"]:
            raise ValueError("pilot copy checksum mismatch")
    else:
        identity, source_id = frozen.source_identity(index, master_seed)
        draw = frozen.draw_source(frozen.D, frozen.N_TOKENS, identity["sequence_seed"])
        oracle = dict(initial_probs=draw["initial_probs"].tolist(),
                      transition_probs=draw["transition_probs"].tolist(),
                      initial_probs_sha256=frozen.sha_array(draw["initial_probs"], "<f8"),
                      transition_probs_sha256=frozen.sha_array(draw["transition_probs"], "<f8"),
                      indexing="observed labels: P(next=b | current=a)=transition_probs[a][b]",
                      note="Evaluation oracle only; never supplied to the Bayesian predictor or transformer.")
        record = dict(schema=frozen.SOURCE_SCHEMA, source_id=source_id, identity=identity,
                      tokens=draw["tokens"].tolist(), tokens_sha256=frozen.sha_array(draw["tokens"], "<i2"),
                      generated_source=dict(seed=identity["sequence_seed"], parameters=draw["parameters"]),
                      oracle=oracle)
        frozen.validate_source_record(record, identity)
        frozen.atomic_json(path, record)
        item = read_source(path, index, master_seed)
    if original.exists() and item["file_sha256"] != frozen.sha_file(original):
        raise ValueError("extension changed a frozen pilot source")
    observation_path = Path(out) / "observations" / path.name
    observations = dict(tokens=item["record"]["tokens"])
    if observation_path.exists():
        if json.loads(observation_path.read_text()) != observations:
            raise ValueError("token-only Bayesian input differs from the frozen sequence")
    else:
        frozen.atomic_json(observation_path, observations)
    item["observation_path"] = observation_path
    return item


def prepare_cohort(out, n_seq, master_seed, pilot):
    records = []
    depths = np.zeros(9, dtype=int)
    alphas = []
    seeds = []
    for index in range(n_seq):
        item = prepare_source(out, index, master_seed, pilot)
        record = item["record"]
        params = record["generated_source"]["parameters"]
        depths[params["ls"]] += 1
        alphas.append(params["alpha"])
        seeds.append(record["identity"]["sequence_seed"])
        records.append(dict(index=index, source_id=record["source_id"],
                            path=str(item["path"].resolve()), file_sha256=item["file_sha256"],
                            observations_path=str(item["observation_path"].resolve()),
                            observations_sha256=frozen.sha_file(item["observation_path"])))
    if len(set(seeds)) != n_seq:
        raise ValueError("per-sequence RNG seed collision")
    declaration = dict(schema="shared-base-full-context-cohort-v1", n_sequences=n_seq,
                       master_seed=master_seed, prefixes=PREFIXES, n_tokens_per_sequence=frozen.N_TOKENS,
                       selection="The first n_seq independent full-prior draws; retain every draw regardless of parameters or inference difficulty.",
                       records=records, pilot=str(Path(pilot).resolve()),
                       depth_counts=depths.tolist(), mean_alpha=float(np.mean(alphas)),
                       driver_sha256=frozen.sha_file(__file__))
    frozen.atomic_json(Path(out) / "cohort.json", declaration)
    return declaration


def model_specification(case, device, threads):
    import torch
    return dict(schema=frozen.MODEL_SCHEMA, run=str(case["run"]),
                checkpoint=str(case["checkpoint"]), checkpoint_sha256=case["checkpoint_sha256"],
                config=case["config"], config_sha256=frozen.sha_file(case["config_path"]),
                model_sha256=frozen.sha_file(frozen.HERE / "model.py"),
                driver_sha256=frozen.sha_file(__file__), frozen_helpers_sha256=frozen.sha_file(frozen.__file__),
                prefixes=PREFIXES, device=device, torch_version=torch.__version__,
                numpy_version=np.__version__, torch_threads=threads,
                logits_dtype="float32", log_probability_dtype="float64 computed from saved logits",
                alignment="prefix t uses input tokens[:t], logits[t-1], and target tokens[t]")


def cached_prediction(path, cache_id):
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    arrays = path.with_suffix(".npz")
    if stored.get("cache_id") != cache_id:
        raise ValueError(f"model cache inputs changed: {path}")
    if not arrays.exists() or frozen.sha_file(arrays) != stored["arrays_sha256"]:
        raise ValueError(f"model array checksum failed: {arrays}")
    return True


def evaluate_run(out, n_seq, master_seed, run, *, step=100000, device="cpu", batch=4, threads=1):
    import torch
    from model import GPT
    torch.set_num_threads(threads)
    case = frozen.model_case(run, step)
    spec = model_specification(case, device, threads)
    folder = Path(out) / "models" / case["name"]
    prefixes = np.asarray(PREFIXES, dtype=np.int64)
    model = None
    alignment = None
    new_records = reused = 0
    start = time.monotonic()
    for lo in range(0, n_seq, batch):
        missing = []
        for index in range(lo, min(n_seq, lo + batch)):
            item = read_source(Path(out) / "data" / f"seq_{index:06d}.json", index, master_seed)
            identity = dict(source_id=item["record"]["source_id"], source_file_sha256=item["file_sha256"], model=spec)
            cache_id = frozen.fingerprint(identity)
            path = folder / f"seq_{index:06d}.json"
            if cached_prediction(path, cache_id):
                reused += 1
            else:
                missing.append((index, item, identity, cache_id, path))
        if not missing:
            continue
        if model is None:
            checkpoint = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
            frozen.validate_checkpoint(checkpoint, case)
            model = GPT(**checkpoint["cfg"]["model"])
            model.load_state_dict(checkpoint["model"])
            model.to(device).eval()
        tokens = np.asarray([x[1]["record"]["tokens"] for x in missing], dtype=np.int64)
        inputs = torch.as_tensor(tokens[:, :-1], device=device)
        with torch.no_grad():
            logits = model(inputs)
            if alignment is None:
                alignment = frozen.check_model_alignment(model, inputs, logits)
        selected = frozen.selected_predictions(logits.detach().cpu().numpy(), tokens, prefixes)
        for j, (index, item, identity, cache_id, path) in enumerate(missing):
            source = item["record"]
            transitions = np.asarray(source["oracle"]["transition_probs"])
            oracle = transitions[tokens[j, prefixes - 1]]
            oracle_loss = -np.log2(oracle[np.arange(len(prefixes)), tokens[j, prefixes]])
            expected_regret = np.sum(oracle * (np.log(oracle) - selected["logp"][j]), axis=1) / np.log(2)
            arrays_path = path.with_suffix(".npz")
            frozen.atomic_npz(arrays_path, prefixes=prefixes, targets=selected["targets"][j],
                              logits=selected["logits"][j], logp=selected["logp"][j],
                              predictive_probs=selected["predictive_probs"][j],
                              selected_loss_bits=selected["selected_loss_bits"][j],
                              oracle_probs=oracle, oracle_loss_bits=oracle_loss,
                              expected_regret_bits=expected_regret)
            report = dict(schema=frozen.MODEL_SCHEMA, cache_id=cache_id, identity=identity,
                          sequence_index=index, source_path=str(item["path"].resolve()),
                          source_id=source["source_id"], tokens_sha256=source["tokens_sha256"],
                          prefixes=PREFIXES, target_indices=PREFIXES, targets=selected["targets"][j].tolist(),
                          prefix_tokens_sha256={str(t): frozen.sha_array(tokens[j, :t], "<i2") for t in prefixes},
                          predictive_probs=selected["predictive_probs"][j].tolist(),
                          selected_loss_bits=selected["selected_loss_bits"][j].tolist(),
                          oracle_probs=oracle.tolist(), oracle_loss_bits=oracle_loss.tolist(),
                          expected_regret_bits=expected_regret.tolist(),
                          logits_sha256=frozen.sha_array(selected["logits"][j], "<f4"),
                          logp_sha256=frozen.sha_array(selected["logp"][j], "<f8"),
                          arrays_file=arrays_path.name, arrays_sha256=frozen.sha_file(arrays_path),
                          checks=dict(probability_sum_error=selected["normalization_error"], alignment=alignment),
                          runtime=dict(python=sys.version.split()[0], torch=torch.__version__,
                                       numpy=np.__version__, device=device))
            frozen.atomic_json(path, report)
            new_records += 1
        if (lo + batch) % 64 == 0 or lo + batch >= n_seq:
            print(json.dumps(dict(run=case["name"], completed=new_records + reused,
                                  total=n_seq, elapsed_seconds=round(time.monotonic() - start, 2))), flush=True)
    summary = dict(run=case["name"], new_records=new_records, reused_records=reused,
                   seconds=time.monotonic() - start, checkpoint_sha256=case["checkpoint_sha256"],
                   alignment=alignment)
    frozen.atomic_json(folder / "run_receipt.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "models", "all"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--n-seq", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "results/transformer/runs/d64", ROOT / "results/transformer/runs/d64_seed1"])
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.n_seq, args.batch, args.torch_threads) < 1 or args.seed < 0:
        parser.error("n-seq, batch and torch-threads must be positive; seed must be nonnegative")
    if args.out.resolve() == args.pilot.resolve():
        parser.error("the extension must use a separate output folder")
    if args.stage in ("sources", "all"):
        declaration = prepare_cohort(args.out, args.n_seq, args.seed, args.pilot)
        print(json.dumps({key: declaration[key] for key in ("n_sequences", "depth_counts", "mean_alpha")}), flush=True)
    if args.stage in ("models", "all"):
        for run in args.runs:
            print(json.dumps(evaluate_run(args.out, args.n_seq, args.seed, run,
                                         step=args.step, device=args.device, batch=args.batch,
                                         threads=args.torch_threads)), flush=True)


if __name__ == "__main__":
    main()

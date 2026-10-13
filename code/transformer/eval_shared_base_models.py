"""Fixed independent source draws and paired existing-transformer predictions.

Generate each 1025-token sequence once, before selecting prediction prefixes.
The prediction after a prefix of length t uses model position t-1 and scores
tokens[t]. Oracle matrices are stored only for later evaluation; the Bayesian
CLI reads only the tokens field from these source files. No training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

from source import Source, seed_of


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_SCHEMA = "shared-base-full-context-source-v1"
MODEL_SCHEMA = "shared-base-full-context-model-v1"
SEED_TAG = "shared-base-full-context-v1"
D = 64
N_TOKENS = 1025


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def draw_source(d, n_tokens, sequence_seed):
    """Mirror Source.batch, retaining the matrix needed by the oracle.

    The permutation is sampled after the entire fixed-length path, exactly
    as in Source.batch. Its inverse maps observed labels back to latent ones.
    """
    source = Source(d, n_tokens, alpha=(0.5, 1.5), ls=(0, 8), permute=True)
    rng = np.random.default_rng(sequence_seed)
    alpha, depth = source.draw_params(1, rng)
    q, transitions = source.matrices(alpha, depth, rng)
    original = source.sequences(q, transitions, rng)[0]
    permutation = rng.permutation(d)
    inverse = np.argsort(permutation)
    tokens = permutation[original]
    observed_q = q[0, inverse]
    observed_transitions = transitions[0][inverse][:, inverse]
    return dict(tokens=tokens, initial_probs=observed_q,
                transition_probs=observed_transitions,
                parameters=dict(alpha=float(alpha[0]), ls=int(depth[0]),
                                perm=permutation.tolist()))


def source_spec():
    # Model-stage edits cannot invalidate an already frozen source draw.
    # Only the source implementation, source-construction function and its
    # explicit protocol/RNG inputs contribute to the source fingerprint.
    return dict(schema=SOURCE_SCHEMA, d=D, n_tokens=N_TOKENS,
                alpha_prior=[0.5, 1.5], depth_prior=list(range(9)),
                depth_scope="one shared depth for every transition row",
                rank_prior="uniform alphabet permutation",
                first_token_law="base q; generally not stationary",
                seed_tag=SEED_TAG, seed_rule="seed_of(seed_tag, master_seed, sequence_index)",
                source_sha256=sha_file(HERE / "source.py"),
                construction_sha256=hashlib.sha256(inspect.getsource(draw_source).encode()).hexdigest(),
                numpy_version=np.__version__, token_hash_encoding="little-endian int16",
                oracle_hash_encoding="little-endian float64, C order")


def source_identity(index, master_seed):
    spec = source_spec()
    sequence_seed = seed_of(SEED_TAG, master_seed, index)
    identity = dict(source=spec, sequence_index=index, master_seed=master_seed,
                    sequence_seed=sequence_seed)
    return identity, fingerprint(identity)


def validate_source_record(record, expected_identity=None):
    if record.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unexpected source schema")
    identity = record["identity"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("source cache inputs changed; use a new output folder")
    if record["source_id"] != fingerprint(identity):
        raise ValueError("source identity checksum failed")
    tokens = np.asarray(record["tokens"])
    if (tokens.shape != (N_TOKENS,) or tokens.dtype.kind not in "iu"
            or np.any(tokens < 0) or np.any(tokens >= D)):
        raise ValueError("invalid source tokens")
    if sha_array(tokens, "<i2") != record["tokens_sha256"]:
        raise ValueError("source token checksum failed")
    oracle = record["oracle"]
    q = np.asarray(oracle["initial_probs"], dtype=float)
    transitions = np.asarray(oracle["transition_probs"], dtype=float)
    if q.shape != (D,) or transitions.shape != (D, D):
        raise ValueError("invalid oracle dimensions")
    for values in (q[None, :], transitions):
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or np.max(np.abs(values.sum(axis=1) - 1)) > 1e-12):
            raise ValueError("invalid oracle probabilities")
    if (sha_array(q, "<f8") != oracle["initial_probs_sha256"]
            or sha_array(transitions, "<f8") != oracle["transition_probs_sha256"]):
        raise ValueError("oracle probability checksum failed")
    return record


def sources(out, n_seq, master_seed, *, create):
    records = []
    for index in range(n_seq):
        identity, source_id = source_identity(index, master_seed)
        path = Path(out) / "data" / f"seq_{index:06d}.json"
        if path.exists():
            record = validate_source_record(json.loads(path.read_text()), identity)
        elif not create:
            raise FileNotFoundError(f"missing frozen source draw: {path}")
        else:
            draw = draw_source(D, N_TOKENS, identity["sequence_seed"])
            oracle = dict(initial_probs=draw["initial_probs"].tolist(),
                          transition_probs=draw["transition_probs"].tolist(),
                          initial_probs_sha256=sha_array(draw["initial_probs"], "<f8"),
                          transition_probs_sha256=sha_array(draw["transition_probs"], "<f8"),
                          indexing="observed labels: P(next=b | current=a)=transition_probs[a][b]",
                          note="Evaluation oracle only; never supplied to the Bayesian predictor or transformer.")
            record = dict(schema=SOURCE_SCHEMA, source_id=source_id, identity=identity,
                          tokens=draw["tokens"].tolist(), tokens_sha256=sha_array(draw["tokens"], "<i2"),
                          generated_source=dict(seed=identity["sequence_seed"], parameters=draw["parameters"]),
                          oracle=oracle)
            validate_source_record(record, identity)
            atomic_json(path, record)
        records.append(dict(path=path, record=record, file_sha256=sha_file(path)))
    return records


def selected_predictions(logits, tokens, prefixes):
    """Convert selected raw logits into aligned probabilities and losses."""
    logits = np.asarray(logits)
    tokens = np.asarray(tokens)
    prefixes = np.asarray(prefixes, dtype=int)
    if (logits.ndim != 3 or tokens.shape != (logits.shape[0], logits.shape[1] + 1)
            or np.any(prefixes < 1) or np.any(prefixes > logits.shape[1])):
        raise ValueError("logits, tokens and prefixes are not aligned")
    selected = logits[:, prefixes - 1, :].astype(np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("nonfinite model logits")
    selected64 = selected.astype(np.float64)
    logp = selected64 - logsumexp(selected64, axis=2, keepdims=True)
    targets = tokens[:, prefixes]
    loss = -np.take_along_axis(logp, targets[:, :, None], axis=2)[:, :, 0] / np.log(2)
    probabilities = np.exp(logp)
    error = float(np.max(np.abs(probabilities.sum(axis=2) - 1)))
    if error > 1e-12:
        raise ValueError("model probability normalization failed")
    return dict(logits=selected, logp=logp, predictive_probs=probabilities,
                targets=targets, selected_loss_bits=loss, normalization_error=error)


def check_model_alignment(model, inputs, full_logits, *, tolerance=2e-5):
    """Independently check batch equivalence, prefix truncation and causality."""
    import torch
    if inputs.ndim != 2 or inputs.shape[1] < 2:
        raise ValueError("alignment check needs at least two input positions")
    probe = min(128, max(1, inputs.shape[1] // 2))
    with torch.no_grad():
        single = model(inputs[:1])
        short = model(inputs[:1, :probe])
        changed = inputs[:1].clone()
        changed[:, probe:] = (changed[:, probe:] + 1) % model.cfg["vocab"]
        altered = model(changed)
    errors = dict(
        batch_vs_single_logits=float((full_logits[0] - single[0]).abs().max().item()),
        full_vs_truncated_prefix_logits=float((single[0, probe - 1] - short[0, -1]).abs().max().item()),
        future_tokens_causality_logits=float((single[0, :probe] - altered[0, :probe]).abs().max().item()))
    if any(not np.isfinite(value) or value > tolerance for value in errors.values()):
        raise ValueError(f"model alignment/causality failed: {errors}")
    return dict(passed=True, probe_prefix_length=probe, tolerance=tolerance, **errors)


def model_case(run, step=100000):
    run = Path(run).resolve()
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    if (config.get("family", "lsa") != "lsa" or config.get("d") != D
            or config.get("alpha") != [0.5, 1.5] or config.get("ls") != [0, 8]
            or config.get("no_permute") is not False or config.get("n_ctx") != N_TOKENS - 1):
        raise ValueError(f"run does not match the original shared-base source: {run}")
    if config["model"]["vocab"] != D or config["model"]["n_ctx"] != N_TOKENS - 1:
        raise ValueError("model alphabet/context does not match source")
    checkpoint = run / f"model_{step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return dict(name=run.name, run=run, config=config, config_path=config_path,
                checkpoint=checkpoint, checkpoint_sha256=sha_file(checkpoint), step=step)


def validate_checkpoint(checkpoint, case):
    config = case["config"]
    if checkpoint["step"] != case["step"] or checkpoint["cfg"]["model"] != config["model"]:
        raise ValueError("checkpoint architecture or step disagrees with run configuration")
    for field in ("d", "n_ctx", "alpha", "ls", "no_permute", "seed"):
        # JSON stores intervals as lists; the checkpoint stores tuples.
        if canonical(checkpoint["cfg"].get(field)) != canonical(config.get(field)):
            raise ValueError(f"checkpoint source/run mismatch: {field}")
    if checkpoint["cfg"].get("family", "lsa") != config.get("family", "lsa"):
        raise ValueError("checkpoint source family mismatch")


def models(out, records, prefixes, runs, *, step=100000, device="cpu", batch=4, torch_threads=1):
    import torch
    from model import GPT
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(torch_threads)
    prefixes = np.asarray(prefixes, dtype=np.int64)
    summaries = []
    for run in runs:
        case = model_case(run, step)
        model_spec = dict(schema=MODEL_SCHEMA, run=str(case["run"]),
                          checkpoint=str(case["checkpoint"]), checkpoint_sha256=case["checkpoint_sha256"],
                          config=case["config"], config_sha256=sha_file(case["config_path"]),
                          model_sha256=sha_file(HERE / "model.py"), driver_sha256=sha_file(Path(__file__)),
                          prefixes=prefixes.tolist(), device=device, torch_version=torch.__version__,
                          numpy_version=np.__version__, torch_threads=torch_threads,
                          logits_dtype="float32", log_probability_dtype="float64 computed from saved logits",
                          alignment="prefix t uses input tokens[:t], logits[t-1], and target tokens[t]")
        folder = Path(out) / "models" / case["name"]
        missing = []
        for index, item in enumerate(records):
            identity = dict(source_id=item["record"]["source_id"], source_file_sha256=item["file_sha256"],
                            model=model_spec)
            cache_id = fingerprint(identity)
            path = folder / f"seq_{index:06d}.json"
            arrays_path = path.with_suffix(".npz")
            if path.exists():
                stored = json.loads(path.read_text())
                if stored.get("cache_id") != cache_id:
                    raise ValueError(f"model cache inputs changed: {path}")
                if not arrays_path.exists() or sha_file(arrays_path) != stored["arrays_sha256"]:
                    raise ValueError(f"model array checksum failed: {arrays_path}")
            else:
                missing.append((index, item, identity, cache_id, path, arrays_path))
        if not missing:
            print(f"{case['name']}: reused {len(records)} model records", flush=True)
            continue
        checkpoint = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, case)
        cfg = checkpoint["cfg"]["model"]
        model = GPT(**cfg)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        start = time.monotonic()
        alignment = None
        for lo in range(0, len(missing), batch):
            chunk = missing[lo:lo + batch]
            tokens = np.asarray([entry[1]["record"]["tokens"] for entry in chunk], dtype=np.int64)
            inputs = torch.as_tensor(tokens[:, :-1], device=device)
            with torch.no_grad():
                logits = model(inputs)
                if alignment is None:
                    alignment = check_model_alignment(model, inputs, logits)
            result = selected_predictions(logits.detach().cpu().numpy(), tokens, prefixes)
            for j, (index, item, identity, cache_id, path, arrays_path) in enumerate(chunk):
                source = item["record"]
                transitions = np.asarray(source["oracle"]["transition_probs"])
                oracle = transitions[tokens[j, prefixes - 1]]
                oracle_loss = -np.log2(oracle[np.arange(len(prefixes)), tokens[j, prefixes]])
                values = dict(prefixes=prefixes, targets=result["targets"][j],
                              logits=result["logits"][j], logp=result["logp"][j],
                              predictive_probs=result["predictive_probs"][j],
                              selected_loss_bits=result["selected_loss_bits"][j],
                              oracle_probs=oracle, oracle_loss_bits=oracle_loss)
                atomic_npz(arrays_path, **values)
                report = dict(schema=MODEL_SCHEMA, cache_id=cache_id, identity=identity,
                              sequence_index=index, source_path=str(item["path"].resolve()),
                              source_id=source["source_id"], tokens_sha256=source["tokens_sha256"],
                              prefixes=prefixes.tolist(), target_indices=prefixes.tolist(),
                              targets=result["targets"][j].tolist(),
                              prefix_tokens_sha256={str(t): sha_array(tokens[j, :t], "<i2") for t in prefixes},
                              predictive_probs=result["predictive_probs"][j].tolist(),
                              selected_loss_bits=result["selected_loss_bits"][j].tolist(),
                              oracle_probs=oracle.tolist(), oracle_loss_bits=oracle_loss.tolist(),
                              logits_sha256=sha_array(result["logits"][j], "<f4"),
                              logp_sha256=sha_array(result["logp"][j], "<f8"),
                              arrays_file=arrays_path.name, arrays_sha256=sha_file(arrays_path),
                              checks=dict(probability_sum_error=result["normalization_error"], alignment=alignment),
                              runtime=dict(python=sys.version.split()[0], torch=torch.__version__,
                                           numpy=np.__version__, device=device))
                atomic_json(path, report)
        elapsed = time.monotonic() - start
        summaries.append(dict(run=case["name"], new_records=len(missing), seconds=elapsed,
                              checkpoint_sha256=case["checkpoint_sha256"], alignment=alignment))
        print(json.dumps(summaries[-1]), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sources", "model"), required=True)
    parser.add_argument("--out", type=Path, default=Path("output/shared_base_full_context"))
    parser.add_argument("--n-seq", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[1024])
    parser.add_argument("--runs", type=Path, nargs="+",
                        default=[ROOT / "results/transformer/runs/d64", ROOT / "results/transformer/runs/d64_seed1"])
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.n_seq < 1 or args.batch < 1 or args.torch_threads < 1 or args.seed < 0:
        parser.error("require positive n-seq/batch/threads and nonnegative seed")
    prefixes = sorted(set(args.prefixes))
    if not prefixes or min(prefixes) < 1 or max(prefixes) >= N_TOKENS:
        parser.error("prefixes must lie between1 and1024")
    records = sources(args.out, args.n_seq, args.seed, create=args.stage == "sources")
    if args.stage == "sources":
        declaration = dict(schema="shared-base-full-context-cohort-v1", n_sequences=args.n_seq,
                           master_seed=args.seed, prefixes=prefixes, n_tokens_per_sequence=N_TOKENS,
                           selection="The first n_seq independent full-prior draws; retain every draw regardless of parameters or inference difficulty.",
                           records=[dict(index=i, source_id=item["record"]["source_id"],
                                         path=str(item["path"].resolve()), file_sha256=item["file_sha256"])
                                    for i, item in enumerate(records)])
        atomic_json(args.out / "cohort.json", declaration)
        print(json.dumps(dict(out=str(args.out), sequences=len(records), tokens_per_sequence=N_TOKENS,
                              prefixes=prefixes, source_ids=[item["record"]["source_id"] for item in records])), flush=True)
    else:
        models(args.out, records, prefixes, args.runs, step=args.step, device=args.device,
               batch=args.batch, torch_threads=args.torch_threads)


if __name__ == "__main__":
    main()

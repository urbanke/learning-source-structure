"""Frozen held-out cohorts and paired evaluations for pooled-family training.

No source family label, parameters, or oracle probabilities enter the GPT.
A context length t means that tokens[0:t] predict tokens[t]. Cumulative
loss through t sums predictions of tokens[1:t+1]; the first token is unscored.
Raw results retain per-sequence values at declared times, not full logit cubes.
Bayesian references remain explicitly pending until marginal likelihoods of
all three families have been computed and independently validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from families import ProductFamily
from source import Source

HERE = Path(__file__).resolve().parent
SCHEMA = "pooled-three-family-evaluation-v1"
SEED_NAMESPACE = "pooled3family-heldout-v1"
FAMILIES = ("shared_base", "uniform_base", "product")
PREFIXES = (1, 4, 16, 64, 256, 1024)
LN2 = np.log(2.0)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def condition_plan(n_prior=1024, n_conditional=256):
    if min(n_prior, n_conditional) < 1:
        raise ValueError("cohorts must contain at least one sequence")
    conditions = [dict(id=f"{family}_prior", family=family, kind="prior", n_seq=n_prior)
                  for family in FAMILIES]
    for alpha in (.5, 1., 1.5):
        for depth in (0, 1, 4, 8):
            conditions.append(dict(id=f"shared_base_a{alpha:g}_L{depth}", family="shared_base",
                                   kind="conditional", alpha=alpha, ls=depth, n_seq=n_conditional))
    for first, second in ((1, 1), (1, 8), (4, 4), (8, 8)):
        conditions.append(dict(id=f"product_L{first}_{second}", family="product", kind="conditional",
                               ls_pair=[first, second], n_seq=n_conditional))
    for depth in (0, 1, 4, 8):
        conditions.append(dict(id=f"uniform_base_L{depth}", family="uniform_base", kind="conditional",
                               ls=depth, n_seq=n_conditional))
    return conditions


def check_prefixes(prefixes, n_ctx):
    prefixes = tuple(int(t) for t in prefixes)
    if (not prefixes or list(prefixes) != sorted(set(prefixes))
            or prefixes[0] < 1 or prefixes[-1] != n_ctx):
        raise ValueError("prefixes must be increasing, positive, and include the final context length")
    return prefixes


def heldout_seed(master_seed, condition_id, index):
    """128-bit case identity; separate namespace from training and monitoring."""
    digest = hashlib.sha256(canonical([SEED_NAMESPACE, master_seed, condition_id, index]).encode()).digest()
    return np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32).tolist()


def draw_source(condition, n_tokens, sequence_seed):
    """Use the existing generators, retaining oracle matrices in observed labels."""
    rng = np.random.default_rng(sequence_seed)
    family = condition["family"]
    if family == "product":
        source = ProductFamily(8, 8, n_tokens, ls=(1, 8),
                               fixed_ls=condition.get("ls_pair"))
        u, _, first, perm1, ls1 = source._draw(source.s1, 1, rng)
        v, _, second, perm2, ls2 = source._draw(source.s2, 1, rng)
        inv1, inv2 = np.argsort(perm1[0]), np.argsort(perm2[0])
        transition = np.kron(first[0][inv1][:, inv1], second[0][inv2][:, inv2])
        return dict(tokens=(8 * u[0] + v[0]), initial_probs=np.full(64, 1 / 64),
                    transition_probs=transition,
                    parameters=dict(ls1=int(ls1[0]), ls2=int(ls2[0]),
                                    perm1=perm1[0].tolist(), perm2=perm2[0].tolist()))
    if family not in ("shared_base", "uniform_base"):
        raise ValueError(f"unknown family: {family}")
    alpha = condition.get("alpha", (.5, 1.5)) if family == "shared_base" else 0.
    source = Source(64, n_tokens, alpha=alpha, ls=condition.get("ls", (0, 8)), permute=True)
    exponent, depth = source.draw_params(1, rng)
    q, transition = source.matrices(exponent, depth, rng)
    original = source.sequences(q, transition, rng)[0]
    permutation = rng.permutation(64)
    inverse = np.argsort(permutation)
    return dict(tokens=permutation[original], initial_probs=q[0, inverse],
                transition_probs=transition[0][inverse][:, inverse],
                parameters=dict(alpha=float(exponent[0]), ls=int(depth[0]), perm=permutation.tolist()))


def validate_oracle(tokens, q, transition):
    if (tokens.ndim != 1 or tokens.dtype.kind not in "iu" or len(tokens) < 2
            or np.any(tokens < 0) or np.any(tokens >= 64)):
        raise ValueError("invalid source tokens")
    if q.shape != (64,) or transition.shape != (64, 64):
        raise ValueError("invalid source oracle shapes")
    for values in (q[None, :], transition):
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or not np.allclose(values.sum(axis=-1), 1., atol=1e-12, rtol=0)):
            raise ValueError("source oracle must be positive and normalized")


def cohort_arrays(condition, n_ctx, prefixes, master_seed):
    prefixes = check_prefixes(prefixes, n_ctx)
    n = condition["n_seq"]
    arrays = dict(tokens=np.empty((n, n_ctx + 1), dtype=np.int16),
                  initial_probs=np.empty((n, 64), dtype=np.float64),
                  transition_probs=np.empty((n, 64, 64), dtype=np.float64),
                  oracle_rows=np.empty((n, len(prefixes), 64), dtype=np.float64),
                  source_entropy_bits=np.empty((n, len(prefixes)), dtype=np.float64),
                  oracle_next_loss_bits=np.empty((n, len(prefixes)), dtype=np.float64),
                  oracle_cumulative_loss_bits=np.empty((n, len(prefixes)), dtype=np.float64),
                  sequence_seed=np.empty((n, 4), dtype=np.uint32), prefixes=np.asarray(prefixes, dtype=np.int64))
    parameters = []
    for index in range(n):
        seed = heldout_seed(master_seed, condition["id"], index)
        record = draw_source(condition, n_ctx + 1, seed)
        x, q, matrix = record["tokens"], record["initial_probs"], record["transition_probs"]
        validate_oracle(x, q, matrix)
        rows = matrix[x[np.asarray(prefixes) - 1]]
        losses = -np.log2(matrix[x[:-1], x[1:]])
        arrays["tokens"][index] = x
        arrays["initial_probs"][index] = q
        arrays["transition_probs"][index] = matrix
        arrays["oracle_rows"][index] = rows
        arrays["source_entropy_bits"][index] = -(rows * np.log2(rows)).sum(-1)
        arrays["oracle_next_loss_bits"][index] = losses[np.asarray(prefixes) - 1]
        arrays["oracle_cumulative_loss_bits"][index] = np.cumsum(losses)[np.asarray(prefixes) - 1]
        arrays["sequence_seed"][index] = seed
        parameters.append(record["parameters"])
    return arrays, parameters


def source_spec(conditions, n_ctx, prefixes, master_seed):
    return dict(schema=SCHEMA, conditions=conditions, alphabet_size=64, n_tokens=n_ctx + 1,
                prefixes=list(check_prefixes(prefixes, n_ctx)), master_seed=master_seed,
                seed_namespace=SEED_NAMESPACE,
                seed_rule="first128bits SHA256(canonicalJSON([namespace, master_seed, condition_id, sequence_index])); four little-endian uint32 words passed to numpy.default_rng",
                priors=dict(shared_base=dict(alpha=[.5, 1.5], depth=list(range(9)), uniform_rank_permutation=True),
                            uniform_base=dict(alpha=0., depth=list(range(9))),
                            product=dict(coordinates=[8, 8], independent_depths=list(range(1, 9)),
                                         coordinate_permutations=True)),
                evaluation_rule="first token unscored; t observed tokens predict token t+1",
                cumulative_rule="at t sum realized losses predicting tokens 2 through t+1",
                oracle_note="parameters and oracle arrays are evaluation-only, never model inputs",
                source_sha256=file_hash(HERE / "source.py"), families_sha256=file_hash(HERE / "families.py"),
                evaluator_sha256=file_hash(__file__), numpy_version=np.__version__)


def create_sources(out, conditions, n_ctx=1024, prefixes=PREFIXES, master_seed=2026091602):
    out = Path(out)
    spec = source_spec(conditions, n_ctx, prefixes, master_seed)
    identity = fingerprint(spec)
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != spec:
        raise ValueError("source plan changed; use a new output directory")
    write_json(plan_path, spec)
    entries = []
    for condition in conditions:
        data_path, record_path = out / f"{condition['id']}.npz", out / f"{condition['id']}.json"
        if data_path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record["plan_id"] != identity or record["condition"] != condition
                    or record["data_sha256"] != file_hash(data_path)):
                raise ValueError(f"frozen source mismatch: {condition['id']}")
        elif data_path.exists() or record_path.exists():
            raise ValueError(f"incomplete source write: {condition['id']}; inspect before removing")
        else:
            arrays, parameters = cohort_arrays(condition, n_ctx, prefixes, master_seed)
            write_npz(data_path, arrays)
            record = dict(schema=SCHEMA, plan_id=identity, condition=condition,
                          parameters=parameters, data_sha256=file_hash(data_path))
            write_json(record_path, record)
        entries.append(dict(condition=condition, data=data_path.name, record=record_path.name,
                            data_sha256=record["data_sha256"], record_sha256=file_hash(record_path)))
    manifest = dict(schema=SCHEMA, plan_id=identity, spec=spec, cohorts=entries,
                    bayesian_references=dict(family_aware="pending", mixture="pending",
                        requirement="validated marginal likelihoods of every family on the same sequences; selected-time predictions alone are insufficient"))
    write_json(out / "manifest.json", manifest)
    return manifest


def load_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["plan_id"] != fingerprint(manifest["spec"]):
        raise ValueError("source manifest identity mismatch")
    if json.loads((root / "plan.json").read_text()) != manifest["spec"]:
        raise ValueError("source plan mismatch")
    for entry in manifest["cohorts"]:
        for path_key, hash_key in (("data", "data_sha256"), ("record", "record_sha256")):
            if file_hash(root / entry[path_key]) != entry[hash_key]:
                raise ValueError(f"source checksum mismatch: {entry[path_key]}")
    return manifest


def scores_from_logits(logits, tokens, oracle_rows, prefixes):
    """NumPy float64 scoring; logits were computed without any oracle input."""
    values = np.asarray(logits, dtype=np.float64)
    tokens, rows = np.asarray(tokens), np.asarray(oracle_rows, dtype=np.float64)
    prefixes = check_prefixes(prefixes, tokens.shape[1] - 1)
    if values.shape != (len(tokens), tokens.shape[1] - 1, 64):
        raise ValueError("model logits must match source tokens and 64-symbol vocabulary")
    if rows.shape != (len(tokens), len(prefixes), 64):
        raise ValueError("oracle rows do not match selected contexts")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite model logits")
    logp = values - values.max(axis=-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    actual = -np.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[:, :, 0] / LN2
    selected = logp[:, np.asarray(prefixes) - 1]
    kl = (rows * (np.log(rows) - selected)).sum(axis=-1) / LN2
    if not np.isfinite(kl).all() or np.min(kl) < -1e-10:
        raise ValueError("invalid KL scores")
    return dict(next_kl_bits=np.maximum(0., kl),
                next_loss_bits=actual[:, np.asarray(prefixes) - 1],
                cumulative_loss_bits=np.cumsum(actual, axis=1)[:, np.asarray(prefixes) - 1])


def load_model(checkpoint_path, device="cpu"):
    import torch
    from model import GPT
    # Checkpoints are trusted local experiment artifacts, not downloaded models.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg", checkpoint.get("config", {}))
    training_cfg = cfg.get("config", cfg)
    expected_step = training_cfg.get("steps")
    if expected_step is None or checkpoint.get("step") != expected_step:
        raise ValueError("held-out evaluation requires the predeclared final checkpoint step")
    model_cfg = cfg.get("model", checkpoint.get("model_config"))
    if model_cfg is None:
        model_cfg = dict(vocab=cfg.get("d", cfg.get("vocab", 64)), n_ctx=cfg["n_ctx"],
                         n_layer=cfg.get("n_layer", 2), n_head=cfg.get("n_head", 4),
                         d_model=cfg.get("d_model", 128), mlp=not cfg.get("no_mlp", False),
                         dropout=cfg.get("dropout", 0.))
    if model_cfg["vocab"] != 64:
        raise ValueError("this experiment requires a 64-symbol model")
    model = GPT(**model_cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, dict(config=cfg, model_config=model_cfg, step=checkpoint.get("step"),
                       predeclared_final_step=expected_step)


def check_causality(model, tokens, prefixes, atol=2e-5):
    """Compare full causal inference against separate exact-prefix inference."""
    import torch
    sample = tokens[:1]
    with torch.inference_mode():
        full = model(sample[:, :-1]).float()
        errors = []
        for t in sorted(set((1, int(prefixes[len(prefixes) // 2]), len(sample[0]) - 1))):
            separate = model(sample[:, :t]).float()[:, -1]
            errors.append(float((separate - full[:, t - 1]).abs().max().item()))
    if max(errors) > atol:
        raise ValueError(f"prefix causality/alignment check failed: {max(errors)} > {atol}")
    return dict(max_absolute_logit_difference=max(errors), tolerance=atol, passed=True)


def evaluate_checkpoint(sources, checkpoint, name, out, device="cpu", batch_size=16, families=None):
    import torch
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("model name must be a simple filename component")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if families is not None and not set(families) <= set(FAMILIES):
        raise ValueError("unknown requested family")
    sources, out = Path(sources), Path(out) / name
    manifest = load_manifest(sources)
    model, checkpoint_record = load_model(checkpoint, device)
    n_ctx = manifest["spec"]["n_tokens"] - 1
    if model.cfg["n_ctx"] < n_ctx:
        raise ValueError("checkpoint context is shorter than the frozen evaluation sequences")
    prefixes = manifest["spec"]["prefixes"]
    identity = dict(schema=SCHEMA, name=name, source_plan_id=manifest["plan_id"],
                    checkpoint_sha256=file_hash(checkpoint), evaluator_sha256=file_hash(__file__),
                    model_code_sha256=file_hash(HERE / "model.py"),
                    device=str(device), batch_size=batch_size, torch_version=torch.__version__,
                    inference_precision="float32; no autocast")
    run_id = fingerprint(identity)
    entries, checked = [], None
    for entry in manifest["cohorts"]:
        condition = entry["condition"]
        if families and condition["family"] not in families:
            continue
        result_path, record_path = out / f"{condition['id']}.npz", out / f"{condition['id']}.json"
        if result_path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record["run_id"] != run_id or record["source_data_sha256"] != entry["data_sha256"]
                    or record["data_sha256"] != file_hash(result_path)):
                raise ValueError("existing evaluation has different inputs or corrupted data")
        elif result_path.exists() or record_path.exists():
            raise ValueError("incomplete evaluation write; inspect before removing")
        else:
            with np.load(sources / entry["data"], allow_pickle=False) as source:
                tokens, oracle_rows = source["tokens"], source["oracle_rows"]
                chunks = []
                for start in range(0, len(tokens), batch_size):
                    xb = torch.as_tensor(tokens[start:start + batch_size].astype(np.int64), device=device)
                    if checked is None:
                        checked = check_causality(model, xb, prefixes)
                    with torch.inference_mode():
                        logits = model(xb[:, :-1]).float().cpu().numpy()
                    chunks.append(scores_from_logits(logits, tokens[start:start + batch_size],
                                                     oracle_rows[start:start + batch_size], prefixes))
                arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
                arrays["cumulative_realized_regret_bits"] = arrays["cumulative_loss_bits"] - source["oracle_cumulative_loss_bits"]
                arrays["prefixes"] = np.asarray(prefixes)
                arrays["sequence_seed"] = source["sequence_seed"]
            write_npz(result_path, arrays)
            record = dict(schema=SCHEMA, run_id=run_id, identity=identity, condition=condition,
                          source_data_sha256=entry["data_sha256"], data_sha256=file_hash(result_path),
                          checkpoint=str(Path(checkpoint).resolve()), checkpoint_record=checkpoint_record,
                          device=str(device), batch_size=batch_size, torch_version=torch.__version__,
                          causality_check=checked)
            write_json(record_path, record)
        entries.append(dict(condition=condition, data=result_path.name, record=record_path.name,
                            data_sha256=record["data_sha256"], record_sha256=file_hash(record_path)))
    result = dict(schema=SCHEMA, run_id=run_id, identity=identity, cohorts=entries,
                  evaluated_families=sorted(set(families or FAMILIES)),
                  checkpoint_record=checkpoint_record, bayesian_references=manifest["bayesian_references"])
    write_json(out / "manifest.json", result)
    return result


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("summary input must contain finite per-sequence scores")
    return dict(n_seq=len(values), mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist() if len(values) > 1 else None)


def summarize(sources, evaluations, out, pairs=()):
    sources, evaluations = Path(sources), Path(evaluations)
    source_manifest = load_manifest(sources)
    source_by_id = {entry["condition"]["id"]: entry for entry in source_manifest["cohorts"]}
    results, arrays = {}, {}
    metrics = ("next_kl_bits", "next_loss_bits", "cumulative_loss_bits", "cumulative_realized_regret_bits")
    for path in sorted(evaluations.glob("*/manifest.json")):
        run = json.loads(path.read_text())
        if (run["schema"] != SCHEMA or run["run_id"] != fingerprint(run["identity"])
                or run["identity"]["source_plan_id"] != source_manifest["plan_id"]):
            raise ValueError(f"evaluation/source manifest mismatch: {path}")
        name = run["identity"]["name"]
        if name in results:
            raise ValueError("duplicate evaluation model name")
        expected_ids = {key for key, entry in source_by_id.items()
                        if entry["condition"]["family"] in run["evaluated_families"]}
        actual_ids = [entry["condition"]["id"] for entry in run["cohorts"]]
        if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
            raise ValueError("evaluation is missing declared source conditions")
        summaries, arrays[name] = {}, {}
        for entry in run["cohorts"]:
            for path_key, hash_key in (("data", "data_sha256"), ("record", "record_sha256")):
                if file_hash(path.parent / entry[path_key]) != entry[hash_key]:
                    raise ValueError("evaluation checksum mismatch")
            record = json.loads((path.parent / entry["record"]).read_text())
            source_entry = source_by_id[entry["condition"]["id"]]
            if (record["condition"] != source_entry["condition"]
                    or entry["condition"] != source_entry["condition"]
                    or record["run_id"] != run["run_id"]
                    or record["source_data_sha256"] != source_entry["data_sha256"]):
                raise ValueError("evaluation cohort provenance mismatch")
            with np.load(path.parent / entry["data"], allow_pickle=False) as data:
                raw = {key: data[key] for key in (*metrics, "sequence_seed", "prefixes")}
            condition = entry["condition"]
            if len(raw["sequence_seed"]) != condition["n_seq"]:
                raise ValueError("incomplete evaluation cohort")
            if not np.array_equal(raw["prefixes"], source_manifest["spec"]["prefixes"]):
                raise ValueError("evaluation context lengths changed")
            arrays[name][condition["id"]] = raw
            summaries[condition["id"]] = dict(condition=condition, metrics={key: mean_se(raw[key]) for key in metrics})
        results[name] = dict(run_id=run["run_id"], checkpoint_sha256=run["identity"]["checkpoint_sha256"], cohorts=summaries)
    if not results:
        raise ValueError("no completed model evaluations found")
    differences = {}
    for left, right in pairs:
        if left not in arrays or right not in arrays:
            raise ValueError(f"missing paired model evaluation: {left}, {right}")
        common = sorted(set(arrays[left]) & set(arrays[right]))
        if not common:
            raise ValueError("paired models have no common cohorts")
        differences[f"{left}_minus_{right}"] = {}
        for condition in common:
            first, second = arrays[left][condition], arrays[right][condition]
            if (not np.array_equal(first["sequence_seed"], second["sequence_seed"])
                    or not np.array_equal(first["prefixes"], second["prefixes"])):
                raise ValueError("paired results do not use the same sequences and times")
            differences[f"{left}_minus_{right}"][condition] = {key: mean_se(first[key] - second[key]) for key in metrics}
    result = dict(schema=SCHEMA, source_plan_id=source_manifest["plan_id"],
                  prefixes=source_manifest["spec"]["prefixes"], models=results, paired_differences=differences,
                  uncertainty="standard error across independent sequences; paired differences taken before averaging",
                  bayesian_references=source_manifest["bayesian_references"],
                  interpretation="transformer minus specialist is a measured training/adaptation difference, not the Bayesian log2(3) bound")
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    sources = stages.add_parser("sources")
    sources.add_argument("--out", required=True)
    sources.add_argument("--n-prior", type=int, default=1024)
    sources.add_argument("--n-conditional", type=int, default=256)
    sources.add_argument("--n-ctx", type=int, default=1024)
    sources.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    sources.add_argument("--seed", type=int, default=2026091602)
    sources.add_argument("--condition", action="append", help="optional declared subset, useful for smoke tests")
    model = stages.add_parser("model")
    model.add_argument("--sources", required=True)
    model.add_argument("--checkpoint", required=True)
    model.add_argument("--name", required=True)
    model.add_argument("--out", required=True)
    model.add_argument("--device", default="cuda")
    model.add_argument("--batch-size", type=int, default=16)
    model.add_argument("--family", action="append", choices=FAMILIES)
    summary = stages.add_parser("summary")
    summary.add_argument("--sources", required=True)
    summary.add_argument("--evaluations", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--pair", action="append", default=[], help="LEFT:RIGHT; report paired LEFT minus RIGHT")
    args = parser.parse_args()
    if args.stage == "sources":
        conditions = condition_plan(args.n_prior, args.n_conditional)
        if args.condition:
            wanted = set(args.condition)
            if not wanted <= {condition["id"] for condition in conditions}:
                parser.error("unknown requested source condition")
            conditions = [condition for condition in conditions if condition["id"] in wanted]
        result = create_sources(args.out, conditions, args.n_ctx, args.prefixes, args.seed)
        print(f"Frozen {len(result['cohorts'])} cohorts; plan {result['plan_id']}")
    elif args.stage == "model":
        result = evaluate_checkpoint(args.sources, args.checkpoint, args.name, args.out,
                                     args.device, args.batch_size, args.family)
        print(f"Evaluated {len(result['cohorts'])} cohorts; run {result['run_id']}")
    else:
        pairs = [tuple(pair.split(":")) for pair in args.pair]
        if any(len(pair) != 2 for pair in pairs):
            parser.error("--pair requires LEFT:RIGHT")
        result = summarize(args.sources, args.evaluations, args.out, pairs)
        print(f"Summarized {len(result['models'])} models; Bayesian references remain pending")


if __name__ == "__main__":
    main()

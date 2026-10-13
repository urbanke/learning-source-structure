"""Reevaluate existing checkpoints against the shared-depth family Bayes rule.

The default cases are the full training priors for uniform-base d=64 chains
and 8x8 product chains. Every sequence has its own seed, independent of sample
count and inference batch size. X1 is observed for free; all reported cumulative
losses score X2 through X(n_ctx+1). A selected prefix length t predicts X(t+1).

Stages may use different Python environments: sources/refs/summary need only
NumPy/SciPy and the reference store, while model imports PyTorch lazily. Each
stage caches one file per sequence in --out; changing its inputs causes a hard
cache mismatch rather than silently mixing results. No training is performed.

Example (set LSA_STORE and PMM_SRC as described in phi_source.py):
    python code/transformer/eval_shared_depth.py --out /private/tmp/shared-depth \
        --family both --n-seq 128 --stage refs
    python code/transformer/eval_shared_depth.py --out /private/tmp/shared-depth \
        --family both --n-seq 128 --stage model --device cpu
    python code/transformer/eval_shared_depth.py --out /private/tmp/shared-depth \
        --family both --n-seq 128 --stage summary
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LN2 = np.log(2.0)
SCHEMA = "shared-depth-evaluation-v1"
DEFAULT_PREFIXES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_tokens(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i2").tobytes()).hexdigest()


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _runtime_versions():
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy_version,
                platform=platform.platform())


def _atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **values)
    temporary.replace(path)


def _read_cache(path, expected_id):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if str(data.get("cache_id", "")) != expected_id:
        raise ValueError(f"Cache inputs changed: {path}. Use a new output folder.")
    return data


def _normalization_error(prob, label, tolerance):
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError(f"Invalid probabilities from {label}")
    error = float(np.max(np.abs(prob.sum(axis=-1) - 1.0)))
    if error > tolerance:
        raise ValueError(f"Unnormalized {label}: maximum error {error:.3g} > {tolerance:.3g}")
    return error


def _case(name, args):
    uniform = name == "uniform"
    run = Path(args.uniform_run if uniform else args.product_run).resolve()
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    wanted_family = "lsa" if uniform else "product:8x8"
    if config.get("family", "lsa") != wanted_family or int(config["d"]) != 64:
        raise ValueError(f"Checkpoint family/alphabet does not match {name}: {config_path}")
    wanted_depths = [0, 8] if uniform else [1, 8]
    if list(config["ls"]) != wanted_depths:
        raise ValueError(f"Checkpoint depth prior does not match {name}: {config_path}")
    if uniform and (list(config["alpha"]) != [0.0, 0.0] or config["no_permute"]):
        raise ValueError(f"Checkpoint does not use the declared uniform-base family: {config_path}")
    if args.n_ctx > int(config["n_ctx"]):
        raise ValueError("Requested prefix exceeds the checkpoint's trained positional-embedding support")
    checkpoint = run / f"model_{args.step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    spec = dict(schema=SCHEMA, family=wanted_family, d=64,
                coordinate_sizes=None if uniform else [8, 8],
                alpha=0.0, depth_range=wanted_depths,
                depth_scope="sequence" if uniform else "one independent depth per coordinate chain",
                relabeling="whole alphabet" if uniform else "independent coordinate permutations",
                initial_distribution="uniform", n_ctx=args.n_ctx, seed=args.seed,
                seed_rule="seed_of(schema, case, master_seed, sequence_index)",
                driver_sha256=_sha_file(Path(__file__)),
                source_numpy_version=np.__version__,
                token_hash_encoding="little-endian int16",
                source_sha256=_sha_file(HERE / "source.py"),
                families_sha256=_sha_file(HERE / "families.py"))
    return dict(name="uniform_d64" if uniform else "product_8x8", uniform=uniform,
                spec=spec, config=config, config_path=config_path,
                checkpoint=checkpoint, checkpoint_sha256=_sha_file(checkpoint))


def _dataset(case, args):
    from families import LSAFamily, ProductFamily
    from source import seed_of

    spec = case["spec"]
    family = (LSAFamily(64, args.n_ctx + 1, alpha=0.0, ls=(0, 8), permute=True)
              if case["uniform"] else ProductFamily(8, 8, args.n_ctx + 1, ls=(1, 8)))
    data = []
    base = Path(args.out) / case["name"] / "data"
    for i in range(args.n_seq):
        sequence_seed = seed_of(SCHEMA, case["name"], args.seed, i)
        metadata = dict(source=spec, index=i, sequence_seed=sequence_seed)
        cache_id = _fingerprint(metadata)
        path = base / f"seq_{i:06d}.npz"
        record = _read_cache(path, cache_id)
        if record is None:
            x, params = family.batch(1, np.random.default_rng(sequence_seed))
            parameter_values = {k: np.asarray(v)[0].tolist() for k, v in params.items() if k != "h_true"}
            _atomic_npz(path, cache_id=cache_id,
                        metadata_json=_json(dict(metadata, runtime=_runtime_versions())),
                        tokens=np.asarray(x[0], dtype=np.int16),
                        tokens_sha256=_sha_tokens(x[0]), h_true=params["h_true"][0],
                        parameters_json=_json(parameter_values))
            record = _read_cache(path, cache_id)
        if _sha_tokens(record["tokens"]) != str(record["tokens_sha256"]):
            raise ValueError(f"Token checksum failed: {path}")
        data.append(record)
    return data


def _reference_spec(case, args, prefixes):
    store = os.environ.get("LSA_STORE")
    source = os.environ.get("PMM_SRC")
    manifest = Path(store).expanduser() / "manifest.json" if store else None
    backend = Path(source).expanduser() / "product_model_with_memory/universal_tables.py" if source else None
    return dict(schema=SCHEMA, depths=list(range(0 if case["uniform"] else 1, 9)),
                global_prior="uniform", local_prior="same depth prior mixed separately in each row",
                u_lo=args.u_lo, u_hi=args.u_hi, rmax=args.n_ctx + 1,
                prefixes=prefixes.tolist(), probability_tolerance=args.prob_tol,
                table_dir=str(Path(args.table_dir).resolve()),
                lsa_store=os.environ.get("LSA_STORE"), pmm_src=os.environ.get("PMM_SRC"),
                store_manifest_sha256=_sha_file(manifest) if manifest and manifest.is_file() else None,
                backend_sha256=_sha_file(backend) if backend and backend.is_file() else None,
                driver_sha256=_sha_file(Path(__file__)),
                core_sha256=_sha_file(HERE / "shared_depth.py"),
                kernel_sha256=_sha_file(HERE / "lsa_seq.py"),
                tables_sha256=_sha_file(HERE / "phi_source.py"))


def _reference_id(record, reference_spec):
    return _fingerprint(dict(data_id=str(record["cache_id"]), reference=reference_spec))


def _references(case, data, args, prefixes):
    from lsa_seq import SeqLSA
    from shared_depth import ProductSharedDepthLSA, SharedDepthLSA

    reference_spec = _reference_spec(case, args, prefixes)
    base = Path(args.out) / case["name"] / "references"
    missing = [i for i, record in enumerate(data)
               if _read_cache(base / f"seq_{i:06d}.npz", _reference_id(record, reference_spec)) is None]
    if not missing:
        print(f"{case['name']}: reusing {len(data)} reference records", flush=True)
        return
    print(f"{case['name']}: preparing shared numerical tables", flush=True)
    kernel = SeqLSA(64 if case["uniform"] else 8, depths=reference_spec["depths"],
                    u_lo=args.u_lo, u_hi=args.u_hi, rmax=args.n_ctx + 1, table_dir=args.table_dir)
    single = SharedDepthLSA(kernel)
    reference = single if case["uniform"] else ProductSharedDepthLSA(single, SharedDepthLSA(kernel))
    start = time.monotonic()
    for count, i in enumerate(missing, 1):
        record = data[i]
        x = record["tokens"].astype(int)
        result = reference.run(x, full_probs=True, positions=prefixes)
        if case["uniform"]:
            old = kernel.run(x, full_probs=True, positions=prefixes)
            local_loss, local_probs = old["lC"], old["pC"]
        else:
            u = kernel.run(x // 8, full_probs=True, positions=prefixes)
            v = kernel.run(x % 8, full_probs=True, positions=prefixes)
            local_loss = u["lC"] + v["lC"]
            local_probs = (u["pC"][:, :, None] * v["pC"][:, None, :]).reshape(len(prefixes), 64)
        errors = [_normalization_error(result["probs"], "shared-depth Bayes", args.prob_tol),
                  _normalization_error(local_probs, "row-local comparator", args.prob_tol)]
        loss = np.asarray(result["loss"], dtype=np.float64)
        if not np.isfinite(loss).all() or not np.isfinite(local_loss).all():
            raise ValueError("Reference returned nonfinite losses")
        # The selected token index t is the prediction after a prefix of length t.
        targets = x[prefixes]
        for label, probabilities, losses in [("Bayes", result["probs"], loss),
                                              ("row-local", local_probs, local_loss)]:
            selected_loss = -np.log2(probabilities[np.arange(len(prefixes)), targets])
            if not np.allclose(selected_loss, np.asarray(losses)[prefixes], atol=2e-6, rtol=0):
                raise ValueError(f"Selected probability/loss alignment failed for {label}")
        metadata = dict(reference=reference_spec, tokens_sha256=str(record["tokens_sha256"]),
                        runtime=_runtime_versions())
        _atomic_npz(base / f"seq_{i:06d}.npz", cache_id=_reference_id(record, reference_spec),
                    metadata_json=_json(metadata), prefixes=prefixes,
                    bayes_loss=loss[1:], local_loss=np.asarray(local_loss, dtype=np.float64)[1:],
                    bayes_probs=np.asarray(result["probs"], dtype=np.float64),
                    local_probs=np.asarray(local_probs, dtype=np.float64),
                    log_evidence=np.asarray(result["log_evidence"], dtype=np.float64),
                    max_normalization_error=np.asarray(errors))
        if count == 1 or count % 8 == 0 or count == len(missing):
            print(f"{case['name']}: references {count}/{len(missing)} new, {time.monotonic()-start:.1f}s", flush=True)


def _model_spec(case, prefixes):
    return dict(checkpoint=str(case["checkpoint"]), checkpoint_sha256=case["checkpoint_sha256"],
                driver_sha256=_sha_file(Path(__file__)),
                config=case["config"], config_sha256=_sha_file(case["config_path"]),
                model_sha256=_sha_file(HERE / "model.py"),
                prefixes=prefixes.tolist(), log_probability_dtype="float32")


def _model_id(record, model_spec):
    return _fingerprint(dict(data_id=str(record["cache_id"]), model=model_spec))


def _models(case, data, args, prefixes):
    model_spec = _model_spec(case, prefixes)
    base = Path(args.out) / case["name"] / "model"
    missing = [i for i, record in enumerate(data)
               if _read_cache(base / f"seq_{i:06d}.npz", _model_id(record, model_spec)) is None]
    if not missing:
        print(f"{case['name']}: reusing {len(data)} model records", flush=True)
        return
    try:
        import torch
        import torch.nn.functional as functional
        from model import GPT
    except ImportError as exc:
        raise RuntimeError("The model stage needs PyTorch; use a PyTorch environment. Reference caches remain reusable.") from exc
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    checkpoint = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]["model"]
    if int(checkpoint["step"]) != args.step or cfg != case["config"]["model"]:
        raise ValueError("Checkpoint step/architecture disagrees with the requested run configuration")
    for field in ("d", "n_ctx", "alpha", "ls", "no_permute", "seed"):
        if _json(checkpoint["cfg"].get(field)) != _json(case["config"].get(field)):
            raise ValueError(f"Checkpoint/run configuration mismatch for {field}")
    if checkpoint["cfg"].get("family", "lsa") != case["config"].get("family", "lsa"):
        raise ValueError("Checkpoint/run source family mismatch")
    model = GPT(cfg["vocab"], cfg["n_ctx"], cfg["n_layer"], cfg["n_head"],
                cfg["d_model"], mlp=cfg["mlp"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"{case['name']}: evaluating checkpoint on {device}", flush=True)
    start = time.monotonic()
    with torch.no_grad():
        for lo in range(0, len(missing), args.batch):
            indices = missing[lo:lo + args.batch]
            x = np.stack([data[i]["tokens"] for i in indices]).astype(np.int64)
            xb = torch.from_numpy(x).to(device)
            lp = functional.log_softmax(model(xb[:, :-1]).float(), dim=-1)
            logp_true = lp.gather(2, xb[:, 1:, None])[:, :, 0].cpu().numpy()
            selected = lp[:, prefixes - 1, :].cpu().numpy().astype(np.float32)
            selected_targets = np.take_along_axis(selected, x[:, prefixes, None], axis=2)[:, :, 0]
            if not np.array_equal(selected_targets, logp_true[:, prefixes - 1]):
                raise ValueError("Transformer selected-prefix/target alignment failed")
            for j, i in enumerate(indices):
                error = _normalization_error(np.exp(selected[j].astype(np.float64)), "transformer", args.prob_tol)
                runtime = dict(_runtime_versions(), torch=torch.__version__, device=device)
                _atomic_npz(base / f"seq_{i:06d}.npz", cache_id=_model_id(data[i], model_spec),
                            metadata_json=_json(dict(model=model_spec, runtime=runtime,
                                                    tokens_sha256=str(data[i]["tokens_sha256"]))),
                            loss=-logp_true[j].astype(np.float64) / LN2,
                            logp=selected[j], prefixes=prefixes, max_normalization_error=error)
            print(f"{case['name']}: model {min(lo+args.batch, len(missing))}/{len(missing)} new, {time.monotonic()-start:.1f}s", flush=True)


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=np.mean(values, axis=0).tolist(),
                se=(np.std(values, axis=0, ddof=1) / np.sqrt(len(values))).tolist())


def _summary(case, data, args, prefixes):
    reference_spec = _reference_spec(case, args, prefixes)
    model_spec = _model_spec(case, prefixes)
    base = Path(args.out) / case["name"]
    bayes, local, tf, kls, local_kls = [], [], [], [], []
    normalization = []
    depth_counts = {}
    runtimes = dict(source=set(), references=set(), model=set())
    for i, record in enumerate(data):
        ref = _read_cache(base / "references" / f"seq_{i:06d}.npz", _reference_id(record, reference_spec))
        net = _read_cache(base / "model" / f"seq_{i:06d}.npz", _model_id(record, model_spec))
        if ref is None or net is None:
            raise FileNotFoundError(f"Missing reference/model stage output for {case['name']} sequence {i}")
        for stage, cache in (("source", record), ("references", ref), ("model", net)):
            runtimes[stage].add(_json(json.loads(str(cache["metadata_json"])).get("runtime")))
        p = ref["bayes_probs"]
        local_p = ref["local_probs"]
        logq = np.asarray(net["logp"], dtype=np.float64)
        errors = [_normalization_error(p, "cached Bayes", args.prob_tol),
                  _normalization_error(local_p, "cached row-local", args.prob_tol),
                  _normalization_error(np.exp(logq), "cached transformer", args.prob_tol)]
        normalization.append(errors)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(p), 0.0)
            kl = np.sum(p * (logp - logq), axis=-1) / LN2
            local_kl = np.sum(p * (logp - np.log(local_p)), axis=-1) / LN2
        if not np.isfinite(kl).all() or not np.isfinite(local_kl).all():
            raise ValueError("Nonfinite predictive KL divergence")
        if min(float(kl.min()), float(local_kl.min())) < -5 * args.prob_tol:
            raise ValueError("Negative predictive KL divergence beyond numerical tolerance")
        bayes.append(ref["bayes_loss"])
        local.append(ref["local_loss"])
        tf.append(net["loss"])
        kls.append(kl)
        local_kls.append(local_kl)
        params = json.loads(str(record["parameters_json"]))
        key = str(params["ls"]) if case["uniform"] else f"{params['ls1']},{params['ls2']}"
        depth_counts[key] = depth_counts.get(key, 0) + 1
    bayes, local, tf = (np.stack(v) for v in (bayes, local, tf))
    curves = dict(tf_minus_bayes=np.cumsum(tf - bayes, axis=1),
                  local_minus_bayes=np.cumsum(local - bayes, axis=1),
                  tf_minus_local=np.cumsum(tf - local, axis=1))
    measured = {key: _mean_se(value[:, prefixes - 1]) for key, value in curves.items()}
    measured["kl_bayes_to_transformer"] = _mean_se(kls)
    measured["kl_bayes_to_local"] = _mean_se(local_kls)
    terminal = {key: _mean_se(value[:, -1]) for key, value in curves.items()}
    terminal.update({f"{key}_total_loss": _mean_se(value.sum(axis=1))
                     for key, value in (("bayes", bayes), ("local", local), ("transformer", tf))})
    tokens = np.stack([r["tokens"] for r in data])
    summary = dict(schema=SCHEMA, source=case["spec"], n_sequences=len(data),
                   prior_draw_depth_counts=depth_counts, tokens_sha256=_sha_tokens(tokens),
                   first_token_scored=False, scored_tokens=f"X2 through X{args.n_ctx+1}",
                   units="bits", prefixes=prefixes.tolist(),
                   comparison="fresh full training-prior draws; one existing trained checkpoint",
                   uncertainty="paired means; standard errors across independent sequence/source draws; no training-seed uncertainty",
                   reference=reference_spec, model=model_spec,
                   runtime_by_stage={stage: [json.loads(value) for value in sorted(values)]
                                     for stage, values in runtimes.items()},
                   summary_runtime=_runtime_versions(),
                   driver_sha256=_sha_file(Path(__file__)),
                   maximum_probability_sum_error=np.max(normalization, axis=0).tolist(),
                   probability_error_order=["bayes", "local", "transformer"],
                   terminal=terminal, by_prefix=measured)
    (base / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fields = [f"{key}_{stat}" for key in measured for stat in ("mean", "se")]
    lines = ["\t".join(["prefix"] + fields)]
    for j, prefix in enumerate(prefixes):
        lines.append("\t".join([str(prefix)] + [f"{measured[key][stat][j]:.10g}"
                                                   for key in measured for stat in ("mean", "se")]))
    (base / "summary.tsv").write_text("\n".join(lines) + "\n")
    print(f"{case['name']}: {json.dumps(terminal, sort_keys=True)}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=("uniform", "product", "both"), default="both")
    ap.add_argument("--stage", choices=("sources", "refs", "model", "summary", "all"), default="all")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--prefixes", default=",".join(str(t) for t in DEFAULT_PREFIXES))
    ap.add_argument("--uniform-run", default=str(ROOT / "results/transformer/runs/lsa_a0_d64"))
    ap.add_argument("--product-run", default=str(ROOT / "results/transformer/runs/prod_8x8"))
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--u-lo", type=float, default=-30.0)
    ap.add_argument("--u-hi", type=float, default=30.0)
    ap.add_argument("--table-dir", default=str(HERE))
    ap.add_argument("--prob-tol", type=float, default=2e-6,
                    help="Fail on larger absolute probability-sum error; never renormalize")
    args = ap.parse_args()
    if args.n_seq < 2 or args.n_ctx < 1 or args.batch < 1 or args.prob_tol <= 0:
        ap.error("Require at least two sequences, positive context/batch, and positive probability tolerance")
    prefixes = np.asarray(sorted(set(int(t) for t in args.prefixes.split(",") if t)), dtype=int)
    prefixes = prefixes[(prefixes >= 1) & (prefixes <= args.n_ctx)]
    prefixes = np.unique(np.append(prefixes, args.n_ctx))
    families = ("uniform", "product") if args.family == "both" else (args.family,)
    summaries = []
    for name in families:
        case = _case(name, args)
        data = _dataset(case, args)
        print(f"{case['name']}: {len(data)} immutable source draws, {args.n_ctx} scored tokens each", flush=True)
        if args.stage in ("refs", "all"):
            _references(case, data, args, prefixes)
        if args.stage in ("model", "all"):
            _models(case, data, args, prefixes)
        if args.stage in ("summary", "all"):
            summaries.append(_summary(case, data, args, prefixes))
    if summaries:
        (Path(args.out) / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

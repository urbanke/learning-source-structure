"""Fresh product-source pilot: exact reference components and transformer fit.

No training. Frozen LSA/shared-depth producers are reused. Every position
t=1..n_ctx predicts tokens[t] from tokens[:t]. Four fitting weights are free
on the simplex; they are predictive contributions, not model posterior odds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import helmert
from scipy.special import logsumexp, xlogy

from eval_conditional_crossing import array_hash, canonical, configure_backend, fingerprint, sha_file
from families import ProductFamily
from source import seed_of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = "product-component-pilot-v1"
H = helmert(4, full=False).T


def distances(p, q):
    return (np.sum(xlogy(p, p) - xlogy(p, q), axis=-1) / np.log(2),
            np.sum(np.abs(p - q), axis=-1) / 2,
            np.linalg.norm(p - q, axis=-1))


def checked_load(path, spec_id):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        record = {name: data[name] for name in data.files}
    meta = json.loads(str(record.pop("metadata_json")))
    index = int(path.stem.rsplit("_", 1)[1])
    if (meta["spec_id"] != spec_id or fingerprint(meta["spec"]) != spec_id
            or meta["index"] != index
            or meta["sequence_seed"] != seed_of(SCHEMA, meta["spec"]["master_seed"], index)):
        raise ValueError(f"Changed inputs at {path}; use a new output directory")
    if set(record) != set(meta["array_hashes"]):
        raise ValueError(f"Unexpected arrays in {path}")
    for name, value in record.items():
        if array_hash(value) != meta["array_hashes"][name]:
            raise ValueError(f"Array checksum failed: {path}/{name}")
    return record


def model_from_checkpoint(path, config_path, threads):
    import torch
    from model import GPT
    torch.set_num_threads(threads)
    config = json.loads(config_path.read_text())
    if (config["family"] != "product:8x8" or config["d"] != 64
            or config["ls"] != [1, 8] or config["seed"] != 0):
        raise ValueError("Checkpoint configuration is not the declared product pilot")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["step"] != 100000:
        raise ValueError("Unexpected training step")
    for key in ("model", "d", "n_ctx", "ls", "family", "seed"):
        if saved["cfg"][key] != config[key]:
            raise ValueError(f"Checkpoint configuration mismatch: {key}")
    model = GPT(**config["model"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, config, torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/product_component_pilot/run_2026-09-15")
    parser.add_argument("--n-seq", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    args = parser.parse_args(argv)
    if args.n_seq < 1 or not 1 <= args.n_ctx <= 1024:
        parser.error("Require positive n-seq and1<=n-ctx<=1024")
    stop = args.n_seq if args.stop is None else args.stop
    if not 0 <= args.start < stop <= args.n_seq:
        parser.error("Invalid sequence-index range")
    store, src = configure_backend(args.store, args.pmm_src)
    from component_fit import fit_components
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA
    from product_components import product_components

    run = ROOT / "results/transformer/runs/prod_8x8"
    checkpoint, config_path = run / "model_100000.pt", run / "config.json"
    model, config, torch = model_from_checkpoint(checkpoint, config_path, args.torch_threads)
    paths = {name: HERE / name for name in (
        "eval_product_components.py", "component_fit.py", "product_components.py", "shared_depth.py",
        "lsa_seq.py", "phi_source.py", "source.py", "families.py", "model.py", "eval_conditional_crossing.py")}
    paths.update(checkpoint=checkpoint, config=config_path)
    paths.update({"store/" + name: store / name for name in ("anchors.json", "manifest.json")})
    paths.update({"pmm/" + name: src / "product_model_with_memory" / name for name in
                  ("production_coding.py", "universal_tables.py", "mellin.py", "kernel.py")})
    spec = dict(schema=SCHEMA, n_seq=args.n_seq, n_ctx=args.n_ctx, master_seed=args.seed,
                source_family="product:8x8", source_depths=list(range(1, 9)),
                depth_scope="independent coordinate depths; each shared across all coordinate rows",
                alphabet_sizes=[8, 8], first_token="supplied free", seed_rule="seed_of(schema,master_seed,index)",
                objective="KL(transformer||four-component mixture),bits; free simplex weights",
                optimality_gap_tolerance_nats=1e-7, identification_sigma_threshold=0.02,
                sensitivity_epsilon_l2=0.001, prior_unchanged=True,
                transformer_probability_calculation="float32 model logits; float64 normalized log-softmax",
                runtime=dict(python=__import__("sys").version, numpy=np.__version__, torch_threads=args.torch_threads,
                             scipy=__import__("scipy").__version__, torch=torch.__version__, device="cpu"),
                provenance={name: dict(path=str(path), sha256=sha_file(path)) for name, path in paths.items()})
    spec_id = fingerprint(spec)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "protocol.json"
    temporary_manifest = manifest.with_suffix(f".{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(spec, indent=2) + "\n")
    try:
        os.link(temporary_manifest, manifest)
    except FileExistsError:
        if json.loads(manifest.read_text()) != spec:
            raise ValueError("Changed run protocol; use a new output directory")
    finally:
        temporary_manifest.unlink()
    kernel = SeqLSA(8, depths=range(1, 9), rmax=args.n_ctx + 1, u_lo=-30, u_hi=30)
    first, second = SharedDepthLSA(kernel), SharedDepthLSA(kernel)
    family = ProductFamily(8, 8, args.n_ctx + 1, ls=(1, 8))
    started = time.monotonic()

    for index in range(args.start, stop):
        path = args.out / f"seq_{index:06d}.npz"
        if checked_load(path, spec_id) is not None:
            print(f"sequence{index}: verified cache", flush=True)
            continue
        sequence_seed = seed_of(SCHEMA, args.seed, index)
        x, params = family.batch(1, np.random.default_rng(sequence_seed))
        tokens = x[0]
        with torch.inference_mode():
            logits = model(torch.from_numpy(x[:, :-1])).cpu().numpy()[0].astype(np.float64)
            logp = logits - logsumexp(logits, axis=1, keepdims=True)
            tf_probs = np.exp(logp)
            # Compare full-pass predictions with separately truncated prefixes.
            causal_error = 0.0
            for prefix in sorted(set((1, min(7, args.n_ctx), min(64, args.n_ctx)))):
                local = model(torch.from_numpy(x[:, :prefix])).cpu().numpy()[0, -1].astype(np.float64)
                local = np.exp(local - logsumexp(local))
                causal_error = max(causal_error, float(np.max(np.abs(local - tf_probs[prefix - 1]))))
        if causal_error > 2e-6:
            raise ValueError(f"Causal prefix replay failed: {causal_error}")
        ref = product_components(first, second, tokens)
        comps, bw = ref["component_probs"], ref["weights"]
        fits = [fit_components(p, c) for p, c in zip(tf_probs, comps)]
        weights = np.array([f.weights for f in fits])
        converged = np.array([f.converged for f in fits])
        gaps = np.array([f.optimality_gap for f in fits])
        if not converged.all() or np.max(gaps) > 1.01e-7:
            raise ValueError(f"Unvalidated convex fits at sequence{index}")
        fitted = np.einsum("tk,tkd->td", weights, comps)
        fit_kl, fit_tv, fit_l2 = distances(tf_probs, fitted)
        bayes_kl, _, _ = distances(tf_probs, ref["joint_probs"])
        if np.max(fit_kl - bayes_kl) > 2e-7:
            raise ValueError("Fitted objective exceeds feasible Bayesian weights")
        singular = np.linalg.svd(comps.transpose(0, 2, 1) @ H, compute_uv=False)
        sigma = singular[:, -1]
        radius = np.ones(args.n_ctx)
        np.divide(np.sqrt(3) * (fit_l2 + 0.001 / 2), sigma, out=radius, where=sigma > 1e-12)
        radius = np.minimum(radius, 1.0)
        coordinate = [ref["coordinate1"], ref["coordinate2"]]
        arrays = dict(tokens=np.asarray(tokens, dtype=np.int16), prefix_lengths=np.arange(1, args.n_ctx + 1),
                      source_depths=np.array([params["ls1"][0], params["ls2"][0]]),
                      transformer_probs=tf_probs, bayes_probs=ref["joint_probs"], component_probs=comps,
                      bayes_weights=bw, transformer_weights=weights,
                      fit_kl_bits=fit_kl, fit_tv=fit_tv, fit_l2=fit_l2, bayes_kl_bits=bayes_kl,
                      fit_optimality_gap=gaps, fit_converged=converged, affine_sigma_min=sigma,
                      coordinate_separation_tv=np.stack([v["uniform_context_tv"] for v in coordinate], axis=1),
                      coordinate_copy_weights=np.stack([v["context_weight"] for v in coordinate], axis=1),
                      row_totals=np.stack([v["row_total"] for v in coordinate], axis=1),
                      eligible=sigma >= 0.02, robustness_radius=radius,
                      copy_lower=np.maximum(0, 1 - weights[:, 0] - radius),
                      copy_upper=np.minimum(1, 1 - weights[:, 0] + radius),
                      both_lower=np.maximum(0, weights[:, 3] - radius),
                      both_upper=np.minimum(1, weights[:, 3] + radius))
        metadata = dict(spec_id=spec_id, spec=spec, index=index, sequence_seed=sequence_seed,
                        causal_prefix_replay_error=causal_error,
                        bayes_reconstruction_error=ref["reconstruction_error"],
                        array_hashes={key: array_hash(value) for key, value in arrays.items()})
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata_json=canonical(metadata), **arrays)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = checked_load(path, spec_id)
            if any(array_hash(existing[key]) != array_hash(value) for key, value in arrays.items()):
                raise ValueError(f"Concurrent publication disagrees: {path}")
        finally:
            temporary.unlink()
        checked_load(path, spec_id)
        print(json.dumps(dict(sequence=index, source_depths=arrays["source_depths"].tolist(),
                              mean_fit_kl=float(fit_kl.mean()), eligible_fraction=float(arrays["eligible"].mean()),
                              max_gap=float(gaps.max()), elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    main()

"""Follow the identified copier along fresh natural uniform-base sequences.

One immutable record per independently drawn source and sequence. All
predictions at time t see exactly tokens[:t] and predict tokens[t]. This
runner retains the trained model and the Bayes depth prior 0..8; source
depth is recorded for analysis, never supplied to either predictor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from eval_conditional_crossing import array_hash, canonical, configure_backend, fingerprint, sha_file
from source import seed_of

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "natural-copier-sequence-v1"


def checked_load(path, spec_id):
    """Verify a completed record before reuse; partial draws are never skipped."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays.pop("metadata_json")))
    index = int(path.stem.rsplit("_", 1)[1])
    if (meta["spec_id"] != spec_id or fingerprint(meta["spec"]) != spec_id
            or meta["index"] != index
            or meta["sequence_seed"] != seed_of(SCHEMA, meta["spec"]["master_seed"], index)):
        raise ValueError(f"Changed source or numerical protocol in {path}")
    if set(arrays) != set(meta["array_hashes"]):
        raise ValueError(f"Incomplete or unexpected record arrays in {path}")
    for key, value in arrays.items():
        if array_hash(value) != meta["array_hashes"][key]:
            raise ValueError(f"Array checksum failed: {path}/{key}")
    return arrays, meta


def publish_protocol(directory, spec):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "protocol.json"
    temporary = directory / f"protocol.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(spec, indent=2) + "\n")
    try:
        os.link(temporary, final)
    except FileExistsError:
        if json.loads(final.read_text()) != spec:
            raise ValueError("Output directory belongs to a different protocol")
    finally:
        temporary.unlink()


def publish_record(path, arrays, metadata):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    metadata = dict(metadata, array_hashes={key: array_hash(value) for key, value in arrays.items()})
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata_json=canonical(metadata), **arrays)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing, _ = checked_load(path, metadata["spec_id"])
            if set(existing) != set(arrays) or any(array_hash(existing[k]) != array_hash(v) for k, v in arrays.items()):
                raise ValueError("Concurrent record publication disagrees")
    finally:
        temporary.unlink(missing_ok=True)


def check_bayes(result, tokens):
    n_ctx = len(tokens) - 1
    for name, shape in (("loss", (n_ctx+1,)), ("log_evidence", (n_ctx+1,)),
                        ("depth_posterior", (n_ctx+1, 9))):
        values = np.asarray(result[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid Bayes {name}")
    if np.any(result["depth_posterior"] < 0) or np.any(result["depth_posterior"] > 1 + 1e-12):
        raise ValueError("Invalid Bayes depth posterior probabilities")
    if not np.array_equal(result["positions"], np.arange(1, n_ctx + 1)):
        raise ValueError("Bayes prediction times are misaligned")
    probs = result["probs"]
    if probs.shape != (n_ctx, 64) or not np.isfinite(probs).all() or np.any(probs <= 0):
        raise ValueError("Invalid raw Bayes probabilities")
    observed = -np.log2(probs[np.arange(n_ctx), tokens[1:]])
    errors = dict(normalization=float(np.max(np.abs(probs.sum(-1) - 1))),
                  observed_loss_bits=float(np.max(np.abs(observed - result["loss"][1:]))),
                  evidence_telescope_bits=float(np.max(np.abs(
                      np.cumsum(result["loss"][1:])
                      + (result["log_evidence"][1:] - result["log_evidence"][0]) / np.log(2)))),
                  first_prediction_uniform=float(np.max(np.abs(probs[0] - 1 / 64))),
                  first_depth_prior=float(np.max(np.abs(result["depth_posterior"][1] - 1 / 9))),
                  posterior_normalization=float(np.max(np.abs(result["depth_posterior"].sum(-1) - 1))))
    if max(errors[k] for k in ("normalization", "first_prediction_uniform", "first_depth_prior", "posterior_normalization")) > 1e-7:
        raise ValueError(f"Bayes normalization/initial conditions failed: {errors}")
    if max(errors["observed_loss_bits"], errors["evidence_telescope_bits"]) > 2e-6:
        raise ValueError(f"Bayes likelihood alignment failed: {errors}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/copier_time/run_2026-09-15")
    parser.add_argument("--n-seq", type=int, default=1024)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    args = parser.parse_args(argv)
    stop = args.n_seq if args.stop is None else args.stop
    if args.n_seq < 1 or not 1 <= args.n_ctx <= 1024 or not 0 <= args.start < stop <= args.n_seq or args.seed < 0:
        parser.error("Invalid sequence count, time horizon, seed or worker range")
    store, src = configure_backend(args.store, args.pmm_src)
    from copier_time_source import draw_natural
    from copier_time_metrics import natural_metrics
    from natural_circuit_probe import load_model, NaturalCircuitProbe
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA

    run = ROOT / "results/transformer/runs/lsa_a0_d64"
    checkpoint, config_path = run / "model_100000.pt", run / "config.json"
    model, model_metadata = load_model(checkpoint, config_path, torch_threads=1)
    probe = NaturalCircuitProbe(model)
    selection_path = ROOT / "results/transformer/copier_circuit/routing_summary.json"
    previous = json.loads(selection_path.read_text())
    if not previous["passed"] or previous["protocol"]["candidate"] != [0, 1]:
        raise ValueError("The causal copier identification must remain validated and fixed")
    for path in (checkpoint, config_path, HERE / "model.py", HERE / "circuit_probe.py", HERE / "source.py"):
        relative = str(path.relative_to(ROOT))
        if previous["protocol"]["provenance"].get(relative) != sha_file(path):
            raise ValueError("The model or instrument differs from the identified copier: " + relative)
    paths = {name: HERE / name for name in (
        "eval_copier_time.py", "copier_time_source.py", "copier_time_metrics.py", "natural_circuit_probe.py",
        "circuit_probe.py", "model.py", "source.py", "shared_depth.py", "lsa_seq.py", "phi_source.py",
        "eval_conditional_crossing.py")}
    paths.update(checkpoint=checkpoint, config=config_path, identification=selection_path,
                 scientific_protocol=ROOT / "notes/reviews/copier_time_protocol_2026-09-15.md")
    paths.update({"store/" + name: store / name for name in ("anchors.json", "manifest.json")})
    paths.update({"pmm/" + name: src / "product_model_with_memory" / name for name in
                  ("production_coding.py", "universal_tables.py", "mellin.py", "kernel.py")})
    spec = dict(schema=SCHEMA, n_seq=args.n_seq, n_ctx=args.n_ctx, master_seed=args.seed,
                d=64, source_family="uniform-base LSA Markov chains", alpha=[0., 0.], source_depths=list(range(9)),
                depth_scope="one depth per sequence shared by all rows, independently drawn from 0..8",
                source_relabelled=True, bayes_prior="uniform shared depth 0..8, uniform base; source parameters withheld",
                conditions=["intact", "copier_deleted", "control_deleted", "bayes"],
                candidate_head=[0, 1], control_head=[0, 0], head_indices="zero based",
                first_token="supplied free", time="t=1..n_ctx: see tokens[:t], predict tokens[t]",
                seed_rule="seed_of(schema,master_seed,index)",
                primary="Probability of previously recorded successors of the current token, minus their uniform mass; compare all conditions on the same sequences",
                secondary="Effect of head deletion on that probability and oracle expected log loss; Bayes versus known uniform base reported separately",
                groups="all sequences; actual depth0 and positive-depth subgroups; all individual depths retained",
                thresholds="No chosen transition, normalized output weight or post-hoc truncation",
                precision="float64 forward with unchanged trained weights; raw Bayes evidence ratios, no clipping or renormalization",
                quadrature=dict(u_lo=-30, u_hi=30, step=1/64, depths=list(range(9)), rmax=args.n_ctx+1),
                tolerances=dict(bayes_probability=1e-7, bayes_loss_bits=2e-6, transformer_probability=1e-12, causal_replay=1e-10),
                model_metadata=model_metadata,
                runtime=dict(python=sys.version, numpy=np.__version__, scipy=__import__("scipy").__version__,
                             torch=__import__("torch").__version__, torch_threads=1, device="cpu"),
                provenance={name: dict(path=str(path), sha256=sha_file(path)) for name, path in paths.items()})
    spec_id = fingerprint(spec)
    publish_protocol(args.out, spec)
    kernel = SeqLSA(64, depths=range(9), rmax=args.n_ctx + 1, u_lo=-30, u_hi=30)
    for value in (kernel.u, kernel.D, kernel.LN0, kernel.lgam):
        value.setflags(write=False)
    bayes = SharedDepthLSA(kernel)
    started = time.monotonic()
    for index in range(args.start, stop):
        path = args.out / f"seq_{index:06d}.npz"
        if checked_load(path, spec_id) is not None:
            print(json.dumps(dict(index=index, state="verified existing record")), flush=True)
            continue
        tick = time.monotonic()
        sequence_seed = seed_of(SCHEMA, args.seed, index)
        source = draw_natural(64, args.n_ctx, sequence_seed)
        tokens = source["tokens"]
        network = probe.run(tokens)
        reference = bayes.run(tokens, full_probs=True, positions=np.arange(1, args.n_ctx+1), depth_posteriors=True)
        bayes_checks = check_bayes(reference, tokens)
        probabilities = {name: condition["probs"] for name, condition in network["conditions"].items()}
        probabilities["bayes"] = reference["probs"]
        metrics, metric_checks = natural_metrics(tokens, probabilities, source["transition_probs"])
        if (not np.array_equal(network["prefix_lengths"], reference["positions"])
                or not np.array_equal(network["matching_successor_count"], metrics["row_totals"])):
            raise ValueError("Attention sites and recorded successor counts are misaligned")
        arrays = {key: np.asarray(value) for key, value in source.items()}
        arrays["prefix_lengths"] = np.arange(1, args.n_ctx+1, dtype=np.int64)
        arrays.update({name + "_probs": value for name, value in probabilities.items()})
        arrays.update({"metric_" + name: value for name, value in metrics.items()})
        arrays["bayes_depth_posterior"] = reference["depth_posterior"][1:]
        arrays["bayes_log_evidence"] = reference["log_evidence"]
        for key in ("previous_token_defined", "matching_successor_count", "matching_successor_defined",
                    "matching_successor_uniform_attention"):
            arrays[key] = network[key]
        for name, condition in network["conditions"].items():
            for key in ("previous_token_attention", "matching_successor_attention"):
                arrays[name + "_" + key] = condition[key]
        metadata = dict(spec=spec, spec_id=spec_id, index=index, sequence_seed=sequence_seed,
                        bayes_checks=bayes_checks, metric_checks=metric_checks,
                        network_checks=network["validation"])
        publish_record(path, arrays, metadata)
        print(json.dumps(dict(index=index, completed_in_worker=index-args.start+1, worker_stop=stop,
                              source_depth=int(source["source_depth"]),
                              sequence_seconds=time.monotonic()-tick, elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()

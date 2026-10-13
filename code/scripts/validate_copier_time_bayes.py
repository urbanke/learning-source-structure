"""Bounded source-aligned grid checks for the natural copier-time Bayes rule.

Reuse the existing production, wider and half-cell-offset grids and the
frozen shared-depth predictor. Select the first depth-eight source in the
declared cohort before inspecting predictions, plus source zero if it has
a different nonzero depth. Smoke rechecks the bound raw arrays and hashes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from copier_time_source import draw_natural, validate_natural
from eval_conditional_crossing import array_hash, configure_backend, sha_file
from source import Source, seed_of
from validate_conditional_crossing import GRIDS

SOURCE_SCHEMA = "natural-copier-sequence-v1"
SOURCE_KEYS = ("tokens", "initial_probs", "transition_probs", "permutation", "source_depth")


def choose_sources(master_seed):
    source = Source(64, 1025, alpha=(0., 0.), ls=(0, 8), permute=True)
    trace = []
    for index in range(1024):
        seed = seed_of(SOURCE_SCHEMA, master_seed, index)
        _, depths = source.draw_params(1, np.random.default_rng(seed))
        depth = int(depths[0])
        trace.append(dict(index=index, sequence_seed=seed, source_depth=depth))
        if depth == 8:
            chosen = [trace[-1]]
            if index != 0 and trace[0]["source_depth"] != 0:
                chosen.insert(0, trace[0])
            return chosen, trace
    raise ValueError("The fixed cohort has no depth-eight draw")


def check_record(arrays):
    source = {key: arrays[key] for key in SOURCE_KEYS}
    source_checks = validate_natural(source, d=64, n_ctx=1024)
    x = source["tokens"]
    oracle = source["transition_probs"][x[:-1]]
    values, checks = {}, {}
    for name in GRIDS:
        p, loss, evidence, posterior = (arrays[name+"_"+key]
                                       for key in ("probs", "loss", "log_evidence", "depth_posterior"))
        if (p.shape != (1024, 64) or loss.shape != (1025,) or evidence.shape != (1025,)
                or posterior.shape != (1025, 9) or not all(np.isfinite(v).all() for v in (p, loss, evidence, posterior))
                or np.any(p <= 0) or np.any(posterior < 0)):
            raise ValueError("Invalid full-path Bayes output arrays")
        errors = dict(raw_probability_sum_error=float(abs(p.sum(-1)-1).max()),
                      raw_posterior_sum_error=float(abs(posterior.sum(-1)-1).max()),
                      first_uniform_error=float(abs(p[0]-1/64).max()),
                      first_depth_prior_error=float(abs(posterior[1]-1/9).max()),
                      observed_loss_error_bits=float(abs(-np.log2(p[np.arange(1024), x[1:]])-loss[1:]).max()),
                      evidence_telescope_error_bits=float(abs(np.cumsum(loss[1:])+(evidence[1:]-evidence[0])/np.log(2)).max()))
        passed = (max(errors[k] for k in ("raw_probability_sum_error", "raw_posterior_sum_error", "first_uniform_error", "first_depth_prior_error")) < 1e-7
                  and max(errors[k] for k in ("observed_loss_error_bits", "evidence_telescope_error_bits")) < 1e-6)
        checks[name] = dict(passed=passed, **errors)
        values[name] = dict(probs=p, loss_bits=loss[1:], cumulative_loss_bits=np.cumsum(loss[1:]),
                            depth_posterior=posterior,
                            oracle_regret_bits=np.sum(oracle*np.log2(oracle/p), axis=-1))
    comparisons = {}
    for name in ("wide", "offset"):
        errors = {key: float(abs(values[name][key]-values["production"][key]).max())
                  for key in values["production"]}
        passed = all(value < (1e-7 if key in ("probs", "depth_posterior") else 1e-6)
                     for key, value in errors.items())
        comparisons[name] = dict(passed=passed, maximum_absolute_differences=errors)
    return dict(passed=all(v["passed"] for v in checks.values()) and all(v["passed"] for v in comparisons.values()),
                source_checks=source_checks, raw_grid_checks=checks, comparisons=comparisons,
                max_current_row_visits=int(np.bincount(x[:-1], minlength=64).max()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/transformer/copier_time/numerical_validation.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT/"output/copier_time/numerical_validation")
    parser.add_argument("--master-seed", type=int, default=20261004)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        previous = json.loads(args.out.read_text())
        if previous["passed"] is not True:
            raise ValueError("The bound numerical receipt did not pass")
        for name, item in previous["provenance"].items():
            if sha_file(item["path"]) != item["sha256"]:
                raise ValueError("Stale numerical/source provenance: "+name)
        for item in previous["sources"]:
            if sha_file(item["raw_path"]) != item["raw_sha256"]:
                raise ValueError("Changed raw grid-validation record")
            with np.load(item["raw_path"], allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            if {key: array_hash(v) for key, v in arrays.items()} != item["array_hashes"]:
                raise ValueError("Changed grid-validation arrays")
            if check_record(arrays) != item["checks"]:
                raise ValueError("Raw readback differs from the saved validation")
        print(json.dumps(dict(passed=True, cached_only=True, sources=len(previous["sources"]))))
        return
    if args.out.exists() or (args.work_dir.exists() and any(args.work_dir.iterdir())):
        raise ValueError("Use new output paths; existing validation records are immutable")
    store, src = configure_backend(args.store, args.pmm_src)
    from lsa_seq import SeqLSA
    from shared_depth import SharedDepthLSA
    chosen, trace = choose_sources(args.master_seed)
    print(json.dumps(dict(selection_before_predictions=chosen)), flush=True)
    paths = {"local/"+name: ROOT/"code/transformer"/name for name in
             ("copier_time_source.py", "source.py", "shared_depth.py", "lsa_seq.py", "phi_source.py", "eval_conditional_crossing.py")}
    paths.update({"local/"+name: ROOT/"code/scripts"/name for name in
                  ("validate_copier_time_bayes.py", "validate_conditional_crossing.py")})
    paths.update({"pmm/"+name: src/"product_model_with_memory"/name for name in
                  ("production_coding.py", "universal_tables.py", "mellin.py", "kernel.py")})
    paths.update({"store/"+name: store/name for name in ("anchors.json", "manifest.json")})
    paths["checkpoint_config"] = ROOT/"results/transformer/runs/lsa_a0_d64/config.json"
    provenance = {name: dict(path=str(path.resolve()), sha256=sha_file(path)) for name, path in paths.items()}
    raw = {item["index"]: {key: np.asarray(v) for key, v in draw_natural(64, 1024, item["sequence_seed"]).items()} for item in chosen}
    for item in chosen:
        if int(raw[item["index"]]["source_depth"]) != item["source_depth"]:
            raise ValueError("The preregistered parameter draw does not match source replay")
    started = time.monotonic()
    timings = {}
    for name, (lo, hi) in GRIDS.items():
        tick = time.monotonic()
        kernel = SeqLSA(64, depths=range(9), rmax=1025, u_lo=lo, u_hi=hi)
        reference = SharedDepthLSA(kernel)
        setup_seconds = time.monotonic()-tick
        per_sequence = {}
        for item in chosen:
            arrays = raw[item["index"]]
            tick = time.monotonic()
            result = reference.run(arrays["tokens"], full_probs=True, positions=np.arange(1, 1025), depth_posteriors=True)
            per_sequence[str(item["index"])] = time.monotonic()-tick
            if not np.array_equal(result["positions"], np.arange(1, 1025)):
                raise ValueError("Wrong Bayesian prediction positions")
            arrays.update({name+"_"+key: result[key] for key in ("probs", "loss", "log_evidence", "depth_posterior")})
        timings[name] = dict(setup_seconds=setup_seconds, per_sequence_seconds=per_sequence)
        del kernel, reference
    args.work_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for item in chosen:
        arrays = raw[item["index"]]
        checks = check_record(arrays)
        path = args.work_dir/f"seq_{item['index']:06d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        sources.append(dict(**item, checks=checks, raw_path=str(path.resolve()), raw_sha256=sha_file(path),
                            array_hashes={key: array_hash(v) for key, v in arrays.items()}))
    if any(sha_file(item["path"]) != item["sha256"] for item in provenance.values()):
        raise ValueError("Source, code or read-only store changed during validation")
    report = dict(schema="natural-copier-bayes-grid-validation-v1", passed=all(v["checks"]["passed"] for v in sources),
                  source_schema=SOURCE_SCHEMA, master_seed=args.master_seed, seed_rule="seed_of(source_schema,master_seed,index)",
                  declared_cohort_size=1024, d=64, n_ctx=1024, alpha=[0., 0.], source_depth_prior=list(range(9)),
                  bayes_depth_prior=list(range(9)), shared_global_depth=True, source_parameters_withheld=True,
                  selection="First depth-eight draw, plus index zero if different and nonzero; selection uses source depth only, before predictions.",
                  selection_trace=trace, grids={k: dict(lower=v[0], upper=v[1], step=1/64, rmax=1025) for k, v in GRIDS.items()},
                  sources=sources, provenance=provenance, timings=timings, elapsed_seconds=time.monotonic()-started,
                  thresholds=dict(probabilities_and_depth_posterior=1e-7, losses_and_cumulative_bits=1e-6),
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=importlib.metadata.version("scipy")),
                  limitations=["Two source-selected finite paths, not all possible count profiles.",
                               "Wider bounds and a half-cell phase shift test the quadrature; all grids share the same sealed table approximation.",
                               "This validates the Bayes/source numerics, not transformer behavior or a causal timing conclusion."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(out=str(args.out), passed=report["passed"], sources=chosen, timings=timings)), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Audit a new surrogate-sampler exponent pair against its complete raw draws.

All probability, scalar, curvature, ESS, Rhat and row-grid tolerances are
inherited from the frozen full-context audit. This exports a selected source
and time, never an average over a resolved subset of the source cohort.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from shared_base_full_context_summary import (
    Audit, NUMERICAL_CODE, RUNS, ROOT, TRANSFORMER, comparison_metrics,
    fingerprint, load_source, metric_refinement, numerical_array_hash,
    probability_refinement, sha_array, sha_file, stable_seed, verify_local_hashes,
)
sys.path.insert(0, str(TRANSFORMER))
from eval_shared_base_reference import load_inputs
from eval_shared_base_surrogate import CODE_FILES, SEED_TAG
from full_context_statistics import diagnostic_gates
from stable_chain_statistics import SUMMARY_SPEC, stable_chain_summary, stable_predictive_moments
from surrogate_chain_runner import proposal_descriptor


def _close(first, second, atol=2e-14, rtol=0):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return first.shape == second.shape and bool(np.allclose(first, second, atol=atol, rtol=rtol))


def _finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def raw_checkpoint(audit, name, path, report):
    """Verify complete saved arrays and recompute summaries without resampling."""
    audit.track(name + ":checkpoint", path)
    with np.load(path, allow_pickle=False) as stored:
        serialized = str(stored["metadata"].item())
        audit.require(name + ":metadata_checksum", hashlib.sha256(serialized.encode()).hexdigest()
                      == str(stored["metadata_sha256"].item()))
        metadata = json.loads(serialized)
        audit.require(name + ":checkpoint_identity", metadata["format"] == "shared_base_chain_run_v1"
                      and metadata["fingerprint"] == report["identity"]["numerical_problem"]
                      and metadata["proposal_descriptor"] == report["identity"]["proposal_descriptor"])
        config = metadata["config"]
        chains = config["chains"]
        expected_names = {"burn_done", "attempts", "accepts", "max_errors"}
        expected_names.update(f"{kind}_{i}" for i in range(chains)
                              for kind in ("predictions", "depths", "alpha_indices", "scalars"))
        audit.require(name + ":complete_raw_arrays", set(metadata["array_hashes"]) == expected_names
                      and set(stored.files) == expected_names | {"metadata", "metadata_sha256"})
        arrays = {}
        for key in expected_names:
            arrays[key] = stored[key].copy()
            audit.require(name + ":array_checksum:" + key,
                          numerical_array_hash(arrays[key]) == metadata["array_hashes"][key])
    settings, original = report["settings"], report["estimate"]["diagnostics"]
    audit.require(name + ":raw_config", all(config[key] == settings[key] == original[key]
                  for key in ("chains", "burn", "thin", "seed")))
    problem = report["identity"]["numerical_problem"]
    d, depths, nodes = problem["d"], len(problem["depths"]), np.asarray(problem["alpha_nodes"])
    draws = original["draws"]
    for i in range(chains):
        index = arrays[f"alpha_indices_{i}"]
        audit.require(name + f":sample_shapes:{i}", arrays[f"predictions_{i}"].shape == (draws, d)
                      and arrays[f"depths_{i}"].shape == (draws, depths)
                      and arrays[f"scalars_{i}"].shape == (draws, 2) and index.shape == (draws,)
                      and index.dtype.kind in "iu" and bool(np.all((index >= 0) & (index < len(nodes)))))
        audit.require(name + f":alpha_trace:{i}", np.array_equal(arrays[f"scalars_{i}"][:, 0], nodes[index]))
        state = metadata["states"][i]
        audit.require(name + f":last_state:{i}", state["alpha_index"] == int(index[-1])
                      and state["log_likelihood"] == float(arrays[f"scalars_{i}"][-1, 1])
                      and sorted(state["ranks"]) == list(range(d))
                      and original["final_states"][i] == dict(alpha_index=state["alpha_index"], ranks=state["ranks"]))
    attempts, accepts = arrays["attempts"], arrays["accepts"]
    audit.require(name + ":transition_counters", arrays["burn_done"].shape == (chains,)
                  and np.all(arrays["burn_done"] == config["burn"])
                  and attempts.shape == accepts.shape == (chains, 4)
                  and np.all(attempts.sum(axis=1) == draws*config["thin"])
                  and np.all((accepts >= 0) & (accepts <= attempts))
                  and np.array_equal(attempts, original["move_attempts"])
                  and np.array_equal(accepts, original["move_accepts"]))
    predictions = np.stack([arrays[f"predictions_{i}"] for i in range(chains)])
    depth_samples = np.stack([arrays[f"depths_{i}"] for i in range(chains)])
    scalar_samples = np.stack([arrays[f"scalars_{i}"] for i in range(chains)])
    alpha_indices = np.stack([arrays[f"alpha_indices_{i}"] for i in range(chains)])
    audit.require(name + ":depth_samples", np.isfinite(depth_samples).all()
                  and np.all(depth_samples >= 0) and np.max(np.abs(depth_samples.sum(2)-1)) <= 1e-7)
    moments = stable_predictive_moments(predictions)
    pred = stable_chain_summary(predictions)
    depth = stable_chain_summary(depth_samples)
    scalar = stable_chain_summary(scalar_samples)
    alpha = stable_chain_summary(np.eye(len(nodes))[alpha_indices])
    checks = (("predictive_probs", pred["mean"]), ("mcse_probs", pred["mcse"]),
              ("posterior_depth", depth["mean"]), ("mcse_depth", depth["mcse"]),
              ("posterior_alpha", alpha["mean"]), ("mcse_alpha", alpha["mcse"]))
    for key, value in checks:
        audit.require(name + ":raw_summary:" + key, _close(value, report["estimate"][key]))
    audit.require(name + ":chain_mean_readback", _close(pred["chain_means"], original["chain_mean_probs"])
                  and _close(pred["chain_means"].mean(0), pred["mean"]))
    for key, value in (("rhat_probs", pred["rhat"]), ("rhat_depth", depth["rhat"]),
                       ("rhat_alpha_and_loglik", scalar["rhat"]), ("ess_probs", pred["ess"])):
        expected = _finite_list(value)
        audit.require(name + ":raw_diagnostic:" + key,
                      all((left is None and right is None) or
                          (left is not None and right is not None and math.isclose(left, right, rel_tol=1e-12, abs_tol=2e-14))
                          for left, right in zip(expected, original[key])) and len(expected) == len(original[key]))
    error = arrays["max_errors"]
    audit.require(name + ":normalization_record", error.shape == (chains,)
                  and np.isfinite(error).all() and np.all(error >= 0)
                  and float(error.max()) == original["max_conditional_probability_sum_error"])
    fresh = deepcopy(report["estimate"])
    fresh["predictive_probs"], fresh["mcse_probs"] = pred["mean"], pred["mcse"]
    for key, value in (("rhat_probs", pred["rhat"]), ("rhat_depth", depth["rhat"]),
                       ("rhat_alpha_and_loglik", scalar["rhat"]), ("ess_probs", pred["ess"])):
        fresh["diagnostics"][key] = _finite_list(value)
    return moments, fresh


def load_models(audit, folder, source, index, prefix):
    """Validate each checkpoint and derive its probabilities from saved logits."""
    models = {}
    for run in RUNS:
        name = f"model{index}:{run}"
        path = folder / "models" / run / f"seq_{index:06d}.json"
        record = audit.read(name, path)
        spec = record["identity"]["model"]
        run_folder = ROOT / "results" / "transformer" / "runs" / run
        audit.require(name + ":checkpoint", sha_file(run_folder / "model_100000.pt") == spec["checkpoint_sha256"]
                      and sha_file(run_folder / "config.json") == spec["config_sha256"]
                      and json.loads((run_folder / "config.json").read_text()) == spec["config"])
        config = spec["config"]
        audit.require(name + ":training_prior", config.get("family", "lsa") == "lsa"
                      and config["d"] == 64 and config["n_ctx"] == 1024
                      and config["alpha"] == [0.5, 1.5] and config["ls"] == [0, 8]
                      and config["no_permute"] is False and config["seed"] == RUNS.index(run))
        producer = "extend_shared_base_models.py" if "frozen_helpers_sha256" in spec else "eval_shared_base_models.py"
        audit.require(name + ":producer", sha_file(TRANSFORMER / producer) == spec["driver_sha256"]
                      and sha_file(TRANSFORMER / "model.py") == spec["model_sha256"])
        if "frozen_helpers_sha256" in spec:
            audit.require(name + ":frozen_helpers", sha_file(TRANSFORMER / "eval_shared_base_models.py") == spec["frozen_helpers_sha256"])
        audit.require(name + ":array_name", Path(record["arrays_file"]).name == record["arrays_file"])
        array_path = path.with_name(record["arrays_file"])
        audit.require(name + ":array_hash", audit.track(name + ":arrays", array_path) == record["arrays_sha256"])
        j = record["prefixes"].index(prefix)
        with np.load(array_path, allow_pickle=False) as arrays:
            logits, logp = arrays["logits"].astype(float), arrays["logp"]
            shifted = logits-logits.max(axis=1, keepdims=True)
            recomputed = shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            audit.require(name + ":logits", sha_array(arrays["logits"], "<f4") == record["logits_sha256"]
                          and sha_array(logp, "<f8") == record["logp_sha256"] and _close(logp, recomputed))
            p = np.exp(logp[j])
            oracle = np.asarray(source["record"]["oracle"]["transition_probs"])[source["tokens"][prefix-1]]
            audit.require(name + ":alignment", np.array_equal(arrays["oracle_probs"][j], oracle)
                          and arrays["prefixes"][j] == prefix
                          and arrays["targets"][j] == source["tokens"][prefix]
                          and _close(p, arrays["predictive_probs"][j], atol=1e-15)
                          and abs(arrays["selected_loss_bits"][j]+logp[j, source["tokens"][prefix]]/np.log(2)) <= 1e-13)
        alignment = record["checks"]["alignment"]
        audit.require(name + ":causal_alignment_checks", alignment["passed"] is True
                      and all(0 <= alignment[key] <= alignment["tolerance"] for key in
                              ("batch_vs_single_logits", "full_vs_truncated_prefix_logits", "future_tokens_causality_logits")))
        models[run] = p
    return models


def load_reference(audit, args, index, prefix, order, tokens, oracle, models, inputs):
    name = f"reference{index}:t{prefix}:alpha{order}"
    stem = f"seq_{index:06d}_t{prefix:04d}"
    path = args.references / "refs" / f"alpha{order}" / (stem + ".json")
    report = audit.read(name, path)
    identity, settings = report["identity"], report["settings"]
    audit.require(name + ":identity", report["schema"] == "shared-base-full-context-reference-v1"
                  and report["reference_id"] == fingerprint(identity)
                  and identity["inputs"] == inputs and report["sequence_index"] == index
                  and report["prefix_length"] == prefix and report["source_id"] == inputs["source_id"])
    verify_local_hashes(audit, name, identity["code_sha256"], required=CODE_FILES)
    problem = identity["numerical_problem"]
    audit.require(name + ":observations", problem["d"] == 64 and problem["prefix_length"] == prefix
                  and problem["token_hash"] == numerical_array_hash(np.asarray(tokens, dtype="<i8")))
    audit.require(name + ":numerical_files", set(problem["numerical_code_hashes"]) == NUMERICAL_CODE)
    for filename, expected in problem["numerical_code_hashes"].items():
        if filename.startswith("store/"):
            current = args.store / filename.split("/", 1)[1]
        elif filename.startswith("pmm/"):
            current = args.pmm_src / "product_model_with_memory" / filename.split("/", 1)[1]
        else:
            current = TRANSFORMER / filename
        audit.require(name + ":numerical_hash:" + filename, sha_file(current) == expected)
    audit.require(name + ":runtime", problem["runtime"] == dict(
        python=list(sys.version_info[:3]), numpy=np.__version__, scipy=importlib.metadata.version("scipy")))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = 1+nodes/2, weights/2
    audit.require(name + ":prior", settings["alpha_order"] == order
                  and all(_close(item["alpha_nodes"], nodes, atol=1e-14)
                          and _close(item["alpha_weights"], weights, atol=1e-14)
                          and item["depths"] == list(range(9))
                          and _close(item["depth_prior"], np.ones(9)/9, atol=1e-14)
                          for item in (settings, problem)))
    audit.require(name + ":row_grid", settings["row_grid"] == dict(lower=-32., upper=32., step=1/32)
                  and all(problem["kernel_grid"][key] == value for key, value in settings["row_grid"].items())
                  and problem["prob_tol"] == 1e-7)
    audit.require(name + ":proposal", identity["proposal_descriptor"] == proposal_descriptor(
        SimpleNamespace(surrogate_steps=settings["surrogate_steps"]))
        and report["estimate"]["diagnostics"]["summary_spec"] == SUMMARY_SPEC)
    audit.require(name + ":seed", identity["sampling_seed_tag"] == settings["sampling_seed_tag"] == SEED_TAG
                  and settings["seed"] == stable_seed(SEED_TAG, settings["sampling_master_seed"], index, prefix, order)
                  and settings["chains"] == 4)
    checkpoint = args.references / "chains" / f"alpha{order}" / (stem + ".npz")
    audit.require(name + ":checkpoint_path", Path(report["chain_checkpoint"]).resolve() == checkpoint.resolve())
    moments, fresh = raw_checkpoint(audit, name, checkpoint, report)
    audit.require(name + ":moments_name", Path(report["moments_file"]).name == report["moments_file"])
    moments_path = path.with_name(report["moments_file"])
    audit.require(name + ":moments_hash", audit.track(name + ":moments", moments_path) == report["moments_sha256"])
    with np.load(moments_path, allow_pickle=False) as saved:
        audit.require(name + ":moments_keys", set(saved.files) == set(moments))
        for key, value in moments.items():
            audit.require(name + ":moments_readback:" + key, _close(value, saved[key], rtol=1e-12))
    gates = diagnostic_gates(fresh)
    audit.require(name + ":stored_diagnostic_gates", gates == report["diagnostic_gates"])
    for gate, passed in gates.items():
        audit.check(name + ":diagnostic:" + gate, passed)
    grids = report["final_state_grid_checks"]
    audit.check(name + ":all_final_grid_checks", len(grids) == 4)
    for i, grid in enumerate(grids):
        passed = all(math.isfinite(grid[key]) and grid[key] >= 0 for key in
                     ("log_likelihood_difference_bits", "max_prediction_difference", "coarse_normalization_error", "fine_normalization_error"))
        passed = passed and grid["log_likelihood_difference_bits"] < 1e-5 and grid["max_prediction_difference"] < 1e-7
        passed = passed and max(grid["coarse_normalization_error"], grid["fine_normalization_error"]) < 1e-7
        audit.require(name + f":stored_grid_gate:{i}", passed == grid["passed"])
        audit.check(name + f":grid:{i}", passed)
    metrics = {run: comparison_metrics(moments["mean"], p, oracle, moments["cov_batch"], moments["cov_chain"])
               for run, p in models.items()}
    for run, metric in metrics.items():
        old = report["model_metrics"][run]
        for key, expected in (("tv", metric["total_variation"]), ("kl_bits", metric["kl_bits"]),
                              ("true_loss_gap_bits", metric["true_expected_loss_gap_bits"]),
                              ("kl_mcse_bits", metric["kl_uncertainty"]["mcse_bits"]),
                              ("gap_mcse_bits", metric["gap_uncertainty"]["mcse_bits"])):
            audit.require(name + f":metric_readback:{run}:{key}", math.isclose(old[key], expected, rel_tol=1e-10, abs_tol=1e-13))
    return dict(order=order, seed=settings["seed"], mean=moments["mean"], mcse=moments["mcse_probs"],
                diagnostics=dict(gates=gates, min_probability_ess=min(fresh["diagnostics"]["ess_probs"]),
                                 max_probability_mcse=float(max(fresh["mcse_probs"]))), metrics=metrics)


def run_audit(args):
    audit, references, rows, failure = Audit(), {}, [], None
    try:
        cohort = audit.read("cohort", args.input / "cohort.json")
        source = load_source(audit, args.input, cohort, args.index)
        _, tokens, oracle, _, inputs = load_inputs(args.input, args.index, args.prefix)
        models = load_models(audit, args.input, source, args.index, args.prefix)
        for order in (args.coarse_order, args.refined_order):
            references[order] = load_reference(audit, args, args.index, args.prefix, order,
                                                tokens, oracle, models, inputs)
        coarse, fine = references[args.coarse_order], references[args.refined_order]
        audit.require("independent_sampling_seeds", coarse["seed"] != fine["seed"])
        vector = probability_refinement(coarse["mean"], fine["mean"], coarse["mcse"], fine["mcse"])
        for key, passed in vector["checks"].items():
            audit.check("probability_refinement:" + key, passed)
        for run in RUNS:
            refined = metric_refinement(coarse["metrics"][run], fine["metrics"][run])
            for key, passed in refined["checks"].items():
                audit.check(f"metric_refinement:{run}:" + key, passed)
            rows.append(dict(sequence_index=args.index, prefix_length=args.prefix, run=run,
                             coarse=coarse["metrics"][run], refined=fine["metrics"][run],
                             probability_refinement=vector, metric_refinement=refined))
    except (ValueError, KeyError, IndexError, OSError) as error:
        failure = str(error)
        audit.check("audit_completed", False, reason=failure)
    for name, tracked in audit.inputs.items():
        audit.check(name + ":unchanged_during_audit", sha_file(tracked["path"]) == tracked["sha256"])
    passed = failure is None and bool(rows) and all(check["passed"] for check in audit.checks)
    for row in rows:
        row["resolved"] = passed
    return dict(schema="shared-base-surrogate-pair-audit-v1", sequence_index=args.index,
                prefix_length=args.prefix, coarse_order=args.coarse_order, refined_order=args.refined_order,
                passed=passed, error=failure, checks=audit.checks, inputs=audit.inputs,
                references=references, rows=rows,
                aggregate=dict(reported=False, reason="Selected-source audit; no resolved-only cohort average."))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output" / "shared_base_full_context")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--prefix", type=int, default=1024)
    parser.add_argument("--coarse-order", type=int, default=64)
    parser.add_argument("--refined-order", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    args = parser.parse_args()
    if args.store is None or args.pmm_src is None:
        parser.error("provide --store/--pmm-src or LSA_STORE/PMM_SRC")
    if not 1 <= args.prefix <= 1024 or args.index < 0 or args.coarse_order < 2 or args.refined_order <= args.coarse_order:
        parser.error("invalid source, prefix or exponent orders")
    args.input, args.references = args.input.resolve(), args.references.resolve()
    result = run_audit(args)
    result["exporter_sha256"] = sha_file(Path(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]),
                          failed=[c["name"] for c in result["checks"] if not c["passed"]], output=str(args.out))))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

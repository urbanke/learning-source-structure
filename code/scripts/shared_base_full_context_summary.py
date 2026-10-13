"""Independent, covariance-aware export of the bounded full-context pilot.

Comparison metrics use the posterior-mean prediction, not mean per-state
divergences. Source variation, Monte Carlo uncertainty and exponent-rule
refinement are distinct. No unresolved source draw may be dropped.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER = ROOT / "code" / "transformer"
LN2 = math.log(2)
RESOLUTION_BITS = 0.01
NUMERICAL_LIMIT_BITS = RESOLUTION_BITS / 3
RUNS = ("d64", "d64_seed1")
REFERENCE_CODE = {"eval_shared_base_reference.py", "full_context_statistics.py",
                  "eval_shared_base_models.py"}
NUMERICAL_CODE = {"shared_base_chain_runner.py", "shared_base_mcmc.py", "ranked_lsa.py",
                  "weighted_lsa.py", "shared_depth.py", "lsa_seq.py", "phi_source.py", "source.py",
                  "store/manifest.json", "store/anchors.json", "pmm/universal_tables.py",
                  "pmm/production_coding.py", "pmm/mellin.py", "pmm/kernel.py"}


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


def probabilities(values, name, *, positive=True):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite probability vector")
    if (positive and np.any(values <= 0)) or (not positive and np.any(values < 0)):
        raise ValueError(f"{name} has invalid probability entries")
    if abs(float(values.sum()) - 1) > 1e-7:
        raise ValueError(f"{name} is not normalized")
    return values


def covariance_of_mean(samples, *, batch_length=None):
    """Two PSD covariance estimates for the sampled predictive mean.

    Between-chain covariance uses independent chain means. Batch covariance
    uses consecutive within-chain batch means, dropping only the incomplete
    final batch. Neither estimator bounds bias from undiscovered modes.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 4 or samples.shape[2] < 2:
        raise ValueError("samples need shape (chains>=2, draws>=4, alphabet>=2)")
    if (not np.isfinite(samples).all() or np.any(samples < 0)
            or np.max(np.abs(samples.sum(axis=2) - 1)) > 1e-7):
        raise ValueError("invalid sampled predictive probabilities")
    chains, draws, d = samples.shape
    length = max(2, int(math.sqrt(draws))) if batch_length is None else batch_length
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1 or draws // length < 2:
        raise ValueError("batch_length must leave at least two complete batches per chain")
    n_batches = draws // length
    means = samples.mean(axis=1)
    mean = samples.mean(axis=(0, 1))
    centered_chains = means - means.mean(axis=0)
    between = centered_chains.T @ centered_chains / ((chains - 1) * chains)
    batches = samples[:, :n_batches * length].reshape(chains, n_batches, length, d).mean(axis=2)
    flat = batches.reshape(chains * n_batches, d)
    centered_batches = flat - flat.mean(axis=0)
    batch = centered_batches.T @ centered_batches / ((len(flat) - 1) * len(flat))
    return dict(mean=mean, covariance_batch=batch, covariance_between=between,
                chain_means=means, batch_length=int(length), batches_per_chain=n_batches,
                discarded_batch_tail_per_chain=draws - n_batches * length,
                chains=chains, draws_per_chain=draws)


def valid_covariance(value, d):
    value = np.asarray(value, dtype=float)
    if value.shape != (d, d) or not np.isfinite(value).all():
        raise ValueError("invalid mean covariance shape or entries")
    if np.max(np.abs(value - value.T)) > 1e-12 or np.linalg.eigvalsh(value).min() < -1e-12:
        raise ValueError("mean covariance must be symmetric positive semidefinite")
    return value


def projected_uncertainty(gradient, covariances):
    values = {}
    for name, covariance in covariances.items():
        variance = float(gradient @ covariance @ gradient)
        if variance < -1e-12:
            raise ValueError("negative projected variance")
        values[name + "_mcse_bits"] = math.sqrt(max(0.0, variance))
    values["mcse_bits"] = max(values.values())
    return values


def second_order_scale(hessian_diagonal, covariances):
    """Local Gaussian quadratic-term scales, not bias corrections or bounds."""
    results = {}
    for name, covariance in covariances.items():
        mean = 0.5 * float(np.dot(hessian_diagonal, np.diag(covariance)))
        # tr(H C H C) = sum_ij H_i H_j C_ij^2 for diagonal H.
        variance = 0.5 * float(np.sum(hessian_diagonal[:, None] * hessian_diagonal[None, :] * covariance ** 2))
        results[name] = dict(mean_scale_bits=mean, sd_scale_bits=math.sqrt(max(0.0, variance)))
    results["mean_scale_bits"] = max(value["mean_scale_bits"] for value in results.values())
    results["sd_scale_bits"] = max(value["sd_scale_bits"] for key, value in results.items() if isinstance(value, dict))
    diagonal_envelope = np.maximum.reduce([np.diag(value) for value in covariances.values()])
    results["noise_scale_bits"] = 0.5 * float(np.dot(hessian_diagonal, np.maximum(diagonal_envelope, 0)))
    return results


def comparison_metrics(bayes, transformer, true_next, covariance_batch, covariance_between):
    b = probabilities(bayes, "Bayes mean")
    t = probabilities(transformer, "transformer")
    r = probabilities(true_next, "true next law", positive=False)
    if b.shape != t.shape or b.shape != r.shape:
        raise ValueError("comparison alphabets do not match")
    covariances = dict(batch=valid_covariance(covariance_batch, len(b)),
                       between=valid_covariance(covariance_between, len(b)))
    log_ratio = np.log2(b / t)
    kl_gradient = log_ratio + 1 / LN2
    gap_gradient = r / (b * LN2)
    # Centering removes the irrelevant normal direction of the simplex.
    kl_gradient = kl_gradient - float(kl_gradient.mean())
    gap_gradient = gap_gradient - float(gap_gradient.mean())
    kl_uncertainty = projected_uncertainty(kl_gradient, covariances)
    gap_uncertainty = projected_uncertainty(gap_gradient, covariances)
    kl_second_order = second_order_scale(1 / (b * LN2), covariances)
    gap_second_order = second_order_scale(r / (b ** 2 * LN2), covariances)
    coordinate_mcse = np.sqrt(np.maximum.reduce([np.maximum(np.diag(value), 0) for value in covariances.values()]))
    return dict(total_variation=float(np.abs(b - t).sum() / 2),
                kl_bits=float(b @ log_ratio), true_expected_loss_gap_bits=float(r @ log_ratio),
                kl_uncertainty=kl_uncertainty, gap_uncertainty=gap_uncertainty,
                kl_second_order=kl_second_order, gap_second_order=gap_second_order,
                max_relative_probability_mcse=float(np.max(coordinate_mcse / b)),
                kl_numerical_scale_bits=max(kl_uncertainty["mcse_bits"], kl_second_order["noise_scale_bits"]),
                gap_numerical_scale_bits=max(gap_uncertainty["mcse_bits"], gap_second_order["noise_scale_bits"]),
                gradient_kl=kl_gradient.tolist(), gradient_gap=gap_gradient.tolist(),
                note="Metrics of the posterior mean. MCSE uses covariance projections; KL quadratic scales are local estimates, not bounds or bias corrections.")


def metric_refinement(coarse, refined):
    checks, differences = {}, {}
    for name, value_key, uncertainty_key in (("kl", "kl_bits", "kl_uncertainty"),
                                            ("gap", "true_expected_loss_gap_bits", "gap_uncertainty")):
        difference = abs(coarse[value_key] - refined[value_key])
        combined = math.hypot(coarse[uncertainty_key]["mcse_bits"], refined[uncertainty_key]["mcse_bits"])
        threshold = 5 * combined + 0.0001
        checks[name + "_numerical_precision"] = max(coarse[name + "_numerical_scale_bits"],
                                                     refined[name + "_numerical_scale_bits"]) <= NUMERICAL_LIMIT_BITS
        checks[name + "_refinement_resolution"] = difference <= NUMERICAL_LIMIT_BITS
        checks[name + "_refinement_mcse"] = difference <= threshold
        differences[name] = dict(absolute_difference_bits=difference, combined_mcse_bits=combined,
                                five_mcse_allowance_bits=threshold, resolution_limit_bits=NUMERICAL_LIMIT_BITS)
    return dict(passed=all(checks.values()), checks=checks, differences=differences)


def probability_refinement(coarse, refined, coarse_mcse, refined_mcse):
    coarse = probabilities(coarse, "coarse predictive mean")
    refined = probabilities(refined, "refined predictive mean")
    first, second = np.asarray(coarse_mcse, dtype=float), np.asarray(refined_mcse, dtype=float)
    if (coarse.shape != refined.shape or first.shape != coarse.shape or second.shape != coarse.shape
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or np.any(first < 0) or np.any(second < 0)):
        raise ValueError("invalid probability-refinement dimensions or MCSE")
    error = np.abs(coarse - refined)
    combined = np.sqrt(first ** 2 + second ** 2)
    tolerance = 5 * combined + 0.0002
    checks = dict(max_absolute_difference_below_0_01=float(error.max()) < 0.01,
                  every_coordinate_within_five_mcse_plus_0_0002=bool(np.all(error <= tolerance)))
    return dict(passed=all(checks.values()), checks=checks,
                max_absolute_difference=float(error.max()), total_variation=float(error.sum() / 2),
                max_combined_mcse=float(combined.max()),
                absolute_difference=error.tolist(), tolerance=tolerance.tolist())


def aggregate_sources(rows):
    """Four source draws are the units; checkpoints remain paired."""
    expected = {(index, run) for index in range(4) for run in RUNS}
    actual = {(row["sequence_index"], row["run"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("all four declared sources and both checkpoints are required")
    unresolved = sorted({row["sequence_index"] for row in rows if not row["resolved"]})
    if unresolved:
        return dict(reported=False, unresolved_source_indices=unresolved,
                    reason="No resolved-only average: every predeclared source remains in the pilot.")
    lookup = {(row["sequence_index"], row["run"]): row["refined"] for row in rows}
    keys = ("total_variation", "kl_bits", "true_expected_loss_gap_bits")
    def mean_se(values):
        values = np.asarray(values, dtype=float)
        return dict(mean=float(values.mean()), source_se=float(values.std(ddof=1) / 2),
                    source_values=values.tolist(), n_sources=4)
    by_model = {run: {key: mean_se([lookup[index, run][key] for index in range(4)]) for key in keys}
                for run in RUNS}
    paired = {key: mean_se([lookup[index, RUNS[1]][key] - lookup[index, RUNS[0]][key]
                            for index in range(4)]) for key in keys}
    return dict(reported=True, by_model=by_model, paired_seed1_minus_seed0=paired,
                note="Source SE uses four independent source draws. The two model checkpoints are paired, not eight independent observations; this pilot does not estimate training-seed variability reliably.")


class Audit:
    def __init__(self):
        self.checks, self.inputs = [], {}

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))
        return bool(passed)

    def require(self, name, passed, **details):
        if not self.check(name, passed, **details):
            raise ValueError(name)

    def track(self, name, path):
        path = Path(path)
        digest = sha_file(path)
        self.inputs[name] = dict(path=str(path.resolve()), sha256=digest)
        return digest

    def read(self, name, path):
        self.track(name, path)
        value = json.loads(Path(path).read_text())
        self.require(name + ":object", isinstance(value, dict))
        return value


def numerical_array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(values.dtype.str.encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def stable_seed(*parts):
    value = 1469598103934665603
    for part in parts:
        for byte in str(part).encode():
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value % ((1 << 32) - 1)


def verify_local_hashes(audit, name, hashes, *, required=None):
    audit.require(name + ":nonempty_code_hashes", bool(hashes))
    if required is not None:
        audit.require(name + ":complete_code_hashes", set(hashes) == set(required))
    for filename, expected in hashes.items():
        audit.require(name + ":code_name", Path(filename).name == filename)
        audit.require(name + ":code:" + filename, sha_file(TRANSFORMER / filename) == expected)


def load_source(audit, folder, cohort, index):
    name = f"source{index}"
    path = folder / "data" / f"seq_{index:06d}.json"
    record = audit.read(name, path)
    audit.require(name + ":schema", record["schema"] == "shared-base-full-context-source-v1")
    identity = record["identity"]
    audit.require(name + ":identity", record["source_id"] == fingerprint(identity))
    declaration = cohort["records"][index]
    audit.require(name + ":declaration", declaration["index"] == index
                  and declaration["source_id"] == record["source_id"]
                  and declaration["file_sha256"] == audit.inputs[name]["sha256"])
    spec = identity["source"]
    audit.require(name + ":prior", spec["d"] == 64 and spec["n_tokens"] == 1025
                  and spec["alpha_prior"] == [0.5, 1.5] and spec["depth_prior"] == list(range(9))
                  and spec["depth_scope"] == "one shared depth for every transition row"
                  and spec["rank_prior"] == "uniform alphabet permutation")
    audit.require(name + ":stable_seed", identity["sequence_index"] == index
                  and identity["master_seed"] == cohort["master_seed"]
                  and identity["sequence_seed"] == stable_seed("shared-base-full-context-v1", cohort["master_seed"], index))
    audit.require(name + ":source_code", spec["source_sha256"] == sha_file(TRANSFORMER / "source.py"))
    producer = (TRANSFORMER / "eval_shared_base_models.py").read_text()
    function = next(node for node in ast.parse(producer).body if isinstance(node, ast.FunctionDef) and node.name == "draw_source")
    construction = "".join(producer.splitlines(keepends=True)[function.lineno - 1:function.end_lineno])
    audit.require(name + ":construction_code", spec["construction_sha256"] == hashlib.sha256(construction.encode()).hexdigest())
    tokens = np.asarray(record["tokens"])
    audit.require(name + ":tokens", tokens.shape == (1025,) and tokens.dtype.kind in "iu"
                  and bool(np.all((tokens >= 0) & (tokens < 64)))
                  and record["tokens_sha256"] == sha_array(tokens, "<i2"))
    oracle = record["oracle"]
    matrix = np.asarray(oracle["transition_probs"], dtype=float)
    initial = probabilities(oracle["initial_probs"], "oracle initial")
    audit.require(name + ":oracle", matrix.shape == (64, 64) and np.isfinite(matrix).all()
                  and np.all(matrix > 0) and np.max(np.abs(matrix.sum(axis=1) - 1)) <= 1e-12
                  and oracle["transition_probs_sha256"] == sha_array(matrix, "<f8")
                  and oracle["initial_probs_sha256"] == sha_array(initial, "<f8"))
    return dict(record=record, tokens=tokens, true_next=matrix[tokens[1023]],
                file_sha256=audit.inputs[name]["sha256"], prefix_sha256=sha_array(tokens[:1024], "<i2"))


def load_model(audit, folder, source, index, run):
    name = f"model{index}:{run}"
    path = folder / "models" / run / f"seq_{index:06d}.json"
    record = audit.read(name, path)
    audit.require(name + ":schema", record["schema"] == "shared-base-full-context-model-v1")
    identity = record["identity"]
    audit.require(name + ":identity", record["cache_id"] == fingerprint(identity)
                  and identity["source_id"] == source["record"]["source_id"]
                  and identity["source_file_sha256"] == source["file_sha256"])
    spec = identity["model"]
    run_path = ROOT / "results" / "transformer" / "runs" / run
    config = json.loads((run_path / "config.json").read_text())
    audit.require(name + ":checkpoint", sha_file(run_path / "model_100000.pt") == spec["checkpoint_sha256"])
    audit.require(name + ":config", sha_file(run_path / "config.json") == spec["config_sha256"] and config == spec["config"])
    audit.require(name + ":training_prior", config.get("family", "lsa") == "lsa"
                  and config["d"] == 64 and config["n_ctx"] == 1024 and config["alpha"] == [0.5, 1.5]
                  and config["ls"] == [0, 8] and config["no_permute"] is False
                  and config["seed"] == RUNS.index(run))
    audit.require(name + ":producer_code", sha_file(TRANSFORMER / "model.py") == spec["model_sha256"]
                  and sha_file(TRANSFORMER / "eval_shared_base_models.py") == spec["driver_sha256"])
    audit.require(name + ":prefix", record["prefixes"] == [1024]
                  and record["target_indices"] == [1024] and record["targets"] == [int(source["tokens"][1024])]
                  and record["tokens_sha256"] == source["record"]["tokens_sha256"]
                  and record["prefix_tokens_sha256"]["1024"] == source["prefix_sha256"])
    audit.require(name + ":array_name", Path(record["arrays_file"]).name == record["arrays_file"])
    array_path = path.with_name(record["arrays_file"])
    audit.require(name + ":arrays_hash", audit.track(name + ":arrays", array_path) == record["arrays_sha256"])
    with np.load(array_path, allow_pickle=False) as arrays:
        logits, logp = arrays["logits"].astype(float), arrays["logp"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        independent_logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        audit.require(name + ":logits", logits.shape == (1, 64) and np.isfinite(logits).all()
                      and sha_array(arrays["logits"], "<f4") == record["logits_sha256"]
                      and sha_array(logp, "<f8") == record["logp_sha256"]
                      and np.max(np.abs(independent_logp - logp)) < 2e-14)
        p = probabilities(np.exp(logp[0]), "transformer")
        audit.require(name + ":prediction_readback", np.array_equal(arrays["predictive_probs"], record["predictive_probs"])
                      and np.max(np.abs(arrays["predictive_probs"][0] - p)) <= 1e-15
                      and arrays["prefixes"].tolist() == [1024]
                      and arrays["targets"].tolist() == record["targets"]
                      and np.array_equal(arrays["oracle_probs"][0], source["true_next"]))
        loss = -logp[0, source["tokens"][1024]] / LN2
        audit.require(name + ":loss_alignment", abs(float(arrays["selected_loss_bits"][0]) - loss) < 1e-13)
    alignment = record["checks"]["alignment"]
    audit.require(name + ":alignment_checks", alignment["passed"] is True
                  and all(0 <= alignment[key] <= alignment["tolerance"] for key in
                          ("batch_vs_single_logits", "full_vs_truncated_prefix_logits", "future_tokens_causality_logits")))
    return dict(record=record, prediction=p, file_sha256=audit.inputs[name]["sha256"],
                checkpoint_sha256=spec["checkpoint_sha256"])


def reference_diagnostics(estimate):
    d = estimate["diagnostics"]
    values = []
    for key, length in (("rhat_probs", 64), ("rhat_depth", 9), ("rhat_alpha_and_loglik", 2)):
        value = np.asarray(d[key], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f"missing/nonfinite reference diagnostic {key}")
        values.extend(value.tolist())
    ess = np.asarray(d["ess_probs"], dtype=float)
    se = np.asarray(estimate["mcse_probs"], dtype=float)
    if ess.shape != (64,) or se.shape != (64,) or not np.isfinite(ess).all() or not np.isfinite(se).all() or np.any(se < 0):
        raise ValueError("invalid reference ESS or MCSE")
    raw = float(d["max_conditional_probability_sum_error"])
    gates = dict(split_rhat_below_1_05=min(values) >= 0 and max(values) < 1.05,
                 probability_ess_at_least_200=float(ess.min()) >= 200,
                 maximum_probability_mcse_at_most_0_005=float(se.max()) <= 0.005,
                 raw_normalization=math.isfinite(raw) and 0 <= raw <= 1e-7
                 and abs(float(np.sum(estimate["predictive_probs"])) - 1) <= 1e-7)
    return dict(gates=gates, max_rhat=max(values), min_probability_ess=float(ess.min()),
                max_probability_mcse=float(se.max()), raw_normalization_error=raw)


def load_reference(audit, folder, source, models, cohort_hash, index, order, args, prior_receipts):
    name = f"reference{index}:alpha{order}"
    path = folder / "refs" / f"alpha{order}" / f"seq_{index:06d}.json"
    report = audit.read(name, path)
    start_check = len(audit.checks)
    audit.require(name + ":schema", report["schema"] == "shared-base-full-context-reference-v1")
    identity = report["identity"]
    audit.require(name + ":identity", report["reference_id"] == fingerprint(identity))
    inputs = identity["inputs"]
    audit.require(name + ":inputs", inputs["cohort_sha256"] == cohort_hash
                  and inputs["source_file_sha256"] == source["file_sha256"]
                  and inputs["source_id"] == source["record"]["source_id"]
                  and inputs["tokens_sha256"] == source["record"]["tokens_sha256"]
                  and inputs["prefix_tokens_sha256"] == source["prefix_sha256"]
                  and inputs["sequence_index"] == index and inputs["prefix_length"] == 1024)
    for run, model in models.items():
        audit.require(name + ":model_input:" + run, inputs["model_files"][run]["file_sha256"] == model["file_sha256"]
                      and inputs["model_files"][run]["cache_id"] == model["record"]["cache_id"])
    verify_local_hashes(audit, name, identity["code_sha256"], required=REFERENCE_CODE)
    problem = identity["numerical_problem"]
    audit.require(name + ":problem_tokens", problem["d"] == 64 and problem["prefix_length"] == 1024
                  and problem["token_hash"] == numerical_array_hash(np.asarray(source["tokens"][:1024], dtype="<i8")))
    audit.require(name + ":complete_numerical_hashes", set(problem["numerical_code_hashes"]) == NUMERICAL_CODE)
    audit.require(name + ":numerical_runtime", problem["runtime"] == dict(
        python=list(sys.version_info[:3]), numpy=np.__version__, scipy=importlib.metadata.version("scipy")))
    for filename, expected in problem["numerical_code_hashes"].items():
        if filename.startswith("store/"):
            current = args.store / filename.split("/", 1)[1]
        elif filename.startswith("pmm/"):
            current = args.pmm_src / "product_model_with_memory" / filename.split("/", 1)[1]
        else:
            current = TRANSFORMER / filename
        audit.require(name + ":numerical_code:" + filename, sha_file(current) == expected)
    settings = report["settings"]
    expected_nodes, expected_weights = np.polynomial.legendre.leggauss(order)
    expected_nodes, expected_weights = 1 + expected_nodes / 2, expected_weights / 2
    audit.require(name + ":prior", settings["alpha_order"] == order
                  and np.allclose(settings["alpha_nodes"], expected_nodes, atol=1e-14, rtol=0)
                  and np.allclose(settings["alpha_weights"], expected_weights, atol=1e-14, rtol=0)
                  and settings["depths"] == list(range(9))
                  and np.allclose(settings["depth_prior"], np.ones(9) / 9, atol=1e-14, rtol=0)
                  and np.allclose(problem["alpha_nodes"], expected_nodes, atol=1e-14, rtol=0)
                  and np.allclose(problem["alpha_weights"], expected_weights, atol=1e-14, rtol=0)
                  and problem["depths"] == list(range(9))
                  and np.allclose(problem["depth_prior"], np.ones(9) / 9, atol=1e-14, rtol=0)
                  and all(problem["kernel_grid"][key] == value for key, value in
                          dict(lower=-32.0, upper=32.0, step=1 / 32).items())
                  and settings["row_grid"] == dict(lower=-32.0, upper=32.0, step=1 / 32))
    audit.require(name + ":moment_name", Path(report["moments_file"]).name == report["moments_file"])
    moments_path = path.with_name(report["moments_file"])
    moment_hash = audit.track(name + ":moments", moments_path)
    audit.require(name + ":moments_hash", moment_hash == report["moments_sha256"])
    with np.load(moments_path, allow_pickle=False) as stored:
        moments = {key: stored[key].copy() for key in stored.files}
    chains, draws = int(moments["chains"]), int(moments["draws"])
    diagnostics = report["estimate"]["diagnostics"]
    audit.require(name + ":sampling_configuration", chains == settings["chains"] == diagnostics["chains"] == 4
                  and draws == diagnostics["draws"] and draws >= 16
                  and all(settings[key] == diagnostics[key] for key in ("burn", "thin", "seed"))
                  and settings["seed"] == stable_seed("shared-base-full-context-reference-v1",
                      settings["sampling_master_seed"], index, 1024, order))
    audit.require(name + ":batch_configuration", int(moments["batch_length"]) == max(2, int(math.sqrt(draws)))
                  and int(moments["batches_per_chain"]) == draws // int(moments["batch_length"]))
    mean = probabilities(moments["mean"], "Bayes mean")
    c_batch, c_between = valid_covariance(moments["cov_batch"], 64), valid_covariance(moments["cov_chain"], 64)
    se = np.sqrt(np.maximum(0, np.maximum(np.diag(c_batch), np.diag(c_between))))
    audit.require(name + ":coordinate_mcse", np.allclose(se, moments["mcse_probs"], atol=1e-14, rtol=1e-10)
                  and np.allclose(se, report["estimate"]["mcse_probs"], atol=1e-14, rtol=1e-10)
                  and np.allclose(mean, report["estimate"]["predictive_probs"], atol=2e-14, rtol=0))
    chain_means = np.asarray(moments["chain_means"], dtype=float)
    audit.require(name + ":chain_mean_readback", chain_means.shape == (chains, 64)
                  and np.allclose(chain_means.mean(axis=0), mean, atol=2e-14, rtol=0)
                  and np.allclose(chain_means, diagnostics["chain_mean_probs"], atol=2e-14, rtol=0))
    centered_chain_means = chain_means - chain_means.mean(axis=0)
    audit.require(name + ":between_chain_covariance", np.allclose(
        centered_chain_means.T @ centered_chain_means / ((chains - 1) * chains),
        c_between, atol=1e-14, rtol=1e-10))
    binding = dict(reference_json_sha256=audit.inputs[name]["sha256"], moments_sha256=moment_hash,
                   reference_id=report["reference_id"])
    if args.smoke:
        receipt = prior_receipts[name]
        audit.require(name + ":prior_raw_receipt", receipt["binding"] == binding and receipt["passed"] is True)
    else:
        chain_path = folder / "chains" / f"alpha{order}" / f"seq_{index:06d}.npz"
        chain_hash = audit.track(name + ":raw_chains", chain_path)
        with np.load(chain_path, allow_pickle=False) as stored:
            metadata_text = str(stored["metadata"].item())
            audit.require(name + ":raw_metadata_hash", hashlib.sha256(metadata_text.encode()).hexdigest()
                          == str(stored["metadata_sha256"].item()))
            metadata = json.loads(metadata_text)
            audit.require(name + ":raw_fingerprint", metadata["format"] == "shared_base_chain_run_v1"
                          and metadata["fingerprint"] == problem)
            for key, expected in metadata["array_hashes"].items():
                audit.require(name + ":raw_array:" + key, numerical_array_hash(stored[key]) == expected)
            samples = np.stack([stored[f"predictions_{chain}"] for chain in range(chains)])
            audit.require(name + ":raw_configuration", metadata["config"] ==
                          {key: settings[key] for key in ("chains", "burn", "thin", "seed")})
        independent = covariance_of_mean(samples)
        errors = dict(mean=float(np.max(np.abs(independent["mean"] - mean))),
                      covariance_batch=float(np.max(np.abs(independent["covariance_batch"] - c_batch))),
                      covariance_between=float(np.max(np.abs(independent["covariance_between"] - c_between))))
        audit.require(name + ":independent_covariances", errors["mean"] <= 2e-14
                      and max(errors["covariance_batch"], errors["covariance_between"]) <= 1e-13
                      and np.allclose(independent["chain_means"], moments["chain_means"], atol=2e-14, rtol=0)
                      and independent["batch_length"] == int(moments["batch_length"])
                      and independent["draws_per_chain"] == int(moments["draws"]))
        receipt = dict(passed=True, binding=binding, chain_sha256=chain_hash,
                       chain_metadata_sha256=hashlib.sha256(metadata_text.encode()).hexdigest(),
                       independent_errors=errors, chains=chains, draws_per_chain=independent["draws_per_chain"])
        del samples
    diagnostic = reference_diagnostics(report["estimate"])
    for key, passed in diagnostic["gates"].items():
        audit.check(name + ":diagnostic:" + key, passed)
    audit.require(name + ":stored_diagnostic_gates", report["diagnostic_gates"] == diagnostic["gates"])
    grid = report["final_state_grid_checks"]
    audit.check(name + ":grid_count", len(grid) == settings["chains"])
    for j, check in enumerate(grid):
        values = [check[key] for key in ("log_likelihood_difference_bits", "max_prediction_difference",
                                         "coarse_normalization_error", "fine_normalization_error")]
        passed = all(np.isfinite(value) and value >= 0 for value in values)
        passed = passed and values[0] < 1e-5 and max(values[1:]) < 1e-7
        audit.check(name + f":grid:{j}", passed)
        audit.require(name + f":stored_grid:{j}", check["passed"] is bool(passed))
    local_pass = all(check["passed"] for check in audit.checks[start_check:])
    return dict(report=report, mean=mean, mcse_probs=se, covariance_batch=c_batch, covariance_between=c_between,
                diagnostics=diagnostic, raw_validation=receipt, local_pass=local_pass,
                sampling=dict(chains=int(moments["chains"]), draws_per_chain=int(moments["draws"]),
                              seed=settings["seed"], elapsed_seconds=report["estimate"]["elapsed_seconds"]))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/shared_base_full_context"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coarse-order", type=int, default=16)
    parser.add_argument("--refined-order", type=int, default=32)
    parser.add_argument("--order-selection", type=Path,
                        help="explicit per-source JSON overrides: index -> coarse_order, refined_order, reason")
    parser.add_argument("--store", type=Path, default=os.environ.get("LSA_STORE"))
    parser.add_argument("--pmm-src", type=Path, default=os.environ.get("PMM_SRC"))
    parser.add_argument("--smoke", action="store_true", help="reuse a bound prior raw-covariance validation receipt")
    parser.add_argument("--raw-validation", type=Path, help="previous full summary, required with --smoke")
    args = parser.parse_args()
    if args.store is None or args.pmm_src is None:
        parser.error("provide --store/--pmm-src or LSA_STORE/PMM_SRC")
    if args.smoke and args.raw_validation is None:
        parser.error("--smoke requires --raw-validation from a completed full export")
    if args.coarse_order < 2 or args.refined_order <= args.coarse_order:
        parser.error("require increasing exponent orders of at least two")
    audit, rows, references, receipts = Audit(), [], {}, {}
    prior_receipts, overrides = {}, {}
    try:
        if args.smoke:
            previous = audit.read("prior_raw_validation", args.raw_validation)
            audit.require("prior_receipt_script", previous["provenance"]["script_sha256"] == sha_file(Path(__file__)))
            prior_receipts = previous["raw_validation"]
        if args.order_selection:
            overrides = audit.read("order_selection", args.order_selection)
            audit.require("order_selection:indices", set(overrides) <= {"0", "1", "2", "3"})
            for index, selected in overrides.items():
                audit.require("order_selection:" + index,
                              isinstance(selected["coarse_order"], int) and isinstance(selected["refined_order"], int)
                              and 2 <= selected["coarse_order"] < selected["refined_order"]
                              and isinstance(selected["reason"], str) and bool(selected["reason"].strip()))
        cohort = audit.read("cohort", args.input / "cohort.json")
        audit.require("cohort:protocol", cohort["schema"] == "shared-base-full-context-cohort-v1"
                      and cohort["n_sequences"] == 4 and cohort["master_seed"] == 20260922
                      and cohort["prefixes"] == [1024] and cohort["n_tokens_per_sequence"] == 1025
                      and len(cohort["records"]) == 4)
    except (OSError, KeyError, TypeError, ValueError) as error:
        audit.check("cohort_or_selection:read_failure", False, error=str(error))
        cohort = None
    for index in range(4):
        try:
            if cohort is None:
                raise ValueError("declared cohort or validation settings unavailable")
            selected = overrides.get(str(index), dict(coarse_order=args.coarse_order, refined_order=args.refined_order,
                                                     reason="predeclared default exponent refinement"))
            coarse_order, refined_order = selected["coarse_order"], selected["refined_order"]
            source = load_source(audit, args.input, cohort, index)
            models = {run: load_model(audit, args.input, source, index, run) for run in RUNS}
            estimates = {}
            for order in (coarse_order, refined_order):
                estimates[order] = load_reference(audit, args.input, source, models,
                    audit.inputs["cohort"]["sha256"], index, order, args, prior_receipts)
                ref = estimates[order]
                key = f"reference{index}:alpha{order}"
                receipts[key] = ref["raw_validation"]
                references[key] = dict(diagnostics=ref["diagnostics"], sampling=ref["sampling"],
                                       local_pass=ref["local_pass"])
            coarse, refined = estimates[coarse_order], estimates[refined_order]
            seed_pass = audit.check(f"source{index}:independent_refinement_seeds", coarse["sampling"]["seed"] != refined["sampling"]["seed"])
            vector_comparison = probability_refinement(coarse["mean"], refined["mean"],
                                                       coarse["mcse_probs"], refined["mcse_probs"])
            for key, passed in vector_comparison["checks"].items():
                audit.check(f"source{index}:probability_refinement:" + key, passed)
            for run in RUNS:
                metrics = {}
                for label, estimate in (("coarse", coarse), ("refined", refined)):
                    metrics[label] = comparison_metrics(estimate["mean"], models[run]["prediction"],
                        source["true_next"], estimate["covariance_batch"], estimate["covariance_between"])
                    stored = estimate["report"]["model_metrics"][run]
                    pairs = (("tv", metrics[label]["total_variation"]), ("kl_bits", metrics[label]["kl_bits"]),
                             ("true_loss_gap_bits", metrics[label]["true_expected_loss_gap_bits"]),
                             ("kl_mcse_bits", metrics[label]["kl_uncertainty"]["mcse_bits"]),
                             ("gap_mcse_bits", metrics[label]["gap_uncertainty"]["mcse_bits"]),
                             ("kl_noise_scale_bits", metrics[label]["kl_second_order"]["noise_scale_bits"]))
                    for key, actual in pairs:
                        audit.require(f"source{index}:{run}:{label}:metric_readback:{key}",
                                      np.isclose(stored[key], actual, rtol=1e-7, atol=1e-11))
                comparison = metric_refinement(metrics["coarse"], metrics["refined"])
                for key, passed in comparison["checks"].items():
                    audit.check(f"source{index}:{run}:" + key, passed)
                resolved = (coarse["local_pass"] and refined["local_pass"] and seed_pass
                            and vector_comparison["passed"] and comparison["passed"])
                rows.append(dict(sequence_index=index, run=run, source_id=source["record"]["source_id"],
                                 resolved=resolved, coarse_order=coarse_order, refined_order=refined_order,
                                 order_selection_reason=selected["reason"], **metrics, refinement=comparison,
                                 probability_refinement=vector_comparison,
                                 checkpoint_sha256=models[run]["checkpoint_sha256"]))
        except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
            audit.check(f"source{index}:unresolved_input", False, error=str(error))
            existing = {row["run"] for row in rows if row["sequence_index"] == index}
            for row in rows:
                if row["sequence_index"] == index:
                    row["resolved"] = False
            rows.extend(dict(sequence_index=index, run=run, resolved=False, error=str(error))
                        for run in RUNS if run not in existing)
    aggregate = aggregate_sources(rows)
    passed = all(check["passed"] for check in audit.checks)
    if not passed and aggregate["reported"]:
        aggregate = dict(reported=False, reason="Export provenance or global validation remains unresolved.")
    report = dict(schema="shared-base-full-context-summary-v1", passed=passed,
                  prefix_length=1024, n_sources=4, models=list(RUNS), rows=rows, references=references,
                  aggregate=aggregate, raw_validation=receipts, checks=audit.checks,
                  order_selection=overrides,
                  thresholds=dict(practical_resolution_bits=RESOLUTION_BITS, numerical_limit_bits=NUMERICAL_LIMIT_BITS,
                                  refinement_combined_mcse_multiplier=5, refinement_allowance_bits=0.0001),
                  additional_export_validation="Gap curvature noise scale uses r/b^2 and the covariance diagonal envelope; this check extends the frozen producer's first-order gap precision check.",
                  provenance=dict(script_sha256=sha_file(Path(__file__)), inputs=audit.inputs,
                                  smoke_reuses_prior_raw_validation=args.smoke),
                  limitations=["Four source draws form a descriptive pilot, not a full-prior equivalence result.",
                               "Two checkpoint seeds are paired on the same four draws, not eight independent sources.",
                               "MCSE, curvature scales and convergence diagnostics do not bound undiscovered-mode bias.",
                               "Exponent-refinement differences include independent Monte Carlo variation.",
                               "TV is descriptive; no naive normal equivalence test is attached to it.",
                               "No cumulative regret or copying-onset conclusion follows from this selected-prefix comparison."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    table = ["sequence\tmodel\tresolved\ttv\tkl_bits\tkl_mcse_bits\tkl_noise_bits\ttrue_gap_bits\tgap_mcse_bits\tgap_noise_bits"]
    for row in rows:
        m = row.get("refined")
        numbers = ([m["total_variation"], m["kl_bits"], m["kl_uncertainty"]["mcse_bits"],
                    m["kl_second_order"]["noise_scale_bits"], m["true_expected_loss_gap_bits"],
                    m["gap_uncertainty"]["mcse_bits"], m["gap_second_order"]["noise_scale_bits"]] if m else [""] * 7)
        table.append("\t".join(str(value) for value in [row["sequence_index"], row["run"], row["resolved"], *numbers]))
    args.out.with_suffix(".tsv").write_text("\n".join(table) + "\n")
    failed = [check["name"] for check in audit.checks if not check["passed"]]
    print(json.dumps(dict(out=str(args.out), passed=report["passed"], checks=len(audit.checks), failed=failed)))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

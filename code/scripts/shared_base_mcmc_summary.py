"""Independently check and export the bounded shared-base MCMC pilots.

This script reads stored probabilities and diagnostics; it does not import
the predictor or trust its pass flags. A failed or missing required input
produces passed=false and a nonzero exit. The 16/32-node comparison mixes
exponent-quadrature differences with sampling variability and is not a
convergence certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss


REPOSITORY = Path(__file__).resolve().parents[2]
TRANSFORMER = REPOSITORY / "code" / "transformer"
SOURCE_PRIOR = {
    "alpha_prior": "uniform on [0.5,1.5]",
    "depth_prior": "one shared uniform depth in 0,...,8",
    "rank_prior": "uniform over all label permutations",
}
PREDICTION_CODE = {
    "ranked_lsa.py", "weighted_lsa.py", "phi_source.py", "lsa_seq.py",
    "shared_base_mcmc.py", "refs_shared_base_mcmc.py", "source.py",
    "validate_shared_base_mcmc.py",
}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []
        self.inputs = {}

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))

    def load(self, name, path):
        try:
            content = path.read_bytes()
            self.inputs[name] = dict(path=str(path.resolve()),
                                     sha256=hashlib.sha256(content).hexdigest())
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("top-level JSON must be an object")
            self.check(name + ":read", True)
            return value
        except (OSError, ValueError) as error:
            self.check(name + ":read", False, error=str(error))
            return None


def vector(value, length, name, *, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def finite_number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def check_provenance(audit, name, report, *, prediction=False):
    provenance = report["provenance"]
    local = provenance["code_sha256"]
    required = PREDICTION_CODE if prediction else {"ranked_lsa.py", "shared_base_mcmc.py", "weighted_lsa.py", "lsa_seq.py", "phi_source.py"}
    audit.check(name + ":required_code_hashes", required <= set(local))
    for filename, expected in local.items():
        if Path(filename).name != filename or Path(filename).suffix != ".py":
            raise ValueError(f"invalid local provenance filename {filename!r}")
        actual = sha_file(TRANSFORMER / filename)
        audit.check(name + ":code:" + filename, actual == expected)
    store = Path(provenance["store"]).expanduser().resolve()
    source = Path(provenance["source"]).expanduser().resolve()
    if prediction:
        store_hashes = {"manifest.json": provenance["store_manifest_sha256"],
                        "anchors.json": provenance["anchors_sha256"]}
        backend_hashes = {"universal_tables.py": provenance["backend_sha256"],
                          "production_coding.py": provenance["production_config_sha256"]}
    else:
        store_hashes = provenance["store_files_sha256"]
        backend_hashes = provenance["backend_files_sha256"]
    for filename in ("manifest.json", "anchors.json"):
        audit.check(name + ":store:" + filename,
                    sha_file(store / filename) == store_hashes[filename])
    for filename in ("universal_tables.py", "production_coding.py"):
        audit.check(name + ":backend:" + filename,
                    sha_file(source / "product_model_with_memory" / filename) == backend_hashes[filename])
    return dict(store=str(store), source=str(source),
                store_hashes=store_hashes, backend_hashes=backend_hashes)


def assess_prediction(audit, name, report, prefix_length, alpha_order):
    audit.check(name + ":schema", report["schema"] == "shared_base_mcmc_prediction_v1")
    audit.check(name + ":alphabet", report["d"] == 64)
    audit.check(name + ":prefix_length", report["prefix_length"] == prefix_length)
    tokens = report["tokens"]
    valid_tokens = (isinstance(tokens, list) and len(tokens) == prefix_length
                    and all(isinstance(token, int) and not isinstance(token, bool)
                            and 0 <= token < 64 for token in tokens))
    audit.check(name + ":tokens", valid_tokens)
    settings = report["settings"]
    audit.check(name + ":source_prior", all(settings[key] == value for key, value in SOURCE_PRIOR.items()))
    audit.check(name + ":alpha_order", settings["alpha_order"] == alpha_order)
    expected_nodes, expected_weights = leggauss(alpha_order)
    expected_nodes, expected_weights = 1 + expected_nodes / 2, expected_weights / 2
    nodes = vector(settings["alpha_nodes"], alpha_order, "alpha_nodes")
    weights = vector(settings["alpha_weights"], alpha_order, "alpha_weights", positive=True)
    audit.check(name + ":alpha_rule", np.max(np.abs(nodes - expected_nodes)) <= 1e-14
                and np.max(np.abs(weights - expected_weights)) <= 1e-14)
    grid = settings["row_grid"]
    audit.check(name + ":row_grid", grid == dict(lower=-32.0, upper=32.0, step=1 / 32))
    provenance = check_provenance(audit, name, report, prediction=True)
    estimate = report["estimate"]
    p = vector(estimate["predictive_probs"], 64, "predictive_probs", positive=True)
    se = vector(estimate["mcse_probs"], 64, "mcse_probs")
    audit.check(name + ":probability_sum", abs(float(p.sum()) - 1) <= 1e-7,
                absolute_error=abs(float(p.sum()) - 1), tolerance=1e-7)
    audit.check(name + ":mcse_nonnegative", bool(np.all(se >= 0)))
    for key, count in (("posterior_depth", 9), ("posterior_alpha", alpha_order)):
        probabilities = vector(estimate[key], count, key)
        audit.check(name + ":" + key, np.all(probabilities >= 0)
                    and abs(float(probabilities.sum()) - 1) <= 1e-7)
    diagnostic = estimate["diagnostics"]
    rhat = np.concatenate([vector(diagnostic["rhat_probs"], 64, "rhat_probs"),
                           vector(diagnostic["rhat_depth"], 9, "rhat_depth"),
                           vector(diagnostic["rhat_alpha_and_loglik"], 2, "rhat_alpha_and_loglik")])
    ess = vector(diagnostic["ess_probs"], 64, "ess_probs")
    max_rhat, min_ess, max_mcse = float(rhat.max()), float(ess.min()), float(se.max())
    computed_thresholds = dict(split_rhat_below_1_05=bool(np.all(rhat >= 0) and max_rhat < 1.05),
                               probability_ess_at_least_200=min_ess >= 200,
                               maximum_probability_mcse_at_most_0_005=max_mcse <= 0.005)
    for key, passed in computed_thresholds.items():
        audit.check(name + ":" + key, passed)
    audit.check(name + ":stored_threshold_flags",
                report["diagnostic_threshold_checks"] == computed_thresholds
                and report["diagnostic_thresholds_passed"] is all(computed_thresholds.values()))
    raw_error = finite_number(diagnostic["max_conditional_probability_sum_error"],
                               "max_conditional_probability_sum_error", nonnegative=True)
    audit.check(name + ":raw_conditional_probabilities", raw_error <= 1e-7,
                absolute_error=raw_error, tolerance=1e-7)
    recorded_sum_error = finite_number(diagnostic["probability_sum_error"], "probability_sum_error")
    audit.check(name + ":reported_probability_sum", abs(recorded_sum_error - (float(p.sum()) - 1)) <= 1e-12)
    chains = integer(diagnostic["chains"], "chains", 2)
    budget = dict(chains=chains, burn=integer(diagnostic["burn"], "burn"),
                  draws=integer(diagnostic["draws"], "draws", 16),
                  thin=integer(diagnostic["thin"], "thin", 1),
                  seed=integer(diagnostic["seed"], "seed"))
    chain_means = np.asarray(diagnostic["chain_mean_probs"], dtype=float)
    if chain_means.shape != (chains, 64) or not np.isfinite(chain_means).all():
        raise ValueError("chain_mean_probs must be a finite chains-by-64 matrix")
    reconstructed_mean_error = float(np.max(np.abs(chain_means.mean(axis=0) - p)))
    between_chain_se = chain_means.std(axis=0, ddof=1) / math.sqrt(chains)
    audit.check(name + ":mean_from_chain_means", reconstructed_mean_error <= 1e-12,
                max_absolute_error=reconstructed_mean_error, tolerance=1e-12)
    audit.check(name + ":mcse_covers_between_chain_se", np.all(se + 1e-12 >= between_chain_se),
                max_between_chain_se=float(between_chain_se.max()), numerical_allowance=1e-12)
    checks = report["final_state_grid_checks"]
    audit.check(name + ":grid_checks_present", isinstance(checks, list) and len(checks) == chains)
    grid_bits, grid_probs, grid_norms = [], [], []
    for index, check in enumerate(checks):
        error_bits = finite_number(check["log_likelihood_difference_bits"], "grid likelihood error", nonnegative=True)
        error_p = finite_number(check["max_prediction_difference"], "grid probability error", nonnegative=True)
        coarse = finite_number(check["coarse_normalization_error"], "coarse normalization error", nonnegative=True)
        fine = finite_number(check["fine_normalization_error"], "fine normalization error", nonnegative=True)
        passed = error_bits < 1e-5 and error_p < 1e-7 and max(coarse, fine) <= 1e-7
        audit.check(name + f":grid_check:{index}", passed)
        audit.check(name + f":stored_grid_flag:{index}", check["passed"] is passed)
        grid_bits.append(error_bits)
        grid_probs.append(error_p)
        grid_norms.extend((coarse, fine))
    if not grid_bits:
        raise ValueError("required row-grid checks are missing")
    summary = dict(prefix_length=prefix_length, alpha_order=alpha_order,
                   max_rhat=max_rhat, min_probability_ess=min_ess, max_probability_mcse=max_mcse,
                   elapsed_seconds=finite_number(estimate["elapsed_seconds"], "elapsed_seconds", nonnegative=True),
                   max_raw_probability_sum_error=raw_error,
                   max_between_chain_se=float(between_chain_se.max()),
                   mean_reconstruction_error=reconstructed_mean_error,
                   max_grid_error_bits=max(grid_bits), max_grid_probability_error=max(grid_probs),
                   max_grid_normalization_error=max(grid_norms), budget=budget,
                   observed_labels=len(set(tokens)))
    return dict(summary=summary, p=p, se=se, tokens=tokens, settings=settings,
                provenance=provenance, generated_source=report["generated_source"])


def assess_pair(audit, name, coarse, refined):
    audit.check(name + ":identical_tokens", coarse["tokens"] == refined["tokens"])
    audit.check(name + ":identical_source_prior", all(coarse["settings"][key] == refined["settings"][key]
                                                      for key in SOURCE_PRIOR))
    audit.check(name + ":identical_generated_source", coarse["generated_source"] == refined["generated_source"])
    audit.check(name + ":identical_backend", coarse["provenance"] == refined["provenance"])
    audit.check(name + ":independent_sampling_seeds", coarse["summary"]["budget"]["seed"]
                != refined["summary"]["budget"]["seed"])
    difference = np.abs(coarse["p"] - refined["p"])
    combined = np.sqrt(coarse["se"] ** 2 + refined["se"] ** 2)
    tolerance = 5 * combined + 0.0002
    audit.check(name + ":refinement_absolute_cap", float(difference.max()) < 0.01,
                max_absolute_error=float(difference.max()), tolerance=0.01)
    audit.check(name + ":refinement_mcse", np.all(difference <= tolerance),
                absolute_error=difference.tolist(), tolerance=tolerance.tolist())
    return dict(prefix_length=coarse["summary"]["prefix_length"],
                alpha16=coarse["summary"], alpha32=refined["summary"],
                max_absolute_probability_difference=float(difference.max()),
                max_combined_mcse=float(combined.max()),
                max_comparison_tolerance=float(tolerance.max()),
                probability_total_variation=float(difference.sum() / 2),
                kl_alpha16_to_alpha32_bits=float(np.sum(coarse["p"] * np.log2(coarse["p"] / refined["p"]))),
                generated_source=coarse["generated_source"])


def assess_validation(audit, name, report, expected_schema, expected_cases):
    audit.check(name + ":schema", report["schema"] == expected_schema)
    check_provenance(audit, name, report)
    cases = report["cases"]
    audit.check(name + ":case_count", len(cases) == expected_cases)
    audit.check(name + ":reported_pass", report["passed"] is True)
    for index, case in enumerate(cases):
        audit.check(name + f":case:{index}:pass", case["passed"] is True)
        checks = case["checks"]
        audit.check(name + f":case:{index}:nonempty_checks", bool(checks))
        audit.check(name + f":case:{index}:all_checks", all(check["passed"] is True for check in checks.values()))
    return dict(cases=len(cases), checks=sum(len(case["checks"]) for case in cases),
                elapsed_seconds=report["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = Audit()
    validation_summaries, predictions, pairs = {}, {}, []
    for filename, schema, count in (("validation.json", "shared_base_mcmc_validation_v1", 4),
                                    ("d64_control.json", "shared_base_mcmc_d64_control_v1", 2)):
        value = audit.load(filename, args.input / filename)
        if value is not None:
            try:
                validation_summaries[filename] = assess_validation(audit, filename, value, schema, count)
            except (KeyError, TypeError, ValueError, OSError) as error:
                audit.check(filename + ":structure", False, error=str(error))
    for prefix in (32, 128):
        for order, directory in ((16, args.input), (32, args.refined)):
            name = f"d64_n{prefix}_alpha{order}"
            value = audit.load(name, directory / f"d64_n{prefix}.json")
            if value is not None:
                try:
                    predictions[prefix, order] = assess_prediction(audit, name, value, prefix, order)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    audit.check(name + ":structure", False, error=str(error))
        if (prefix, 16) in predictions and (prefix, 32) in predictions:
            pairs.append(assess_pair(audit, f"d64_n{prefix}_pair",
                                     predictions[prefix, 16], predictions[prefix, 32]))
    audit.check("both_prefix_pairs_available", len(pairs) == 2)
    report = dict(schema="shared_base_mcmc_summary_v1", passed=all(check["passed"] for check in audit.checks),
                  validation=validation_summaries, prefixes=pairs, checks=audit.checks,
                  provenance=dict(inputs=audit.inputs, script=str(Path(__file__).resolve()),
                                  script_sha256=sha_file(Path(__file__).resolve())),
                  limitations=[
                      "These are two selected d=64 prefixes; they do not establish accuracy across the source prior.",
                      "Passing split-Rhat, ESS and MCSE thresholds is not a convergence certificate or a bound on undiscovered-mode bias.",
                      "The paired 16/32-node difference includes independent sampling variability; it does not isolate pure exponent-quadrature error.",
                      "Row-grid refinements assess final chain states only, not every visited or unvisited posterior state.",
                      "This export recomputes prediction, diagnostic and refinement comparisons; separate validation artifacts retain their own recorded checks.",
                  ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    rows = ["prefix_tokens\talpha16_max_rhat\talpha32_max_rhat\talpha16_min_ess\talpha32_min_ess\talpha16_max_mcse\talpha32_max_mcse\tmax_probability_difference\tmax_combined_mcse\ttotal_variation\tkl_16_to_32_bits\talpha16_seconds\talpha32_seconds\tpassed"]
    for pair in pairs:
        left, right = pair["alpha16"], pair["alpha32"]
        rows.append("\t".join(str(value) for value in (
            pair["prefix_length"], left["max_rhat"], right["max_rhat"],
            left["min_probability_ess"], right["min_probability_ess"],
            left["max_probability_mcse"], right["max_probability_mcse"],
            pair["max_absolute_probability_difference"], pair["max_combined_mcse"],
            pair["probability_total_variation"], pair["kl_alpha16_to_alpha32_bits"],
            left["elapsed_seconds"], right["elapsed_seconds"], report["passed"])))
    args.out.with_suffix(".tsv").write_text("\n".join(rows) + "\n")
    failed = [check["name"] for check in audit.checks if not check["passed"]]
    print(json.dumps(dict(out=str(args.out), passed=report["passed"], checks=len(audit.checks), failed=failed)))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

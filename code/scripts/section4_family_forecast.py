"""Finite-population forecast for the early cost of an unknown source family.

Enumerate all S8 x S8 orbits of sequences of one through five tokens. The
three family likelihoods reuse the validated LSA store and short shared-base
catalog; no source draws, transformer inference, or posterior sampling occur.
Finite enumeration eliminates sampling error, not inherited quadrature error.
Numerical probability masses are never rescaled.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
sys.path.insert(0, str(ROOT / "code/scripts"))
from mixture_family_evidence import FamilyEvidence
from shared_base_short_evidence import SharedBaseShortEvidence
from section4_family_cost import audit, recorded_path

FAMILIES = ["product", "shared_base", "uniform_base"]
MODELS = ["pooled_seed0", "pooled_seed1"]
OUT = ROOT / "results/transformer/section4_family_forecast_2026-09-18"
RAW = ROOT / "output/section4_family_forecast_2026-09-18/orbits.npz"
CATALOG = ROOT / "results/transformer/shared_base_conditional_curves/shortprefix_reference.json"
LN2 = math.log(2)
TOLERANCE = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(values):
    names = {}
    return tuple(names.setdefault(int(value), len(names)) for value in values)


def patterns(length, alphabet_size):
    """All equality patterns realizable on the specified alphabet."""
    require(length >= 0 and alphabet_size >= 1, "invalid pattern dimensions")
    result = [()]
    for _ in range(length):
        result = [prefix + (value,) for prefix in result
                  for value in range(min(max(prefix, default=-1) + 2, alphabet_size))]
    return result


def falling(size, distinct):
    return math.prod(range(size - distinct + 1, size + 1))


def orbits(length, d1=8, d2=8):
    """Representatives and exact cardinalities under coordinate relabelling."""
    representatives, multiplicities = [], []
    for u, v in itertools.product(patterns(length, d1), patterns(length, d2)):
        representatives.append(np.asarray(u, dtype=np.int64) * d2 + np.asarray(v))
        multiplicities.append(falling(d1, max(u, default=-1) + 1)
                              * falling(d2, max(v, default=-1) + 1))
    require(sum(multiplicities) == (d1 * d2) ** length,
            "orbit multiplicities do not partition all labelled sequences")
    return np.asarray(representatives), np.asarray(multiplicities, dtype=np.int64)


def summarize(log_probabilities, multiplicities):
    """Integrate with the raw family masses, preserving quadrature residuals."""
    logs = np.asarray(log_probabilities, dtype=float)
    counts = np.asarray(multiplicities)
    require(logs.ndim == 2 and logs.shape[1] == 3 and counts.shape == logs.shape[:1]
            and np.isfinite(logs).all() and np.all(counts > 0), "invalid orbit probabilities")
    log_mix = logsumexp(logs, axis=1) - math.log(3)
    weighted = counts[:, None] * np.exp(logs)
    mixture_mass = counts * np.exp(log_mix)
    posterior = np.exp(logs - logsumexp(logs, axis=1)[:, None])
    cost = (logs - log_mix[:, None]) / LN2
    masses = np.array([math.fsum(weighted[:, k]) for k in range(3)])
    costs = np.array([math.fsum(weighted[:, k] * cost[:, k]) for k in range(3)])
    posterior_matrix = np.array([[math.fsum(weighted[:, k] * posterior[:, j])
                                 for j in range(3)] for k in range(3)])
    entropy = math.fsum(mixture_mass * -np.sum(posterior * np.log2(posterior), axis=1))
    classification = math.fsum(mixture_mass * np.max(posterior, axis=1))
    mixture_total = math.fsum(mixture_mass)
    identity_error = abs(float(costs.mean()) - (math.log2(3) * mixture_total - entropy))
    posterior_error = float(np.max(abs(cost - math.log2(3) - np.log2(posterior))))
    require(max(abs(masses - 1)) < TOLERANCE, "family probability mass is not one")
    require(float(cost.max()) <= math.log2(3) + TOLERANCE, "mixture coding bound failed")
    require(float(costs.min()) >= -TOLERANCE, "negative population family cost")
    require(identity_error < TOLERANCE and posterior_error < TOLERANCE,
            "posterior or entropy identity failed")
    return dict(family_probability_mass=masses.tolist(),
                family_cost_bits=costs.tolist(), equal_family_cost_bits=float(costs.mean()),
                posterior_entropy_bits=float(entropy),
                posterior_weight_mean_by_generating_family=posterior_matrix.tolist(),
                mean_true_family_posterior_weight=float(np.trace(posterior_matrix) / 3),
                optimal_family_classification_accuracy=float(classification),
                maximum_sequence_cost_bits=float(cost.max()),
                entropy_identity_error_bits=float(identity_error),
                posterior_cost_identity_error_bits=posterior_error,
                total_mixture_probability_mass=float(mixture_total))


def enumerate_curves(references, d1=8, d2=8, maximum_predictions=4):
    records, arrays = [], {}
    for length in range(1, maximum_predictions + 2):
        tokens, counts = orbits(length, d1, d2)
        logs = np.array([[reference.log_joint(x) for reference in references] for x in tokens])
        record = summarize(logs, counts)
        record.update(scored_predictions=length - 1, observed_tokens=length,
                      n_orbits=len(tokens), labelled_sequences=(d1*d2) ** length)
        records.append(record)
        for name, value in (("tokens", tokens), ("multiplicity", counts), ("log_probability", logs)):
            arrays[f"m{length}_{name}"] = value
    return records, arrays


class SavedSharedMarginals:
    """Independent backward marginalization of an archived refinement report."""
    def __init__(self, report):
        require(report["d"] == 64 and report["time"] == 4
                and len(report["records"]) == 15, "wrong shared-base refinement scope")
        self.joint = {(): 1.0}
        for entry in report["records"]:
            pattern = tuple(entry["pattern"])
            self.joint[pattern] = entry["prefix_probability"]
            for token in range(max(pattern) + 2):
                self.joint[pattern + (token,)] = (entry["prefix_probability"]
                                                * entry["predictive_probs"][token])
        for length in [3, 2, 1]:
            for pattern in patterns(length, 64):
                k = max(pattern) + 1
                self.joint[pattern] = (math.fsum(self.joint[pattern + (a,)] for a in range(k))
                                       + (64-k) * self.joint[pattern + (k,)])

    def log_joint(self, tokens):
        return math.log(self.joint[canonical(tokens)])


def maximum_metric_difference(first, second):
    keys = ["family_cost_bits", "equal_family_cost_bits", "posterior_entropy_bits",
            "posterior_weight_mean_by_generating_family", "optimal_family_classification_accuracy"]
    return {key: max(float(np.max(abs(np.asarray(a[key]) - np.asarray(b[key]))))
                     for a, b in zip(first, second)) for key in keys}


def validate_relabelling(references):
    worst, checked = 0., 0
    for length in range(1, 6):
        x, _ = orbits(length)
        for tokens in x[np.unique(np.linspace(0, len(x)-1, min(12, len(x)), dtype=int))]:
            changed = ((7 - tokens // 8) * 8 + (tokens % 8 + 3) % 8)
            for reference in references:
                worst = max(worst, abs(reference.log_joint(tokens) - reference.log_joint(changed)) / LN2)
                checked += 1
    require(worst < TOLERANCE, "family likelihood is not invariant under coordinate relabelling")
    return dict(checked=checked, maximum_absolute_error_bits=worst)


def validate_saved_endpoints(references):
    """Bind current likelihood evaluation to all archived paired endpoints."""
    worst, checked = 0., 0
    for family in FAMILIES:
        cohort_path = ROOT / f"output/multifamily_results_2026-09-16/heldout/{family}_prior.npz"
        saved_path = ROOT / f"output/multifamily_bayes_2026-09-16/short_comparison/{family}_prior.npz"
        with np.load(cohort_path, allow_pickle=False) as source, np.load(saved_path, allow_pickle=False) as saved:
            require(np.array_equal(source["sequence_seed"], saved["sequence_seed"]),
                    "endpoint source identities differ")
            for index, tokens in enumerate(source["tokens"]):
                for column, time in enumerate(saved["prefixes"]):
                    recomputed = np.array([reference.log_joint(tokens[:int(time)+1])
                                           for reference in references])
                    expected = saved["log_marginals_after_prediction"][index, column]
                    worst = max(worst, float(np.max(abs(recomputed - expected))) / LN2)
                    checked += 3
    require(worst < TOLERANCE, "enumerated likelihood providers disagree with saved endpoints")
    return dict(component_log_likelihoods_checked=checked, maximum_absolute_error_bits=worst)


def empirical_comparison(existing, curves):
    """Raw estimates plus the unbiased fixed-coefficient-one control variate."""
    times = existing["prediction_times"]
    by_time = {record["scored_predictions"]: record for record in curves}
    result = {}
    groups = [("equal_family_average", existing["equal_family_average"], None)]
    groups += [(family, existing["cohorts"][family], k) for k, family in enumerate(FAMILIES)]
    for name, group, index in groups:
        deterministic = [by_time[t]["equal_family_cost_bits"] if index is None
                         else by_time[t]["family_cost_bits"][index] for t in times]
        sample = group["bayes_mixture_minus_known_family_bits"]
        deviation = np.asarray(sample["mean"]) - deterministic
        item = dict(bayes_forecast_bits=deterministic, empirical_bayes_cost=sample,
                    empirical_minus_forecast_bits=deviation.tolist(),
                    empirical_minus_forecast_in_SE=(deviation / sample["se"]).tolist(), models={})
        for model in MODELS:
            saved = group["models"][model]
            residual = saved["transformer_minus_mixture_bits"]
            item["models"][model] = dict(
                raw_transformer_cost=saved["transformer_minus_known_family_bits"],
                paired_transformer_minus_mixture=residual,
                control_variate_transformer_cost=dict(
                    mean=(np.asarray(deterministic) + residual["mean"]).tolist(),
                    se=residual["se"], n_seq=residual["n_seq"], coefficient=1))
        result[name] = item
    return result


def build():
    require(os.environ.get("LSA_STORE") and os.environ.get("PMM_SRC"),
            "set LSA_STORE and PMM_SRC to the existing read-only anchor store and package")
    existing = audit()
    hashes = dict(existing["verified_file_sha256"])
    hashes[str(ROOT / "code/scripts/section4_family_cost.py")] = sha(ROOT / "code/scripts/section4_family_cost.py")
    numerical_path = ROOT / "output/multifamily_bayes_2026-09-16/family_evidence/validation.json"
    numerical = json.loads(numerical_path.read_text())
    require(numerical["passed"] is True, "prior family quadrature validation failed")
    for name, digest in numerical["code_sha256"].items():
        path = ROOT / "code/transformer" / name
        require(sha(path) == digest, "validated family evidence code changed: " + name)
        hashes[str(path)] = digest
    shared = SharedBaseShortEvidence()
    hashes.update(shared.validation["verified_file_sha256"])
    store_manifest = Path(os.environ["LSA_STORE"]) / "manifest.json"
    backend = Path(os.environ["PMM_SRC"]) / "product_model_with_memory/universal_tables.py"
    # Bind the runtime backend to the archived reference configuration, not
    # merely to its present contents.
    reference_record = ROOT / "output/multifamily_bayes_2026-09-16/family_evidence/product_on_product_prior.json"
    bound = json.loads(reference_record.read_text())
    require(sha(backend) == bound["backend_sha256"], "reference backend hash changed")
    for path in (store_manifest, backend, Path(__file__)):
        hashes[str(path)] = sha(path)
    references = [FamilyEvidence("product"), shared, FamilyEvidence("uniform_base")]
    curves, arrays = enumerate_curves(references)
    checks = dict(coordinate_relabelling=validate_relabelling(references),
                  saved_endpoint_replay=validate_saved_endpoints(references),
                  maximum_mass_error=max(abs(x - 1) for row in curves for x in row["family_probability_mass"]),
                  maximum_entropy_identity_error_bits=max(row["entropy_identity_error_bits"] for row in curves),
                  maximum_posterior_identity_error_bits=max(row["posterior_cost_identity_error_bits"] for row in curves))
    require([row["n_orbits"] for row in curves] == [1, 4, 25, 225, 2704], "unexpected orbit counts")
    require(np.min(np.diff([row["equal_family_cost_bits"] for row in curves])) >= -TOLERANCE,
            "expected cumulative family information must be nondecreasing")
    expanded_refs = [FamilyEvidence("product", bounds=(-36., 36.)), shared,
                     FamilyEvidence("uniform_base", bounds=(-36., 36.))]
    expanded, _ = enumerate_curves(expanded_refs)
    refined_refs = [FamilyEvidence("product", bounds=(-36., 36.), step=1/128), shared,
                    FamilyEvidence("uniform_base", bounds=(-36., 36.), step=1/128)]
    refined, _ = enumerate_curves(refined_refs)
    refinements = dict(lsa_expanded_bounds=maximum_metric_difference(curves, expanded),
                       lsa_halved_step=maximum_metric_difference(expanded, refined))
    catalog = json.loads(CATALOG.read_text())
    shared_reports = []
    for record in catalog["provenance"]["numerical_reports"]:
        path = recorded_path(record["path"])
        require(path.is_file() and sha(path) == record["sha256"], "shared refinement report missing or changed")
        hashes[str(path)] = record["sha256"]
        report = json.loads(path.read_text())
        require(report["prior"] == catalog["prior"]
                and report["script_sha256"] == catalog["provenance"]["exact_script_sha256"],
                "shared refinement prior/code identity changed")
        alternative, _ = enumerate_curves([references[0], SavedSharedMarginals(report), references[2]])
        refinements[path.name] = maximum_metric_difference(curves, alternative)
        shared_reports.append(dict(path=str(path), alpha_order=report["alpha_order"],
                                   bounds=report["bounds"], step=report["step"]))
    require(max(value for check in refinements.values() for value in check.values()) < TOLERANCE,
            "population metrics fail quadrature refinement")
    summary = dict(schema="section4-family-forecast-v1", families=FAMILIES,
                   family_prior_weights=[1/3]*3, coordinate_sizes=[8, 8],
                   scored_predictions=list(range(5)), records=curves,
                   convention="n predictions score X2 through X(n+1); X1 is supplied. Every family has exact initial probability 1/64.",
                   method="Exhaustive S8 x S8 orbit sum of full-prior family likelihoods. Quadrature-backed finite population calculation, not an analytic zero-error result.",
                   symmetry="Independent permutations of the eight row and eight column labels preserve all three priors. A pair of coordinate equality patterns with ku,kv distinct labels represents (8)_ku (8)_kv labelled sequences.",
                   numerical_masses_rescaled=False, numerical_uncertainty="No sampling uncertainty for the finite sums. Normalization residuals and quadrature refinement differences are reported separately; refinement is not a rigorous error bound.",
                   empirical_times=existing["prediction_times"], empirical=empirical_comparison(existing, curves),
                   control_variate="C_TF estimate = deterministic population C_B + paired sample mean(loss_TF - loss_mix), with fixed coefficient one. Its SE is the paired gap SE; it is unbiased conditional on reference numerical accuracy. Raw estimates are retained. No agreement between transformer and Bayes is assumed.",
                   empirical_scope="Existing 1024 independent sequences per full-prior family (3072 total), both pooled training seeds; no new transformer evaluation. Standard errors exclude variation across training runs and numerical reference error.",
                   empirical_deviation_policy="Report deviations in SE units without treating a collection of two-SE comparisons as a pass/fail rule.",
                   checks=checks, refinements=refinements,
                   quadrature=dict(production_lsa_bounds=[-30, 30], production_lsa_step=1/64,
                                   expanded_lsa_bounds=[-36, 36], refined_lsa_step=1/128,
                                   shared_reports=shared_reports),
                   verified_file_sha256=hashes, no_new_sampling=True,
                   no_new_transformer_inference=True, maximum_token_length=5)
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary, arrays = build()
    if args.check_only:
        require(json.loads((args.out / "summary.json").read_text()) == summary, "saved forecast differs")
        receipt = json.loads((args.out / "validation.json").read_text())
        require(receipt["passed"] is True and receipt["summary_sha256"] == sha(args.out / "summary.json")
                and receipt["script_sha256"] == sha(__file__) and receipt["raw_sha256"] == sha(args.raw),
                "saved forecast receipt differs")
        with np.load(args.raw, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items()),
                    "raw orbit enumeration differs")
        print("Finite-orbit forecast, quadrature refinements, paired estimates and provenance verified.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw, **arrays)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    receipt = dict(schema="section4-family-forecast-validation-v1", passed=True,
                   summary_sha256=sha(args.out / "summary.json"), script_sha256=sha(__file__),
                   raw_path=str(args.raw), raw_sha256=sha(args.raw), checks=summary["checks"],
                   refinements=summary["refinements"], tolerance=TOLERANCE,
                   inherited_source_audit="section4_family_cost.audit: complete saved 3072-sequence comparison and all linked identities rechecked.",
                   scientific_scope="All labelled sequences through five tokens; no statement about longer contexts or transformer family representations.")
    (args.out / "validation.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.out),
                          bayes_cost_bits=[r["equal_family_cost_bits"] for r in summary["records"]],
                          checks=summary["checks"]), indent=2))


if __name__ == "__main__":
    main()

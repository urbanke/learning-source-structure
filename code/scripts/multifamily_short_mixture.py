"""Compare saved pooled transformers with the actual three-family Bayes rule.

All three normalized family evidences are available through five tokens,
giving cumulative losses at prediction times 1 and 4. Product/uniform-base
references are read from validated endpoint files; the shared-base adapter
reuses the validated exhaustive short-prefix quadrature catalog. Longer
three-family-mixture comparisons are deliberately not inferred from these.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from mixture_family_evidence import FamilyEvidence, sha256
from shared_base_short_evidence import SharedBaseShortEvidence
from mixture_bayes import (EVIDENCE_KIND, SequenceEvidence, cumulative_mixture_losses,
                          mixture_prediction, posterior_family_weights)

FAMILIES = ("product", "shared_base", "uniform_base")
TIMES = (1, 4)
PRIOR = np.full(3, 1 / 3)
LN2 = np.log(2.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stats(values):
    values = np.asarray(values)
    require(values.ndim >= 1 and len(values) > 1 and np.isfinite(values).all(),
            "statistics require finite independent sequence records")
    return dict(mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist(), n_seq=len(values))


def balanced_stats(records):
    return dict(mean=np.mean([r["mean"] for r in records], axis=0).tolist(),
                se=(np.sqrt(np.sum(np.square([r["se"] for r in records]), axis=0)) / 3).tolist(),
                n_seq=sum(r["n_seq"] for r in records), family_weights=PRIOR.tolist())


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def evidence(prefix, references):
    return SequenceEvidence(prefix_length=len(prefix), evidence_kind=EVIDENCE_KIND,
                            log_marginals=np.array([references[f].log_joint(prefix) for f in FAMILIES]))


def sequential_validation(cohorts, references):
    """Independent product of posterior-weighted next-token predictions."""
    maximum_difference = 0.
    maximum_normalization_error = 0.
    examples = []
    checked = 0
    for family, source in cohorts.items():
        for i, tokens in enumerate(source["tokens"][:8]):
            sequential_loss = 0.
            fixed_weight_loss = 0.
            first = evidence(tokens[:1], references)
            for time in range(1, 5):
                parent = evidence(tokens[:time], references)
                children = np.stack([[references[name].log_joint(np.r_[tokens[:time], symbol])
                                      for symbol in range(64)] for name in FAMILIES])
                component_probs = np.exp(children - parent.log_marginals[:, None])
                maximum_normalization_error = max(maximum_normalization_error,
                                                  float(np.max(abs(component_probs.sum(axis=1) - 1))))
                prediction = mixture_prediction(parent, PRIOR, component_probs,
                                                successor_log_marginals=children)
                sequential_loss -= np.log2(prediction[tokens[time]])
                fixed_weight_loss -= np.log2(component_probs[:, tokens[time]].mean())
                after = evidence(tokens[:time + 1], references)
                endpoint_loss = cumulative_mixture_losses(after, PRIOR, first_token_evidence=first)["mixture_loss_bits"]
                error = float(abs(sequential_loss - endpoint_loss))
                maximum_difference = max(maximum_difference, error)
                checked += 1
            examples.append(dict(cohort=f"{family}_prior", sequence_index=i,
                                 sequential_mixture_loss_bits=float(sequential_loss),
                                 fixed_weight_average_loss_bits=float(fixed_weight_loss),
                                 fixed_minus_bayes_bits=float(fixed_weight_loss - sequential_loss)))
    distinguish = max(examples, key=lambda item: abs(item["fixed_minus_bayes_bits"]))
    require(maximum_difference < 1e-8 and maximum_normalization_error < 1e-9,
            "mixture sequential/endpoint or normalization validation failed")
    require(abs(distinguish["fixed_minus_bayes_bits"]) > 1e-4,
            "validation sample does not distinguish fixed from posterior family weights")
    return dict(passed=True, independent_sequences=24, prediction_steps_checked=checked,
                maximum_endpoint_vs_sequential_error_bits=maximum_difference,
                maximum_component_probability_sum_error=maximum_normalization_error,
                fixed_weight_comparison_example=distinguish,
                interpretation="Bayesian weights update after observations; fixed averaging is a different predictor.")


def run(raw, family_root, output, curated):
    numerical = json.loads((family_root / "validation.json").read_text())
    require(numerical.get("passed") is True, "family evidence quadrature validation is missing")
    for name, digest in numerical["code_sha256"].items():
        require(sha256(ROOT / "code/transformer" / name) == digest, "validated family code changed")
    prior_validation = ROOT / "results/transformer/multifamily_2026-09-16/validation.json"
    previous_audit = json.loads(prior_validation.read_text())
    require(previous_audit.get("status") == "passed", "original transformer/cohort audit must pass")
    source_manifest = json.loads((raw / "heldout/manifest.json").read_text())
    source_plan_id = source_manifest["plan_id"]
    require(fingerprint(source_manifest["spec"]) == source_plan_id, "source plan fingerprint mismatch")
    require(previous_audit["source_plan_id"] == source_plan_id
            and previous_audit["source_manifest_sha256"] == sha256(raw / "heldout/manifest.json"),
            "original audit belongs to a different source cohort")
    shared = SharedBaseShortEvidence()
    references = {"product": FamilyEvidence("product"), "shared_base": shared,
                  "uniform_base": FamilyEvidence("uniform_base")}
    verified = {str(path): sha256(path) for path in
                (family_root / "validation.json", prior_validation, raw / "heldout/manifest.json")}
    cohorts, records, raw_arrays = {}, {}, {}
    first_token_error = 0.
    saved_endpoint_error = 0.
    raw_file_hashes = {}
    output.mkdir(parents=True, exist_ok=True)
    curated.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        cohort = f"{family}_prior"
        path = raw / "heldout" / f"{cohort}.npz"
        source_record_path = path.with_suffix(".json")
        source_record = json.loads(source_record_path.read_text())
        require(source_record["plan_id"] == source_plan_id, "source record belongs to another plan")
        require(sha256(path) == source_record["data_sha256"], "frozen source hash mismatch")
        verified[str(path)] = sha256(path)
        verified[str(source_record_path)] = sha256(source_record_path)
        source = load_npz(path)
        require(source["tokens"].shape == (1024, 1025), "unexpected full-prior cohort size")
        require(np.array_equal(source["prefixes"], [1, 4, 16, 64, 256, 1024]), "unexpected time grid")
        cohorts[family] = source
        marginals = np.empty((1024, 2, 3))
        for reference_family in ("product", "uniform_base"):
            ref_path = family_root / f"{reference_family}_on_{cohort}.npz"
            ref_record_path = ref_path.with_suffix(".json")
            ref_record = json.loads(ref_record_path.read_text())
            require(ref_record["complete"] is True and ref_record["source_plan_id"] == source_plan_id,
                    "family reference incomplete or belongs to another source plan")
            require(ref_record["source_data_sha256"] == sha256(path)
                    and ref_record["validation_sha256"] == sha256(family_root / "validation.json")
                    and ref_record["reference_family"] == reference_family,
                    "family reference has a different source, validation, or prior")
            require(ref_record["evidence_kind"] == EVIDENCE_KIND, "absolute sequence evidence required")
            require(sha256(ref_path) == ref_record["data_sha256"], "family reference hash mismatch")
            values = load_npz(ref_path)
            require(np.array_equal(values["sequence_seed"], source["sequence_seed"])
                    and np.array_equal(values["prefixes"], source["prefixes"]), "family reference sequence/time alignment changed")
            require(values["log_marginals"].shape == (1024, 6), "family evidence has an unexpected shape")
            marginals[:, :, FAMILIES.index(reference_family)] = values["log_marginals"][:, :2]
            for item in (ref_path, ref_record_path):
                verified[str(item)] = sha256(item)
        before = np.empty_like(marginals)
        weights = np.empty_like(marginals)
        mixture_loss = np.empty((1024, 2))
        component_losses = np.empty_like(marginals)
        for i, tokens in enumerate(source["tokens"]):
            first = evidence(tokens[:1], references)
            first_token_error = max(first_token_error,
                                    float(np.max(abs(first.log_marginals + np.log(64)))))
            for j, time in enumerate(TIMES):
                marginals[i, j, 1] = shared.log_joint(tokens[:time + 1])
                current = evidence(tokens[:time], references)
                before[i, j] = current.log_marginals
                weights[i, j] = posterior_family_weights(current, PRIOR)
                endpoint = SequenceEvidence(prefix_length=time + 1, log_marginals=marginals[i, j],
                                            evidence_kind=EVIDENCE_KIND)
                computed = cumulative_mixture_losses(endpoint, PRIOR, first_token_evidence=first)
                mixture_loss[i, j] = computed["mixture_loss_bits"]
                component_losses[i, j] = computed["component_loss_bits"]
                if i < 8:
                    independent = evidence(tokens[:time + 1], references)
                    saved_endpoint_error = max(saved_endpoint_error,
                                                float(np.max(abs(independent.log_marginals - marginals[i, j]))))
        known_loss = component_losses[:, :, FAMILIES.index(family)]
        ambiguity_penalty = mixture_loss - known_loss
        require(np.max(ambiguity_penalty) <= np.log2(3) + 1e-9, "Bayesian mixture bound failed")
        require(np.allclose(weights.sum(axis=-1), 1, rtol=0, atol=1e-12), "family weights do not sum to one")
        models = {}
        arrays = dict(prefixes=np.asarray(TIMES), sequence_seed=source["sequence_seed"],
                      family_names=np.asarray(FAMILIES), log_marginals_after_prediction=marginals,
                      log_marginals_before_prediction=before, posterior_family_weights_before_prediction=weights,
                      mixture_cumulative_loss_bits=mixture_loss,
                      component_cumulative_loss_bits=component_losses,
                      mixture_minus_generating_family_bits=ambiguity_penalty)
        for seed in (0, 1):
            model = f"pooled_seed{seed}"
            model_root = raw / "eval" / model
            model_manifest = json.loads((model_root / "manifest.json").read_text())
            tf_record_path = model_root / f"{cohort}.json"
            tf_path = model_root / f"{cohort}.npz"
            tf_record = json.loads(tf_record_path.read_text())
            require(tf_record["run_id"] == model_manifest["run_id"] == fingerprint(model_manifest["identity"]),
                    "transformer run identity mismatch")
            require(tf_record["identity"] == model_manifest["identity"]
                    and tf_record["identity"]["source_plan_id"] == source_plan_id,
                    "transformer source plan mismatch")
            require(tf_record["source_data_sha256"] == sha256(path)
                    and tf_record["data_sha256"] == sha256(tf_path), "transformer source/evaluation hash mismatch")
            model_entry = next(r for r in model_manifest["cohorts"] if r["condition"]["id"] == cohort)
            require(model_entry["record_sha256"] == sha256(tf_record_path)
                    and model_entry["data_sha256"] == sha256(tf_path), "transformer manifest hash mismatch")
            scores = load_npz(tf_path)
            require(np.array_equal(scores["prefixes"], source["prefixes"])
                    and np.array_equal(scores["sequence_seed"], source["sequence_seed"]),
                    "transformer/reference sequence or time mismatch")
            tf_loss = scores["cumulative_loss_bits"][:, :2]
            gap = tf_loss - mixture_loss
            models[model] = dict(transformer_cumulative_loss_bits=stats(tf_loss),
                                transformer_minus_mixture_bits=stats(gap),
                                run_id=tf_record["run_id"], checkpoint_sha256=tf_record["identity"]["checkpoint_sha256"])
            arrays[f"{model}_cumulative_loss_bits"] = tf_loss
            arrays[f"{model}_minus_mixture_bits"] = gap
            for item in (model_root / "manifest.json", tf_record_path, tf_path):
                verified[str(item)] = sha256(item)
        raw_target = output / f"{cohort}.npz"
        np.savez_compressed(raw_target, **arrays)
        raw_file_hashes[str(raw_target)] = sha256(raw_target)
        raw_arrays[family] = arrays
        records[family] = dict(cohort=cohort, n_seq=1024,
                              mixture_cumulative_loss_bits=stats(mixture_loss),
                              generating_family_cumulative_loss_bits=stats(known_loss),
                              mixture_minus_generating_family_bits=stats(ambiguity_penalty),
                              posterior_family_weights_before_prediction=stats(weights), models=models)
    checks = sequential_validation(cohorts, references)
    require(first_token_error < 1e-9 and saved_endpoint_error < 1e-10,
            "first-token or saved-endpoint anchor failed")
    checks.update(first_token_uniform_max_abs_error_nats=first_token_error,
                  replayed_saved_endpoint_max_abs_error_nats=saved_endpoint_error)
    aggregate = {metric: balanced_stats([records[f][metric] for f in FAMILIES]) for metric in
                 ("mixture_cumulative_loss_bits", "generating_family_cumulative_loss_bits",
                  "mixture_minus_generating_family_bits", "posterior_family_weights_before_prediction")}
    aggregate["models"] = {model: {metric: balanced_stats([records[f]["models"][model][metric] for f in FAMILIES])
                                   for metric in ("transformer_cumulative_loss_bits", "transformer_minus_mixture_bits")}
                           for model in ("pooled_seed0", "pooled_seed1")}
    shared_validation_target = curated / "shared_base_evidence_validation.json"
    shared_validation_target.write_text(json.dumps(shared.validation, indent=2, allow_nan=False) + "\n")
    copied_validation = curated / "family_evidence_validation.json"
    shutil.copyfile(family_root / "validation.json", copied_validation)
    verified.update(shared.validation["verified_file_sha256"])
    code_files = [Path(__file__), ROOT / "code/transformer/mixture_bayes.py",
                  ROOT / "code/transformer/mixture_family_evidence.py", ROOT / "code/transformer/shared_base_short_evidence.py"]
    receipt = dict(schema="multifamily-short-mixture-validation-v1", passed=True,
                   source_plan_id=source_plan_id, independent_sequences=3072,
                   direct_comparison_prediction_times=list(TIMES), uncomputed_long_prediction_times=[16, 64, 256, 1024],
                   numerical_validations={str(p): sha256(p) for p in (copied_validation, shared_validation_target)},
                   checks=checks, verified_file_sha256=verified,
                   code_sha256={str(p): sha256(p) for p in code_files}, raw_output_sha256=raw_file_hashes,
                   no_new_transformer_inference=True, no_true_source_parameters_used_by_predictors=True)
    receipt_path = curated / "short_validation.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    summary = dict(schema="multifamily-short-mixture-comparison-v1", source_plan_id=source_plan_id,
                   family_order=list(FAMILIES), family_prior_weights=PRIOR.tolist(),
                   prediction_times=list(TIMES), scored_tokens="At time t, score tokens 2,...,t+1; token 1 is given.",
                   direct_mixture_evidence_available_through_tokens=5,
                   long_context_mixture_comparison="not computed",
                   uncertainty="Sequence-paired SE within each family; equal-family-average SE is sqrt(sum(SE_f^2))/3. Training seeds are reported separately.",
                   interpretation="The target is the equal-prior Bayesian mixture, not specialist transformers. Conditional-family sample gaps may be negative; the balanced-prior expectation is nonnegative. The mixture-minus-known-family penalty is bounded above by log2(3) per sequence, but need not equal that bound.",
                   cohorts=records, equal_family_average=aggregate,
                   validation_receipt=dict(path=str(receipt_path), sha256=sha256(receipt_path)))
    summary_path = curated / "short_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(out=str(summary_path), direct_times=list(TIMES),
                          gaps={model: values["transformer_minus_mixture_bits"] for model, values in aggregate["models"].items()},
                          ambiguity_penalty=aggregate["mixture_minus_generating_family_bits"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "output/multifamily_results_2026-09-16")
    parser.add_argument("--families", type=Path, default=ROOT / "output/multifamily_bayes_2026-09-16/family_evidence")
    parser.add_argument("--out", type=Path, default=ROOT / "output/multifamily_bayes_2026-09-16/short_comparison")
    parser.add_argument("--curated", type=Path, default=ROOT / "results/transformer/multifamily_bayes_2026-09-16")
    args = parser.parse_args()
    run(args.raw, args.families, args.out, args.curated)


if __name__ == "__main__":
    main()

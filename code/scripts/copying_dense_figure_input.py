"""Read only a fully validated copying extension on the original 256 histories.

This adapter recomputes plotted means and paired source errors; it does not
run Bayesian inference, approve a partial cohort, or estimate a payoff crossing.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

CONDITIONS = ("a1_L0", "a1_L2", "a1_L3")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_dense_copying(source, old, neural_source):
    source, old, neural_source = map(Path, (source, old, neural_source))
    paths = [source / name for name in
             ("summary.json", "arrays.npz", "input_bindings.json", "validation.json")]
    summary, validation = read(paths[0]), read(paths[3])
    require(validation["schema"] == "copying-dense16-release-validation-v1"
            and validation["passed"] is True
            and validation["complete_original_cohorts"] is True
            and validation["retained_early_scores_exact"] is True,
            "Dense copying release has no independent complete-cohort approval")
    for name in ("summary", "arrays", "input_bindings"):
        path = source / (name + (".npz" if name == "arrays" else ".json"))
        require(sha(path) == validation[name + "_sha256"], "Dense copying release changed")
    require(sha(old / "arrays.npz") == validation["old_arrays_sha256"],
            "Dense copying uses another early reference cohort")
    paths.append(old / "arrays.npz")
    require(summary["schema"] == "copying-dense16-recovery-v1"
            and all(summary[key] is True for key in
                    ("complete", "aggregate_precision_passed", "population_reference_approved"))
            and summary["publication_status"] == "approved"
            and not any(summary[key] for key in
                        ("missing", "failed_audits", "failed_paired_orders"))
            and summary["times"] == list(range(1, 17))
            and summary["original_histories"] == 768
            and summary["arrays_sha256"] == sha(paths[1])
            and summary["input_bindings_sha256"] == sha(paths[2]),
            "Incomplete or unapproved dense copying reference")
    require(summary["payoff_forecast_approved"] is False
            and "surrogate" in summary["comparator"].lower()
            and "not exact finite-shuffle" in summary["comparator"].lower(),
            "Dense copying comparator or approved scope changed")
    groups = summary["groups"]
    require(len(groups) == 3 and {g["condition"] for g in groups} == set(CONDITIONS),
            "Dense copying omitted or duplicated a source condition")
    result, checks = {}, {}
    with np.load(paths[1], allow_pickle=False) as arrays, np.load(old / "arrays.npz", allow_pickle=False) as prior:
        for group in groups:
            condition = group["condition"]
            depth = int(condition.split("L")[-1])
            score = {}
            for name in ("stationary_loss", "bayes_loss_a32", "ordered_seed0", "shuffled_seed0"):
                score[name] = arrays[condition + "_" + name].copy()
                require(score[name].shape == (256, 16) and np.isfinite(score[name]).all(),
                        "Dense copying must retain exactly 256 paired histories")
                prior_name = "bayes_loss" if name == "bayes_loss_a32" else name
                require(np.array_equal(score[name][:, :8], prior[condition + "_" + prior_name]),
                        "Dense copying changed retained early scores or source order")
            for role in ("ordered", "shuffled"):
                path = neural_source / (role + "_seed0") / (condition + ".npz")
                paths.append(path)
                with np.load(path, allow_pickle=False) as raw:
                    require(raw["expected_loss_bits"].shape == (256, 1024)
                            and np.array_equal(raw["times"], np.arange(1, 1025))
                            and np.array_equal(score[role + "_seed0"], raw["expected_loss_bits"][:, :16]),
                            "Dense copying neural scores are not the original paired cohort")
            gains = {"bayes": score["stationary_loss"] - score["bayes_loss_a32"],
                     "neural_seed0": score["shuffled_seed0"] - score["ordered_seed0"]}
            result[depth] = {}
            for metric in ("instantaneous", "cumulative"):
                gate = group[metric]
                require(gate["passed"] is True, "Dense copying metric gate failed")
                record = {}
                for predictor, values in gains.items():
                    values = values.cumsum(1) if metric == "cumulative" else values
                    record[predictor + "_mean"] = values.mean(0)
                    record[predictor + "_source_se"] = values.std(0, ddof=1) / 16
                mean, se = record["bayes_mean"], record["bayes_source_se"]
                saved_mean, saved_se = (np.asarray(gate[key]) for key in ("mean_bits", "source_se_bits"))
                require(saved_mean.shape == saved_se.shape == (16,), "Incomplete dense copying coordinate summary")
                error = max(float(np.max(abs(mean - saved_mean))), float(np.max(abs(se - saved_se))))
                allowance, budget = (np.asarray(gate[key]) for key in
                                     ("numerical_allowance_bits", "numerical_budget_bits"))
                require(allowance.shape == budget.shape == (16,)
                        and np.isfinite(allowance).all() and np.isfinite(budget).all()
                        and np.all(allowance >= 0) and np.all(allowance <= budget)
                        and np.allclose(budget, np.minimum(.01 / 3, .25 * se), rtol=0, atol=2e-14)
                        and error < 2e-12, "Dense copying mean, source error or numerical gate differs")
                result[depth][metric] = record
                checks[f"dense_copying_{depth}_{metric}_max_error"] = error
    return result, checks, paths

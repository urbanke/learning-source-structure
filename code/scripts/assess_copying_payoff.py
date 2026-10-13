"""Validate copying-payoff curves through eight observations.

The publication estimate is the refined stationary-frequency surrogate minus
the informed shared-base Bayes loss.  Times 1--4 use the validated exact short
reference; times 5--8 use complete-cohort alpha-32 diagnostics.  This reader
does no sampling and is independent of any saved-chain recovery mechanism.

Every sequence remains paired across all eight times.  Common source-bank
jackknife covariance is retained across time, while independently seeded
MCMC calculations at different times add in variance.  The comparator is a
stationary-frequency surrogate, not exact finite-shuffle Bayes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("a1_L0", "a1_L2", "a1_L3")
TIMES = np.arange(1, 9, dtype=np.int64)
DIAGNOSTIC_TASK_COUNTS = {5:76,6:162,7:318,8:496}
N_SEQUENCES = 256
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260921
ZERO_TOLERANCE_BITS = 1e-12
ABSOLUTE_NUMERICAL_BUDGET_BITS = .01/3

DEFAULT_SHORT = ROOT/"output/copying_frequency_calibration_2026-09-19/refined"
DEFAULT_SHORT_VALIDATION = ROOT/"results/transformer/copying_frequency_calibration_2026-09-19/validation.json"
DEFAULT_STATIONARY_REFINED = ROOT/"output/copying_frequency_extension_2026-09-19/refined"
DEFAULT_STATIONARY_COARSE = ROOT/"output/copying_frequency_extension_2026-09-19/stratified"
DEFAULT_STATIONARY_VALIDATION = ROOT/"results/transformer/copying_frequency_extension_2026-09-19/validation.json"
PAIRED_STATIONARY_POLICY_SOURCE = ROOT/"output/copying_payoff_assessment_2026-09-21/summary.json"
DEFAULT_T5 = ROOT/"output/copying_payoff_dense_diagnostics_2026-09-20/t5"
DEFAULT_T6 = ROOT/"output/copying_payoff_parallel_2026-09-20/diagnostics/t6"
DEFAULT_OUT = ROOT/"output/copying_payoff_assessment_2026-09-21"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def stats(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (N_SEQUENCES,) and np.isfinite(values).all(),
            "Every paired sequence value is required")
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(N_SEQUENCES))


class Bindings:
    """Direct file hashes plus collision checks on producer hash manifests."""
    def __init__(self):
        self.direct = {}
        self.claimed = {}

    @staticmethod
    def _path(path):
        path = Path(path)
        return str((path if path.is_absolute() else ROOT/path).resolve())

    def bind(self, path, expected=None):
        name = self._path(path)
        if name in self.direct:
            actual=self.direct[name]
            require(expected is None or actual == expected, "Input hash changed: "+name)
            return actual
        actual = sha(name)
        require(expected is None or actual == expected, "Input hash changed: "+name)
        self.direct[name] = actual
        return actual

    def claims(self, values, source):
        require(isinstance(values, dict) and values, "Empty provenance manifest: "+source)
        for path, digest in values.items():
            require(isinstance(digest, str) and len(digest) == 64,
                    "Malformed producer hash: "+str(path))
            name = self._path(path); previous = self.claimed.get(name)
            require(previous in (None, digest), "Producer provenance conflict: "+name)
            self.claimed[name] = digest
            self.bind(name,digest)

    def receipt(self):
        claims = dict(sorted(self.claimed.items()))
        return dict(direct_input_sha256=dict(sorted(self.direct.items())),
                    transitive_claim_count=len(claims),
                    transitive_claims_sha256=json_digest(claims))


def jackknife_metrics(full_loss, leave_one_out_loss):
    """Mean-level delete-one covariance and bias, retaining all time covariance."""
    full = np.asarray(full_loss, dtype=float)
    leave = np.asarray(leave_one_out_loss, dtype=float)
    require(full.shape == (N_SEQUENCES, 8) and leave.ndim == 3
            and leave.shape[1:] == full.shape and leave.shape[0] >= 2
            and np.isfinite(full).all() and np.isfinite(leave).all(),
            "Invalid source-bank arrays")
    estimates = leave.mean(axis=1)
    center = estimates.mean(axis=0); banks = len(estimates)
    covariance = (banks-1)/banks * (estimates-center).T@(estimates-center)
    covariance = (covariance+covariance.T)/2
    require(np.linalg.eigvalsh(covariance).min() >= -1e-14,
            "Invalid source-bank covariance")
    bias = (banks-1)*(center-full.mean(axis=0))
    return dict(covariance=covariance, mcse=np.sqrt(np.maximum(0, np.diag(covariance))),
                bias=np.abs(bias), signed_bias=bias, estimates=estimates)


def cumulative_jackknife_metrics(full_loss, leave_one_out_loss):
    return jackknife_metrics(np.cumsum(full_loss, axis=1),
                             np.cumsum(leave_one_out_loss, axis=2))


def numerical_accounting(source_se, mcmc32_mcse, mcmc32_bias,
                         bank32_mcse, bank32_bias,
                         alpha_shift, alpha_comparison_mcse, alpha_comparison_bias,
                         stationary_shift, stationary_comparison_mcse,
                         stationary_comparison_bias):
    """Apply the predeclared pooled-mean-style numerical allowance."""
    fields = [source_se,mcmc32_mcse,mcmc32_bias,bank32_mcse,bank32_bias,
              alpha_shift,alpha_comparison_mcse,alpha_comparison_bias,
              stationary_shift,stationary_comparison_mcse,
              stationary_comparison_bias]
    values = [np.asarray(v,dtype=float) for v in fields]
    require(all(v.shape == (8,) and np.isfinite(v).all() for v in values)
            and all(np.min(v) >= 0 for i,v in enumerate(values) if i not in (5,8)),
            "Invalid numerical-accounting curve")
    (source_se,m32,b32,bank,bbank,ashift,acmp,abias,
     sshift,scmp,sbias) = values
    final_mcse = np.hypot(m32, bank)
    final_bias = b32+bbank
    final_allowance = 2*final_mcse+2*final_bias
    alpha_allowance = np.abs(ashift)+3*acmp+abias
    stationary_allowance = np.abs(sshift)+3*scmp+sbias
    total = final_allowance+alpha_allowance+stationary_allowance
    budget = np.minimum(ABSOLUTE_NUMERICAL_BUDGET_BITS, .25*source_se)
    return dict(final_numerical_mcse=final_mcse, final_log_bias_proxy=final_bias,
                final_estimate_allowance=final_allowance,
                alpha16_32_refinement_allowance=alpha_allowance,
                stationary_coarse_fine_refinement_allowance=stationary_allowance,
                total_numerical_allowance=total, numerical_budget=budget,
                precision_passed=total <= budget)


def crossing_draws(values):
    """First positive and start of final positive run; infinity is censoring."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    require(values.ndim == 2 and values.shape[1] == 8 and np.isfinite(values).all(),
            "Crossing curves must be finite through time 8")
    positive = values > ZERO_TOLERANCE_BITS
    first = np.where(positive.any(axis=1), positive.argmax(axis=1)+1, np.inf)
    last_nonpositive = np.where(~positive, np.arange(8), -1).max(axis=1)
    final = np.where(positive[:,-1], last_nonpositive+2, np.inf)
    return first.astype(float), final.astype(float)


def serial_time(value):
    return int(value) if np.isfinite(value) else None


def crossing_percentiles(draws):
    value = np.quantile(draws, [.025,.5,.975], method="inverted_cdf")
    return dict(lower=serial_time(value[0]), median=serial_time(value[1]),
                upper=serial_time(value[2]))


def crossing_report(mean, first_draws, final_draws, eligible):
    if not eligible:
        withheld = dict(time=None, status="withheld_numerical_precision")
        return dict(eligible=False, first_positive=withheld,
                    final_positive_run=withheld,
                    bootstrap_repetitions=len(first_draws),
                    bootstrap_scope="Saved for diagnostics; crossing inference withheld.")
    point_first,point_final = crossing_draws(mean)
    def one(point, draws, final=False):
        point = point[0]
        if not np.isfinite(point):
            status = "not_positive_at_horizon" if final else "no_positive_point"
        elif point == 1:
            status = "positive_from_start"
        else:
            status = "upward_crossing_after_nonpositive_values"
        return dict(time=serial_time(point),status=status,
                    bootstrap_percentiles=crossing_percentiles(draws),
                    bootstrap_censored_count=int(np.isinf(draws).sum()),
                    bootstrap_positive_from_start_count=int((draws == 1).sum()))
    return dict(eligible=True, first_positive=one(point_first,first_draws),
                final_positive_run=one(point_final,final_draws,True),
                bootstrap_repetitions=len(first_draws),
                bootstrap_scope="Whole-sequence paired resampling; unreached draws remain right-censored.")


def bootstrap_curves(curves, repetitions, seed):
    """One resampled sequence index is shared by every supplied curve and time."""
    require(repetitions == N_BOOTSTRAP, "The production bootstrap uses exactly 2000 draws")
    require(curves and all(np.asarray(v).shape == (N_SEQUENCES,8) for v in curves.values()),
            "Bootstrap curves must share all 256 sequences and eight times")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(N_SEQUENCES, np.full(N_SEQUENCES,1/N_SEQUENCES),
                              size=repetitions).astype(float)/N_SEQUENCES
    return {name:weights@np.asarray(value,dtype=float) for name,value in curves.items()}


def load_short(folder, bindings):
    folder = Path(folder); summary_path=folder/"summary.json"; plan_path=folder/"plan.json"
    summary,plan = read(summary_path),read(plan_path)
    bindings.bind(summary_path); bindings.bind(plan_path, summary["plan_sha256"])
    exact=summary.get("exact_bayes_validation",{})
    require(summary.get("complete") is True and summary.get("stage") == "refined"
            and summary.get("times") == [1,2,3,4] and exact.get("passed") is True,
            "Exact short-reference gate failed")
    require(exact.get("verified_file_sha256"),"Exact short-reference provenance is empty")
    for name,digest in exact["verified_file_sha256"].items(): bindings.bind(name,digest)
    require(plan.get("stage") == "refined" and plan.get("times") == [1,2,3,4]
            and summary["reference"]["banks"] == 16
            and summary["reference"]["alpha_order"] == 16,
            "Exact short-reference identity changed")
    bindings.claims(summary["scoring_inputs_sha256"], str(summary_path))
    bindings.claims(plan["code_sha256"], str(plan_path))
    bank = folder/"bank.npz"; bindings.bind(bank,summary["reference"]["bank_sha256"])
    groups={g["id"]:g for g in summary["groups"]}
    require(set(groups) == set(CONDITIONS), "Short-reference conditions changed")
    result={}
    for condition in CONDITIONS:
        path=folder/(condition+".npz"); bindings.bind(path); a=load_archive(path)
        required=("times","stationary","bayes","stationary_leave_one_bank_out",
                  "ordered_seed0","shuffled_seed0","ordered_seed1","shuffled_seed1")
        require(all(key in a for key in required) and np.array_equal(a["times"],np.arange(1,5))
                and all(a[key].shape == (N_SEQUENCES,4) for key in required[1:3]+required[4:])
                and a["stationary_leave_one_bank_out"].shape == (16,N_SEQUENCES,4)
                and all(np.isfinite(a[key]).all() for key in required[1:]),
                "Invalid exact short arrays: "+condition)
        require(groups[condition]["n_sequences"] == N_SEQUENCES
                and groups[condition]["maximum_predictive_normalization_error"] <= 1e-8,
                "Short-reference summary changed")
        result[condition]=a
    return result


def validate_short_receipt(path, bindings):
    path=Path(path);receipt=read(path);bindings.bind(path)
    require(receipt.get("passed") is True and receipt.get("scientific_timing_validated") is False
            and receipt.get("raw_arrays_committed") is False,
            "Short-reference independent validation failed or changed scope")
    for name,digest in receipt["input_sha256"].items(): bindings.bind(ROOT/name,digest)
    publisher=ROOT/"code/scripts/publish_copying_frequency_calibration.py"
    bindings.bind(publisher,receipt["script_sha256"])
    bindings.bind(path.parent/"summary.json",receipt["summary_sha256"])


def load_stationary(folder, stage, expected_banks, bindings):
    folder=Path(folder); summary_path=folder/"summary.json";plan_path=folder/"plan.json"
    summary,plan=read(summary_path),read(plan_path)
    bindings.bind(summary_path);bindings.bind(plan_path,summary["plan_sha256"])
    require(summary.get("complete") is True and summary.get("times") == list(range(1,9))
            and summary.get("finite_shuffle_exact") is False and plan.get("stage") == stage
            and plan.get("times") == list(range(1,9)), "Stationary extension gate failed")
    bindings.claims(plan["code_sha256"],str(plan_path));bindings.claims(plan["inputs_sha256"],str(plan_path))
    bindings.claims(plan["original_scoring_inputs_sha256"],str(plan_path))
    result={}
    for condition in CONDITIONS:
        path=folder/(condition+".npz");bindings.bind(path);a=load_archive(path)
        require(a.get("stationary_loss",np.empty(0)).shape == (N_SEQUENCES,8)
                and a.get("stationary_leave_one_bank_out_loss",np.empty(0)).shape == (expected_banks,N_SEQUENCES,8)
                and np.isfinite(a["stationary_loss"]).all()
                and np.isfinite(a["stationary_leave_one_bank_out_loss"]).all(),
                "Invalid stationary arrays: "+condition)
        result[condition]=a
    return result


def validate_stationary_receipt(path, bindings):
    path=Path(path);receipt=read(path);bindings.bind(path)
    require(receipt.get("passed") is True and receipt.get("scientific_timing_validated") is False
            and receipt.get("independent_confirmation") is False,
            "Stationary independent validation failed or changed scope")
    for name,digest in receipt["input_sha256"].items(): bindings.bind(ROOT/name,digest)
    publisher=ROOT/"code/scripts/publish_copying_frequency_extension.py"
    bindings.bind(publisher,receipt["script_sha256"])
    published=path.parent/"summary.json";bindings.bind(published,receipt["summary_sha256"])


def load_stationary_paired(folder, bindings):
    """Load the explicitly paired 16/32-node stationary refinement."""
    folder=Path(folder);validation_path=folder/"validation.json"
    validation=read(validation_path);bindings.bind(validation_path)
    require(validation.get("schema") == "copying-frequency-paired-refinement-validation-v1"
            and validation.get("passed") is True
            and validation.get("input_integrity_passed") is True
            and validation.get("stationary_component_precision_passed") is True
            and validation.get("separate_bayesian_alpha_refinement_included") is False
            and validation.get("payoff_forecast_approved") is False,
            "Paired stationary refinement validation failed or changed scope")
    validator=ROOT/"code/scripts/validate_copying_frequency_paired_refinement.py"
    bindings.bind(validator,validation["validator_sha256"])
    bindings.bind(PAIRED_STATIONARY_POLICY_SOURCE,validation["assessment_policy_sha256"])
    for name,digest in validation["artifact_sha256"].items():
        bindings.bind(folder/name,digest)
    plan_path=folder/"plan.json";summary_path=folder/"summary.json"
    plan,summary=read(plan_path),read(summary_path)
    identity={key:value for key,value in plan.items() if key != "plan_id"}
    require(plan.get("schema") == "copying-frequency-paired-refinement-plan-v1"
            and plan.get("plan_id") == json_digest(identity)
            and plan.get("banks") == 16 and plan.get("coarse_order") == 16
            and plan.get("fine_order") == 32 and plan.get("times") == list(range(1,9))
            and plan.get("resources",{}).get("max_workers") <= 4
            and plan.get("resources",{}).get("maximum_attempts_per_bank") == 1,
            "Paired stationary plan changed")
    require(summary.get("schema") == "copying-frequency-paired-refinement-v1"
            and summary.get("complete") is True and summary.get("plan_id") == plan["plan_id"]
            and summary.get("banks") == 16 and summary.get("times") == list(range(1,9)),
            "Paired stationary summary changed")
    bindings.claims(plan["inputs_sha256"],str(plan_path))
    bindings.claims(plan["code_sha256"],str(plan_path))
    result={}
    for condition in CONDITIONS:
        path=folder/(condition+".npz");a=load_archive(path)
        fine=a.get("fine_stationary_loss",np.empty(0))
        fine_leave=a.get("fine_stationary_leave_one_bank_out_loss",np.empty(0))
        coarse=a.get("coarse_stationary_loss",np.empty(0))
        coarse_leave=a.get("coarse_stationary_leave_one_bank_out_loss",np.empty(0))
        paired_leave=a.get("paired_difference_leave_one_bank_out_loss",np.empty(0))
        require(fine.shape == coarse.shape == (N_SEQUENCES,8)
                and fine_leave.shape == coarse_leave.shape == paired_leave.shape == (16,N_SEQUENCES,8)
                and all(np.isfinite(value).all() for value in
                        (fine,fine_leave,coarse,coarse_leave,paired_leave))
                and np.array_equal(paired_leave,fine_leave-coarse_leave),
                "Invalid paired stationary arrays: "+condition)
        result[condition]=dict(stationary_loss=fine,
            stationary_leave_one_bank_out_loss=fine_leave,
            stationary_coarse_loss=coarse,
            stationary_coarse_leave_one_bank_out_loss=coarse_leave,
            stationary_comparison_paired=True)
    return result


def _replay_stats(value, saved, label):
    mean,se=stats(value)
    require(abs(mean-saved["candidate_mean_bits"]) <= 2e-12
            and abs(se-saved["observed_sequence_se_bits"]) <= 2e-12,
            "Diagnostic statistic changed: "+label)


def diagnostic_seeds(producer, observed, bindings):
    """Read the bound task identities and prove independence across times."""
    tasks={}
    marker=f"/tasks/t{observed:04d}_"
    for path,digest in producer["input_sha256"].items():
        normalized=str(Path(path)).replace("\\","/")
        if marker not in normalized or not normalized.endswith("/identity.json"):
            continue
        bindings.bind(path,digest);identity=read(path);task=identity.get("task",{})
        require(task.get("time") == observed and task.get("order") in (16,32)
                and isinstance(task.get("seed"),int) and task.get("id") in normalized,
                "Bound MCMC task identity changed")
        value=(task["seed"],task["order"]);previous=tasks.get(task["id"])
        require(previous in (None,value),"Recovered and parent task identities differ")
        tasks[task["id"]]=value
    require(len(tasks) == DIAGNOSTIC_TASK_COUNTS[observed]
            and len({value[0] for value in tasks.values()}) == len(tasks),
            "Incomplete or repeated MCMC seed family at t="+str(observed))
    return {value[0] for value in tasks.values()}


def load_diagnostic(folder, observed, bindings):
    folder=Path(folder); arrays_path=folder/"arrays.npz"
    summary_path=folder/"summary.json"
    if summary_path.is_file():
        summary=read(summary_path); producer=summary
        finite_shuffle_exact=summary.get("finite_shuffle_exact")
    else:
        # The immutable recovery merger writes one common receipt and separate
        # t7/t8 arrays.  Normalize its per-wave record to the original
        # diagnostic interface without importing recovery code.
        summary_path=folder.parent/"summary.json";producer=read(summary_path)
        require(producer.get("schema") == "copying-payoff-recovery-merge-v1"
                and producer.get("complete_cohorts_passed") is True
                and producer.get("diagnostic_arrays_validated") is True
                and producer.get("individual_audits_passed") is True
                and producer.get("paired_orders_passed") is True,
                "Recovery diagnostic family did not pass")
        summary=producer.get("waves",{}).get(str(observed),{})
        require(summary.get("comparator") ==
                "Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.",
                "Recovery comparator scope changed")
        finite_shuffle_exact=False
    bindings.bind(summary_path);bindings.bind(arrays_path,summary.get("arrays_sha256"))
    gates=(summary.get("complete"),summary.get("individual_audits_passed"),
           summary.get("paired_orders_passed"),summary.get("diagnostic_arrays_validated"))
    require(all(value is True for value in gates) and summary.get("time") == observed
            and summary.get("original_histories") == 768
            and finite_shuffle_exact is False,
            "Incomplete or failed complete-cohort diagnostics at t="+str(observed))
    bindings.claims(producer["input_sha256"],str(summary_path))
    seeds=diagnostic_seeds(producer,observed,bindings)
    arrays=load_archive(arrays_path); groups={g["condition"]:g for g in summary["diagnostic_groups"]}
    require(set(groups) == set(CONDITIONS), "Diagnostic conditions changed")
    for condition in CONDITIONS:
        group=groups[condition];require(group["n_sequences"] == N_SEQUENCES,"Incomplete condition")
        names=[condition+"_stationary_loss",condition+"_bayes_loss_a16",condition+"_bayes_loss_a32",
               condition+"_ordered_seed0",condition+"_shuffled_seed0",
               condition+"_ordered_seed1",condition+"_shuffled_seed1"]
        require(all(arrays.get(name,np.empty(0)).shape == (N_SEQUENCES,) for name in names)
                and all(np.isfinite(arrays[name]).all() for name in names),
                "Incomplete diagnostic score arrays: "+condition)
        for order in (16,32):
            probs=arrays[condition+f"_original_label_probs_a{order}"]
            require(probs.shape == (N_SEQUENCES,64) and np.isfinite(probs).all()
                    and probs.min()>0 and np.max(abs(probs.sum(1)-1)) <= 1e-7,
                    "Invalid original-label probabilities")
            loss=arrays[condition+f"_bayes_loss_a{order}"]
            _replay_stats(loss,group["rules"][str(order)]["bayes_loss"],condition)
            _replay_stats(arrays[condition+"_stationary_loss"]-loss,
                          group["rules"][str(order)]["stationary_minus_bayes"],condition)
        _replay_stats(arrays[condition+"_bayes_loss_a16"]-arrays[condition+"_bayes_loss_a32"],
                      group["order_loss_difference"],condition)
        for seed in (0,1):
            gain=arrays[condition+f"_shuffled_seed{seed}"]-arrays[condition+f"_ordered_seed{seed}"]
            _replay_stats(gain,group["neural"][str(seed)]["gain"],condition)
    return dict(summary=summary,groups=groups,arrays=arrays,seeds=seeds)


def combine_inputs(short, stationary, diagnostics, stationary_paired=None):
    combined={}
    for condition in CONDITIONS:
        exact=short[condition]
        if stationary_paired is None:
            refined=stationary["refined"][condition];coarse=stationary["coarse"][condition]
            require(np.allclose(exact["stationary"],refined["stationary_loss"][:,:4],rtol=0,atol=5e-12)
                    and np.allclose(exact["stationary_leave_one_bank_out"],
                                    refined["stationary_leave_one_bank_out_loss"][:,:,:4],rtol=0,atol=5e-12),
                    "Early stationary arrays do not replay: "+condition)
            paired=False
        else:
            paired_data=stationary_paired[condition]
            refined=dict(stationary_loss=paired_data["stationary_loss"],
                stationary_leave_one_bank_out_loss=paired_data["stationary_leave_one_bank_out_loss"])
            coarse=dict(stationary_loss=paired_data["stationary_coarse_loss"],
                stationary_leave_one_bank_out_loss=paired_data["stationary_coarse_leave_one_bank_out_loss"])
            paired=True
        bayes32=np.empty((N_SEQUENCES,8));bayes16=np.empty_like(bayes32)
        bayes32[:,:4]=exact["bayes"];bayes16[:,:4]=exact["bayes"]
        neural={f"{role}_seed{seed}":np.empty((N_SEQUENCES,8))
                for role in ("ordered","shuffled") for seed in (0,1)}
        for key,value in neural.items(): value[:,:4]=exact[key]
        m16=np.zeros(8);m32=np.zeros(8);b16=np.zeros(8);b32=np.zeros(8)
        for t in range(5,9):
            diagnostic=diagnostics[t];a=diagnostic["arrays"];g=diagnostic["groups"][condition]
            bayes16[:,t-1]=a[condition+"_bayes_loss_a16"]
            bayes32[:,t-1]=a[condition+"_bayes_loss_a32"]
            if not paired:
                require(np.allclose(a[condition+"_stationary_loss"],refined["stationary_loss"][:,t-1],rtol=0,atol=2e-12),
                        "Stationary diagnostic pairing changed")
            for key,value in neural.items(): value[:,t-1]=a[condition+"_"+key]
            for order,target_m,target_b in ((16,m16,b16),(32,m32,b32)):
                rule=g["rules"][str(order)]
                target_m[t-1]=rule["numerical_mcse_bits"]
                target_b[t-1]=rule["log_bias_proxy_bits"]
                require(target_m[t-1] >= 0 and target_b[t-1] >= 0,"Invalid MCMC uncertainty")
        combined[condition]=dict(stationary=refined["stationary_loss"],
            stationary_leave=refined["stationary_leave_one_bank_out_loss"],
            stationary_coarse=coarse["stationary_loss"],
            stationary_coarse_leave=coarse["stationary_leave_one_bank_out_loss"],
            stationary_comparison_paired=paired,
            bayes32=bayes32,bayes16=bayes16,mcmc16_mcse=m16,mcmc32_mcse=m32,
            mcmc16_bias=b16,mcmc32_bias=b32,**neural)
    return combined


def _curve_rows(mean,se,accounting,extra):
    rows=[]
    for i,time in enumerate(TIMES):
        row=dict(time=int(time),mean_bits=float(mean[i]),paired_source_se_bits=float(se[i]))
        row.update({key+"_bits" if key != "precision_passed" else key:
                    (bool(value[i]) if key == "precision_passed" else float(value[i]))
                    for key,value in accounting.items()})
        row.update({key+"_bits":float(value[i]) for key,value in extra.items()})
        rows.append(row)
    return rows


def stationary_refinement_metrics(data):
    fine,coarse=data["stationary"],data["stationary_coarse"]
    fine_leave,coarse_leave=data["stationary_leave"],data["stationary_coarse_leave"]
    shift=(fine-coarse).mean(0)
    shift_cumulative=np.cumsum(fine-coarse,axis=1).mean(0)
    if data.get("stationary_comparison_paired",False):
        comparison=jackknife_metrics(fine-coarse,fine_leave-coarse_leave)
        comparison_cumulative=cumulative_jackknife_metrics(
            fine-coarse,fine_leave-coarse_leave)
        mode="paired_delete_bank"
    else:
        fine_metrics=jackknife_metrics(fine,fine_leave)
        coarse_metrics=jackknife_metrics(coarse,coarse_leave)
        fine_cumulative=cumulative_jackknife_metrics(fine,fine_leave)
        coarse_cumulative=cumulative_jackknife_metrics(coarse,coarse_leave)
        comparison=dict(mcse=np.hypot(fine_metrics["mcse"],coarse_metrics["mcse"]),
            bias=fine_metrics["bias"]+coarse_metrics["bias"],covariance=np.zeros((8,8)))
        comparison_cumulative=dict(mcse=np.hypot(fine_cumulative["mcse"],coarse_cumulative["mcse"]),
            bias=fine_cumulative["bias"]+coarse_cumulative["bias"],covariance=np.zeros((8,8)))
        mode="independent_banks"
    return (dict(shift=shift,mcse=comparison["mcse"],bias=comparison["bias"],
                 covariance=comparison["covariance"],mode=mode),
            dict(shift=shift_cumulative,mcse=comparison_cumulative["mcse"],
                 bias=comparison_cumulative["bias"],
                 covariance=comparison_cumulative["covariance"],mode=mode))


def assess_condition(condition, data, bootstrap_seed):
    gain=data["stationary"]-data["bayes32"]
    gain_cumulative=np.cumsum(gain,axis=1)
    mean=gain.mean(0);source_se=gain.std(0,ddof=1)/np.sqrt(N_SEQUENCES)
    cmean=gain_cumulative.mean(0);csource_se=gain_cumulative.std(0,ddof=1)/np.sqrt(N_SEQUENCES)

    bank=jackknife_metrics(data["stationary"],data["stationary_leave"])
    bank_cumulative=cumulative_jackknife_metrics(data["stationary"],data["stationary_leave"])
    coarse=jackknife_metrics(data["stationary_coarse"],data["stationary_coarse_leave"])
    coarse_cumulative=cumulative_jackknife_metrics(data["stationary_coarse"],data["stationary_coarse_leave"])
    stationary_refinement,stationary_refinement_cumulative=stationary_refinement_metrics(data)

    alpha_shift=(data["bayes16"]-data["bayes32"]).mean(0)
    alpha_shift_cumulative=np.cumsum(data["bayes16"]-data["bayes32"],axis=1).mean(0)
    alpha_mcse=np.hypot(data["mcmc16_mcse"],data["mcmc32_mcse"])
    alpha_mcse_cumulative=np.sqrt(np.cumsum(data["mcmc16_mcse"]**2+data["mcmc32_mcse"]**2))
    alpha_bias=data["mcmc16_bias"]+data["mcmc32_bias"]
    alpha_bias_cumulative=np.cumsum(alpha_bias)

    stationary_shift=stationary_refinement["shift"]
    stationary_shift_cumulative=stationary_refinement_cumulative["shift"]
    stationary_mcse=stationary_refinement["mcse"]
    stationary_mcse_cumulative=stationary_refinement_cumulative["mcse"]
    stationary_bias=stationary_refinement["bias"]
    stationary_bias_cumulative=stationary_refinement_cumulative["bias"]

    mcmc32_cumulative=np.sqrt(np.cumsum(data["mcmc32_mcse"]**2))
    mcmc32_bias_cumulative=np.cumsum(data["mcmc32_bias"])
    inst=numerical_accounting(source_se,data["mcmc32_mcse"],data["mcmc32_bias"],
        bank["mcse"],bank["bias"],alpha_shift,alpha_mcse,alpha_bias,
        stationary_shift,stationary_mcse,stationary_bias)
    cumulative=numerical_accounting(csource_se,mcmc32_cumulative,mcmc32_bias_cumulative,
        bank_cumulative["mcse"],bank_cumulative["bias"],alpha_shift_cumulative,
        alpha_mcse_cumulative,alpha_bias_cumulative,stationary_shift_cumulative,
        stationary_mcse_cumulative,stationary_bias_cumulative)

    neural={f"neural_seed{seed}":data[f"shuffled_seed{seed}"]-data[f"ordered_seed{seed}"]
            for seed in (0,1)}
    boots=bootstrap_curves(dict(bayes=gain,**neural),N_BOOTSTRAP,bootstrap_seed)
    bootstrap_arrays={};crossings={}
    for name,values in dict(bayes=gain,**neural).items():
        boot_inst=boots[name];boot_cum=np.cumsum(boot_inst,axis=1)
        first_i,final_i=crossing_draws(boot_inst);first_c,final_c=crossing_draws(boot_cum)
        for suffix,value in (("instantaneous_mean",boot_inst),("cumulative_mean",boot_cum),
                             ("instantaneous_first_positive",first_i),("instantaneous_final_positive_run",final_i),
                             ("cumulative_first_positive",first_c),("cumulative_final_positive_run",final_c)):
            bootstrap_arrays[name+"_"+suffix]=value
        eligible = bool(np.all(inst["precision_passed"]) and np.all(cumulative["precision_passed"])) if name=="bayes" else True
        value_mean=mean if name=="bayes" else values.mean(0)
        crossings[name]=dict(
            instantaneous=crossing_report(value_mean,first_i,final_i,eligible),
            cumulative=crossing_report(np.cumsum(values,axis=1).mean(0),first_c,final_c,eligible))

    arrays=dict(stationary_loss=data["stationary"],stationary_coarse_loss=data["stationary_coarse"],
        bayes_loss=data["bayes32"],bayes_loss_a16=data["bayes16"],bayes_gain=gain,
        stationary_leave_one_bank_out_loss=data["stationary_leave"],
        stationary_coarse_leave_one_bank_out_loss=data["stationary_coarse_leave"],
        **{key:data[key] for key in ("ordered_seed0","shuffled_seed0","ordered_seed1","shuffled_seed1")})
    curves=dict(bayes_instantaneous_mean=mean,bayes_instantaneous_source_se=source_se,
        bayes_cumulative_mean=cmean,bayes_cumulative_source_se=csource_se,
        refined_bank_covariance=bank["covariance"],coarse_bank_covariance=coarse["covariance"],
        stationary_comparison_covariance=stationary_refinement["covariance"],
        stationary_comparison_cumulative_covariance=stationary_refinement_cumulative["covariance"])
    for kind,account in (("instantaneous",inst),("cumulative",cumulative)):
        for key,value in account.items(): curves[f"bayes_{kind}_{key}"]=value
    neural_summary={}
    for seed in (0,1):
        value=neural[f"neural_seed{seed}"];cvalue=np.cumsum(value,axis=1)
        curves[f"neural_seed{seed}_instantaneous_mean"]=value.mean(0)
        curves[f"neural_seed{seed}_instantaneous_source_se"]=value.std(0,ddof=1)/16
        curves[f"neural_seed{seed}_cumulative_mean"]=cvalue.mean(0)
        curves[f"neural_seed{seed}_cumulative_source_se"]=cvalue.std(0,ddof=1)/16
        neural_summary[str(seed)]=dict(crossings=crossings[f"neural_seed{seed}"],
            scope="One trained model; intervals resample test sequences and exclude training uncertainty.")
    extra_inst=dict(mcmc_mcse=data["mcmc32_mcse"],source_bank_mcse=bank["mcse"],
        mcmc_log_bias_proxy=data["mcmc32_bias"],source_bank_bias_proxy=bank["bias"],
        alpha16_32_mean_shift=alpha_shift,stationary_coarse_fine_mean_shift=stationary_shift)
    extra_cum=dict(mcmc_mcse=mcmc32_cumulative,source_bank_mcse=bank_cumulative["mcse"],
        mcmc_log_bias_proxy=mcmc32_bias_cumulative,source_bank_bias_proxy=bank_cumulative["bias"],
        alpha16_32_mean_shift=alpha_shift_cumulative,
        stationary_coarse_fine_mean_shift=stationary_shift_cumulative)
    eligible=bool(np.all(inst["precision_passed"]) and np.all(cumulative["precision_passed"]))
    summary=dict(condition=condition,n_sequences=N_SEQUENCES,
        stationary_refinement_mode=stationary_refinement["mode"],
        aggregate_precision_passed=eligible,
        instantaneous=_curve_rows(mean,source_se,inst,extra_inst),
        cumulative=_curve_rows(cmean,csource_se,cumulative,extra_cum),
        crossings=dict(bayes=crossings["bayes"],neural=neural_summary))
    return summary,arrays,curves,bootstrap_arrays


def assess(short_path,short_validation_path,stationary_refined_path,stationary_coarse_path,
           stationary_validation_path,diagnostic_paths,stationary_paired_path=None):
    bindings=Bindings();validate_short_receipt(short_validation_path,bindings)
    short=load_short(short_path,bindings)
    stationary_paired=None
    if stationary_paired_path is None:
        validate_stationary_receipt(stationary_validation_path,bindings)
        stationary=dict(refined=load_stationary(stationary_refined_path,"refined",16,bindings),
                        coarse=load_stationary(stationary_coarse_path,"stratified",8,bindings))
    else:
        stationary={};stationary_paired=load_stationary_paired(stationary_paired_path,bindings)
    diagnostics={time:load_diagnostic(diagnostic_paths[time],time,bindings) for time in range(5,9)}
    seen=set()
    for time in range(5,9):
        require(not (seen & diagnostics[time]["seeds"]),
                "MCMC seeds overlap across observation times")
        seen.update(diagnostics[time]["seeds"])
    seed_receipts={str(time):json_digest(sorted(diagnostics[time]["seeds"]))
                   for time in range(5,9)}
    combined=combine_inputs(short,stationary,diagnostics,stationary_paired)
    summaries=[];arrays={"times":TIMES};curves={"times":TIMES};bootstrap={}
    for index,condition in enumerate(CONDITIONS):
        summary,a,c,b=assess_condition(condition,combined[condition],
            np.random.SeedSequence([BOOTSTRAP_SEED,index]))
        summaries.append(summary)
        arrays.update({condition+"_"+key:value for key,value in a.items()})
        curves.update({condition+"_"+key:value for key,value in c.items()})
        bootstrap.update({condition+"_"+key:value for key,value in b.items()})
    approved=all(group["aggregate_precision_passed"] for group in summaries)
    report=dict(schema="copying-payoff-assessment-v1",complete=True,input_validation_passed=True,
        times=TIMES.tolist(),horizon=8,n_sequences=768,n_per_condition=N_SEQUENCES,
        conditions=list(CONDITIONS),publication_bayes_rule="alpha32 for t5--8; validated exact short evidence for t1--4",
        comparator="Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.",
        estimate="stationary-frequency surrogate loss minus informed shared-base Bayes loss",
        numerical_policy=dict(
            budget="min(0.01/3 bits, one quarter of the paired source SE)",
            final_estimate_allowance="2*sqrt(MCMC_MCSE^2 + common-bank_MCSE^2) + 2*(MCMC_log_bias + bank_jackknife_bias)",
            alpha_refinement_allowance="absolute alpha16/32 mean shift + 3*comparison_MCSE + both log-bias proxies",
            stationary_refinement_allowance=(
                "absolute paired coarse/fine mean shift + 3*paired delete-bank comparison_MCSE + paired delete-bank bias proxy"
                if stationary_paired is not None else
                "absolute coarse/fine mean shift + 3*comparison_MCSE + both bank-bias proxies"),
            cumulative_covariance="Common source-bank delete-one estimates are summed before jackknifing; independently seeded per-time MCMC variances add.",
            mcmc_seed_sets_sha256=seed_receipts,
            every_crossing_coordinate_must_pass=True),
        crossings=dict(first_positive="first t in 1..8 with mean gain > 1e-12 bits",
            final_positive_run="first t whose mean gain remains > 1e-12 through t=8",
            horizon_warning="A final positive run through t=8 does not establish permanence after t=8.",
            unreached="Retained as right-censored infinity in bootstrap_draws.npz."),
        bootstrap=dict(repetitions=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,
            condition_streams="SeedSequence([seed, condition_index])",
            method="whole-sequence paired multinomial bootstrap; all times and predictors share each resample",
            numerical_uncertainty_resampled=False),
        stationary_reference=("paired 16/32-node refinement" if stationary_paired is not None
                              else "legacy independent coarse/fine refinement"),
        reused_development_histories=True,independent_confirmation=False,
        finite_shuffle_exact=False,groups=summaries,
        aggregate_precision_passed=approved,payoff_forecast_approved=approved,
        publication_status="approved" if approved else "diagnostic_nonpublication",
        scope="Assessment ends at t=8 and cannot support a claim beyond that horizon.")
    return report,arrays,curves,bootstrap,bindings.receipt()


def check_output(out):
    out=Path(out);validation=read(out/"validation.json")
    require(validation.get("integrity_passed") is True,
            "Saved assessment integrity receipt did not pass")
    require(validation.get("input_validation_passed") is True,
            "Saved assessment input validation did not pass")
    for name,digest in validation["artifact_sha256"].items():
        require(sha(out/name) == digest,"Saved assessment artifact changed: "+name)
    summary=read(out/"summary.json");require(summary.get("complete") is True,"Incomplete saved assessment")
    approved=summary.get("payoff_forecast_approved") is True
    expected_status="approved" if approved else "diagnostic_nonpublication"
    require(validation.get("passed") is approved
            and validation.get("scientific_precision_passed") is
                (summary.get("aggregate_precision_passed") is True)
            and validation.get("payoff_forecast_approved") is approved
            and validation.get("publication_status") == expected_status
            and summary.get("publication_status") == expected_status,
            "Scientific or publication flags disagree between summary and validation")
    arrays=load_archive(out/"arrays.npz");curves=load_archive(out/"curves.npz")
    for condition in CONDITIONS:
        gain=arrays[condition+"_stationary_loss"]-arrays[condition+"_bayes_loss"]
        require(np.array_equal(gain,arrays[condition+"_bayes_gain"]),"Saved gain identity changed")
        require(np.allclose(gain.mean(0),curves[condition+"_bayes_instantaneous_mean"],rtol=0,atol=1e-14)
                and np.allclose(np.cumsum(gain,axis=1).mean(0),curves[condition+"_bayes_cumulative_mean"],rtol=0,atol=1e-14),
                "Saved curves do not replay")
    scientific=approved
    print(json.dumps(dict(status="scientific_passed" if scientific else
                          "integrity_passed_scientific_precision_failed",
                          integrity_passed=True,artifacts=len(validation["artifact_sha256"]),
                          payoff_forecast_approved=scientific)))
    return 0 if scientific else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--short",type=Path,default=DEFAULT_SHORT)
    p.add_argument("--short-validation",type=Path,default=DEFAULT_SHORT_VALIDATION)
    p.add_argument("--stationary-refined",type=Path,default=DEFAULT_STATIONARY_REFINED)
    p.add_argument("--stationary-coarse",type=Path,default=DEFAULT_STATIONARY_COARSE)
    p.add_argument("--stationary-validation",type=Path,default=DEFAULT_STATIONARY_VALIDATION)
    p.add_argument("--stationary-paired",type=Path,default=None,
        help="Validated paired stationary refinement; preserves legacy defaults when omitted")
    p.add_argument("--t5",type=Path,default=DEFAULT_T5);p.add_argument("--t6",type=Path,default=DEFAULT_T6)
    p.add_argument("--t7",type=Path,required=False);p.add_argument("--t8",type=Path,required=False)
    p.add_argument("--out",type=Path,default=DEFAULT_OUT);p.add_argument("--check-only",action="store_true")
    a=p.parse_args()
    if a.check_only:
        return check_output(a.out)
    require(a.t7 is not None and a.t8 is not None,
            "Validated overlay diagnostics must be supplied explicitly with --t7 and --t8")
    require(not a.out.exists(),"Use a new output directory; preserve every prior assessment")
    report,arrays,curves,bootstrap,receipt=assess(a.short,a.short_validation,a.stationary_refined,
        a.stationary_coarse,a.stationary_validation,{5:a.t5,6:a.t6,7:a.t7,8:a.t8},
        a.stationary_paired)
    # No output directory is created until every producer and array check passes.
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/"arrays.npz",**arrays)
    np.savez_compressed(a.out/"curves.npz",**curves)
    np.savez_compressed(a.out/"bootstrap_draws.npz",**bootstrap)
    write(a.out/"summary.json",report)
    artifacts={name:sha(a.out/name) for name in
               ("arrays.npz","curves.npz","bootstrap_draws.npz","summary.json")}
    validation=dict(schema="copying-payoff-assessment-validation-v1",
        integrity_passed=True,passed=report["payoff_forecast_approved"],
        input_validation_passed=True,scientific_precision_passed=report["aggregate_precision_passed"],
        payoff_forecast_approved=report["payoff_forecast_approved"],
        publication_status=report["publication_status"],
        artifact_sha256=artifacts,assessor_sha256=sha(__file__),**receipt)
    write(a.out/"validation.json",validation)
    print(json.dumps(dict(complete=True,input_validation_passed=True,
        aggregate_precision_passed=report["aggregate_precision_passed"],
        payoff_forecast_approved=report["payoff_forecast_approved"])))
    return 0 if report["payoff_forecast_approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

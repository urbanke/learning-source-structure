"""Independent raw-data audit of complete shared-base conditional cohorts.

Audit one compressed reference at a time. Missing, running, corrupt or
nonmixing cases remain explicit; no average over the available subset is
released. Full-cohort transformer summaries are separate from validated
transformer--Bayes comparisons.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/"code/transformer"
sys.path.insert(0, str(HERE))
from shared_base_conditional_cohort import inputs, load_source
from shared_base_precision_cohort import case_precision, validate_cohort
from shared_base_full_context_summary import Audit
from shared_base_surrogate_summary import raw_checkpoint, load_models
from shared_base_chain_runner import _fingerprint
from shared_base_curve_runner import descriptor
from shared_base_surrogate import SharedBaseSurrogateMCMC
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import PrefixPosterior
from full_context_statistics import compare_prediction, diagnostic_gates
from eval_shared_base_reference import selected_grid_checks
from eval_shared_base_models import atomic_json, fingerprint, sha_file
from validate_shared_base_mcmc import json_safe
from source import seed_of

SCHEMA = "shared-base-curve-audit-v1"
REFERENCE_SCHEMA = "shared-base-curve-reference-v1"


def require(condition, description):
    if not condition:
        raise ValueError(description)


def same(actual, saved, *, atol=2e-13, rtol=1e-10):
    """Recursive numerical comparison, including all saved keys and flags."""
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()
    if isinstance(actual, dict):
        return (isinstance(saved, dict) and set(actual) == set(saved)
                and all(same(v, saved[k], atol=atol, rtol=rtol) for k,v in actual.items()))
    if isinstance(actual, (list, tuple)):
        return (isinstance(saved, (list, tuple)) and len(actual) == len(saved)
                and all(same(a,b,atol=atol,rtol=rtol) for a,b in zip(actual,saved)))
    if actual is None or isinstance(actual, (bool, str)):
        return type(actual) is type(saved) and actual == saved
    return (isinstance(saved, (int,float)) and not isinstance(saved,bool)
            and math.isclose(actual,saved,rel_tol=rtol,abs_tol=atol))


@contextmanager
def read_lock(path):
    # The worker uses this same advisory lock. Do not read a checkpoint while
    # its report and moments are being replaced by a resumed sampling round.
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@lru_cache(maxsize=8)
def sampler_for(order, proposal_json):
    proposal = json.loads(proposal_json)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    kernel = RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384)
    classes = {
        "shared_base_surrogate.SharedBaseSurrogateMCMC": SharedBaseSurrogateMCMC,
        "shared_base_efficiency.SharedBaseEfficiencyMCMC": SharedBaseEfficiencyMCMC,
        "shared_base_efficiency.SharedBaseEfficiencyRowMCMC": SharedBaseEfficiencyRowMCMC,
    }
    require(proposal["sampler_class"] in classes,"unrecognized proposal class")
    spec = proposal["proposal_spec"]
    options = dict(surrogate_steps=spec["surrogate_steps"])
    if "auxiliary_power" in spec:
        options["auxiliary_power"] = spec["auxiliary_power"]
    if "concentration" in spec:
        options["concentration"] = spec["concentration"]
    sampler = classes[proposal["sampler_class"]](64,kernel,weights/2,**options)
    require(descriptor(sampler) == proposal,"proposal implementation or density specification changed")
    require(sampler.depths.tolist() == list(range(9)),"wrong depth support")
    require(np.allclose(sampler.depth_prior,np.ones(9)/9,rtol=0,atol=1e-15),"wrong depth prior")
    return sampler


def audit_reference(source_folder, reference_root, index, time, order):
    """Return one independently checked record; do not accumulate raw arrays."""
    folder = Path(reference_root)/Path(source_folder).name/f"alpha{order}"
    stem = f"seq_{index:06d}_t{time:04d}"
    path = folder/(stem+".json")
    if not path.exists():
        lock_path=folder/(stem+".lock")
        if lock_path.exists():
            try:
                with read_lock(lock_path):
                    pass
            except BlockingIOError:
                return dict(index=index,time=time,order=order,status="running",ready=False,
                            todo="Wait for the first complete sampling report.")
        return dict(index=index,time=time,order=order,status="missing",ready=False,
                    todo="Compute this declared reference.")
    try:
        with read_lock(folder/(stem+".lock")):
            return _audit_locked(source_folder,path,index,time,order)
    except BlockingIOError:
        return dict(index=index,time=time,order=order,status="running",ready=False,
                    todo="Wait for the current sampling round and audit its complete checkpoint.")
    except (ValueError,KeyError,TypeError,AssertionError,OSError) as error:
        return dict(index=index,time=time,order=order,status="invalid",ready=False,
                    report=str(path.resolve()),error=str(error),
                    todo="Repair the reported identity or raw-data failure before aggregation.")


def _audit_locked(source_folder,path,index,time,order):
    audit = Audit()
    report = audit.read("reference",path)
    require(report["schema"] == REFERENCE_SCHEMA,"wrong reference schema")
    require((report["index"],report["time"],report["order"],report["setting"]) ==
            (index,time,order,Path(source_folder).name),"reference case identity differs")
    tokens,oracle,models,identity = inputs(source_folder,index,time)
    require(report["identity"]["inputs"] == identity,"reference input identity differs")
    require(report["reference_id"] == fingerprint(report["identity"]),"reference identity checksum differs")
    code = report["identity"]["scoring_code"]
    require(set(code) == {"shared_base_curve_reference.py","shared_base_conditional_cohort.py",
                         "shared_base_precision_cohort.py"},"incomplete scoring provenance")
    for name,expected in code.items():
        require(sha_file(Path(__file__).with_name(name)) == expected,"scoring code changed: "+name)
    sampler = sampler_for(order,json.dumps(report["identity"]["proposal"],sort_keys=True))
    target = _fingerprint(sampler,tokens)
    require(target == report["identity"]["target"],"target prior, observations, numerical code, store or runtime differs")
    source = load_source(source_folder,index)
    # Independently check checkpoint/training-prior/logit identities in
    # addition to the bounded cohort loader's source and causal alignment.
    checked_models = load_models(audit,Path(source_folder),
        dict(record=source,tokens=np.asarray(source["tokens"])),index,time)
    for name,values in checked_models.items():
        require(np.array_equal(values,models[name]["prediction"]),"model predictions differ between loaders")
    checkpoint = path.with_name(path.stem+"_chains.npz")
    moment_path = path.with_name(path.stem+"_moments.npz")
    require(str(checkpoint.resolve()) == report["checkpoint"],"checkpoint path differs")
    require(sha_file(checkpoint) == report["checkpoint_sha256"],"checkpoint checksum differs")
    require(sha_file(moment_path) == report["moments_sha256"],"moment checksum differs")
    config = report["estimate"]["diagnostics"]
    require(config["chains"] == 4 and config["thin"] == 1,"wrong chain count or thinning")
    require(config["seed"] == report["identity"]["sampling_seed"] and
            config["burn"] == report["identity"]["burn"],"sampling identity differs")
    expected_seed = seed_of(REFERENCE_SCHEMA,report["identity"]["master_sampling_seed"],
                           Path(source_folder).name,index,time,order)
    require(config["seed"] == expected_seed,"independent sampling-seed construction differs")
    adapter = deepcopy(report)
    adapter["identity"] = dict(numerical_problem=target,proposal_descriptor=report["identity"]["proposal"])
    adapter["settings"] = {k:config[k] for k in ("chains","burn","thin","seed")}
    moments,fresh = raw_checkpoint(audit,"raw",checkpoint,adapter)
    with np.load(moment_path,allow_pickle=False) as saved:
        require(set(saved.files) == set(moments),"incomplete saved moments")
        for name,value in moments.items():
            require(same(value,saved[name].tolist()),"saved moments differ: "+name)
    # raw_checkpoint has released its local raw arrays before the scalar
    # audit reloads just predictions. At most one reference is in memory.
    with np.load(checkpoint,allow_pickle=False) as saved:
        samples = np.stack([saved[f"predictions_{j}"] for j in range(config["chains"])])
        metadata = json.loads(str(saved["metadata"].item()))
    precision = case_precision(samples,oracle)
    del samples
    require(same(precision,report["record"]["precision"]),"recomputed scalar precision differs")
    require(np.allclose(precision["predictive_probs"],moments["mean"],rtol=0,atol=2e-14),"stable predictive means differ")
    # Bind each recorded last draw to the actual full posterior target. This
    # also checks the shared-depth integration and unseen-label prediction.
    target_object = PrefixPosterior(sampler,tokens)
    for j,state in enumerate(metadata["states"]):
        actual = target_object.evaluate(state["alpha_index"],state["ranks"])
        p,depth,error = target_object.predict(actual)
        require(math.isclose(actual.log_likelihood,state["log_likelihood"],rel_tol=0,abs_tol=1e-10),"final-state likelihood differs")
        require(math.isclose(actual.log_target,state["log_target"],rel_tol=0,abs_tol=1e-10),"final-state target differs")
        with np.load(checkpoint,allow_pickle=False) as saved:
            require(np.allclose(p,saved[f"predictions_{j}"][-1],rtol=0,atol=2e-13),"last predictive draw differs from target")
            require(np.allclose(depth,saved[f"depths_{j}"][-1],rtol=0,atol=2e-13),"last depth draw differs from target")
    grids = selected_grid_checks(sampler,tokens,fresh)
    require(same(grids,report["fine_grid_checks"],atol=1e-11,rtol=1e-6),"independent finer-grid values differ")
    row_passed = len(grids) == 4 and all(v["passed"] for v in grids)
    metrics = {name:compare_prediction(moments["mean"],value["prediction"],oracle,moments)
               for name,value in models.items()}
    require(same(metrics,report["model_metrics"]),"model comparison metrics differ")
    gates = diagnostic_gates(fresh)
    mixing = all(gates.values()) and precision["scalar_mixing_passed"]
    old_precision = all(gates.values()) and all(v["metric_precision_passed"] for v in metrics.values()) and row_passed
    record = dict(case_id=index,sampling_seed=config["seed"],precision=precision,
        full_prior_valid=True,provenance_valid=True,mixing_passed=bool(mixing),
        row_grid_passed=bool(row_passed),per_sequence_precision_passed=bool(old_precision))
    require(same(record,report["record"]),"reference validity flags differ from independent audit")
    ready = bool(mixing and row_passed and precision["jensen_local_check_passed"])
    require(ready == report["ready_for_cohort_check"],"worker readiness flag differs")
    require(all(v["passed"] for v in audit.checks),"raw/model audit failure")
    draws = config["draws"]
    min_ess = min(fresh["diagnostics"]["ess_probs"]+[precision["scalar_ess"]])
    factor = max(1.,200/max(min_ess,1e-30))
    requested = int(2**math.ceil(math.log2(max(draws*factor,draws*2 if not ready else draws))))
    return dict(index=index,time=time,order=order,status="ready" if ready else "needs_more_draws",ready=ready,
        record=record,raw_checks=len(audit.checks),report=str(path.resolve()),report_sha256=sha_file(path),
        reference_id=report["reference_id"],draws_per_chain=draws,
        estimated_minimum_draws_for_ess=int(math.ceil(draws*factor)),suggested_next_draws=requested,
        draw_deficit=max(0,requested-draws),gates=gates,
        scalar_ess=precision["scalar_ess"],minimum_probability_ess=min(fresh["diagnostics"]["ess_probs"]),
        recomputed_final_states=True,recomputed_finer_grid=True,
        todo=None if ready else "Extend this same reference; the draw suggestion is an ESS heuristic, not a guarantee.")


def moments(values):
    values = np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),
                standard_error=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)


def aggregate_time(case_audits, expected_ids, orders):
    """Incomplete or invalid data yields no candidate or published mean."""
    found = {(r["index"],r["order"]):r for r in case_audits}
    require(len(found) == len(case_audits),"duplicate audited case/order")
    require(set(found) <= {(i,o) for i in expected_ids for o in orders},"undeclared audited case/order")
    missing = [(i,o) for i in expected_ids for o in orders if (i,o) not in found or
               found[i,o]["status"] in ("missing","running","invalid")]
    if missing:
        return dict(complete=False,passed=False,reported_mean_bayes_regret_bits=None,
                    missing_or_invalid=[dict(index=i,order=o) for i,o in missing],
                    explanation="Every declared case and both exponent rules are required; no partial-cohort Bayesian mean is computed.")
    result = validate_cohort([found[i,orders[0]]["record"] for i in expected_ids],
                            [found[i,orders[1]]["record"] for i in expected_ids],expected_ids)
    result["complete"] = True
    if not result["passed"]:
        hints=[]
        for rule in result["rules"]:
            budget=rule["numerical_budget_bits"]
            allowance=rule["numerical_error_allowance_bits"]
            if allowance > budget:
                hints.append(dict(rule=rule["rule"],reason="cohort_numerical_budget",
                    suggested_uniform_draw_multiplier=max(1.,(allowance/budget)**2) if budget>0 else None,
                    note="Heuristic assumes stationary-chain variance and local Jensen bias decay; re-audit every case after extension."))
        if any(not c["passed"] and "refinement" in c["name"] for c in result["checks"]):
            hints.append(dict(reason="exponent_refinement",note="Resolve numerical disagreement; aggregate mean agreement alone cannot overrule failed case checks."))
        result["continuation_hints"]=hints
    return result


def run(source_folder,reference_root,times,orders,cohort_file=None):
    source_folder = Path(source_folder)
    declaration = source_folder/"cohort.json" if cohort_file is None else Path(cohort_file)
    cohort = json.loads(declaration.read_text())
    protocol = json.loads((source_folder/"protocol.json").read_text())
    require(cohort["protocol_sha256"] == sha_file(source_folder/"protocol.json"),"cohort protocol differs")
    require(times == sorted(set(times)) and all(t in cohort["prefixes"] for t in times),"undeclared or repeated times")
    require(len(orders) == 2 and orders[0] < orders[1],"require two increasing exponent orders")
    expected = list(range(cohort["n_sequences"]))
    require([r["index"] for r in cohort["records"]] == expected,"incomplete cohort declaration")
    # A saved earlier complete declaration may be audited after extension;
    # its immutable existing source entries must agree with the live cohort.
    live = json.loads((source_folder/"cohort.json").read_text())
    require(live["records"][:len(expected)] == cohort["records"],"saved cohort differs from live source identities")
    rows = []
    for time in times:
        tf = {name:[] for name in ("d64","d64_seed1")}
        entropies = []
        for index in expected:
            _,oracle,predictions,_ = inputs(source_folder,index,time)
            mask = oracle>0
            entropies.append(float(-oracle[mask]@np.log2(oracle[mask])))
            for name,value in predictions.items():
                tf[name].append(float(oracle[mask]@np.log2(oracle[mask]/value["prediction"][mask])))
        audits = []
        for index in expected:
            for order in orders:
                checked = audit_reference(source_folder,reference_root,index,time,order)
                audits.append(checked)
                if checked["status"] != "missing":
                    print(json.dumps(dict(index=index,time=time,order=order,status=checked["status"])),flush=True)
        aggregate = aggregate_time(audits,expected,orders)
        matched = None
        if aggregate["passed"]:
            by_id = {r["index"]:r["record"]["precision"]["bayes_regret_bits"] for r in audits if r["order"] == orders[1]}
            bayes = np.array([by_id[i] for i in expected])
            matched = {name:dict(transformer_regret=moments(values),bayes_regret=moments(bayes),
                         transformer_minus_bayes=moments(np.asarray(values)-bayes)) for name,values in tf.items()}
        rows.append(dict(time=time,cohort_validation=aggregate,matched_comparisons=matched,
            full_cohort_transformer_only={name:moments(values) for name,values in tf.items()},
            full_cohort_oracle_entropy=moments(entropies),case_audits=audits,
            status_counts={s:sum(v["status"]==s for v in audits) for s in
                           ("missing","running","invalid","needs_more_draws","ready")}))
    return dict(schema=SCHEMA,setting=source_folder.name,alpha=protocol["alpha"],depth=protocol["depth"],
        n_sequences=len(expected),n_tokens_per_sequence=1025,times=times,orders=orders,rows=rows,
        all_requested_times_validated=all(r["cohort_validation"]["passed"] for r in rows),
        provenance=dict(cohort_file=str(declaration.resolve()),cohort_sha256=sha_file(declaration),
                        protocol_sha256=sha_file(source_folder/"protocol.json"),audit_sha256=sha_file(__file__)),
        interpretation="Only matched_comparisons with passed complete-cohort validation are scientific transformer--Bayes means. Other records describe outstanding work; there is no available-case mean.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--references",type=Path,required=True)
    p.add_argument("--times",type=int,nargs="+",default=[4,16,64,256,1024])
    p.add_argument("--orders",type=int,nargs=2,default=[16,32])
    p.add_argument("--cohort-file",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.input,a.references,a.times,a.orders,a.cohort_file)
    atomic_json(a.out,json_safe(result))
    print(json.dumps(dict(out=str(a.out),complete=result["all_requested_times_validated"],
         times=[dict(time=r["time"],counts=r["status_counts"]) for r in result["rows"]])),flush=True)


if __name__ == "__main__":
    main()

"""Publish only complete, numerically validated long-context Bayes cohorts.

Replays probability-score and cohort-precision arithmetic from the transferred
snapshot, including the untouched pilot sequences. Raw-chain audits remain
cluster provenance; no chain loading, posterior sampling or inference occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

import shared_base_campaign_results as campaign
from shared_base_curve_audit import aggregate_time, same
from shared_base_long_context_results import select_references, array_sha
from source import seed_of
from validate_shared_base_mcmc import json_safe

ROOT = campaign.ROOT
INPUT = ROOT/"output/shared_base_posterior_full_download_2026-09-19"
PLAN = ROOT/"code/transformer/configs/shared_base_posterior_full_plan_2026-09-17.json"
OUT = ROOT/"results/transformer/shared_base_full_cohort_2026-09-19"
COHORTS = ROOT/"output/shared_base_conditional_curves_2026-09-15"
PILOT_PLAN = ROOT/"code/transformer/configs/shared_base_posterior_plan_2026-09-16.json"
REFINE_PLAN = ROOT/"code/transformer/configs/shared_base_posterior_refinement_plan_2026-09-17.json"
BASE = ROOT/"output/shared_base_refinement_base_2026-09-17.json"
REFINEMENT = ROOT/"output/shared_base_refinement_completed_2026-09-17.json"
require, sha, fingerprint = campaign.require, campaign.sha, campaign.fingerprint


def recompute_audit(audit, oracle, index, time, order, expected_seed=None):
    """Independently score a saved forecast and check its numerical receipt."""
    a = deepcopy(audit)
    require((a["index"], a["time"], a["order"]) == (index,time,order) and
            a["ready"] is True and a["status"] == "ready", "Wrong or unresolved reference case")
    require(a["recomputed_final_states"] is True and a["recomputed_finer_grid"] is True and
            a["gates"] == dict(split_rhat_below_1_05=True, probability_ess_at_least_200=True,
                maximum_probability_mcse_at_most_0_005=True, raw_normalization=True),
            "Raw numerical audit did not pass")
    r = a["record"]; m = r["precision"]
    require(r["case_id"] == index and all(r[k] is True for k in
            ("full_prior_valid","provenance_valid","mixing_passed","row_grid_passed")),
            "Reference provenance or basic checks failed")
    if expected_seed is not None:
        require(r["sampling_seed"] == expected_seed, "Reference sampling seed differs from the frozen task")
    p, q = np.asarray(m["predictive_probs"],float), np.asarray(oracle,float)
    require(p.shape == q.shape == (64,) and np.isfinite(p).all() and np.isfinite(q).all() and
            np.all(p > 0) and np.all(q >= 0), "Invalid saved probability law")
    normalization = max(abs(float(p.sum())-1),abs(float(q.sum())-1))
    require(normalization <= 1e-7, "Saved probability law does not normalize")
    positive = q > 0
    entropy = float(-q[positive] @ np.log2(q[positive]))
    loss = float(-q[positive] @ np.log2(p[positive]))
    regret = float(q[positive] @ (np.log2(q[positive])-np.log2(p[positive])))
    errors = [abs(m[k]-v) for k,v in (("oracle_entropy_bits",entropy),
        ("bayes_cross_entropy_bits",loss),("bayes_regret_bits",regret))]
    require(max(errors) < 2e-12, "Saved Bayesian score differs from the frozen true-row expectation")
    variances = list(m["variance_by_estimator_bits2"].values())
    require(variances and all(np.isfinite(v) and v >= 0 for v in variances) and
            same(m["mc_variance_bits2"],max(variances)) and m["mcse_bits"] >= 0 and
            same(m["mcse_bits"]**2,m["mc_variance_bits2"]), "Invalid Monte Carlo variance receipt")
    require(all(np.isfinite(m[k]) and m[k] >= 0 for k in
                ("jensen_curvature_scale_bits","chain_jackknife_bias_bits","jensen_proxy_bits")) and
            same(m["jensen_proxy_bits"],max(m["jensen_curvature_scale_bits"],m["chain_jackknife_bias_bits"])),
            "Invalid Jensen diagnostic receipt")
    scalar_ok = (m["scalar_split_rhat"] is not None and np.isfinite(m["scalar_split_rhat"]) and
                 0 <= m["scalar_split_rhat"] < 1.05 and m["scalar_ess"] >= 200)
    jensen_ok = (0 <= m["max_relative_probability_mcse"] <= .1 and
                 m["chain_jackknife_bias_bits"] <= 2*m["jensen_curvature_scale_bits"]+1e-12)
    require(scalar_ok and m["scalar_mixing_passed"] is True and jensen_ok and
            m["jensen_local_check_passed"] is True and m["deterministic"] is False and m["chains"] == 4 and
            m["draws_per_chain"] == a["draws_per_chain"] and a["minimum_probability_ess"] >= 200,
            "Scalar mixing, probability ESS or local bias checks failed")
    # Preserve informational per_sequence_precision_passed unchanged. The frozen
    # acceptance criterion is the complete-cohort numerical budget, not that flag.
    m.update(oracle_entropy_bits=entropy,bayes_cross_entropy_bits=loss,bayes_regret_bits=regret)
    return a, regret, dict(max_normalization_error=normalization,max_score_discrepancy_bits=max(errors))


def validate_group(audits, oracles, time, orders, saved_audit, expected_ids=None):
    expected = list(range(256)) if expected_ids is None else list(expected_ids)
    require(saved_audit.get("complete") is True and saved_audit.get("passed") is True,
            "Cannot publish an incomplete or failed cohort")
    require(len(orders)==2 and orders[0]<orders[1], "Expected two increasing integration orders")
    lookup = {(a["index"],a["order"]):a for a in audits}
    require(len(lookup)==len(audits) and set(lookup)=={(i,o) for i in expected for o in orders},
            "Every declared sequence and both orders are required exactly once")
    require(np.asarray(oracles).shape==(len(expected),64), "Oracle cohort is incomplete")
    corrected=[]; values={}; errors=dict(max_normalization_error=0.,max_score_discrepancy_bits=0.)
    for j,index in enumerate(expected):
        for order in orders:
            a,r,check=recompute_audit(lookup[index,order],oracles[j],index,time,order)
            corrected.append(a); values[index,order]=r
            for k,v in check.items(): errors[k]=max(errors[k],v)
    fresh=json_safe(aggregate_time(corrected,expected,orders))
    require(fresh["complete"] is True and fresh["passed"] is True and same(fresh,json_safe(saved_audit)),
            "Replayed full-cohort numerical budget/refinement differs or fails")
    return fresh,np.asarray([values[i,orders[-1]] for i in expected]),errors


def moments(values):
    x=np.asarray(values,float)
    require(x.shape==(256,) and np.isfinite(x).all(),"A published cohort must contain all 256 finite scores")
    return dict(n=256,mean=float(x.mean()),se=float(x.std(ddof=1)/16))


def build(folder=INPUT, campaign_folder=campaign.OUT, campaign_input=campaign.INPUT):
    inputs=campaign.Inputs(); plan=inputs.read(PLAN)
    require(plan["schema"]=="shared-base-posterior-full-plan-v1" and
            plan["reference_prior"]==dict(alpha=[.5,1.5],depths=list(range(9)),ranks="uniform permutation") and
            plan["conditions"]["depths"]==campaign.DEPTHS and plan["conditions"]["indices"]==list(range(2,256)) and
            plan["conditions"]["observed_tokens"]==[256,1024] and plan["conditions"]["alpha"]==1.,
            "Frozen full-cohort prior or source scope differs")
    launch=inputs.read(Path(folder)/"launch.json"); snapshot=inputs.read(Path(folder)/"summary.json")
    status=inputs.read(Path(folder)/"status.json")
    require(launch["schema"]=="shared-base-posterior-full-launch-v1" and launch["plan_sha256"]==sha(PLAN) and
            launch["launch_id"]==fingerprint({k:v for k,v in launch.items() if k!="launch_id"}),"Launch identity differs")
    require(snapshot["schema"]=="shared-base-posterior-full-summary-v1" and
            snapshot["launch_id"]==launch["launch_id"] and snapshot["expected_new_tasks"]==4064 and
            snapshot["reused_pilot_sequences"]==8 and [r["task"] for r in snapshot["rows"]]==plan["tasks"],
            "Full-cohort snapshot task coverage differs")
    require(snapshot["counts"]==dict(Counter(r["status"] for r in snapshot["rows"])) and
            status["counts"]==snapshot["counts"] and status["expected_tasks"]==4064 and
            status["coverage_complete"]==snapshot["coverage_complete"] and status["status"]==snapshot["status"],
            "Snapshot/status accounting differs")
    covered=all(r["status"] in ("ready","unresolved","failed","timeout_exhausted") for r in snapshot["rows"])
    passed=all(g["numerical_audit"]["passed"] for g in snapshot["groups"])
    require(snapshot["coverage_complete"] is covered and snapshot["all_numerical_checks_passed"] is passed and
            status["scientific_reference_validated"] is (covered and passed) and
            snapshot["status"] == ("validated" if covered and passed else "failed_numerical_checks" if covered else "incomplete"),
            "Global coverage/status does not match the declared task and group outcomes")
    for name,digest in plan["code_sha256"].items(): inputs.bind(ROOT/name,digest)
    inputs.bind(ROOT/plan["completed_pilot"]["path"],plan["completed_pilot"]["sha256"])
    base=inputs.read(BASE,plan["inputs"]["files"]["pilot/base.json"]["sha256"])
    refinement=inputs.read(REFINEMENT,plan["inputs"]["files"]["pilot/refinement.json"]["sha256"])
    pilot=inputs.read(PILOT_PLAN); refine_plan=inputs.read(REFINE_PLAN)
    old=select_references(base,refinement,pilot,refine_plan)
    all_audits=dict(old)
    for row in snapshot["rows"]:
        t=row["task"]
        if "audit" in row:
            require(row["status"] in ("ready","unresolved"),"Unexpected audit status")
            a=row["audit"]
            require((a["index"],a["time"],a["order"])==(t["index"],t["time"],t["order"]),"Task/audit identity differs")
            raw_path=Path(a["report"])
            require(raw_path.name==f"seq_{t['index']:06d}_t{t['time']:04d}.json" and
                    raw_path.parent.name==f"alpha{t['order']}" and raw_path.parent.parent.name==t["setting"] and
                    campaign.digest_value(a["report_sha256"]) and campaign.digest_value(a["reference_id"]),
                    "Remote audited reference locator or digest is malformed")
            require(a["record"]["sampling_seed"]==seed_of("shared-base-curve-reference-v1",plan["worker"]["seed"],
                t["setting"],t["index"],t["time"],t["order"]),"Unplanned or reused full-cohort sampling seed")
            all_audits[t["depth"],t["index"],t["time"],t["order"]]=a
    expected_groups={(d,t) for d in campaign.DEPTHS for t in (256,1024)}
    require(len(snapshot["groups"])==8 and {(g["depth"],g["time"]) for g in snapshot["groups"]}==expected_groups,
            "Missing or repeated full-cohort group")
    accepted=[]; withheld=[]
    for g in snapshot["groups"]:
        d,t=g["depth"],g["time"]; orders=([32,64] if (d,t)==(0,1024) else [16,32])
        require(g["orders"]==orders,"Integration orders differ from frozen design")
        selected=[a for (depth,i,time,o),a in all_audits.items() if (depth,time)==(d,t) and o in orders]
        replay=json_safe(aggregate_time(selected,list(range(256)),orders))
        require(same(replay,g["numerical_audit"]),"Saved cohort audit differs from its source records")
        if replay.get("complete") and replay.get("passed"): accepted.append((g,selected))
        else: withheld.append(dict(depth=d,time=t,complete=replay["complete"],passed=False,
            reason="Full 256-sequence coverage and all numerical checks are required; no subset mean released."))
    paths, campaign_report, _=campaign.load_campaign(campaign_folder)
    for p in paths: inputs.bind(p)
    campaign_audit=inputs.read(Path(campaign_folder)/"validation.json")
    campaign_plan=inputs.read(campaign.PLAN)
    frozen=campaign.load_frozen(campaign_plan,campaign.COHORTS,inputs)
    baseline=campaign.load_baseline(campaign.BASELINE,frozen,inputs)
    source_lookup={(int(s["setting"].split("L")[-1]),s["index"]):s for p in (plan,pilot) for s in p["source_sequences"]}
    checked_sources=set(); published=[]; arrays={}; max_errors=dict(max_normalization_error=0.,max_score_discrepancy_bits=0.)
    for group,audits in accepted:
        d,t=group["depth"],group["time"]; j=campaign.TIMES.index(t); oracles=[]
        for i in range(256):
            relative=f"a1_L{d}/data/seq_{i:06d}.json"; source_plan=pilot if i<2 else plan
            source=inputs.read(COHORTS/relative,source_plan["inputs"]["files"][relative]["sha256"])
            identity=source_lookup[d,i]; tokens=np.asarray(source["tokens"],dtype=int)
            transition=np.asarray(source["oracle"]["transition_probs"],float)
            require(tokens.shape==(1025,) and source["source_id"]==identity["source_id"]==fingerprint(source["identity"]) and
                source["tokens_sha256"]==identity["tokens_sha256"]==array_sha(tokens,"<i2") and
                transition.shape==(64,64) and array_sha(transition,"<f8")==source["oracle"]["transition_probs_sha256"],
                "Frozen source identity/tokens/oracle differs")
            oracle=transition[tokens[t-1]]
            require(frozen[d]["source_ids"][i]==source["source_id"] and np.array_equal(frozen[d]["tokens"][i],tokens) and
                    np.array_equal(frozen[d]["oracle_probs"][i,j],oracle),"Bayesian/transformer cohort pairing differs")
            oracles.append(oracle);checked_sources.add((d,i))
        audit,regrets,errors=validate_group(audits,np.asarray(oracles),t,group["orders"],group["numerical_audit"])
        for k,v in errors.items():max_errors[k]=max(max_errors[k],v)
        numerical=audit["rules"][-1]
        b=dict(**moments(regrets),numerical_mcse_bits=numerical["numerical_mcse_bits"],
            numerical_error_allowance_bits=numerical["numerical_error_allowance_bits"],
            numerical_budget_bits=numerical["numerical_budget_bits"],denoised_sequence_se_bits=numerical["denoised_sequence_se_bits"],
            mean_jensen_proxy_bits=numerical["mean_jensen_proxy_bits"])
        require(same(b["mean"],audit["reported_mean_bayes_regret_bits"]) and
                same(b["se"],numerical["observed_sequence_se_bits"]),"Published Bayes mean/SE differs")
        name=f"L{d}_t{t}";arrays[name+"_bayes_regret"]=regrets;arrays[name+"_source_ids"]=frozen[d]["source_ids"]
        comparisons=[]
        for seed in campaign.SEEDS:
            model_scores={"original100k":baseline[d]["baseline_regret_bits"][:,j,seed],
                          "large100k":baseline[d]["larger_regret_bits"][:,j,seed]}
            for arm,steps in campaign_report["milestones"].items():
                for step in steps:
                    path=Path(campaign_input)/f"eval/{arm}/seed{seed}/step{step}/a1_L{d}.npz"
                    digest=campaign_audit["inputs"][str(path.relative_to(ROOT))]
                    saved=inputs.arrays(path,digest)
                    rebuilt,_,_=campaign.score_arrays(saved,frozen[d])
                    model_scores[f"{arm}_{step}"]=rebuilt[:,j]
            expected=next(g for g in campaign_report["groups"] if g["depth"]==d and g["seed"]==seed)
            for model,values in model_scores.items():
                model_mean=moments(values); gap=moments(values-regrets)
                require(same(model_mean["mean"],expected["predictors"][model]["mean"][j]) and
                    same(model_mean["se"],expected["predictors"][model]["se"][j]),"Campaign released score differs")
                comparisons.append(dict(seed=seed,model=model,regret=model_mean,excess_over_bayes=gap))
                arrays[name+f"_{model}_seed{seed}"]=values
        published.append(dict(depth=d,time=t,orders=group["orders"],n_sequences=256,indices=list(range(256)),
            source_ids=frozen[d]["source_ids"].tolist(),bayes=b,comparisons=comparisons,
            numerical_checks=audit["checks"],refinement={k:v for k,v in audit["refinement"].items() if k!="per_case"},
            numerical_budget=audit["budget"],all_cases_retained=True))
    report=dict(schema="shared-base-full-cohort-results-v1",passed=True,plan_sha256=sha(PLAN),launch_id=launch["launch_id"],
        source_snapshot_sha256=sha(Path(folder)/"summary.json"),n_sequences_per_group=256,groups=published,withheld=withheld,
        complete_campaign=not withheld,cohort_role="Reused diagnostic cohorts, with every declared source retained within each released condition.",
        uncertainty="One pointwise standard error across source sequences; paired differences use the same histories. Numerical MCMC allowance reported separately.",
        validation_scope="Predictive normalization, source identity, true-row scoring and whole-cohort numerical budgets independently replayed locally. Raw chains, final-state and finer-grid recomputation are frozen cluster audit provenance, not rerun locally.",
        sampling_performed=False,inference_performed=False)
    return report,arrays,dict(schema="shared-base-full-cohort-validation-v1",passed=True,inputs=inputs.hashes,
        checks=max_errors,source_records_verified=len(checked_sources),raw_chains_reaudited=False,
        groups_published=len(published),groups_withheld=len(withheld))


def load_full_cohort(source=OUT):
    source=Path(source)
    if (source/"reports.json.gz").exists():
        paths=[source/"reports.json.gz",PLAN]
        with gzip.open(paths[0],"rt") as stream:
            saved=json.load(stream)
        for name in ("summary.json","validation.json"):
            if (source/name).exists():
                require((source/name).read_text()==saved[name],"Archived full-cohort report differs")
        report,audit=[json.loads(saved[name]) for name in ("summary.json","validation.json")]
        summary_sha=hashlib.sha256(saved["summary.json"].encode()).hexdigest()
    else:
        paths=[source/"summary.json",source/"validation.json",PLAN]
        report,audit=[json.loads(p.read_text()) for p in paths[:2]]
        summary_sha=sha(paths[0])
    require(report["schema"]=="shared-base-full-cohort-results-v1" and report["passed"] is True and
            audit["schema"]=="shared-base-full-cohort-validation-v1" and audit["passed"] is True and
            audit["summary_sha256"]==summary_sha and report["plan_sha256"]==sha(PLAN),"Full-cohort release integrity differs")
    curves={}
    for depth in campaign.DEPTHS:
        rows=sorted([g for g in report["groups"] if g["depth"]==depth],key=lambda g:g["time"])
        require(len({g["time"] for g in rows})==len(rows) and all(g["n_sequences"]==256 and g["all_cases_retained"] is True and
                all(c["passed"] for c in g["numerical_checks"]) for g in rows),"Incomplete displayed full cohort")
        curves[depth]=dict(times=[g["time"] for g in rows],mean=[g["bayes"]["mean"] for g in rows],se=[g["bayes"]["se"] for g in rows])
    return paths,report,curves


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=INPUT);parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--campaign-input",type=Path,default=campaign.INPUT,
                        help="Transferred neural arrays bound by the campaign validation receipt")
    args=parser.parse_args();report,arrays,audit=build(args.input,campaign_input=args.campaign_input.resolve());args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(args.out/"paired_scores.npz",**arrays)
    rows=[dict(depth=g["depth"],time=g["time"],seed=c["seed"],model=c["model"],bayes=g["bayes"]["mean"],
          bayes_se=g["bayes"]["se"],regret=c["regret"]["mean"],regret_se=c["regret"]["se"],
          excess=c["excess_over_bayes"]["mean"],excess_se=c["excess_over_bayes"]["se"],
          numerical_allowance=g["bayes"]["numerical_error_allowance_bits"]) for g in report["groups"] for c in g["comparisons"]]
    with (args.out/"comparisons.tsv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(rows)
    audit.update(script_sha256=sha(__file__),summary_sha256=sha(args.out/"summary.json"),
        outputs={n:sha(args.out/n) for n in ("summary.json","paired_scores.npz","comparisons.tsv")})
    (args.out/"validation.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True,published=[(g["depth"],g["time"]) for g in report["groups"]],
                         withheld=len(report["withheld"]),checks=audit["checks"]),indent=2))


if __name__=="__main__":main()

"""Focused checks for the cross-time copying-payoff assessment."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import assess_copying_payoff as assessment


def test_crossings_keep_unreached_values_right_censored():
    values=np.array([
        [-1,1,-1,1,1,1,1,1],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [1,1,1,1,1,1,1,1],
    ],dtype=float)
    first,final=assessment.crossing_draws(values)
    assert np.array_equal(first[[0,2]],[2,1]) and np.isinf(first[1])
    assert np.array_equal(final[[0,2]],[4,1]) and np.isinf(final[1])
    interval=assessment.crossing_percentiles(np.array([1.,2.,np.inf,np.inf]))
    assert interval == dict(lower=1,median=2,upper=None)


def test_numerical_zero_tolerance_is_not_a_positive_crossing():
    first,final=assessment.crossing_draws(np.array([0,1e-13,1e-12,2e-12,3e-12,4e-12,5e-12,6e-12]))
    assert first[0] == 4 and final[0] == 4


def test_common_bank_covariance_is_summed_before_cumulative_mcse():
    full=np.zeros((256,8));leave=np.zeros((4,256,8))
    signs=np.array([-3.,-1.,1.,3.])
    leave[:,:,0]=signs[:,None];leave[:,:,1]=-signs[:,None]
    instant=assessment.jackknife_metrics(full,leave)
    cumulative=assessment.cumulative_jackknife_metrics(full,leave)
    assert instant["mcse"][0] > 0 and instant["mcse"][1] > 0
    assert cumulative["mcse"][0] > 0
    assert cumulative["mcse"][1] == pytest.approx(0,abs=1e-14)
    assert instant["covariance"][0,1] < 0


def test_paired_stationary_refinement_cancels_common_bank_noise():
    full=np.zeros((256,8));leave=np.empty((4,256,8))
    for bank,value in enumerate((-3.,-1.,1.,3.)):
        leave[bank]=value
    data=dict(stationary=full,stationary_leave=leave,
        stationary_coarse=full.copy(),stationary_coarse_leave=leave.copy(),
        stationary_comparison_paired=True)
    instant,cumulative=assessment.stationary_refinement_metrics(data)
    assert np.array_equal(instant["shift"],np.zeros(8))
    assert np.array_equal(instant["mcse"],np.zeros(8))
    assert np.array_equal(cumulative["mcse"],np.zeros(8))
    assert np.array_equal(instant["bias"],np.zeros(8))


def test_allowance_matches_predeclared_formula_and_budget():
    source=np.full(8,.008)
    m32=np.full(8,.0001);b32=np.full(8,.00001)
    bank=np.full(8,.0002);bbank=np.full(8,.00002)
    ashift=np.full(8,-.0003);acmp=np.full(8,.00004);abias=np.full(8,.00005)
    sshift=np.full(8,.0002);scmp=np.full(8,.00003);sbias=np.full(8,.00004)
    got=assessment.numerical_accounting(source,m32,b32,bank,bbank,
                                        ashift,acmp,abias,sshift,scmp,sbias)
    final=2*np.hypot(.0001,.0002)+2*(.00001+.00002)
    alpha=.0003+3*.00004+.00005
    stationary=.0002+3*.00003+.00004
    assert np.allclose(got["total_numerical_allowance"],final+alpha+stationary)
    assert np.allclose(got["numerical_budget"],.002)
    assert np.all(got["precision_passed"] == ((final+alpha+stationary)<=.002))


def test_whole_sequence_bootstrap_preserves_cross_time_pairing():
    x=np.linspace(-1,1,256)
    paired=np.zeros((256,8));paired[:,0]=x;paired[:,1]=-x
    other=np.arange(256*8,dtype=float).reshape(256,8)
    draws=assessment.bootstrap_curves(dict(paired=paired,other=other),2000,7)
    # Any paired resample has exactly cancelling first two cumulative values.
    assert np.max(abs(np.cumsum(draws["paired"],axis=1)[:,1])) < 1e-14
    assert draws["paired"].shape == draws["other"].shape == (2000,8)


def test_complete_zero_error_assessment_releases_crossings():
    row=np.linspace(-.02,.02,256)[:,None]
    gain=np.broadcast_to(row,(256,8)).copy()+np.linspace(-.01,.01,8)
    bayes=np.full((256,8),5.)
    stationary=bayes+gain
    data=dict(stationary=stationary,stationary_leave=np.tile(stationary,(16,1,1)),
        stationary_coarse=stationary.copy(),stationary_coarse_leave=np.tile(stationary,(8,1,1)),
        bayes32=bayes,bayes16=bayes.copy(),mcmc16_mcse=np.zeros(8),mcmc32_mcse=np.zeros(8),
        mcmc16_bias=np.zeros(8),mcmc32_bias=np.zeros(8),
        ordered_seed0=bayes,shuffled_seed0=stationary,
        ordered_seed1=bayes+.1,shuffled_seed1=stationary+.1)
    summary,arrays,curves,bootstrap=assessment.assess_condition("a1_L0",data,11)
    assert summary["aggregate_precision_passed"]
    assert summary["crossings"]["bayes"]["instantaneous"]["eligible"]
    assert summary["crossings"]["bayes"]["instantaneous"]["first_positive"]["time"] == 5
    assert summary["crossings"]["bayes"]["instantaneous"]["final_positive_run"]["time"] == 5
    assert np.array_equal(arrays["bayes_gain"],stationary-bayes)
    assert curves["bayes_cumulative_mean"].shape == (8,)
    assert bootstrap["bayes_instantaneous_mean"].shape == (2000,8)


def test_failed_gate_withholds_crossing_even_when_candidate_is_positive():
    mean=np.ones(8);draws=np.ones(2000)
    report=assessment.crossing_report(mean,draws,draws,False)
    assert not report["eligible"]
    assert report["first_positive"] == dict(time=None,status="withheld_numerical_precision")
    assert report["final_positive_run"]["time"] is None


def test_recovery_parent_summary_is_normalized_to_per_time_interface(tmp_path,monkeypatch):
    out=tmp_path/"merge";folder=out/"t7";folder.mkdir(parents=True)
    arrays={};groups=[]
    def saved_stats(value):
        return dict(candidate_mean_bits=float(value.mean()),
                    observed_sequence_se_bits=float(value.std(ddof=1)/16))
    for condition in assessment.CONDITIONS:
        stationary=np.full(256,6.);a16=np.full(256,5.);a32=np.full(256,5.)
        arrays[condition+"_stationary_loss"]=stationary
        arrays[condition+"_bayes_loss_a16"]=a16
        arrays[condition+"_bayes_loss_a32"]=a32
        arrays[condition+"_original_label_probs_a16"]=np.full((256,64),1/64)
        arrays[condition+"_original_label_probs_a32"]=np.full((256,64),1/64)
        neural={}
        for seed in (0,1):
            ordered=np.full(256,5.5);shuffled=np.full(256,5.75)
            arrays[condition+f"_ordered_seed{seed}"]=ordered
            arrays[condition+f"_shuffled_seed{seed}"]=shuffled
            neural[str(seed)]=dict(gain=saved_stats(shuffled-ordered))
        groups.append(dict(condition=condition,n_sequences=256,
            rules={str(order):dict(bayes_loss=saved_stats(arrays[condition+f"_bayes_loss_a{order}"]),
                stationary_minus_bayes=saved_stats(stationary-arrays[condition+f"_bayes_loss_a{order}"]))
                for order in (16,32)},order_loss_difference=saved_stats(a16-a32),neural=neural))
    np.savez_compressed(folder/"arrays.npz",**arrays)
    wave=dict(time=7,complete=True,original_histories=768,individual_audits_passed=True,
        paired_orders_passed=True,diagnostic_arrays_validated=True,diagnostic_groups=groups,
        arrays_sha256=assessment.sha(folder/"arrays.npz"),
        comparator="Source-induced stationary-frequency surrogate, not exact finite-shuffle Bayes.")
    claimed=tmp_path/"claimed";claimed.write_bytes(b"bound input")
    producer=dict(schema="copying-payoff-recovery-merge-v1",complete_cohorts_passed=True,
        diagnostic_arrays_validated=True,individual_audits_passed=True,paired_orders_passed=True,
        input_sha256={str(claimed):assessment.sha(claimed)},waves={"7":wave})
    (out/"summary.json").write_text(json.dumps(producer))
    monkeypatch.setattr(assessment,"diagnostic_seeds",lambda *args:{1,2})
    loaded=assessment.load_diagnostic(folder,7,assessment.Bindings())
    assert loaded["summary"] == wave
    assert set(loaded["groups"]) == set(assessment.CONDITIONS)


def test_bound_mcmc_identity_family_requires_distinct_seeds(tmp_path,monkeypatch):
    monkeypatch.setitem(assessment.DIAGNOSTIC_TASK_COUNTS,5,2)
    manifest={}
    for index,seed in enumerate((17,29)):
        task_id=f"t0005_case{index}_a{16+16*index}"
        folder=tmp_path/"tasks"/task_id;folder.mkdir(parents=True)
        path=folder/"identity.json"
        path.write_text(json.dumps(dict(task=dict(id=task_id,time=5,order=16+16*index,seed=seed))))
        manifest[str(path)]=assessment.sha(path)
    producer=dict(input_sha256=manifest)
    assert assessment.diagnostic_seeds(producer,5,assessment.Bindings()) == {17,29}
    second=Path(list(manifest)[1]);value=json.loads(second.read_text());value["task"]["seed"]=17
    second.write_text(json.dumps(value));manifest[str(second)]=assessment.sha(second)
    with pytest.raises(ValueError,match="repeated MCMC seed"):
        assessment.diagnostic_seeds(producer,5,assessment.Bindings())


def test_producer_claims_are_hashed_not_only_recorded(tmp_path):
    path=tmp_path/"producer.bin";path.write_bytes(b"frozen producer input")
    digest=assessment.sha(path);bindings=assessment.Bindings()
    bindings.claims({str(path):digest},"producer")
    assert bindings.direct[str(path.resolve())] == digest
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Input hash changed"):
        assessment.Bindings().claims({str(path):digest},"producer")
    with pytest.raises(ValueError,match="provenance conflict"):
        bindings.claims({str(path):"0"*64},"other producer")


def test_paired_stationary_loader_hashes_every_bound_artifact(tmp_path,monkeypatch):
    folder=tmp_path/"paired";folder.mkdir()
    source=tmp_path/"source.bin";source.write_bytes(b"frozen source")
    policy=tmp_path/"policy.json";policy.write_text("{}")
    monkeypatch.setattr(assessment,"PAIRED_STATIONARY_POLICY_SOURCE",policy)
    arrays={
        "fine_stationary_loss":np.ones((256,8)),
        "fine_stationary_leave_one_bank_out_loss":np.ones((16,256,8)),
        "coarse_stationary_loss":np.ones((256,8)),
        "coarse_stationary_leave_one_bank_out_loss":np.ones((16,256,8)),
        "paired_difference_leave_one_bank_out_loss":np.zeros((16,256,8)),
    }
    for condition in assessment.CONDITIONS:
        np.savez_compressed(folder/(condition+".npz"),**arrays)
    producer=Path(assessment.__file__).with_name("copying_frequency_paired_refinement.py")
    validator=Path(assessment.__file__).with_name("validate_copying_frequency_paired_refinement.py")
    plan=dict(schema="copying-frequency-paired-refinement-plan-v1",banks=16,
        coarse_order=16,fine_order=32,times=list(range(1,9)),
        resources=dict(max_workers=4,maximum_attempts_per_bank=1),
        inputs_sha256={str(source):assessment.sha(source)},
        code_sha256={str(producer):assessment.sha(producer)})
    plan["plan_id"]=assessment.json_digest(plan)
    (folder/"plan.json").write_text(json.dumps(plan))
    summary=dict(schema="copying-frequency-paired-refinement-v1",complete=True,
        plan_id=plan["plan_id"],banks=16,times=list(range(1,9)))
    (folder/"summary.json").write_text(json.dumps(summary))
    artifacts={name:assessment.sha(folder/name) for name in
        ["plan.json","summary.json",*(condition+".npz" for condition in assessment.CONDITIONS)]}
    validation=dict(schema="copying-frequency-paired-refinement-validation-v1",
        passed=True,input_integrity_passed=True,stationary_component_precision_passed=True,
        separate_bayesian_alpha_refinement_included=False,payoff_forecast_approved=False,
        validator_sha256=assessment.sha(validator),assessment_policy_sha256=assessment.sha(policy),
        artifact_sha256=artifacts)
    (folder/"validation.json").write_text(json.dumps(validation))
    loaded=assessment.load_stationary_paired(folder,assessment.Bindings())
    assert set(loaded)==set(assessment.CONDITIONS)
    (folder/"a1_L0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Input hash changed"):
        assessment.load_stationary_paired(folder,assessment.Bindings())


def test_check_output_distinguishes_integrity_from_failed_scientific_gate(tmp_path,capsys):
    arrays={};curves={"times":np.arange(1,9)}
    for condition in assessment.CONDITIONS:
        stationary=np.ones((256,8));bayes=np.zeros((256,8));gain=stationary-bayes
        arrays[condition+"_stationary_loss"]=stationary
        arrays[condition+"_bayes_loss"]=bayes
        arrays[condition+"_bayes_gain"]=gain
        curves[condition+"_bayes_instantaneous_mean"]=gain.mean(0)
        curves[condition+"_bayes_cumulative_mean"]=np.cumsum(gain,axis=1).mean(0)
    np.savez_compressed(tmp_path/"arrays.npz",**arrays)
    np.savez_compressed(tmp_path/"curves.npz",**curves)
    np.savez_compressed(tmp_path/"bootstrap_draws.npz",placeholder=np.zeros(1))
    (tmp_path/"summary.json").write_text(json.dumps(dict(
        complete=True,aggregate_precision_passed=False,payoff_forecast_approved=False,
        publication_status="diagnostic_nonpublication")))
    artifacts={name:assessment.sha(tmp_path/name) for name in
               ("arrays.npz","curves.npz","bootstrap_draws.npz","summary.json")}
    (tmp_path/"validation.json").write_text(json.dumps(dict(
        integrity_passed=True,input_validation_passed=True,passed=False,
        scientific_precision_passed=False,payoff_forecast_approved=False,
        publication_status="diagnostic_nonpublication",artifact_sha256=artifacts)))
    assert assessment.check_output(tmp_path) == 2
    status=json.loads(capsys.readouterr().out)
    assert status["integrity_passed"] and not status["payoff_forecast_approved"]
    assert status["status"] == "integrity_passed_scientific_precision_failed"


def test_check_output_rejects_publication_flag_mismatch(tmp_path):
    arrays={};curves={"times":np.arange(1,9)}
    for condition in assessment.CONDITIONS:
        value=np.ones((256,8));arrays[condition+"_stationary_loss"]=value
        arrays[condition+"_bayes_loss"]=np.zeros_like(value)
        arrays[condition+"_bayes_gain"]=value
        curves[condition+"_bayes_instantaneous_mean"]=value.mean(0)
        curves[condition+"_bayes_cumulative_mean"]=np.cumsum(value,axis=1).mean(0)
    np.savez_compressed(tmp_path/"arrays.npz",**arrays);np.savez_compressed(tmp_path/"curves.npz",**curves)
    np.savez_compressed(tmp_path/"bootstrap_draws.npz",placeholder=np.zeros(1))
    (tmp_path/"summary.json").write_text(json.dumps(dict(complete=True,
        aggregate_precision_passed=False,payoff_forecast_approved=False,
        publication_status="diagnostic_nonpublication")))
    artifacts={name:assessment.sha(tmp_path/name) for name in
               ("arrays.npz","curves.npz","bootstrap_draws.npz","summary.json")}
    (tmp_path/"validation.json").write_text(json.dumps(dict(integrity_passed=True,
        input_validation_passed=True,passed=True,scientific_precision_passed=False,
        payoff_forecast_approved=False,publication_status="diagnostic_nonpublication",
        artifact_sha256=artifacts)))
    with pytest.raises(ValueError,match="flags disagree"):
        assessment.check_output(tmp_path)

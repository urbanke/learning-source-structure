"""Shared-reference uncertainty, alphabet mapping and fail-closed diagnostics."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import merge_copying_payoff_dense as m


def moments():
    p=np.array([.3,.2]+[.5/62]*62)
    v=np.array([1.,-1.]+[0.]*62)*.0001
    cm=p+np.array([-1.,1.,-2.,2.])[:,None]*v
    covariance=np.outer(v,v)
    return dict(mean=p,chain_means=cm,cov_chain=covariance,cov_batch=2*covariance,
                mcse_probs=np.sqrt(2*np.diag(covariance)),chains=np.array(4),draws=np.array(16384))


def test_repeated_prefix_error_does_not_shrink_as_independent_sequences():
    ref=moments();r=np.eye(64)[0]
    one=m.shared_error(np.array([r]),['a'],{'a':ref})
    repeated=m.shared_error(np.tile(r,(256,1)),['a']*256,{'a':ref})
    assert repeated==pytest.approx(one)
    independent=m.shared_error(np.tile(r,(2,1)),['a','b'],{'a':ref,'b':ref})
    assert independent['numerical_mcse_bits']==pytest.approx(one['numerical_mcse_bits']/np.sqrt(2))
    assert independent['log_bias_proxy_bits']==pytest.approx(one['log_bias_proxy_bits'])


def test_gradients_combine_before_covariance_and_can_cancel():
    ref=moments();a=np.zeros(64);a[0]=1
    b=np.zeros(64);b[1]=1
    # Weight in ratio p0:p1 makes their aggregate derivative constant on
    # the only uncertain coordinates. Independent-per-use variance is wrong.
    oracle=np.array([a,a,a,b,b]);error=m.shared_error(oracle,['a']*5,{'a':ref})
    assert error['numerical_mcse_bits']<1e-12
    assert m.shared_error(oracle,list('abcde'),{k:ref for k in 'abcde'})['numerical_mcse_bits']>0
    assert error['log_bias_proxy_bits']>0  # Curvature need not cancel.


def test_common_bank_error_is_computed_after_mean_not_divided_by_population():
    loss=np.ones(256);leave=np.tile(np.array([.99,1.01,1.02,.98])[:,None],(1,256))
    result=m.bank_error(loss,leave)
    expected=np.sqrt(3/4*np.sum((leave[:,0]-1)**2))
    assert result['prior_bank_mcse_bits']==pytest.approx(expected)
    assert result['prior_bank_jackknife_bias_proxy_bits']==pytest.approx(0)


def test_raw_moments_match_and_preserve_equal_unseen_labels():
    ref=moments();estimate=dict(predictive_probs=ref['mean'].tolist())
    m.moments_checked(ref,estimate,2)
    ref['mean'][8]+=1e-4;ref['mean'][9]-=1e-4
    with pytest.raises(ValueError):m.moments_checked(ref,estimate,2)


@pytest.mark.parametrize('field',['draws','covariance','nonfinite','unseen_chain'])
def test_invalid_moment_diagnostics_rejected(field):
    ref=moments();estimate=dict(predictive_probs=ref['mean'].tolist())
    if field=='draws':ref['draws']=np.array(4096)
    elif field=='covariance':ref['cov_chain'][0,0]=-1
    elif field=='nonfinite':ref['mean'][0]=np.nan
    else:ref['chain_means'][0,5]+=.001;ref['chain_means'][0,6]-=.001
    with pytest.raises(ValueError):m.moments_checked(ref,estimate,2)


def test_mapping_preserves_last_context_and_all_unseen_successors():
    import hashlib
    tokens=np.array([41,19,41,7,19]);pattern,labels=m.canonicalize(tokens)
    law=np.arange(1,65,dtype=float);law/=law.sum()
    transition=np.full((64,64),1/64);transition[19]=law
    use=dict(reference_key=m.reference_key(pattern),canonical_to_original_seen=list(labels),
             original_prefix_sha256=hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest(),queried_original_label=19)
    mapping,oracle=m.map_history(tokens,transition,use,dict(tokens=list(pattern)))
    assert mapping[:3].tolist()==[41,19,7] and np.array_equal(oracle,law[mapping])
    prediction=moments()['mean'];prediction[2]=prediction[3]  # Map without renormalizing.
    original=np.empty(64);original[mapping]=prediction
    assert original[41]==prediction[0] and original[19]==prediction[1]
    assert -law@np.log2(original)==pytest.approx(-oracle@np.log2(prediction))
    bad=deepcopy(use);bad['queried_original_label']=7
    with pytest.raises(ValueError):m.map_history(tokens,transition,bad,dict(tokens=list(pattern)))


def test_missing_full_wave_returns_no_moments_or_available_case_mean(tmp_path,monkeypatch):
    tasks=[dict(id=f't{i}',seed=i) for i in range(76)]
    monkeypatch.setattr(m.dense,'load_plan',lambda folder:dict(observed=5,tasks=tasks))
    _,result,refs,bindings=m.load_references(tmp_path)
    assert not result['complete'] and len(result['missing_task_ids'])==76
    assert refs==bindings=={} and result['published_mean_gain_bits'] is None
    assert not result['aggregate_precision_passed'] and not result['payoff_forecast_approved']


@pytest.mark.parametrize('gate',['complete','individual_audits_passed','paired_orders_passed'])
def test_any_failed_required_gate_prevents_all_scoring(gate,tmp_path,monkeypatch):
    result=dict(complete=True,individual_audits_passed=True,paired_orders_passed=True)
    result[gate]=False
    monkeypatch.setattr(m,'load_references',lambda path:(dict(plan_id='bound'),result,{},{}))
    monkeypatch.setattr(m,'frozen_inputs',lambda plan:pytest.fail('Scoring incomplete/failed references'))
    out,arrays=m.merge(tmp_path)
    assert arrays=={} and not out[gate]


def test_sequence_variation_is_not_reported_as_mcmc_error():
    value=np.arange(256,dtype=float)
    out=m.stats(value)
    assert out['observed_sequence_se_bits']==pytest.approx(value.std(ddof=1)/16)
    assert 'numerical_mcse_bits' not in out
    with pytest.raises(ValueError):m.stats(value[:-1])


@pytest.mark.parametrize('complete,individual,paired,expected',[
    (False,False,False,3),(False,True,True,3),(True,False,True,2),
    (True,True,False,2),(True,True,True,0)])
def test_exit_status_distinguishes_coverage_audit_and_diagnostics(complete,individual,paired,expected):
    report=dict(complete=complete,individual_audits_passed=individual,paired_orders_passed=paired,
                diagnostic_arrays_validated=complete and individual and paired,
                population_reference_approved=False,payoff_forecast_approved=False)
    assert m.exit_status(report)==expected


def test_success_requires_validated_arrays_even_when_checks_pass():
    with pytest.raises(ValueError,match='not validated'):
        m.exit_status(dict(complete=True,individual_audits_passed=True,paired_orders_passed=True,
                           diagnostic_arrays_validated=False))

"""Context indices, complete-cohort gating and full-campaign multiplicity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import merge_copying_payoff_parallel as m


def reference():
    p = np.full(64,1/64); c = np.zeros((64,64))
    return dict(mean=p,chain_means=np.tile(p,(4,1)),cov_batch=c,cov_chain=c,
                mcse_probs=np.zeros(64))


def use_for(tokens):
    pattern,labels = m.old.canonicalize(tokens)
    return dict(reference_key=m.old.reference_key(pattern),
        canonical_to_original_seen=list(labels),queried_original_label=int(tokens[-1]),
        original_prefix_sha256=hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()),dict(tokens=list(pattern))


@pytest.mark.parametrize('observed',[6,7,8])
def test_context_mapping_uses_current_query_not_t5(observed):
    tokens = np.array([41,19,41,7,19,8,12,45])[:observed]
    use,task = use_for(tokens); law = np.arange(1,65,dtype=float); law /= law.sum()
    transition = np.full((64,64),1/64); transition[tokens[-1]] = law
    mapping,oracle = m.map_history(tokens,transition,use,task,observed)
    assert np.array_equal(oracle,law[mapping])
    assert mapping[:len(set(tokens))].tolist() == list(dict.fromkeys(tokens))
    original = np.empty(64); p = np.arange(64,0,-1,dtype=float); p /= p.sum(); original[mapping] = p
    assert -law@np.log2(original) == pytest.approx(-oracle@np.log2(p))
    wrong = deepcopy(use); wrong['queried_original_label'] = 19
    with pytest.raises(ValueError,match='history'):
        m.map_history(tokens,transition,wrong,task,observed)


def test_full_campaign_multiplicity_is_declared():
    assert m.FAMILY_PAIRS == 81+159+248
    expected = NormalDist().inv_cdf(1-.05/(2*488*64))
    assert m.Z_MULTIPLIER == pytest.approx(expected)
    assert m.Z_MULTIPLIER > m.dense.accuracy.Z_MULTIPLIER
    a = reference(); b = reference(); b['mean'][0] += 1e-5; b['mean'][1] -= 1e-5
    result = m.paired_order_check(a,b)
    assert not result['passed'] and result['failing_coordinates'] == [0,1]
    assert result['family_comparisons'] == 488*64


@pytest.mark.parametrize('field',['mean','mcse_probs'])
def test_paired_moments_reject_invalid_vectors(field):
    a,b = reference(),reference(); b[field][0] = np.nan
    with pytest.raises(ValueError):
        m.paired_order_check(a,b)


def test_incomplete_wave_never_scores_available_cases(tmp_path,monkeypatch):
    tasks = [dict(id=f't{i}',seed=i) for i in range(162)]
    plan = dict(observed=6,tasks=tasks,runtime=m.dense.pilot.runtime())
    monkeypatch.setattr(m,'campaign_binding',lambda *args:({},{}))
    monkeypatch.setattr(m.dense,'load_plan',lambda path:plan)
    monkeypatch.setattr(m,'sha',lambda path:'digest')
    _,result,refs,_ = m.load_references(tmp_path,tmp_path)
    assert not result['complete'] and len(result['missing_task_ids']) == 162 and refs == {}
    assert result['published_mean_gain_bits'] is None and not result['aggregate_precision_checked']
    assert m.old.exit_status(result) == 3


def test_wrong_runtime_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'campaign_binding',lambda *args:({},{}))
    monkeypatch.setattr(m.dense,'load_plan',lambda path:dict(observed=6,runtime={}))
    with pytest.raises(ValueError,match='runtime'):
        m.load_references(tmp_path,tmp_path)


def test_invalid_allocation_rejected_before_wave_read(tmp_path,monkeypatch):
    def refuse(*args):
        raise ValueError('Allocation differs')
    monkeypatch.setitem(sys.modules,'copying_payoff_parallel',SimpleNamespace(validate_wave=refuse))
    monkeypatch.setattr(m.dense,'load_plan',lambda path:pytest.fail('Read an unbound wave'))
    with pytest.raises(ValueError,match='Allocation'):
        m.load_references(tmp_path,tmp_path)


@pytest.mark.parametrize('gate',['complete','individual_audits_passed','paired_orders_passed'])
def test_failed_gate_prevents_all_scoring(gate,tmp_path,monkeypatch):
    report = dict(complete=True,individual_audits_passed=True,paired_orders_passed=True)
    report[gate] = False
    monkeypatch.setattr(m,'load_references',lambda *args:(dict(observed=7,plan_id='bound'),report,{},{}))
    monkeypatch.setattr(m.old,'frozen_inputs',lambda plan:pytest.fail('Scored a failed wave'))
    result,arrays = m.merge(tmp_path,tmp_path)
    assert not result[gate] and arrays == {}


def test_neural_scores_take_requested_time(tmp_path,monkeypatch):
    folder = tmp_path/'eval/ordered_seed0'; folder.mkdir(parents=True)
    protocol = dict(role='ordered',seed=0,evaluation_code_sha256='h',model_code_sha256='h')
    (folder/'protocol.json').write_text(json.dumps(protocol))
    (folder/'status.json').write_text(json.dumps(dict(complete=True,scientific_results=True)))
    np.savez(folder/'a1_L0.npz',protocol_sha256='h',source_tokens_sha256='tokens',
             times=np.arange(1,1025),expected_loss_bits=np.tile(np.arange(1,1025),(256,1)))
    monkeypatch.setattr(m.old,'SCORES',tmp_path); monkeypatch.setattr(m,'sha',lambda p:'h')
    bindings = {}
    for observed in (6,7,8):
        value = m.model_loss('a1_L0',0,'ordered',dict(tokens_sha256='tokens'),bindings,observed)
        assert np.array_equal(value,np.full(256,observed))
    assert len(bindings) == 3


class Archive(dict):
    def __enter__(self): return self
    def __exit__(self,*args): return False


@pytest.mark.parametrize('observed',[6,7,8])
def test_complete_merge_retains_every_history_and_stays_diagnostic(observed,tmp_path,monkeypatch):
    prefix = np.array([4,2,4,7,8,2,3,4])[:observed]; use,task = use_for(prefix)
    key = use['reference_key']; names = ('a1_L0','a1_L2','a1_L3')
    tokenrow = np.zeros(1025,dtype=int); tokenrow[:observed] = prefix
    tokens = np.tile(tokenrow,(256,1)); transition = np.full((256,64,64),1/64)
    stationary = np.tile(np.arange(1,9,dtype=float),(256,1))
    plan = dict(observed=observed,plan_id='bound',tasks=[dict(reference_key=key,**task)],
        uses=[dict(use,condition=name,index=i) for name in names for i in range(256)],
        source_bindings={name:dict(path=str(tmp_path/(name+'.npz')),sha256='h') for name in names})
    report = dict(complete=True,individual_audits_passed=True,paired_orders_passed=True,
        aggregate_precision_checked=False,aggregate_precision_passed=False,
        population_reference_approved=False,payoff_forecast_approved=False,published_mean_gain_bits=None)
    monkeypatch.setattr(m,'load_references',lambda *args:(plan,report,{16:{key:reference()},32:{key:reference()}},{}))
    monkeypatch.setattr(m.old,'frozen_inputs',lambda plan:({},{})); monkeypatch.setattr(m,'sha',lambda p:'h')
    def archive(path,**kwargs):
        if Path(path).parent == tmp_path:
            return Archive(tokens=tokens,transition=transition)
        return Archive(stationary_loss=stationary,
                       stationary_leave_one_bank_out_loss=np.tile(stationary,(16,1,1)))
    monkeypatch.setattr(m.np,'load',archive)
    monkeypatch.setattr(m,'model_loss',lambda name,seed,role,source,bindings,t:
                        np.full(256,t+(role == 'shuffled')))
    result,arrays = m.merge(tmp_path,tmp_path)
    assert result['diagnostic_arrays_validated'] and len(result['diagnostic_groups']) == 3
    assert not result['aggregate_precision_checked'] and not result['population_reference_approved']
    assert not result['payoff_forecast_approved'] and result['published_mean_gain_bits'] is None
    for group in result['diagnostic_groups']:
        assert group['n_sequences'] == 256
        assert group['rules']['32']['stationary_minus_bayes']['candidate_mean_bits'] == pytest.approx(observed-6)
        assert group['neural']['0']['gain']['candidate_mean_bits'] == pytest.approx(1)
        assert arrays[group['condition']+'_original_label_probs_a32'].shape == (256,64)
    plan['uses'].pop()
    with pytest.raises(ValueError,match='histories'):
        m.merge(tmp_path,tmp_path)

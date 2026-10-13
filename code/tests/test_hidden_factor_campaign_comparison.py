"""Reused-cohort isolation, paired scientific scores, immutable resumes and claims."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
import hidden_factor_campaign_comparison_eval as comparison
import train_hidden_factor as training
from train_mixture import digest_json, runtime_contract
from model import GPT

STRICT_LOAD_INPUTS = comparison.load_inputs


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    before = torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Small networks and short sequences, but the actual 256:4096 selection."""
    cfg = training.validate_config(dict(schema='hidden-factor-training-v1', seed=0,
        source_dimensions=[2,2], n_ctx=4, batch=2, steps=400000, warmup=1000,
        lr=.0003, wd=.1, clip=1., n_layer=1, n_head=1, d_model=8, val_n=2,
        log_every=100, val_every=5000, ckpt_every=10000, weight_steps=[200000,400000],
        validation_seed=123, val_chunk=1))
    config = {k:v for k,v in cfg.items() if k not in ('seed','source_dimensions')}
    model = GPT(4,4,1,1,8)
    plan = dict(source_dimensions=[2,2], implementation_sha256=training.implementation_hashes(),
        arms={'xlarge':dict(config=config,n_params=model.n_params(),milestones=[200000,400000])},
        primary={'final_steps':{'xlarge':400000}}, interpretation='Synthetic test of paired scoring.')
    tokens = np.random.default_rng(1729).integers(0,4,size=(4096,5),dtype=np.int64)
    times = np.array([1,4],dtype=np.int64)
    oracle = np.tile([.4,.3,.2,.1],(4096,2,1))
    bayes = comparison.previous.score_log_probabilities(np.tile(np.log2([.4,.3,.2,.1]),(4096,4,1)),tokens,oracle,times)
    uniform = comparison.previous.score_log_probabilities(np.full((4096,4,4),-2.),tokens,oracle,times)
    references = {'oracle_probs':oracle}
    for name in comparison.previous.REFERENCES:
        for metric in comparison.METRICS:
            references[name+'_'+metric]=(uniform if name=='flat_dirichlet' else bayes)[metric]
    old = dict(indices=np.arange(256,4096),times=times,reference_records_sha256='full-references',plan_sha256='original-plan',
        plan={'source_dimensions':[2,2], 'test':dict(cohort_id='cohort',data_sha256='tokens',times=[1,4])},
        references=references, frozen=dict(reference_hash='prefix-references',
            data=dict(tokens=tokens,stratum=np.arange(4096)%4),cohort={'stratum_labels':['a','b','c','d']}))
    parents={f'large_seed{s}':{m:uniform[m][256:].copy() for m in comparison.METRICS} for s in (0,1)}
    parent_receipts={f'large_seed{s}':dict(identity={'name':f'large_seed{s}'},data_sha256='original-'+str(s)) for s in (0,1)}
    inputs=dict(plan=plan,plan_sha256='campaign-plan',old=old,parents=parents,parent_receipts=parent_receipts,
                implementation_sha256=comparison.code_hashes())
    campaign=tmp_path/'campaign'
    for seed in (0,1):
        torch.manual_seed(13+seed); model=GPT(4,4,1,1,8)
        config_seed=dict(cfg,seed=seed,campaign_provenance=dict(plan_sha256='campaign-plan',arm='xlarge',seed=seed,
            trainer_sha256=comparison.sha(ROOT/'code/transformer/hidden_factor_campaign_train.py'),parent=None))
        contract=training.source_contract([2,2])
        validation=training.draw_validation([2,2],4,2,123)
        receipt=dict(config=config_seed,config_hash=digest_json(config_seed),model=model.cfg,n_params=model.n_params(),
            implementation_hashes=plan['implementation_sha256'],source_contract=contract,source_contract_hash=digest_json(contract),
            precision='float32',runtime=runtime_contract(torch.device('cpu')),
            validation_hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in validation.items()})
        path=campaign/f'train/xlarge/seed{seed}/model_200000.pt';path.parent.mkdir(parents=True)
        torch.save(dict(schema='hidden-factor-weights-v1',step=200000,model=model.state_dict(),cfg=receipt,
            completed_sequences=400000,config_hash=receipt['config_hash'],implementation_hashes=plan['implementation_sha256']),path)
        dest=campaign/f'eval/xlarge/seed{seed}/step200000';dest.mkdir(parents=True)
        comparison.original.save_npz(dest/'scores.npz',{'test':np.arange(256)})
        identity=dict(plan_sha256='campaign-plan',arm='xlarge',seed=seed,step=200000,test_role='existing_development_prefix',
            n_sequences=256,cohort_id='cohort',times=[1,4],evaluator_sha256=comparison.sha(comparison.campaign_eval.__file__),
            reference_records_sha256='prefix-references',checkpoint_sha256=comparison.sha(path))
        comparison.original.save_json(dest/'summary.json',dict(identity=identity,complete=True,
            data_sha256=comparison.sha(dest/'scores.npz'),causality_check=dict(passed=True,max_absolute_logit_difference=0.)))
    monkeypatch.setattr(comparison,'load_inputs',lambda *a:inputs)
    return dict(inputs=inputs,campaign=campaign,out=tmp_path/'comparison',root=tmp_path)


def run(e, **kwargs):
    return comparison.run('unused',e['root']/'base',e['root']/'scaling',e['root']/'original-comparison',
                          e['campaign'],e['out'],'xlarge',200000,device='cpu',batch_size=128,**kwargs)


def test_real_inference_pairing_uncertainty_and_completed_model_reuse(saved,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('No training or new source/reference calculation')
    monkeypatch.setattr(training,'train',forbidden)
    monkeypatch.setattr(comparison.original,'create_cohort',forbidden)
    monkeypatch.setattr(comparison.original,'reference_one',forbidden)
    before={str(p):p.read_bytes() for p in saved['campaign'].rglob('*') if p.is_file()}
    result=run(saved)
    assert result['complete'] and result['evaluated_n_sequences']==3840 and result['test_role']=='reused_comparison'
    assert not result['final_checkpoint'] and not result['fresh_confirmation'] and not result['new_bayesian_computation']
    assert result['sequence_start']==256 and result['sequence_stop']==4096 and len(result['groups'])==5
    assert set(result['model_receipts'])=={'xlarge_seed0','xlarge_seed1'}
    with np.load(saved['out']/'models/xlarge_seed0.npz') as z:
        np.testing.assert_array_equal(z['sequence_indices'],np.arange(256,4096))
        oracle=saved['inputs']['old']['references']['oracle_probs'][256:]
        expected=(oracle*np.log2(oracle/z['predictive'])).sum(-1)
        np.testing.assert_allclose(z['next_kl_bits'],expected,atol=1e-12)
        difference=expected-saved['inputs']['parents']['large_seed0']['next_kl_bits']
        actual=result['groups'][0]['paired']['xlarge_seed0_minus_large_seed0']['next_kl_bits']
        np.testing.assert_allclose(actual['mean'],difference.mean(0),atol=1e-12)
        np.testing.assert_allclose(actual['se'],difference.std(0,ddof=1)/np.sqrt(3840),atol=1e-12)
    assert before=={str(p):p.read_bytes() for p in saved['campaign'].rglob('*') if p.is_file()}
    monkeypatch.setattr(comparison,'GPT',forbidden)
    assert run(saved)['groups']==result['groups']


@pytest.mark.parametrize('change',['weights','config','development'])
def test_checkpoint_and_development_provenance_cannot_drift(saved,change):
    path=saved['campaign']/'train/xlarge/seed0/model_200000.pt'
    pilot=saved['campaign']/'eval/xlarge/seed0/step200000/summary.json'
    if change=='development':
        receipt=json.loads(pilot.read_text());receipt['identity']['n_sequences']=255
        comparison.original.save_json(pilot,receipt)
    else:
        weights=torch.load(path,weights_only=False)
        if change=='weights': weights['model']['head.weight'][0,0]+=.1
        else: weights['config_hash']='changed'
        torch.save(weights,path)
        if change=='config':
            # Updating the file hash cannot get around the independent contract checker.
            receipt=json.loads(pilot.read_text());receipt['identity']['checkpoint_sha256']=comparison.sha(path)
            comparison.original.save_json(pilot,receipt)
    with pytest.raises(ValueError,match='Weights changed|configuration hash|development evaluation identity'):
        run(saved)
    assert not (saved['out']/'summary.json').exists()


@pytest.mark.parametrize('change',['pairing','metric','bytes'])
def test_saved_arrays_are_replayed_before_reuse(saved,change):
    run(saved)
    path=saved['out']/'models/xlarge_seed0.npz'
    if change=='bytes':
        with path.open('ab') as f:f.write(b'changed')
    else:
        with np.load(path) as z: arrays={k:z[k] for k in z.files}
        if change=='pairing':arrays['sequence_indices'][0]=0
        else:arrays['next_kl_bits'][0,0]+=.1
        comparison.original.save_npz(path,arrays)
        receipt=json.loads(path.with_suffix('.json').read_text());receipt['data_sha256']=comparison.sha(path)
        comparison.original.save_json(path.with_suffix('.json'),receipt)
    with pytest.raises(ValueError,match='pairing|replay|checksum'):
        run(saved)


def test_selector_and_output_cannot_silently_switch(saved):
    for arm,step in [('continue_large',200000),('xlarge',100000),('unplanned',400000)]:
        with pytest.raises(ValueError,match='Select'):comparison.selection(arm,step)
    run(saved)
    with pytest.raises(ValueError,match='selection or implementation changed'):
        comparison.run('unused',saved['root']/'base',saved['root']/'scaling',saved['root']/'original-comparison',
            saved['campaign'],saved['out'],'xlarge',400000,device='cpu')
    with pytest.raises(ValueError,match='separate'):
        comparison.protect_output(saved['campaign']/'nested',[saved['campaign']])


def test_frozen_range_and_original_parent_binding(saved,monkeypatch):
    inputs=saved['inputs'];old=inputs['old'];plan=deepcopy(inputs['plan'])
    plan['parent']=dict(plan_path='unused',checkpoint_sha256={'0':'parent0','1':'parent1'})
    plan['evaluation']=dict(development_prefix=256,planned_cohort=4096,cohort_id='cohort',
                            cohort_data_sha256='tokens',times=[1,4])
    old['pilots']={f'large_seed{s}':{'checkpoint_sha256':f'parent{s}'} for s in (0,1)}
    monkeypatch.setattr(comparison,'read_plan',lambda path:(plan,'campaign-plan'))
    monkeypatch.setattr(comparison.previous,'load_inputs',lambda *a:old)
    monkeypatch.setattr(comparison.previous,'_load_model_result',
        lambda _,name,__:(inputs['parents'][name],inputs['parent_receipts'][name]))
    assert STRICT_LOAD_INPUTS('unused','unused','unused','unused')['old'] is old
    old['indices'][0]=0
    with pytest.raises(ValueError,match='exactly the existing indices'):
        STRICT_LOAD_INPUTS('unused','unused','unused','unused')
    old['indices'][0]=256;old['pilots']['large_seed0']['checkpoint_sha256']='changed'
    with pytest.raises(ValueError,match='another parent'):
        STRICT_LOAD_INPUTS('unused','unused','unused','unused')


def test_slurm_atomic_claim_refuses_live_owner_and_allows_terminal_retry(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'code/scripts'))
    import campaign_job_guard as guard
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(guard,'confirmed_terminal',lambda job:False)
    with comparison.output_claim(tmp_path/'claim','xlarge'):
        with pytest.raises(guard.GuardError):
            with comparison.output_claim(tmp_path/'claim','xlarge'): pass
        with pytest.raises(guard.GuardError):
            with comparison.output_claim(tmp_path/'claim','continue_large'): pass
    monkeypatch.setenv('SLURM_JOB_ID','124')
    monkeypatch.setattr(guard,'confirmed_terminal',lambda job:job=='123')
    with comparison.output_claim(tmp_path/'claim','xlarge'): pass
    claims=sorted((tmp_path/'claim/.campaign_claims/hidden_factor/eval/xlarge/0').glob('attempt-*'))
    assert len(claims)==2
    assert [json.loads((p/'allocation.json').read_text())['job_id'] for p in claims]==['123','124']


def test_launcher_is_lightweight_explicit_and_print_only(tmp_path):
    launcher=ROOT/'code/transformer/kuma/submit_hidden_factor_campaign_comparison.sh'
    out=tmp_path/'new output'
    command=['bash',str(launcher),'--arm','xlarge','--step','200000','--out',str(out)]
    completed=subprocess.run(command,capture_output=True,text=True)
    assert completed.returncode==0 and not out.exists()
    words=shlex.split(completed.stdout)
    assert '--output='+str(out/'xlarge/step200000/logs/comparison-%j.out') in words
    assert words[-3:]==['xlarge','200000','8']
    for extra in [('--step','100000'),('--out',str(ROOT/'code/unwanted')),('--batch-size','0')]:
        assert subprocess.run(command+list(extra),capture_output=True).returncode!=0

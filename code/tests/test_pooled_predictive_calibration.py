"""Covariance, frozen provenance and completeness; no full-prior local SMC."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'code/scripts/pooled_predictive_calibration.py'
spec = importlib.util.spec_from_file_location('pooled_predictive_calibration', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def reference(probabilities, z=.04):
    p = np.asarray(probabilities)
    return dict(log_evidence=math.log(z), predictive_probs=p.tolist(),
                log_joint_next=(math.log(z)+np.log(p)).tolist())


def records(z, p):
    return [dict(log_evidence=math.log(a), predictive_probs=list(b)) for a,b in zip(z,p)]


def test_log_ratio_mcse_cancels_common_evidence_noise():
    values = records([.01,.04,.02,.07], [[.2,.8]]*4)
    report = c.summarize(values, reference([.2,.8]), 4)
    assert report['evidence_loss_mcse_bits'] > .1
    np.testing.assert_allclose(report['predictive_log_mcse_bits'], [0,0], atol=1e-15)
    assert not report['passed']  # Four replicates cannot pass the protocol.


def test_evidence_weighted_vector_and_covariant_mcse_match_independent_derivative():
    z = np.array([.01,.03,.02,.04])
    p = np.array([[.9,.1],[.1,.9],[.5,.5],[.2,.8]])
    joint = z[:,None]*p
    wanted = joint.mean(0)/z.mean()
    report = c.summarize(records(z,p), reference(wanted, z.mean()), 4)
    np.testing.assert_allclose(report['pooled']['predictive_probs'], wanted, atol=1e-15)
    assert not np.allclose(wanted, p.mean(0))
    eps = 1e-5
    directions=[]
    for zz,jj in zip(z,joint):
        plus=np.log2((joint.mean(0)+eps*(jj-joint.mean(0)))/(z.mean()+eps*(zz-z.mean())))
        minus=np.log2((joint.mean(0)-eps*(jj-joint.mean(0)))/(z.mean()-eps*(zz-z.mean())))
        directions.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(report['predictive_log_mcse_bits'],
                               np.std(directions,axis=0,ddof=1)/2, rtol=1e-9)


def test_every_coordinate_gate_rejects_normalized_but_wrong_prediction():
    p=[.501,.499]
    report=c.summarize(records([.04]*32,[p]*32), reference([.5,.5]), 32)
    assert report['passed']
    report=c.summarize(records([.04]*32,[[.51,.49]]*32), reference([.5,.5]), 32)
    assert report['checks']['evidence_precision']
    assert not report['checks']['every_predictive_log_probability']
    assert not report['passed']


def test_plan_cases_full_prior_and_exact_catalog_are_frozen(tmp_path):
    plan,digest=c.read_plan(c.PLAN)
    assert len(plan['cases'])==16
    assert plan['cases'][0]['tokens']==[0]
    assert [0,1,0,2] in [x['tokens'] for x in plan['cases']]
    assert plan['numerics']['depths']==list(range(9))
    assert plan['numerics']['alpha_bounds']==[.5,1.5]
    assert c.task_spec(plan,'initial',127)['replica_stop']==32
    assert c.task_spec(plan,'stronger',255)['replica_stop']==64
    with pytest.raises(ValueError,match='outside'):
        c.task_spec(plan,'initial',128)
    catalog=c.SharedBaseShortEvidence()
    assert catalog.validation['passed']
    for case in plan['cases']:
        ref=c.exact_reference(catalog,case['tokens'])
        np.testing.assert_allclose(np.exp(ref['log_joint_next']).sum(),
                                   math.exp(ref['log_evidence']),rtol=1e-9)
    modified=copy.deepcopy(plan); modified['criteria']['tolerance_bits']=.02
    path=tmp_path/'changed.json'; path.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match='settings changed'):c.read_plan(path)
    modified=copy.deepcopy(plan); modified['reference_sha256'][c.REFERENCES[0]]='wrong'
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match='Changed input'):c.read_plan(path)


def fake_estimate(plan,attempt,case,replica,catalog=None):
    catalog=catalog or c.SharedBaseShortEvidence()
    ref=c.exact_reference(catalog,case['tokens'])
    settings=plan['attempts'][attempt]
    return dict(schema='shared-base-predictive-smc-repetition-v1',n_observed=len(case['tokens']),
        **ref, particles=settings['particles'],temperatures=settings['temperatures'],moves=settings['moves'],
        seed=c.replica_seed(plan,attempt,case['id'],replica),resample_fraction=.75,
        maximum_conditional_normalization_error=0.,sampler_seconds=1.,prediction_seconds=1.,elapsed_seconds=2.,
        effective_sample_sizes=[settings['particles']]*settings['temperatures'],scientific_reference_validated=False)


def wrapped(plan,attempt,case,replica,catalog=None):
    value=fake_estimate(plan,attempt,case,replica,catalog)
    return dict(case_id=case['id'],replica=replica,estimate=value,estimate_sha256=c.digest(value))


def test_live_tiny_sampler_output_matches_record_contract():
    # Real API/schema test with64probabilities; depth0only avoids backend work.
    # This is not the full-prior scientific calibration.
    plan,_=c.read_plan(c.PLAN)
    plan=copy.deepcopy(plan)
    plan['attempts']['initial']=dict(particles=8,temperatures=2,moves=1,replicates=32)
    case=plan['cases'][0]
    sampler=c.SharedBaseMCMC(64,c.RankedLSARowEvidence(64,[.6,1.3],depths=[0]),[.5,.5])
    value=c.estimate(sampler,case['tokens'],particles=8,temperatures=2,moves=1,
                     seed=c.replica_seed(plan,'initial',case['id'],0),seconds_limit=30.)
    record=dict(case_id=case['id'],replica=0,estimate=value,estimate_sha256=c.digest(value))
    c.validate_record(record,plan,'initial',case,0)
    assert len(value['predictive_probs'])==64
    assert value['log_evidence']==pytest.approx(-math.log(64),abs=1e-14)


def test_saved_replica_hash_and_seed_are_checked():
    plan,_=c.read_plan(c.PLAN); case=plan['cases'][0]
    record=wrapped(plan,'initial',case,0)
    c.validate_record(record,plan,'initial',case,0)
    modified=copy.deepcopy(record); modified['estimate']['predictive_probs'][0]*=2
    with pytest.raises(ValueError,match='replica changed'):
        c.validate_record(modified,plan,'initial',case,0)
    with pytest.raises(ValueError,match='identity changed'):
        c.validate_record(record,plan,'initial',case,1)
    assert c.replica_seed(plan,'initial',case['id'],0)!=c.replica_seed(plan,'stronger',case['id'],0)


@pytest.fixture
def no_dependency_work(monkeypatch):
    monkeypatch.setattr(c,'dependencies',lambda *args:{'synthetic_test_fixture':'unchanged'})
    monkeypatch.setattr(c,'numerical_environment',lambda *args:{})


def test_resume_preserves_completed_replicates(tmp_path,monkeypatch,no_dependency_work):
    plan,_=c.read_plan(c.PLAN); case=plan['cases'][0]; catalog=c.SharedBaseShortEvidence()
    monkeypatch.setattr(c,'RankedLSARowEvidence',lambda *args,**kw:None)
    monkeypatch.setattr(c,'SharedBaseMCMC',lambda *args,**kw:None)
    calls=[]
    def sample(_sampler,tokens,**kw):
        calls.append(kw['seed'])
        if len(calls)==2:raise TimeoutError('test interruption')
        replica=next(i for i in range(4) if c.replica_seed(plan,'initial',case['id'],i)==kw['seed'])
        return fake_estimate(plan,'initial',case,replica,catalog)
    monkeypatch.setattr(c,'estimate',sample)
    assert c.run(c.PLAN,'initial',0,tmp_path,tmp_path,tmp_path)==75
    path=tmp_path/'initial/shards/task-0000.json'
    first=json.loads(path.read_text())['records'][0]
    assert c.run(c.PLAN,'initial',0,tmp_path,tmp_path,tmp_path)==0
    result=json.loads(path.read_text())
    assert result['records'][0]==first and result['complete']
    assert len(calls)==5 and calls[1]==calls[2]
    assert c.run(c.PLAN,'initial',0,tmp_path,tmp_path,tmp_path)==0
    assert len(calls)==5


def test_merge_requires_every_shard_and_rejects_tampering(tmp_path,no_dependency_work):
    plan,digest=c.read_plan(c.PLAN); catalog=c.SharedBaseShortEvidence()
    provenance=c.identity(plan,digest,'initial',{'synthetic_test_fixture':'unchanged'},{})
    for index in range(128):
        task=c.task_spec(plan,'initial',index)
        report=dict(schema='pooled-predictive-exact-short-shard-v1',identity=provenance,task=task,
                    exact_reference=c.exact_reference(catalog,task['case']['tokens']),complete=True,status='completed',
                    records=[wrapped(plan,'initial',task['case'],r,catalog)
                             for r in range(task['replica_start'],task['replica_stop'])],scientific_reference_validated=False)
        c.atomic_json(tmp_path/'initial/shards'/f'task-{index:04d}.json',report)
    result=c.merge(c.PLAN,'initial',tmp_path,tmp_path,tmp_path)
    assert result['complete'] and result['exact_short_passed']
    assert not result['scientific_reference_validated'] and not result['population_reference_validated']
    last=tmp_path/'initial/shards/task-0127.json'; data=last.read_bytes(); last.unlink()
    result=c.merge(c.PLAN,'initial',tmp_path,tmp_path,tmp_path)
    assert not result['complete'] and not result['exact_short_passed'] and result['missing_tasks']==[127]
    altered=json.loads(data); altered['records'][0]['estimate']['log_evidence']+=.1
    c.atomic_json(last,altered)
    with pytest.raises(ValueError,match='replica changed'):
        c.merge(c.PLAN,'initial',tmp_path,tmp_path,tmp_path)


def test_writer_claim_excludes_duplicates_and_is_not_stolen(tmp_path):
    path=tmp_path/'task.json'
    with c.writer_claim(path):
        with pytest.raises(RuntimeError,match='Writer claim exists'):
            with c.writer_claim(path):pass
    assert not path.with_suffix('.writer').exists()


def test_launcher_dry_run_and_selected_preflight_task():
    launcher=ROOT/'code/transformer/scitas/submit_pooled_predictive_calibration.sh'
    def command(*extra):
        result=subprocess.run(['bash',str(launcher),*extra],text=True,capture_output=True,check=True)
        return result.stdout
    assert '--array=0-127%64' in command('--dry-run')
    assert '--array=0-255%64' in command('--attempt','stronger')
    assert '--array=0%64' in command('--only-task','0')
    merged=command('--merge','--after-job','12345')
    assert '--dependency=afterany:12345' in merged and '--array=' not in merged
    batch=(ROOT/'code/transformer/scitas/pooled_predictive_calibration.sbatch').read_text()
    assert '#SBATCH --time=02:00:00' in batch and '#SBATCH --cpus-per-task=1' in batch

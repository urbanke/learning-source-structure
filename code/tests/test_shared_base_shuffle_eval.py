"""Independent scoring, pairing, crossing, and deployment checks for the control."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
import shared_base_shuffle_eval as evaluation


def test_expected_losses_match_explicit_transition_sum_and_targets():
    tokens=np.array([[0,1,0,1],[1,1,0,0]])
    matrix=np.array([[[.8,.2],[.3,.7]],[[.4,.6],[.9,.1]]])
    probs=np.array([[[.6,.4],[.2,.8],[.7,.3]],[[.2,.8],[.6,.4],[.1,.9]]])
    result=evaluation.expected_losses(np.log(probs),tokens,matrix)
    for i in range(2):
        for time in range(3):
            expected=sum(-matrix[i,tokens[i,time],b]*np.log2(probs[i,time,b]) for b in range(2))
            entropy=sum(-matrix[i,tokens[i,time],b]*np.log2(matrix[i,tokens[i,time],b]) for b in range(2))
            assert result['expected_loss_bits'][i,time]==pytest.approx(expected)
            assert result['regret_bits'][i,time]==pytest.approx(expected-entropy)
            assert result['realized_loss_bits'][i,time]==pytest.approx(-np.log2(probs[i,time,tokens[i,time+1]]))


def test_identical_predictors_have_exactly_zero_gap_and_no_crossing():
    values=np.zeros((4,7));counts=np.array([[0,1,2,1],[4,0,0,0],[1,1,1,1]],float)
    summary=evaluation.paired_summary(values,counts)
    assert summary['mean']==[0.]*7
    assert summary['lower95']==summary['upper95']==[0.]*7
    assert summary['crossing']['first_upcrossing'] is None


def test_no_forced_delayed_crossing_and_multiple_crossings_retained():
    assert evaluation.crossing_description([1,2,3])['status']=='positive_from_first_scored_token'
    assert evaluation.crossing_description([-2,-1,0])['first_upcrossing'] is None
    report=evaluation.crossing_description([-2,-1,1,-1,1])
    assert report['all_upcrossings']==[3,5]
    assert report['all_downcrossings']==[4]


def test_cumulative_uncertainty_preserves_within_sequence_dependence():
    # Perfect anticorrelation across times makes total uncertainty zero.
    differences=np.array([[-2,2],[-1,1],[1,-1],[2,-2]],float)
    counts=np.array([[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]],float)
    report=evaluation.paired_summary(differences.cumsum(1),counts)
    assert report['lower95'][-1]==report['upper95'][-1]==report['se'][-1]==0


def test_source_replay_matches_original_and_depth_zero_is_iid():
    arrays=evaluation.source_arrays(0,[0,1],20260930,d=4,n_tokens=17)
    assert np.array_equal(arrays['transition'],np.broadcast_to(arrays['base'][:,None,:],(2,4,4)))
    assert np.array_equal(arrays['tokens'],evaluation.source_arrays(0,[0,1],20260930,d=4,n_tokens=17)['tokens'])


def test_prepare_rejects_changed_source_hash(tmp_path):
    manifest=dict(schema=evaluation.SCHEMA,source_code_sha256='wrong')
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Source implementation'):
        evaluation.prepare(p,tmp_path/'cohorts')


def test_prepare_reuses_transferred_exact_arrays_without_regeneration(tmp_path,monkeypatch):
    arrays=evaluation.source_arrays(0,[0,1],20260930,d=4,n_tokens=17)
    manifest=dict(schema=evaluation.SCHEMA,source_code_sha256=evaluation.sha(ROOT/'code/transformer/source.py'),
                  d=4,n_tokens=17,master_seed=20260930,
                  conditions=[dict(id='frozen',depth=0,indices=[0,1],arrays_sha256=evaluation.checksums(arrays))])
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    folder=tmp_path/'cohorts';folder.mkdir()
    evaluation.save_npz(folder/'frozen.npz',**arrays,indices=np.array([0,1]),manifest_sha256=evaluation.sha(p))
    def no_regeneration(*args,**kwargs):
        raise AssertionError('Transferred cohorts must be reused, not generated again')
    monkeypatch.setattr(evaluation,'source_arrays',no_regeneration)
    evaluation.prepare(p,folder)
    arrays['transition'][0,0,0]=np.nextafter(arrays['transition'][0,0,0],1.)
    evaluation.save_npz(folder/'frozen.npz',**arrays,indices=np.array([0,1]),manifest_sha256=evaluation.sha(p))
    with pytest.raises(ValueError,match='arrays differ'):
        evaluation.prepare(p,folder)


def test_control_must_name_the_actual_ordered_training_pair(tmp_path):
    plan=dict(config={'steps':100000},source={'d':64},implementation_sha256={'code':'frozen'},
              shuffle_contract={'law':'frozen'},ordered_runs={'0':{'files_sha256':{'model_100000.pt':'ordered'}}})
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    cfg=dict(config={'schema':'shared-base-shuffle-training-v1','steps':100000,'d':64,'seed':0},
             model=dict(vocab=64,n_ctx=1024,n_layer=4,n_head=8,d_model=256,mlp=True,dropout=0.),
             n_params=3454464,control='independent-tail-position-shuffle',source_contract=plan['shuffle_contract'],
             plan_sha256=evaluation.sha(plan_path),implementation_hashes=plan['implementation_sha256'],
             ordered_training={'original_files_sha256':{'model_100000.pt':'ordered'}})
    (tmp_path/'status.json').write_text(json.dumps(dict(complete=True,step=100000,completed_sequences=6400000)))
    saved=dict(cfg=cfg,step=100000)
    evaluation.validate_checkpoint(saved,cfg,0,'shuffled',checkpoint=tmp_path/'model_100000.pt',plan_path=plan_path)
    cfg['ordered_training']['original_files_sha256']['model_100000.pt']='different'
    with pytest.raises(ValueError,match='paired with another'):
        evaluation.validate_checkpoint(saved,cfg,0,'shuffled',checkpoint=tmp_path/'model_100000.pt',plan_path=plan_path)


def test_end_to_end_smoke_marked_nonscientific(tmp_path):
    evaluation.smoke(tmp_path)
    report=json.loads((tmp_path/'summary/summary.json').read_text())
    assert report['scientific_results'] is False
    assert len(report['groups'])==1
    assert (tmp_path/'summary/gap_seed0.png').is_file()
    assert (tmp_path/'summary/regret_seed0.png').is_file()
    assert np.isfinite(report['groups'][0]['cumulative']['mean']).all()


def test_eval_launcher_dependency_is_compute_node_only_and_dry(tmp_path):
    command=['bash',str(ROOT/'code/transformer/kuma/submit_shared_base_shuffle_eval.sh'),
             '--out',str(tmp_path/'output'),'--after-job','12345']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--dependency=afterok:12345' in result.stdout
    assert '--kill-on-invalid-dep=yes' in result.stdout
    assert not (tmp_path/'output').exists()
    invalid=subprocess.run(command[:-1]+['invalid'],capture_output=True,text=True)
    assert invalid.returncode!=0

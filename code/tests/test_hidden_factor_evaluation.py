"""Exact-reference arithmetic, held-out alignment and paired evaluation checks."""
from pathlib import Path
import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transformer'))
import hidden_factor as source
import hidden_factor_gpu as gpu
import eval_hidden_factor as evaluation
import hidden_factor_cache as cache
import train_hidden_factor as training


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('dimensions',[(2,2),(2,3),(3,3)])
def test_gamma_selected_matches_independent_sequential_reference(tmp_path,dimensions):
    xs=source.draw_batch(np.random.default_rng(113),1,65,dimensions)['xs'][0]
    times=[1,4,16,64]
    expected=source.exact_curve(xs,times,dimensions)
    actual=gpu.exact_selected(xs,dimensions,times,chunk_size=257)
    np.testing.assert_allclose(actual['predictive'],expected['predictive'],atol=2e-12,rtol=0)
    np.testing.assert_allclose(actual['log_evidence_bits'],expected['log_evidence_bits'][actual['evidence_times']],atol=2e-11,rtol=0)
    np.testing.assert_allclose(actual['cumulative_loss_bits'],-expected['log_evidence_bits'][np.array(times)+1]-np.log2(np.prod(dimensions)),atol=2e-11,rtol=0)
    path=tmp_path/'encodings.npy'
    cache.prepare_encoding_cache(path,dimensions,chunk_size=193)
    cached=cache.open_encoding_cache(path,dimensions)
    other=gpu.exact_selected(xs,dimensions,times,chunk_size=113,encoding_cache=cached)
    np.testing.assert_allclose(other['predictive'],actual['predictive'],atol=2e-12,rtol=0)
    np.testing.assert_allclose(other['cumulative_loss_bits'],actual['cumulative_loss_bits'],atol=2e-11,rtol=0)


def test_streaming_control_causality_and_symbol_relabelling():
    dimensions=(2,3);times=[1,4,16]
    xs=source.draw_batch(np.random.default_rng(115),1,17,dimensions)['xs'][0]
    selected=gpu.exact_selected(xs,dimensions,times,chunk_size=17)
    streamed=gpu.exact_streaming(xs,dimensions,times,chunk_size=13)
    np.testing.assert_allclose(selected['predictive'],streamed['predictive'],atol=1e-12,rtol=0)
    changed=xs.copy();changed[4:]=(changed[4:]+1)%6
    causal=gpu.exact_selected(changed,dimensions,times,chunk_size=17)
    np.testing.assert_array_equal(causal['predictive'][:2],selected['predictive'][:2])
    perm=np.array([4,1,3,0,5,2])
    relabelled=gpu.exact_selected(perm[xs],dimensions,times,chunk_size=17)
    np.testing.assert_allclose(relabelled['predictive'][:,perm],selected['predictive'],atol=1e-12,rtol=0)


def test_reference_bounds_memory_and_timeout():
    xs=np.array([0,1,0,2,4],dtype=np.int64)
    with pytest.raises(TimeoutError):gpu.exact_selected(xs,(2,3),[1,4],max_seconds=1e-12,chunk_size=1)
    with pytest.raises(ValueError):gpu.exact_selected(xs,(2,3),[1,5])
    with pytest.raises(ValueError):gpu.exact_selected(xs,(2,3),[1,4],encoding_cache=np.zeros((3,6),dtype=np.uint8))
    assert gpu.memory_estimate((2,9),1024)==gpu.memory_estimate((2,9),1024)
    assert gpu.memory_estimate((2,9),1024)<gpu.memory_estimate((2,9),2048)


def test_frozen_cohort_entropy_strata_and_changed_plan(tmp_path):
    record=evaluation.create_cohort(tmp_path,(2,3),n_seq=40,length=17,times=[1,4,16])
    assert evaluation.create_cohort(tmp_path,(2,3),n_seq=40,length=17,times=[1,4,16])==record
    _,data=evaluation.load_cohort(tmp_path)
    entropy,medians,strata=evaluation.entropy_strata((data['first_matrix'],data['second_matrix']))
    expected=(entropy[:,0]>=medians[0])*2+(entropy[:,1]>=medians[1])
    np.testing.assert_array_equal(strata,expected)
    assert (entropy>=0).all() and (entropy<=1).all()
    assert record['entropy_medians']==medians.tolist()
    with pytest.raises(ValueError):evaluation.create_cohort(tmp_path,(2,3),n_seq=40,length=17,times=[1,4,16],seed=1)
    assert not record['identity']['production_size']


def tiny_config(seed=0):
    return dict(schema='hidden-factor-training-v1',seed=seed,source_dimensions=[2,3],
        n_ctx=16,batch=3,steps=3,warmup=1,lr=3e-4,wd=.1,clip=1.,
        n_layer=1,n_head=1,d_model=8,val_n=2,log_every=3,val_every=3,
        ckpt_every=3,val_chunk=1,weight_steps=[3])


def test_complete_paired_pipeline_and_checkpoint_guards(tmp_path):
    out=tmp_path/'eval';run=tmp_path/'train'
    evaluation.create_cohort(out,(2,3),n_seq=12,length=17,times=[1,4,16])
    evaluation.references(out,device='cpu',chunk_size=17,cache_root=tmp_path/'cache')
    bayes=evaluation.summarize(out,bayes_only=True)
    assert bayes['missing_training_seeds']==[0,1]
    training.train(tiny_config(),run,device='cpu',workers=0)
    checkpoint=run/'model_3.pt'
    evaluation.evaluate_model(out,checkpoint,device='cpu',batch_size=4)
    summary=evaluation.summarize(out)
    assert summary['present_training_seeds']==[0] and summary['missing_training_seeds']==[1]
    assert not summary['two_repeat_comparison_complete']
    record,data=evaluation.load_cohort(out)
    refs,_=evaluation.read_complete(out,'references',12,record['cohort_id'])
    models,_=evaluation.read_complete(out,Path('models')/'seed0',12,record['cohort_id'])
    full=next(g for g in summary['groups'] if g['id']=='full_prior')
    diff=models['cumulative_loss_bits']-refs['bayes_hidden_cumulative_loss_bits']
    stats=full['paired']['seed0_minus_bayes_hidden']['cumulative_loss_bits']
    np.testing.assert_allclose(stats['mean'],diff.mean(0),rtol=0,atol=1e-14)
    np.testing.assert_allclose(stats['se'],diff.std(0,ddof=1)/np.sqrt(12),rtol=0,atol=1e-14)
    known=evaluation.moments(refs['bayes_hidden_cumulative_loss_bits']-refs['bayes_encoding_given_cumulative_loss_bits'])
    assert full['paired']['bayes_hidden_minus_bayes_encoding_given']['cumulative_loss_bits']==known
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    bad=copy.deepcopy(ckpt);bad['config_hash']='bad'
    with pytest.raises(ValueError,match='configuration hash'):evaluation.validate_checkpoint(bad,record['identity'])
    bad=copy.deepcopy(ckpt);bad['completed_sequences']=1
    with pytest.raises(ValueError,match='sample count'):evaluation.validate_checkpoint(bad,record['identity'])
    bad=copy.deepcopy(ckpt);bad['implementation_hashes']['hidden_factor.py']='bad'
    with pytest.raises(ValueError,match='differs|receipt'):evaluation.validate_checkpoint(bad,record['identity'])
    # A smoke checkpoint cannot be promoted to a production reference claim.
    production=dict(record['identity'],production_size=True)
    with pytest.raises(ValueError,match='production evaluation'):evaluation.validate_checkpoint(ckpt,production)
    # All records resume unchanged and summaries retain the same paired values.
    evaluation.references(out,device='cpu',chunk_size=17,cache_root=tmp_path/'cache')
    evaluation.evaluate_model(out,checkpoint,device='cpu',batch_size=4)
    assert evaluation.summarize(out)==summary


def test_pilot_then_models_then_full_reuses_frozen_paired_records(tmp_path,monkeypatch):
    out=tmp_path/'eval';cache_root=tmp_path/'cache';planned=24;pilot=8
    record=evaluation.create_cohort(out,(2,3),n_seq=planned,length=17,times=[1,4,16])
    cohort_hash=evaluation.file_hash(out/'cohort.npz')
    original_reference=evaluation.reference_one;computed=[]
    def counted_reference(data,index,*args,**kwargs):
        computed.append(index)
        return original_reference(data,index,*args,**kwargs)
    monkeypatch.setattr(evaluation,'reference_one',counted_reference)
    evaluation.references(out,device='cpu',chunk_size=17,cache_root=cache_root,max_sequences=pilot)
    assert computed==list(range(pilot))
    manifest=json.loads((out/'references'/'manifest.json').read_text())
    assert not manifest['complete'] and manifest['partial_evaluation']
    assert manifest['evaluated_n_sequences']==pilot and manifest['planned_n_sequences']==planned
    pilot_bayes=evaluation.summarize(out,bayes_only=True,max_sequences=pilot)
    assert pilot_bayes['cohort']==record and pilot_bayes['partial_evaluation']
    assert pilot_bayes['evaluated_n_sequences']==pilot and pilot_bayes['planned_n_sequences']==planned
    assert not pilot_bayes['evaluation_complete'] and not pilot_bayes['two_repeat_comparison_complete']
    assert not (out/'summary.json').exists()
    assert json.loads((out/'summary_prefix_000008.json').read_text())==pilot_bayes
    with pytest.raises(ValueError,match='incomplete'):
        evaluation.summarize(out,bayes_only=True)
    with pytest.raises(ValueError,match='incomplete'):
        evaluation.read_complete(out,'references',pilot,record['cohort_id'])
    checkpoints=[]
    for seed in [0,1]:
        run=tmp_path/f'train{seed}'
        training.train(tiny_config(seed),run,device='cpu',workers=0)
        checkpoint=run/'model_3.pt';checkpoints.append(checkpoint)
        evaluation.evaluate_model(out,checkpoint,device='cpu',batch_size=4,max_sequences=pilot)
    pilot_summary=evaluation.summarize(out,max_sequences=pilot)
    assert pilot_summary['present_training_seeds']==[0,1]
    assert not pilot_summary['two_repeat_comparison_complete']
    refs,_=evaluation.read_complete(out,'references',pilot,record['cohort_id'],planned)
    models,_=evaluation.read_complete(out,Path('models')/'seed0',pilot,record['cohort_id'],planned)
    expected=evaluation.moments(models['cumulative_loss_bits']-refs['bayes_hidden_cumulative_loss_bits'])
    assert pilot_summary['groups'][0]['paired']['seed0_minus_bayes_hidden']['cumulative_loss_bits']==expected
    _,data=evaluation.load_cohort(out)
    assert [g['n_seq'] for g in pilot_summary['groups'][1:]]==np.bincount(data['stratum'][:pilot],minlength=4).tolist()
    assert pilot_summary['cohort']['entropy_medians']==record['entropy_medians']
    pilot_files=list((out/'references').glob('seq_*'))+list((out/'models').glob('*/seq_*'))
    old_records={path:(path.read_bytes(),path.stat().st_mtime_ns) for path in pilot_files}
    computed.clear()
    evaluation.references(out,device='cpu',chunk_size=17,cache_root=cache_root)
    assert computed==list(range(pilot,planned))
    with pytest.raises(ValueError,match='incomplete'):
        evaluation.summarize(out)
    for checkpoint in checkpoints:
        evaluation.evaluate_model(out,checkpoint,device='cpu',batch_size=4)
    full=evaluation.summarize(out)
    assert full['two_repeat_comparison_complete'] and full['evaluation_complete']
    assert not full['partial_evaluation'] and full['evaluated_n_sequences']==planned
    assert full['cohort']==record and evaluation.file_hash(out/'cohort.npz')==cohort_hash
    assert all((path.read_bytes(),path.stat().st_mtime_ns)==before for path,before in old_records.items())
    # Reviewing a pilot after completion cannot replace or downgrade full results.
    evaluation.references(out,device='cpu',chunk_size=17,cache_root=cache_root,max_sequences=pilot)
    for checkpoint in checkpoints:
        evaluation.evaluate_model(out,checkpoint,device='cpu',batch_size=4,max_sequences=pilot)
    assert evaluation.summarize(out,max_sequences=pilot)==pilot_summary
    assert json.loads((out/'summary.json').read_text())==full
    assert evaluation.summarize(out)==full


def test_pilot_cli_keeps_full_cohort_and_needs_no_checkpoints(tmp_path):
    out=tmp_path/'pilot'
    result=subprocess.run([sys.executable,str(Path(evaluation.__file__)),
        '--out',str(out),'--dimensions','2x3','--device','cpu','--n-seq','24',
        '--length','17','--times','1','4','16','--chunk-size','17',
        '--bayes-only','--max-sequences','6'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    record=json.loads((out/'cohort.json').read_text())
    summary=json.loads((out/'summary_prefix_000006.json').read_text())
    assert record['identity']['n_seq']==24 and summary['cohort']==record
    assert summary['bayes_only'] and summary['partial_evaluation']
    assert summary['missing_training_seeds']==[0,1]
    assert len(list((out/'references').glob('seq_*.npz')))==6
    assert not (out/'models').exists() and not (out/'summary.json').exists()


def test_production_cohort_pilot_is_never_full_production_evidence(tmp_path):
    evaluation.create_cohort(tmp_path,(2,2))
    evaluation.references(tmp_path,device='cpu',max_sequences=2)
    summary=evaluation.summarize(tmp_path,bayes_only=True,max_sequences=2)
    assert summary['production_size'] and summary['partial_evaluation']
    assert summary['evaluated_n_sequences']==2 and summary['planned_n_sequences']==4096
    assert not summary['evaluation_complete'] and not summary['two_repeat_comparison_complete']
    assert summary['cohort']['identity']['length']==1025
    assert not (tmp_path/'summary.json').exists()


@pytest.mark.parametrize('cap',[0,1,25])
def test_pilot_rejects_invalid_caps_before_expensive_work(tmp_path,cap):
    evaluation.create_cohort(tmp_path,(2,3),n_seq=24,length=17,times=[1,4,16])
    with pytest.raises(ValueError,match='max_sequences'):
        evaluation.references(tmp_path,max_sequences=cap)
    assert not (tmp_path/'encoding_cache').exists()


def test_safe_orphan_recovery_and_missing_checkpoint_fails_early(tmp_path):
    path=tmp_path/'result.npz';evaluation.save_npz(path,dict(a=np.arange(3)))
    with pytest.raises(ValueError):evaluation.checked_existing(path,{'id':1})
    assert evaluation.checked_existing(path,{'id':1},recover_orphan=True) is None
    assert not path.exists() and list(tmp_path.glob('result.npz.orphan.*'))
    out=tmp_path/'no_work'
    result=subprocess.run([sys.executable,str(Path(evaluation.__file__)),'--out',str(out),'--dimensions','2x3','--device','cpu'],capture_output=True,text=True)
    assert result.returncode!=0 and 'no expensive computation' in result.stderr
    assert not out.exists()


def test_sizing_cpu_smoke_is_not_production_shape_selection(tmp_path):
    args=SimpleNamespace(device='cpu',out=tmp_path/'capacity.json',length=17,n_eval=8,
        chunk_size=257,repeats=2,max_reference_hours=24.,dimensions=[(2,3)],
        max_candidate_seconds=20.,cache_root=tmp_path/'cache',max_cache_gib=.1,max_cache_build_seconds=30.)
    report=gpu.benchmark(args)
    assert report['validation']['passed'] and len(report['validation']['cases'])==4
    assert report['candidates'][0]['status']=='complete'
    assert len(report['candidates'][0]['measured_seconds'])==2
    assert report['recommended_dimensions'] is None and not report['production_workload']
    assert json.loads(args.out.read_text())==report

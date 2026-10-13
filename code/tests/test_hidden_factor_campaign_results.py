"""Independent arithmetic and rejection tests for the 3840-case publication import."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'))
import hidden_factor_campaign_results as result


@pytest.fixture
def scores():
    n,nt,d=3840,6,16
    i=np.arange(n)[:,None,None];t=np.arange(nt)[None,:,None];b=np.arange(d)[None,None,:]
    p=1+((3*i+5*t+7*b)%17)/11;p=p/p.sum(-1,keepdims=True)
    oracle=np.broadcast_to(1+((5*i+3*t+b)%13),p.shape).astype(float).copy();oracle/=oracle.sum(-1,keepdims=True)
    tokens=np.broadcast_to(np.arange(1025)%16,(n,1025)).copy()
    kl=(oracle*np.log2(oracle/p)).sum(-1)
    observed=-np.log2(np.take_along_axis(p,tokens[:,result.TIMES,None],-1)[...,0])
    cumulative=observed[:,:1]+np.array([0.,10.,60.,220.,900.,4000.])
    return dict(predictive=p,next_kl_bits=kl,next_loss_bits=observed,cumulative_loss_bits=cumulative,
                sequence_indices=np.arange(256,4096),times=result.TIMES.copy()),oracle,tokens


def test_independent_regret_and_realized_loss_with_preserved_pairing(scores):
    a,oracle,tokens=scores
    checked=result.check_scores(a,oracle,tokens,'synthetic')
    # Scalar calculation with a different summation order checks the vector score.
    for i,t in [(0,0),(3811,4),(3839,5)]:
        expected=math.fsum(float(q)*math.log2(float(q)/float(p)) for q,p in zip(oracle[i,t],a['predictive'][i,t]))
        assert checked['next_kl_bits'][i,t]==pytest.approx(expected,abs=1e-14)
        assert checked['next_loss_bits'][i,t]==pytest.approx(-math.log2(a['predictive'][i,t,tokens[i,result.TIMES[t]]]),abs=1e-14)


@pytest.mark.parametrize('mutation',['pairing','times','normalization','regret','realized','cumulative','nonfinite'])
def test_broken_probability_pairing_or_loss_is_not_repaired(scores,mutation):
    a,oracle,tokens=scores
    if mutation=='pairing':a['sequence_indices'][0]=0
    elif mutation=='times':a['times'][0]=2
    elif mutation=='normalization':a['predictive'][0,0]*=1.00001
    elif mutation=='regret':a['next_kl_bits'][1,2]+=.001
    elif mutation=='realized':a['next_loss_bits'][1,2]+=.001
    elif mutation=='cumulative':a['cumulative_loss_bits'][0,1]=-1
    else:a['predictive'][0,0,0]=np.nan
    with pytest.raises(ValueError):result.check_scores(a,oracle,tokens,'synthetic')


def test_paired_mean_and_standard_error_are_computed_per_sequence(scores):
    a,_,_=scores
    # Correlated scores make subtracting independent SEs an incorrect substitute.
    left=a['next_kl_bits'];right=.97*left+np.arange(3840)[:,None]*1e-7
    diff=left-right
    manual=[]
    for col in diff.T:
        mean=math.fsum(map(float,col))/3840
        se=math.sqrt(math.fsum((float(x)-mean)**2 for x in col)/(3840*3839))
        manual.append((mean,se))
    saved=dict(n_seq=3840,mean=[x[0] for x in manual],se=[x[1] for x in manual])
    result.check_stats(diff,saved,'paired')
    wrong=deepcopy(saved);wrong['se']=((left.std(0,ddof=1)**2+right.std(0,ddof=1)**2)/3840)**.5
    with pytest.raises(ValueError,match='se differs'):result.check_stats(diff,wrong,'paired')


@pytest.fixture
def envelope(tmp_path):
    plan=result.read(result.PLAN);arm='xlarge';step=200000
    code={name:result.sha(ROOT/name) for name in result.CODE_FILES}
    baseline=dict(cohort_id='cohort',cohort_data_sha256='cohort-data',reference_records_sha256='references',
                  model_receipts={f'large_seed{s}':{'identity':{'seed':s}} for s in (0,1)})
    selection=dict(schema=result.REMOTE_SCHEMA,arm=arm,step=step,plan_sha256=result.sha(result.PLAN),
                   test_role='reused_comparison',implementation_sha256=code)
    summary=dict(schema=result.REMOTE_SCHEMA,complete=True,scientific_results=True,arm=arm,step=step,
        final_checkpoint=False,test_role='reused_comparison',fresh_confirmation=False,new_training=False,
        new_bayesian_computation=False,plan_sha256=result.sha(result.PLAN),source_dimensions=[2,8],
        sequence_start=256,sequence_stop=4096,evaluated_n_sequences=3840,times=result.TIMES.tolist(),
        first_token_scored=False,implementation_sha256=code,interpretation=plan['interpretation'],
        cohort_id='cohort',cohort_data_sha256='cohort-data',reference_records_sha256='references',
        parent_receipts=baseline['model_receipts'],model_receipts={f'xlarge_seed{s}':{} for s in (0,1)})
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    status=dict(schema=result.REMOTE_SCHEMA,status='completed',complete=True,arm=arm,step=step,
                evaluated_n_sequences=3840,fresh_confirmation=False,summary_sha256=result.sha(tmp_path/'summary.json'))
    (tmp_path/'status.json').write_text(json.dumps(status))
    return tmp_path,plan,baseline


def test_valid_completed_intermediate_envelope(envelope):
    folder,plan,baseline=envelope
    report=result.check_run_envelope(folder,'xlarge',200000,plan,baseline,result.Inputs())
    assert not report['final_checkpoint'] and not report['fresh_confirmation']


def final_envelope(envelope):
    folder,plan,baseline=envelope
    for filename in ('selection.json','summary.json','status.json'):
        p=folder/filename;value=result.read(p);value['step']=400000
        if filename=='summary.json':value['final_checkpoint']=True
        p.write_text(json.dumps(value))
    p=folder/'status.json';value=result.read(p);value['summary_sha256']=result.sha(folder/'summary.json')
    p.write_text(json.dumps(value))
    return folder,plan,baseline


def test_final_400k_envelope_is_accepted_without_relabelling_interim(envelope):
    folder,plan,baseline=final_envelope(envelope)
    report=result.check_run_envelope(folder,'xlarge',400000,plan,baseline,result.Inputs())
    assert report['final_checkpoint'] is True and report['fresh_confirmation'] is False
    with pytest.raises(ValueError,match='selection'):
        result.check_run_envelope(folder,'xlarge',200000,plan,baseline,result.Inputs())


@pytest.mark.parametrize('file,field,value',[
    ('summary.json','final_checkpoint',False),
    ('summary.json','step',200000),
    ('selection.json','step',200000),
    ('status.json','step',200000),
])
def test_final_import_rejects_interim_scope_in_any_receipt(envelope,file,field,value):
    folder,plan,baseline=final_envelope(envelope)
    p=folder/file;s=result.read(p);s[field]=value;p.write_text(json.dumps(s))
    if file=='summary.json':
        p=folder/'status.json';s=result.read(p);s['summary_sha256']=result.sha(folder/'summary.json')
        p.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        result.check_run_envelope(folder,'xlarge',400000,plan,baseline,result.Inputs())


@pytest.mark.parametrize('step,final,sequences',[(200000,False,'12.8M'),(400000,True,'25.6M')])
def test_selected_checkpoint_drives_labels_and_final_flags(step,final,sequences):
    plan=result.read(result.PLAN)
    flags,labels,scope=result.selection_metadata(plan,dict(continue_large=400000,xlarge=step))
    assert flags==dict(continue_large=True,xlarge=final)
    assert sequences in labels['xlarge']
    assert ('interim' in labels['xlarge']) is (not final)
    assert f'{step:,} updates' in scope
    assert '25.6M' in labels['continue_large']


@pytest.mark.parametrize('steps',[
    dict(continue_large=400000,xlarge=100000), dict(continue_large=400000,xlarge=400001),
    dict(continue_large=200000,xlarge=400000), dict(continue_large=400000,xlarge=400000.0),
    dict(xlarge=400000), dict(continue_large=400000,xlarge=400000,extra=400000),
])
def test_unselected_or_partial_campaign_milestones_are_rejected(steps):
    with pytest.raises(ValueError):result.selection_metadata(result.read(result.PLAN),steps)


def panel_data(step):
    plan=result.read(result.PLAN);steps=dict(continue_large=400000,xlarge=step)
    flags,_,_=result.selection_metadata(plan,steps)
    curve=dict(n_seq=3840,mean=[.1]*6,se=[.01]*6)
    predictors={'bayes_hidden':dict(next_kl_bits=curve)};receipts={}
    for j,arm in enumerate(result.ARMS):
        for seed in (0,1):
            name=f'{arm}_seed{seed}';digest=f'{2*j+seed:064x}'
            identity=(dict(pilot=dict(checkpoint_sha256=digest)) if arm=='baseline' else
                      dict(checkpoint=dict(checkpoint_sha256=digest),step=steps[arm],final_checkpoint=flags[arm]))
            receipts[name]=dict(identity=identity);predictors[name]=dict(next_kl_bits=curve)
    return dict(selected_steps=steps,final_checkpoint=flags,cohort_id='cohort',
                groups=[dict(id='full_prior',predictors=predictors)],model_receipts=receipts),plan


@pytest.mark.parametrize('step', [200000,400000])
def test_plot_receipts_keep_actual_checkpoint_step_exposure_and_status(step):
    data,plan=panel_data(step);panel=result.panel(data,plan)
    for case in panel['cases']:
        if case['arm']=='xlarge':
            assert case['checkpoint_step']==step and case['training_sequences']==64*step
            assert case['interim'] is (step==200000)


@pytest.mark.parametrize('mutation',['step','receipt_final','publication_final'])
def test_final_plot_cannot_relabel_an_interim_checkpoint(mutation):
    data,plan=panel_data(400000)
    if mutation=='step':data['model_receipts']['xlarge_seed0']['identity']['step']=200000
    elif mutation=='receipt_final':data['model_receipts']['xlarge_seed1']['identity']['final_checkpoint']=False
    else:data['final_checkpoint']['xlarge']=False
    with pytest.raises(ValueError):result.panel(data,plan)


@pytest.mark.parametrize('mutation',['status','summary_hash','selection_step','selection_code','fresh','final','parent','missing_seed'])
def test_status_selection_or_scope_failure_blocks_publication(envelope,mutation):
    folder,plan,baseline=envelope
    if mutation in ('status','summary_hash'):
        p=folder/'status.json';s=result.read(p)
        if mutation=='status':s['complete']=False
        else:s['summary_sha256']='changed'
    elif mutation.startswith('selection'):
        p=folder/'selection.json';s=result.read(p)
        if mutation=='selection_step':s['step']=400000
        else:s['implementation_sha256'][result.CODE_FILES[0]]='changed'
    else:
        p=folder/'summary.json';s=result.read(p)
        if mutation=='fresh':s['fresh_confirmation']=True
        elif mutation=='final':s['final_checkpoint']=True
        elif mutation=='parent':s['parent_receipts']['large_seed0']={'identity':{'seed':1}}
        else:del s['model_receipts']['xlarge_seed1']
    p.write_text(json.dumps(s))
    if p.name=='summary.json':
        status=result.read(folder/'status.json');status['summary_sha256']=result.sha(p)
        (folder/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError):result.check_run_envelope(folder,'xlarge',200000,plan,baseline,result.Inputs())


@pytest.fixture
def training():
    plan=result.read(result.PLAN);arm='continue_large';seed=0
    runtime=dict(device='cuda',float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,num_threads=1)
    validation={'hidden_factor':'a'*64}
    parent=dict(weights_checkpoint_sha256=plan['parent']['checkpoint_sha256']['0'],step=100000,
                completed_sequences=6400000,validation_hashes=validation,runtime=runtime)
    provenance=dict(plan_sha256=result.sha(result.PLAN),arm=arm,seed=seed,
        trainer_sha256=result.sha(ROOT/'code/transformer/hidden_factor_campaign_train.py'),parent=parent)
    cfg=dict(plan['arms'][arm]['config'],seed=seed,source_dimensions=[2,8],scaling_provenance=provenance)
    contract=dict(version='hidden-factor-dirichlet1-v1',dimensions=[2,8],
        rows='independent Dirichlet(1,...,1) for each coordinate',initial_states='independent uniform coordinates',
        encoding='fresh uniform joint-pair-to-token permutation per sequence',
        parameters='fresh independent transitions and encoding per sequence',model_input='encoded tokens only',sampling='indexed-batch-v1')
    receipt=dict(config=cfg,config_hash=result.fingerprint(cfg),implementation_hashes=plan['implementation_sha256'],
        model=dict(vocab=16,n_ctx=1024,n_layer=4,n_head=8,d_model=256,mlp=True,dropout=0.),n_params=3429888,
        source_contract=contract,source_contract_hash=result.fingerprint(contract),precision='float32',device='cuda',
        runtime=runtime,validation_hashes=validation)
    return plan,receipt,arm,seed


def test_valid_training_receipt_without_mutable_training_status(training):
    result.validate_training_receipt(*training)


@pytest.mark.parametrize('mutation',['source','architecture','parent','seed','precision','implementation'])
def test_training_metadata_tampering_blocks_import(training,mutation):
    plan,r,arm,seed=training
    if mutation=='source':r['config']['source_dimensions']=[4,4]
    elif mutation=='architecture':r['model']['n_layer']=8
    elif mutation=='parent':r['config']['scaling_provenance']['parent']['weights_checkpoint_sha256']='f'*64
    elif mutation=='seed':r['config']['seed']=1
    elif mutation=='precision':r['precision']='bfloat16'
    else:r['implementation_hashes']=dict(r['implementation_hashes'],**{'model.py':'f'*64})
    r['config_hash']=result.fingerprint(r['config'])
    with pytest.raises(ValueError):result.validate_training_receipt(plan,r,arm,seed)

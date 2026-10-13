"""Training-data pairing, independent permutations, target alignment and resume."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transformer'))
import train_shared_base_scaling as ordered
import train_shared_base_shuffle as shuffle
import shared_base_shuffle_source as source
from model import GPT
from source import Source


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def same(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b


def test_shuffle_each_sequence_counts_first_token_and_rng_separation():
    # All token IDs distinct here, so the positional permutation is observable.
    tokens=np.tile(np.arange(1025,dtype=np.int64),(64,1));old=tokens.copy()
    np.random.seed(732);random.seed(321);torch.manual_seed(19)
    before=(np.random.get_state(),random.getstate(),torch.get_rng_state())
    actual,indices=source.tail_shuffle(tokens,0,3,return_indices=True)
    same(before,(np.random.get_state(),random.getstate(),torch.get_rng_state()))
    assert np.array_equal(tokens,old) and np.array_equal(actual[:,0],tokens[:,0])
    assert np.array_equal(np.sort(actual,axis=1),tokens)
    assert len(np.unique(indices[:,1:],axis=0))==64
    assert not np.array_equal(actual,source.tail_shuffle(tokens,0,4))
    assert not np.array_equal(actual,source.tail_shuffle(tokens,1,3))
    assert not np.array_equal(actual,source.tail_shuffle(tokens,0,3,stream=1))
    assert np.array_equal(actual,source.tail_shuffle(tokens,0,3))
    assert not np.array_equal(actual,source.tail_shuffle(tokens,0,3+2**32))


@pytest.mark.parametrize('seed,index',[(0,0),(0,1),(0,49999),(1,99999)])
def test_exact_production_batch_source_replay(seed,index):
    reference_rng=np.random.default_rng(np.random.SeedSequence([
        0x53424C53,seed&0xffffffff,seed>>32,index&0xffffffff,index>>32]))
    original=Source(64,1025,alpha=(.5,1.5),ls=(0,8),permute=True).batch(64,reference_rng)[0]
    expected,indices=source.tail_shuffle(original,seed,index,return_indices=True)
    actual=source.draw_batch(seed,index,64,1024)
    assert np.array_equal(actual,expected)
    recovered=np.take_along_axis(actual,np.argsort(indices,axis=1),axis=1)
    assert np.array_equal(recovered,original)
    assert np.array_equal(original,ordered.IndexedSharedBase(seed,64,1024,index,index+1)[0].numpy())
    assert source.pairing_probe(seed,index)['distinct_tail_permutations']==64


def test_validation_is_held_out_exact_counts_no_train_rng():
    cfg=dict(n_ctx=15,val_n=8,val_settings=[[1,2],[1,4]])
    original=ordered.draw_validation(cfg);actual=source.draw_validation(cfg)
    for index,name in enumerate(original):
        assert np.array_equal(actual[name][:,0],original[name][:,0])
        assert np.array_equal(np.sort(actual[name],axis=1),np.sort(original[name],axis=1))
        assert np.array_equal(actual[name],source.tail_shuffle(original[name].numpy(),0,index,stream=1))


@pytest.fixture
def mini(tmp_path,monkeypatch):
    plan=dict(config=deepcopy(ordered.PRODUCTION),source=deepcopy(ordered.SOURCE))
    plan['source']['n_ctx']=5
    plan['config'].update(batch=3,steps=4,warmup=2,n_layer=1,n_head=1,d_model=8,
        val_n=2,val_chunk=1,log_every=1,val_every=2,ckpt_every=2,weight_steps=[4])
    path=tmp_path/'tiny_test_only.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(shuffle,'read_plan',lambda p:(plan,shuffle.file_hash(p)))
    monkeypatch.setattr(ordered,'read_plan',lambda p:(plan,ordered.file_hash(p)))
    monkeypatch.setattr(shuffle,'verify_ordered_training',lambda *a,**kw:{'test_fixture':True})
    return path,plan


def test_same_initialization_as_original(mini,tmp_path):
    path,_=mini
    assert ordered.train(path,1,tmp_path/'ordered',max_steps=0)==75
    assert shuffle.train(path,1,tmp_path/'shuffled',max_steps=0)==75
    a=load(tmp_path/'ordered/latest.pt');b=load(tmp_path/'shuffled/latest.pt')
    same(a['model'],b['model']);same(a['opt'],b['opt']);same(a['rng'],b['rng'])


def test_exact_resume_workers_and_independent_shifted_target_reference(mini,tmp_path):
    path,plan=mini;whole=tmp_path/'whole';split=tmp_path/'split'
    assert shuffle.train(path,0,whole)==0
    assert shuffle.train(path,0,split,max_steps=2)==75
    assert shuffle.train(path,0,split,resume=True,workers=1)==0
    full,resumed=load(whole/'latest.pt'),load(split/'latest.pt')
    for key in ['model','opt','rng','cfg','validation_hashes','completed_sequences','completed_batches']:
        same(full[key],resumed[key])
    cfg=shuffle.planned_config(plan,0)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    model=GPT(64,5,1,1,8,mlp=True)
    opt=torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.dim()>=2],weight_decay=.1),
        dict(params=[p for p in model.parameters() if p.dim()<2],weight_decay=0.)],lr=.0003,betas=(.9,.95))
    for step in range(4):
        for group in opt.param_groups:group['lr']=ordered.lr_at(cfg,step)
        unshuffled=Source(64,6,alpha=(.5,1.5),ls=(0,8),permute=True).batch(3,ordered.batch_rng(0,step))[0]
        # Independent construction of the agreed training input AND shifted target.
        x=unshuffled.copy()
        for row in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([0x53485346,0,0,0,step,0,row]))
            x[row,1:]=unshuffled[row,1:][rng.permutation(5)]
        x=torch.from_numpy(x)
        loss=F.cross_entropy(model(x[:,:-1]).reshape(-1,64),x[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    same(full['model'],model.state_dict());same(full['opt'],opt.state_dict());same(full['rng'],shuffle.rng_state())
    assert load(whole/'model_4.pt')['schema']=='shared-base-shuffle-weights-v1'
    assert full['completed_sequences']==12


@pytest.mark.parametrize('corruption',['config_hash','implementation_hashes','pairing'])
def test_resume_rejects_changes(mini,tmp_path,corruption):
    path,_=mini;out=tmp_path/'run';assert shuffle.train(path,0,out,max_steps=1)==75
    ck=load(out/'latest.pt')
    if corruption=='pairing':ck['cfg']['ordered_training']={'changed':True}
    else:ck[corruption]='changed'
    torch.save(ck,out/'latest.pt');before=(out/'latest.pt').read_bytes()
    with pytest.raises(ValueError,match='resume refused'):shuffle.train(path,0,out,resume=True)
    assert (out/'latest.pt').read_bytes()==before


def test_archived_production_plan_and_runtime_receipts():
    root=Path(__file__).resolve().parents[2]
    path=root/'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'
    plan,_=shuffle.read_plan(path)
    assert plan['config']==ordered.PRODUCTION and plan['source']==ordered.SOURCE
    assert plan['config']['steps']*plan['config']['batch']==6400000
    # The real downloaded receipts are optional on other checkouts.
    original=root/'output/shared_base_scaling_download_2026-09-17'
    if original.is_dir():
        for seed in [0,1]:
            result=shuffle.verify_ordered_training(path,plan,seed,original,device='cpu')
            assert result['probes']==plan['pairing_probes'][str(seed)]

@pytest.mark.parametrize('field,value',[
    ('source.alpha',[1,1]),('source.ls',[0,0]),('source.n_ctx',512),
    ('config.batch',32),('config.steps',50000),('config.n_layer',2),
    ('implementation_sha256',{}),('shuffle_contract',{})])
def test_changed_pairing_plan_refused(tmp_path,field,value):
    root=Path(__file__).resolve().parents[2]
    path=root/'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'
    plan=json.loads(path.read_text());parts=field.split('.')
    if len(parts)==2:plan[parts[0]][parts[1]]=value
    else:plan[field]=value
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError):shuffle.read_plan(changed)

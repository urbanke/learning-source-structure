"""Source identity and exact CPU continuation for the larger shared-base model.

Small internal-engine fixtures exercise training without changing any production
plan or generating scientific experiment artifacts. Public plan validation is
tested independently against the fixed production settings.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import signal
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transformer'))
import train_shared_base_scaling as scaling
from eval_shared_base_models import validate_checkpoint
from model import GPT
from source import Source,seed_of


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def production_plan():
    return dict(schema='shared-base-scaling-plan-v1',seeds=[0,1],source=deepcopy(scaling.SOURCE),
        config=deepcopy(scaling.PRODUCTION),n_params=3454464,
        implementation_sha256=scaling.implementation_hashes(),benchmark=dict(steps=1000,seed=0))


def test_production_model_source_and_schedule(tmp_path):
    plan=production_plan();path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    saved,_=scaling.read_plan(path);cfg=scaling.planned_config(saved,1)
    assert GPT(64,1024,4,8,256).n_params()==3454464
    assert cfg['steps']*cfg['batch']==6400000 and cfg['weight_steps']==[50000,100000]
    assert cfg['seed']==1 and cfg['alpha']==[.5,1.5] and cfg['ls']==[0,8] and cfg['permute'] is True
    assert scaling.lr_at(cfg,0)==pytest.approx(3e-7)
    assert scaling.lr_at(cfg,999)==pytest.approx(3e-4)
    assert scaling.lr_at(cfg,1000)==pytest.approx(3e-4)
    assert scaling.lr_at(cfg,100000)==pytest.approx(3e-5)


@pytest.mark.parametrize('field,value',[
    ('source.alpha',[1,1]),('source.ls',[1,8]),('source.permute',False),('source.d',16),
    ('config.steps',1000),('config.n_layer',2),('config.val_n',256),('config.batch',64.),
    ('config.lr',1e-3),('n_params',1),('seeds',[False,True]),('implementation_sha256',{}),
])
def test_changed_production_plan_refused(tmp_path,field,value):
    plan=production_plan();parts=field.split('.')
    if len(parts)==2:plan[parts[0]][parts[1]]=value
    else:plan[field]=value
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):scaling.read_plan(path)


def test_indexed_batches_are_exact_original_source_and_disjoint():
    for seed,index in [(0,0),(0,1000),(1,0)]:
        expected,_=Source(64,8,alpha=(.5,1.5),ls=(0,8),permute=True).batch(5,scaling.batch_rng(seed,index))
        assert np.array_equal(scaling.draw_batch(seed,index,5,7),expected)
    assert not np.array_equal(scaling.draw_batch(0,0,5,7),scaling.draw_batch(1,0,5,7))
    assert not np.array_equal(scaling.draw_batch(0,0,5,7),scaling.draw_batch(0,1,5,7))
    # High index bits are retained; the schedule is not truncated to32 bits.
    assert not np.array_equal(scaling.draw_batch(0,0,5,7),scaling.draw_batch(0,2**32,5,7))


def test_historical_monitoring_sets_exactly_reused():
    cfg=dict(n_ctx=7,val_n=3,val_settings=deepcopy(scaling.PRODUCTION['val_settings']))
    actual=scaling.draw_validation(cfg)
    assert len(actual)==4
    for a,L in cfg['val_settings']:
        a=float(a)
        expected,_=Source(64,8,alpha=a,ls=L,permute=True).batch(3,np.random.default_rng(seed_of('val',64,a,L)))
        assert np.array_equal(actual[f'a{a:g}_L{L}'].numpy(),expected)


@pytest.fixture
def mini(tmp_path,monkeypatch):
    plan=production_plan()
    plan['source']['n_ctx']=5
    plan['config'].update(batch=3,steps=4,warmup=2,n_layer=1,n_head=1,d_model=8,
        val_n=2,val_chunk=1,log_every=1,val_every=2,ckpt_every=2,weight_steps=[4])
    path=tmp_path/'tiny_test_only.json';path.write_text(json.dumps(plan))
    # Only this fixture bypasses the independently tested production-plan gate.
    monkeypatch.setattr(scaling,'read_plan',lambda p:(plan,scaling.file_hash(p)))
    return path,plan


def test_exact_cpu_resume_across_worker_count_and_independent_reference(mini,tmp_path):
    path,plan=mini;whole=tmp_path/'whole';split=tmp_path/'split'
    assert scaling.train(path,0,whole)==0
    assert scaling.train(path,0,split,max_steps=2)==75
    first=load(split/'latest.pt')
    assert first['step']==2 and first['completed_sequences']==6
    assert not (split/'model_4.pt').exists()
    assert scaling.train(path,0,split,resume=True,workers=1)==0
    full=load(whole/'latest.pt');resumed=load(split/'latest.pt')
    for key in ['model','opt','rng','cfg','validation_hashes','completed_sequences','completed_batches']:
        same(full[key],resumed[key])
    cfg=scaling.planned_config(plan,0)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    model=GPT(64,5,1,1,8,mlp=True)
    opt=torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.dim()>=2],weight_decay=.1),
        dict(params=[p for p in model.parameters() if p.dim()<2],weight_decay=0.)],lr=.0003,betas=(.9,.95))
    for step in range(4):
        for group in opt.param_groups:group['lr']=scaling.lr_at(cfg,step)
        x=torch.from_numpy(Source(64,6,alpha=(.5,1.5),ls=(0,8),permute=True).batch(3,scaling.batch_rng(0,step))[0])
        loss=F.cross_entropy(model(x[:,:-1]).reshape(-1,64),x[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    same(full['model'],model.state_dict());same(full['opt'],opt.state_dict());same(full['rng'],scaling.rng_state())
    weights=load(split/'model_4.pt');config=json.loads((split/'config.json').read_text())
    validate_checkpoint(weights,dict(config=config,step=4))
    assert weights['schema']=='shared-base-scaling-weights-v1'
    assert weights['completed_batches']==4 and weights['completed_sequences']==12


def test_max_cap_retains_production_schedule_and_cli_receipt(mini,tmp_path):
    path,plan=mini;out=tmp_path/'cap'
    assert scaling.main(['--plan',str(path),'--seed','1','--out',str(out),'--device','cpu','--workers','0','--max-steps','1'])==75
    checkpoint=load(out/'latest.pt');runtime=json.loads((out/'scaling_runtime.json').read_text())
    precision=checkpoint['cfg']['runtime']
    assert precision['float32_matmul_precision']=='highest'
    assert precision['cuda_matmul_allow_tf32'] is False and precision['cudnn_allow_tf32'] is False
    assert checkpoint['cfg']['config']['steps']==4
    assert checkpoint['opt']['param_groups'][0]['lr']==pytest.approx(.00015)
    assert runtime['return_code']==75 and runtime['step']==1 and runtime['seed']==1
    assert runtime['plan_sha256']==scaling.file_hash(path) and runtime['device']=='cpu'
    status=json.loads((out/'status.json').read_text())
    assert status['reason']=='max_steps' and status['complete'] is False


@pytest.mark.parametrize('corruption',['config_hash','plan_sha256','implementation_hashes','runtime','counts','config_file'])
def test_resume_rejects_changed_state(mini,tmp_path,corruption):
    path,_=mini;out=tmp_path/'run';assert scaling.train(path,0,out,max_steps=1)==75
    ck=load(out/'latest.pt')
    if corruption in ['config_hash','plan_sha256']:ck[corruption]='0'*64
    elif corruption=='implementation_hashes':ck['implementation_hashes']={}
    elif corruption=='runtime':ck['cfg']['runtime']['numpy_version']='changed'
    elif corruption=='counts':ck['completed_sequences']=999
    elif corruption=='config_file':(out/'config.json').write_text('{}')
    torch.save(ck,out/'latest.pt');before=(out/'latest.pt').read_bytes()
    with pytest.raises(ValueError,match='resume refused'):scaling.train(path,0,out,resume=True)
    assert (out/'latest.pt').read_bytes()==before


def test_nonempty_missing_checkpoint_lock_and_bad_caps(mini,tmp_path):
    path,_=mini;out=tmp_path/'run';out.mkdir();(out/'keep').write_text('original')
    with pytest.raises(ValueError,match='nonempty'):scaling.train(path,0,out)
    with pytest.raises(ValueError,match='requires latest'):scaling.train(path,0,out,resume=True)
    with (out/'.training.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='another trainer'):scaling.train(path,0,out)
    for cap in [-1,5]:
        with pytest.raises(ValueError,match='step cap'):scaling.train(path,0,tmp_path/'bad',max_steps=cap)
    assert not (tmp_path/'bad').exists()
    assert (out/'keep').read_text()=='original'


def test_signal_stops_at_completed_batch_and_restores_handlers(mini,tmp_path,monkeypatch):
    path,_=mini;out=tmp_path/'signalled';original=torch.optim.AdamW.step
    handler=signal.getsignal(signal.SIGUSR1)
    def step_then_signal(self,*args,**kwargs):
        result=original(self,*args,**kwargs);signal.raise_signal(signal.SIGUSR1);return result
    monkeypatch.setattr(torch.optim.AdamW,'step',step_then_signal)
    assert scaling.train(path,0,out)==75
    ck=load(out/'latest.pt')
    assert ck['step']==1 and ck['stop_reason']=='SIGUSR1' and ck['completed_sequences']==3
    assert signal.getsignal(signal.SIGUSR1)==handler

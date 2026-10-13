"""Campaign ancestry, exact indexed resumes, benchmark gates and safe launch paths."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
import hidden_factor_campaign_train as campaign
import train_hidden_factor as original
from model import GPT


def load(path): return torch.load(path,map_location='cpu',weights_only=False)


def same(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else: assert a==b


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)


@pytest.fixture
def tiny(tmp_path,monkeypatch):
    base=dict(schema='hidden-factor-training-v1',n_ctx=5,batch=3,steps=2,warmup=1,lr=3e-4,
              wd=.1,clip=1.,n_layer=1,n_head=1,d_model=8,val_n=3,log_every=1,val_every=2,
              ckpt_every=1,val_chunk=1,validation_seed=20260917,weight_steps=[2])
    params=GPT(6,5,1,1,8).n_params()
    plan=dict(source_dimensions=[2,3],seeds=[0,1],implementation_sha256=original.implementation_hashes(),
              adapter_dependencies_sha256={'train_hidden_factor_scaling.py':campaign.sha(ROOT/'code/transformer/train_hidden_factor_scaling.py')},
              parent=dict(config=base,plan_sha256='parent-plan',checkpoint_sha256={}),
              arms=dict(continue_large=dict(config=dict(base,steps=6,weight_steps=[4,6]),n_params=params,continuation_lr=3e-5),
                        xlarge=dict(config=dict(base,steps=6,weight_steps=[3,6]),n_params=params)))
    cfg=campaign.parent_adapter(plan,0)['baseline']['config']
    cfg=original.validate_config(dict(cfg,seed=0,source_dimensions=[2,3]));parent=tmp_path/'parent'
    assert original.train(cfg,parent)==0
    plan['parent']['checkpoint_sha256']={'0':campaign.sha(parent/'model_2.pt')}
    monkeypatch.setattr(campaign,'read_plan',lambda p:(deepcopy(plan),'campaign-plan'))
    return plan,parent,cfg


def test_production_architectures_and_budgets():
    p,_=campaign.read_plan(campaign.PLAN)
    assert p['arms']['continue_large']['config']['steps']==400000
    assert p['arms']['continue_large']['additional_sequences']==19200000
    assert p['arms']['xlarge']['total_sequences']==25600000
    for arm,n in [('continue_large',3429888),('xlarge',25760768)]:
        c=p['arms'][arm]['config'];assert GPT(16,c['n_ctx'],c['n_layer'],c['n_head'],c['d_model']).n_params()==n
    assert p['benchmark']['maximum_projected_training_hours']==72


def test_continuation_preserves_parent_and_matches_direct_training_and_resume(tiny,tmp_path,monkeypatch):
    plan,parent,base=tiny;before={f.name:f.read_bytes() for f in parent.glob('*.pt')}
    def run(out,**kw):return campaign.train('unused','continue_large',0,out,parent_checkpoint=parent/'latest.pt',parent_weights=parent/'model_2.pt',**kw)
    full,split,reference=[tmp_path/n for n in ('full','split','reference')]
    assert run(full)==0
    assert run(split,max_steps=3)==75
    assert run(split,resume=True)==0
    cfg=dict(plan['arms']['continue_large']['config'],seed=0,source_dimensions=[2,3])
    old_lr=original.lr_at
    monkeypatch.setattr(original,'lr_at',lambda c,t:old_lr(base,t) if t<2 else 3e-5)
    assert original.train(cfg,reference)==0
    left=load(full/'latest.pt')
    for other in (split,reference):
        right=load(other/'latest.pt')
        for k in ('model','opt','rng','completed_sequences','completed_batches','validation_hashes'):same(left[k],right[k])
    assert before=={f.name:f.read_bytes() for f in parent.glob('*.pt')}
    assert json.loads((full/'status.json').read_text())['campaign_arm']=='continue_large'


def test_fresh_engine_and_resume_match_original_indexed_stream(tiny,tmp_path):
    plan,_,_=tiny
    full,split,direct=[tmp_path/n for n in ('full','split','direct')]
    assert campaign.train('unused','xlarge',0,full)==0
    assert campaign.train('unused','xlarge',0,split,max_steps=3)==75
    assert campaign.train('unused','xlarge',0,split,resume=True)==0
    assert original.train(dict(plan['arms']['xlarge']['config'],source_dimensions=[2,3],seed=0),direct)==0
    for other in (split,direct):
        for k in ('model','opt','rng','completed_sequences'):same(load(full/'latest.pt')[k],load(other/'latest.pt')[k])


def test_parent_mismatch_is_rejected_before_training(tiny,tmp_path):
    _,parent,_=tiny;bad=load(parent/'latest.pt');bad['model']['tok.weight'].add_(1);torch.save(bad,parent/'latest.pt')
    with pytest.raises(ValueError,match='model tensors differ'):
        campaign.train('unused','continue_large',0,tmp_path/'bad',parent_checkpoint=parent/'latest.pt',parent_weights=parent/'model_2.pt')


def test_plan_identity_change_blocks_resume(tiny,tmp_path,monkeypatch):
    plan,_,_=tiny;out=tmp_path/'run';assert campaign.train('unused','xlarge',0,out,max_steps=3)==75
    monkeypatch.setattr(campaign,'read_plan',lambda p:(deepcopy(plan),'changed-plan'))
    with pytest.raises(ValueError,match='resume refused'):
        campaign.train('unused','xlarge',0,out,resume=True)


def test_launcher_prints_without_creating_output_and_protects_original(tmp_path):
    script=ROOT/'code/transformer/kuma/submit_hidden_factor_campaign.sh'
    out=tmp_path/'new output'
    r=subprocess.run(['bash',str(script),'--arm','continue_large','--out',str(out)],capture_output=True,text=True)
    assert r.returncode==0 and not out.exists()
    args=shlex.split(r.stdout);assert '--array=0,1%4' in args
    for extra in [('--out',str(ROOT/'results/new')),('--seed','2'),('--arm','bad'),('--benchmark','--arm','continue_large')]:
        r=subprocess.run(['bash',str(script),*extra],capture_output=True,text=True)
        assert r.returncode!=0


def test_completed_final_pair_is_skipped_without_submission(tmp_path):
    out=tmp_path/'campaign';digest=campaign.sha(campaign.PLAN)
    for seed in (0,1):
        p=out/f'train/continue_large/seed{seed}/status.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(complete=True,campaign_plan_sha256=digest)))
        p=out/f'eval/continue_large/seed{seed}/step400000/summary.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(complete=True,identity=dict(plan_sha256=digest))))
    r=subprocess.run(['bash',str(ROOT/'code/transformer/kuma/submit_hidden_factor_campaign.sh'),'--arm','continue_large','--out',str(out),'--submit'],capture_output=True,text=True)
    assert r.returncode==0 and 'already complete' in r.stdout and not (out/'logs').exists()


def test_benchmark_rejects_slow_or_incomplete_largest_model(tmp_path):
    spec=importlib.util.spec_from_file_location('benchmark',ROOT/'code/scripts/hidden_factor_campaign_benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=json.loads(campaign.PLAN.read_text());digest=campaign.sha(campaign.PLAN);out=tmp_path/'bench';out.mkdir()
    cfg=dict(p['arms']['xlarge']['config'],seed=0,source_dimensions=p['source_dimensions'],campaign_provenance=dict(plan_sha256=digest,arm='xlarge',seed=0))
    (out/'config.json').write_text(json.dumps(dict(config=cfg,n_params=25760768,device='cuda')))
    (out/'status.json').write_text(json.dumps(dict(step=1000,reason='max_steps',complete=False)))
    runtime=dict(plan_sha256=digest,arm='xlarge',seed=0,step=1000,return_code=75,reason='max_steps',max_memory_allocated_bytes=100,
                 max_memory_reserved_bytes=200,total_memory_bytes=1000,cuda_device='test')
    (out/'campaign_runtime.json').write_text(json.dumps(runtime))
    def logs(seconds):
        (out/'log.jsonl').write_text('\n'.join(json.dumps(dict(event='train',step=s,run_elapsed_seconds=t,loss_bits=3.5)) for s,t in [(100,10),(1000,seconds)]))
    logs(100);assert module.collect(campaign.PLAN,out)['passed']
    logs(1000)
    with pytest.raises(ValueError,match='runtime budget'):module.collect(campaign.PLAN,out)
    logs(100);runtime['max_memory_reserved_bytes']=950;(out/'campaign_runtime.json').write_text(json.dumps(runtime))
    with pytest.raises(ValueError,match='headroom'):module.collect(campaign.PLAN,out)


def test_milestone_receipt_rejects_changed_configuration_source_runtime_and_validation(tiny,tmp_path):
    import hidden_factor_campaign_eval as evaluation
    plan,_,_=tiny
    plan['arms']['xlarge']['milestones']=[3,6]
    out=tmp_path/'evaluated';assert campaign.train('unused','xlarge',0,out)==0
    saved=load(out/'model_6.pt')
    assert evaluation.validate_checkpoint(saved,plan,'campaign-plan','xlarge',0,'cpu')==6
    changes=[
        ('configuration hash',lambda x:x.update(config_hash='changed')),
        ('architecture',lambda x:x['cfg']['model'].update(n_head=2)),
        ('source contract',lambda x:x['cfg']['source_contract'].update(encoding='changed')),
        ('runtime',lambda x:x['cfg']['runtime'].update(float32_matmul_precision='changed')),
        ('training-validation cohort',lambda x:x['cfg'].update(validation_hashes={})),
        ('Receipt implementation',lambda x:x['cfg'].update(implementation_hashes={})),
        ('source exposure',lambda x:x.update(completed_sequences=1)),
    ]
    for message,mutate in changes:
        bad=deepcopy(saved);mutate(bad)
        with pytest.raises(ValueError,match=message):
            evaluation.validate_checkpoint(bad,plan,'campaign-plan','xlarge',0,'cpu')

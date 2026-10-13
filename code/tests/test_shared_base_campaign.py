"""Independent optimizer/data continuation, ancestry rejection and launch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transformer'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import shared_base_campaign_train as campaign
import train_shared_base_scaling as old
from model import GPT
from train_mixture import rng_state,runtime_contract,digest_json

ROOT=Path(__file__).resolve().parents[2]


def load(p): return torch.load(p,map_location='cpu',weights_only=False)


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)is type(b) and len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_plan_stream_and_guards(tmp_path):
    plan,_=campaign.read_plan(campaign.PLAN)
    assert plan['arms']['xlarge']['n_params']==GPT(64,1024,8,8,512).n_params()==25809920
    assert plan['arms']['continue_large']['config']['steps']==400000
    assert plan['benchmark']['maximum_projected_hours']==72
    for index in [0,100000,399999]:
        assert np.array_equal(campaign.draw_batch(1,index,2,5),old.draw_batch(1,index,2,5))
    changed=deepcopy(plan);changed['arms']['xlarge']['config']['steps']=399999
    p=tmp_path/'changed.json';p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='schedule'): campaign.read_plan(p)


def mini_cfg(arm='continue_large'):
    return dict(schema='test-only',**dict(old.PRODUCTION,batch=2,steps=4,warmup=1,n_layer=1,n_head=1,d_model=8,
        val_n=2,val_chunk=1,log_every=1,val_every=2,ckpt_every=2,weight_steps=[3,4]),
        **dict(old.SOURCE,n_ctx=5),seed=0,arm=arm,parent={'test_only':True},continuation_lr=3e-5)


def test_continuation_matches_independent_updates_and_resume(tmp_path):
    parent_dir=tmp_path/'parent';parent_dir.mkdir();cfg=mini_cfg()
    old_cfg={k:v for k,v in cfg.items() if k not in ('arm','parent','continuation_lr')};old_cfg.update(steps=2,weight_steps=[2])
    assert old._train_locked(old_cfg,'old-test',parent_dir,resume=False,device='cpu',workers=0,max_steps=None)==0
    parent=load(parent_dir/'latest.pt'); original_bytes=(parent_dir/'latest.pt').read_bytes()
    whole=tmp_path/'whole';split=tmp_path/'split';whole.mkdir();split.mkdir()
    assert campaign._train_locked(cfg,'test',whole,resume=False,device='cpu',workers=0,max_steps=None,parent=parent)==0
    assert campaign._train_locked(cfg,'test',split,resume=False,device='cpu',workers=0,max_steps=3,parent=parent)==75
    assert campaign._train_locked(cfg,'test',split,resume=True,device='cpu',workers=1,max_steps=None,parent=parent)==0
    full=load(whole/'latest.pt'); resumed=load(split/'latest.pt')
    for key in ('model','opt','rng','cfg','validation_hashes','completed_batches','completed_sequences'):equal(full[key],resumed[key])
    assert (parent_dir/'latest.pt').read_bytes()==original_bytes
    model=GPT(64,5,1,1,8);model.load_state_dict(parent['model'])
    opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.dim()>=2],weight_decay=.1),
                          dict(params=[p for p in model.parameters() if p.dim()<2],weight_decay=0.)],lr=3e-4,betas=(.9,.95))
    opt.load_state_dict(parent['opt']);campaign.restore_rng(parent['rng'])
    for index in (2,3):
        for group in opt.param_groups:group['lr']=3e-5
        x=torch.from_numpy(old.draw_batch(0,index,2,5));loss=F.cross_entropy(model(x[:,:-1]).reshape(-1,64),x[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    equal(full['model'],model.state_dict());equal(full['opt'],opt.state_dict());equal(full['rng'],rng_state())
    assert full['step']==4 and full['completed_sequences']==8 and full['opt']['param_groups'][0]['lr']==3e-5
    before=(split/'latest.pt').read_bytes()
    assert campaign._train_locked(cfg,'test',split,resume=True,device='cpu',workers=0,max_steps=None,parent=parent)==0
    assert (split/'latest.pt').read_bytes()==before
    bad=load(split/'latest.pt');bad['cfg']['config']['parent']={'changed':True};torch.save(bad,split/'latest.pt')
    with pytest.raises(ValueError,match='resume refused'):campaign._train_locked(cfg,'test',split,resume=True,device='cpu',workers=0,max_steps=None,parent=parent)


def test_full_parent_contract_uses_real_weight_and_rng_schema(tmp_path):
    plan,_=campaign.read_plan(campaign.PLAN)
    original=ROOT/'output/shared_base_scaling_download_2026-09-17/seed0'
    if not original.is_dir():pytest.skip('local pinned100k weights not downloaded')
    weights=load(original/'model_100000.pt');receipt=deepcopy(weights['cfg'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt['runtime']=runtime_contract(torch.device('cpu'));receipt['device']='cpu';weights['cfg']=receipt
    model=GPT(**receipt['model']);model.load_state_dict(weights['model'])
    opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.dim()>=2],weight_decay=.1),
                         dict(params=[p for p in model.parameters() if p.dim()<2],weight_decay=0.)],lr=3e-5,betas=(.9,.95))
    state=opt.state_dict()
    for group in state['param_groups']:group['lr']=old.lr_at(receipt['config'],99999)
    for i,p in zip([i for group in state['param_groups'] for i in group['params']],
                   [p for group in opt.param_groups for p in group['params']]):
        state['state'][i]=dict(step=torch.tensor(100000.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
    ckpt=dict(weights,schema=old.CHECKPOINT_SCHEMA,opt=state,rng=rng_state(),validation_hashes=receipt['validation_hashes'])
    (tmp_path/'config.json').write_text(json.dumps(receipt));(tmp_path/'status.json').write_text(json.dumps(dict(step=100000,complete=True,reason='completed')))
    torch.save(weights,tmp_path/'model_100000.pt');torch.save(ckpt,tmp_path/'latest.pt')
    plan['parent']['seeds']['0']={name:campaign.file_hash(tmp_path/name) for name in ('config.json','status.json','model_100000.pt')}
    got,ancestry=campaign.verify_parent(plan,0,tmp_path,'cpu');assert got['step']==100000 and ancestry['latest_sha256']==campaign.file_hash(tmp_path/'latest.pt')
    first=next(iter(ckpt['model']));ckpt['model'][first]=ckpt['model'][first]+.01;torch.save(ckpt,tmp_path/'latest.pt')
    with pytest.raises(ValueError,match='tensors differ'):campaign.verify_parent(plan,0,tmp_path,'cpu')
    ckpt['model']=load(tmp_path/'model_100000.pt')['model'];ckpt['opt']['state'][0]['step']=torch.tensor(99999.);torch.save(ckpt,tmp_path/'latest.pt')
    with pytest.raises(ValueError,match='optimizer state'):campaign.verify_parent(plan,0,tmp_path,'cpu')


def test_compact_fixed_inputs_and_tampering(tmp_path):
    import shared_base_campaign_eval as evaluation
    plan,_=campaign.read_plan(campaign.PLAN);folder=ROOT/'output/shared_base_campaign_inputs_2026-09-18'
    if not folder.is_dir():pytest.skip('compact frozen inputs not exported')
    values=evaluation.inputs(plan,folder)
    assert len(values)==4 and values[4]['tokens'].shape==(256,1025)
    for depth,a in values.items():
        source=json.loads((ROOT/f'output/shared_base_conditional_curves_2026-09-15/a1_L{depth}/data/seq_000000.json').read_text())
        expected=np.asarray(source['oracle']['transition_probs'])[a['tokens'][0,a['times']-1]]
        assert np.array_equal(a['oracle_probs'][0],expected)
    changed=deepcopy(plan);changed['evaluation']['inputs_manifest_sha256']='x'
    with pytest.raises(ValueError,match='manifest changed'):evaluation.inputs(changed,folder)


def test_launcher_defaults_ungated_continuation():
    script=ROOT/'code/transformer/kuma/submit_shared_base_campaign.sh'
    out=subprocess.check_output(['bash',str(script)],text=True)
    assert 'continue_large' in out and '--array=0-1%2' in out and 'fresh' in out
    out=subprocess.check_output(['bash',str(script),'--xlarge','--resume','--seed','1'],text=True)
    assert 'xlarge' in out and '--array=1' in out and 'resume' in out
    batch=(ROOT/'code/transformer/kuma/shared_base_campaign_train.sbatch').read_text()
    assert '--time=24:00:00' in batch and '$arm == xlarge' in batch and '--check-only' in batch


def test_benchmark_timing_precision_memory_and_budget_gates(tmp_path):
    import shared_base_campaign_benchmark as benchmark
    plan=json.loads(campaign.PLAN.read_text());plan_hash=campaign.file_hash(campaign.PLAN)
    cfg=dict(plan['arms']['xlarge']['config'],**plan['source'],seed=0,schema='shared-base-campaign-training-v1',arm='xlarge',parent=None,continuation_lr=None)
    receipt=dict(config=cfg,config_hash=digest_json(cfg),plan_sha256=plan_hash,implementation_hashes=plan['implementation_sha256'],
        device='cuda',precision='float32',n_params=25809920,runtime=dict(device='cuda',float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False))
    state=dict(step=1000,reason='max_steps',complete=False,completed_sequences=64000,production_steps=400000,config_hash=digest_json(cfg),plan_sha256=plan_hash)
    runtime=dict(plan_sha256=plan_hash,device='cuda',arm='xlarge',seed=0,step=1000,return_code=75,
        cuda_device='test-only',max_memory_allocated_bytes=1,max_memory_reserved_bytes=2,total_memory_bytes=3)
    for name,value in [('config.json',receipt),('status.json',state),('campaign_runtime.json',runtime)]:
        (tmp_path/name).write_text(json.dumps(value))
    logs=[dict(event='train',step=i,run_elapsed_seconds=i*.2,loss_bits=5.) for i in range(100,1001,100)]
    (tmp_path/'log.jsonl').write_text('\n'.join(json.dumps(r) for r in logs))
    result=benchmark.collect(campaign.PLAN,tmp_path)
    assert result['projected_training_hours']==pytest.approx(22.2222222222)
    (tmp_path/'benchmark.json').write_text(json.dumps(result))
    cli=[sys.executable,str(ROOT/'code/scripts/shared_base_campaign_benchmark.py'),'--plan',str(campaign.PLAN),'--run',str(tmp_path),'--check-only']
    subprocess.run(cli,check=True,capture_output=True)
    (tmp_path/'benchmark.json').write_text('{}')
    assert subprocess.run(cli,capture_output=True).returncode!=0
    for field,value,message in [('device','cpu','GPU'),('max_memory_allocated_bytes',float('nan'),'memory'),('max_memory_reserved_bytes',4,'memory')]:
        changed=dict(runtime);changed[field]=value;(tmp_path/'campaign_runtime.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match=message):benchmark.collect(campaign.PLAN,tmp_path)
    (tmp_path/'campaign_runtime.json').write_text(json.dumps(runtime))
    changed=dict(runtime,max_memory_reserved_bytes=2.9);(tmp_path/'campaign_runtime.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='headroom'):benchmark.collect(campaign.PLAN,tmp_path)
    (tmp_path/'campaign_runtime.json').write_text(json.dumps(runtime))
    changed=dict(state,reason='SIGUSR1');(tmp_path/'status.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='cap'):benchmark.collect(campaign.PLAN,tmp_path)
    (tmp_path/'status.json').write_text(json.dumps(state))
    logs=[dict(r,run_elapsed_seconds=r['step']) for r in logs]
    (tmp_path/'log.jsonl').write_text('\n'.join(json.dumps(r) for r in logs))
    with pytest.raises(ValueError,match='exceeds72'):benchmark.collect(campaign.PLAN,tmp_path)

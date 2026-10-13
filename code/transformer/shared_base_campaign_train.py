"""Additional shared-base training, with explicit parent and resume contracts.

Both arms use the original indexed data stream. A continuation retains the
parent optimizer and RNG at update100000; fresh larger models start at update0.
Exit75 is a saved, incomplete run. Nothing silently starts over or changes budget.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT
import train_shared_base_scaling as previous
from train_shared_base_scaling import (SOURCE, SOURCE_CONTRACT, PRODUCTION, IndexedSharedBase,
    draw_batch, batch_rng, draw_validation, require, file_hash)
from train_mixture import (atomic_save, digest_json, evaluate, initialize_worker,
    loader_iterator, lr_at, restore_rng, rng_state, runtime_contract, write_json)

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'code/transformer/configs/shared_base_campaign_plan_2026-09-18.json'
CHECKPOINT_SCHEMA='shared-base-campaign-checkpoint-v1'


def implementation_hashes():
    return dict(previous.implementation_hashes(), **{'shared_base_campaign_train.py':file_hash(__file__)})


def parameter_count(cfg):
    d=cfg['d_model']
    return cfg['n_layer']*(12*d*d+13*d)+d*(SOURCE['n_ctx']+2*SOURCE['d']+2)


def read_plan(path):
    path=Path(path); plan=json.loads(path.read_text())
    require(plan['schema']=='shared-base-campaign-plan-v1' and plan['seeds']==[0,1]
            and all(type(seed)is int for seed in plan['seeds']), 'unplanned campaign')
    require(plan['source']==SOURCE and plan['implementation_sha256']==implementation_hashes(), 'source or implementation changed')
    old, old_hash=previous.read_plan(ROOT/plan['parent']['plan_path'])
    require(old_hash==plan['parent']['plan_sha256'] and plan['parent']['step']==100000, 'parent plan changed')
    require(set(plan['arms'])=={'continue_large','xlarge'}, 'unexpected arm')
    for name,arm in plan['arms'].items():
        cfg=arm['config']; expected=dict(PRODUCTION,steps=400000,weight_steps=[200000,400000])
        if name=='xlarge': expected.update(n_layer=8,d_model=512,weight_steps=[100000,200000,400000])
        require(cfg==expected, 'planned architecture, schedule or monitoring changed')
        require(all(type(cfg[k])is type(v) for k,v in expected.items()), 'configuration types changed')
        require(arm['n_params']==parameter_count(dict(cfg,**SOURCE)), 'parameter count differs')
    require(plan['arms']['continue_large']['continuation_lr']==3e-5, 'continuation LR changed')
    require(plan['benchmark']==dict(arm='xlarge',seed=0,steps=1000,maximum_projected_hours=72,
                                   maximum_memory_fraction=.92), 'benchmark gate changed')
    return plan,file_hash(path)


def verify_parent(plan, seed, directory, device):
    """Bind the full optimizer checkpoint to the pinned published100k weights."""
    directory=Path(directory); pins=plan['parent']['seeds'][str(seed)]
    for name in ('config.json','status.json','model_100000.pt'):
        require(file_hash(directory/name)==pins[name], 'parent file changed: '+name)
    receipt=json.loads((directory/'config.json').read_text()); state=json.loads((directory/'status.json').read_text())
    weights=torch.load(directory/'model_100000.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    expected_cfg=previous.planned_config({'config':PRODUCTION,'source':SOURCE},seed)
    require(receipt['config']==expected_cfg and receipt['config_hash']==digest_json(expected_cfg), 'parent configuration changed')
    require(state['step']==100000 and state['complete'] is True and state['reason']=='completed', 'parent not complete')
    require(weights['schema']=='shared-base-scaling-weights-v1' and checkpoint['schema']==previous.CHECKPOINT_SCHEMA, 'wrong parent schema')
    for key,value in [('step',100000),('completed_batches',100000),('completed_sequences',6400000),
                      ('cfg',receipt),('config_hash',receipt['config_hash']),('plan_sha256',plan['parent']['plan_sha256']),
                      ('implementation_hashes',previous.implementation_hashes()),('source_contract_hash',digest_json(SOURCE_CONTRACT))]:
        require(checkpoint.get(key)==value and weights.get(key)==value, 'parent metadata differs: '+key)
    require(receipt['runtime']==runtime_contract(torch.device(device)), 'parent runtime differs')
    require(checkpoint['validation_hashes']==receipt['validation_hashes'], 'parent validation differs')
    require(set(checkpoint['model'])==set(weights['model']), 'parent model keys differ')
    for name,tensor in checkpoint['model'].items():
        require(torch.isfinite(tensor).all().item() and torch.equal(tensor,weights['model'][name]), 'parent tensors differ: '+name)
    groups=checkpoint['opt']['param_groups']
    architecture=GPT(64,1024,4,8,256)
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in architecture.parameters() if p.dim()>=2],weight_decay=.1),
        dict(params=[p for p in architecture.parameters() if p.dim()<2],weight_decay=0.)],lr=3e-4,betas=(.9,.95))
    expected_groups=optimizer.state_dict()['param_groups']
    require(len(groups)==2 and all({k:v for k,v in got.items() if k!='lr'}==
            {k:v for k,v in expected.items() if k!='lr'} for got,expected in zip(groups,expected_groups)),
            'parent optimizer settings differ')
    require(all(math.isclose(g['lr'],previous.lr_at(expected_cfg,99999),rel_tol=1e-12) for g in groups),
            'parent final learning rate differs')
    require(len(checkpoint['opt']['state'])==sum(len(g['params']) for g in groups), 'parent optimizer state incomplete')
    for item in checkpoint['opt']['state'].values():
        require(float(item['step'])==100000 and all(torch.isfinite(item[k]).all().item() for k in ('exp_avg','exp_avg_sq')), 'invalid parent optimizer state')
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    for index,parameter in enumerate(parameters):
        item=checkpoint['opt']['state'][index]
        require(all(item[k].shape==parameter.shape and item[k].dtype==torch.float32 for k in ('exp_avg','exp_avg_sq')),
                'parent optimizer tensor shape or precision differs')
    require(set(checkpoint['rng']) == {'torch','numpy','python','cuda'}, 'parent RNG missing')
    expected_model=dict(vocab=64,n_ctx=1024,n_layer=4,n_head=8,d_model=256,mlp=True,dropout=0.)
    require(receipt['model']==expected_model and receipt['n_params']==3454464, 'parent architecture differs')
    ancestry=dict(directory=str(directory.resolve()),latest_sha256=file_hash(directory/'latest.pt'),
                  weights_sha256=pins['model_100000.pt'],config_hash=receipt['config_hash'],step=100000)
    return checkpoint,ancestry


def train(plan_path, arm, seed, out, *, parent_root=None, resume=False, device='cpu', workers=0, max_steps=None):
    plan,plan_hash=read_plan(plan_path)
    require(arm in plan['arms'] and seed in plan['seeds'] and type(seed)is int,'unplanned arm/seed')
    require(device in ('cpu','cuda') and workers>=0,'invalid execution settings')
    require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(out); require(not out.is_symlink(),'output must not be symlink')
    parent=None; ancestry=None
    if arm=='continue_large':
        require(parent_root is not None,'continuation needs the original training directory')
        directory=Path(parent_root)/f'seed{seed}'
        require(not (out.resolve()==directory.resolve() or out.resolve() in directory.resolve().parents
                      or directory.resolve() in out.resolve().parents),'output overlaps parent')
        parent,ancestry=verify_parent(plan,seed,directory,device)
    else:
        require(parent_root is None,'fresh arm cannot have parent')
    cfg=dict(schema='shared-base-campaign-training-v1',**plan['arms'][arm]['config'],**SOURCE,seed=seed,
             arm=arm,parent=ancestry,continuation_lr=plan['arms'][arm].get('continuation_lr'))
    require(max_steps is None or type(max_steps)is int and 0<=max_steps<=cfg['steps'],'invalid absolute step cap')
    if (out/'status.json').is_file() and json.loads((out/'status.json').read_text()).get('complete') is True:
        resume=True  # Validate the complete checkpoint, then skip without rewriting it.
    if resume: require((out/'latest.pt').is_file(),'resume requires latest.pt')
    out.mkdir(parents=True,exist_ok=True)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    with (out/'.training.lock').open('a+') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('another trainer holds this run') from error
        code=_train_locked(cfg,plan_hash,out,resume=resume,device=device,workers=workers,max_steps=max_steps,parent=parent)
    state=json.loads((out/'status.json').read_text())
    write_json(out/'campaign_runtime.json',dict(schema='shared-base-campaign-runtime-v1',plan_sha256=plan_hash,arm=arm,seed=seed,
        step=state['step'],device=device,return_code=code,cuda_device=torch.cuda.get_device_name() if device=='cuda' else None,
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        max_memory_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None))
    return code

def _train_locked(cfg, plan_hash, out, *, resume, device, workers, max_steps, parent=None):
    """Internal engine; public train() has already checked the immutable plan."""
    latest=out/'latest.pt'; target=torch.device(device)
    require(torch.get_default_dtype()==torch.float32, 'training requires the float32 default dtype')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    hashes=implementation_hashes(); runtime=runtime_contract(target)
    config_hash=digest_json(cfg); source_hash=digest_json(SOURCE_CONTRACT)
    checkpoint=None
    if resume:
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
        for key,expected in [('schema',CHECKPOINT_SCHEMA),('plan_sha256',plan_hash),
                ('config_hash',config_hash),('implementation_hashes',hashes),('source_contract_hash',source_hash)]:
            require(checkpoint.get(key)==expected,f'resume refused: {key} differs')
        require(checkpoint['cfg']['runtime']==runtime, 'resume refused: runtime differs')
        require(checkpoint['cfg']['config']==cfg, 'resume refused: saved configuration differs')
        require(json.loads((out/'config.json').read_text())==checkpoint['cfg'], 'resume refused: config.json differs')
        step=checkpoint['step']
        require(type(step) is int and 0<=step<=cfg['steps']
                and checkpoint['completed_batches']==step
                and checkpoint['completed_sequences']==step*cfg['batch'], 'resume refused: completed counts differ')
    else:
        require(not any(p.name!='.training.lock' for p in out.iterdir()), 'output directory is nonempty; use --resume or a new directory')
        step=0 if parent is None else parent['step']
    stop_step=cfg['steps'] if max_steps is None else min(cfg['steps'],max_steps)
    require(stop_step>=step, 'max_steps is below the already completed step')

    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    model=GPT(64,cfg['n_ctx'],cfg['n_layer'],cfg['n_head'],cfg['d_model'],mlp=True).to(target).float()
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.dim()>=2],weight_decay=cfg['wd']),
        dict(params=[p for p in model.parameters() if p.dim()<2],weight_decay=0.)],
        lr=cfg['lr'],betas=(.9,.95))
    total_elapsed=0.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['opt'])
        total_elapsed=checkpoint['training_elapsed_seconds']; restore_rng(checkpoint['rng'])
    elif parent is not None:
        model.load_state_dict(parent['model']); optimizer.load_state_dict(deepcopy(parent['opt'])); restore_rng(parent['rng'])
    validation=draw_validation(cfg)
    validation_hashes={name:hashlib.sha256(x.numpy().tobytes()).hexdigest() for name,x in validation.items()}
    if parent is not None:
        require(parent['validation_hashes']==validation_hashes, 'parent monitoring data changed')
    if checkpoint is not None:
        require(checkpoint['validation_hashes']==validation_hashes, 'resume refused: monitoring data changed')
    receipt=dict(config=cfg,config_hash=config_hash,plan_sha256=plan_hash,
        family='lsa',d=64,n_ctx=cfg['n_ctx'],alpha=[.5,1.5],ls=[0,8],no_permute=False,seed=cfg['seed'],
        model=model.cfg,n_params=model.n_params(),source_contract=SOURCE_CONTRACT,
        source_contract_hash=source_hash,implementation_hashes=hashes,validation_hashes=validation_hashes,
        precision='float32',runtime=runtime,device=device)
    if checkpoint is not None:
        require(receipt==checkpoint['cfg'], 'resume refused: regenerated configuration differs')
        if step==cfg['steps']:
            return 0
    else:
        write_json(out/'config.json',receipt)

    stopped={'signal':None}; previous={}
    def request_stop(signum,_frame):
        stopped['signal']=signal.Signals(signum).name
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:
        previous[sig]=signal.signal(sig,request_stop)
    start=time.monotonic()
    def save(reason):
        saved=dict(schema=CHECKPOINT_SCHEMA,model=model.state_dict(),opt=optimizer.state_dict(),step=step,
            cfg=receipt,config_hash=config_hash,plan_sha256=plan_hash,implementation_hashes=hashes,
            source_contract_hash=source_hash,rng=rng_state(),completed_batches=step,
            completed_sequences=step*cfg['batch'],validation_hashes=validation_hashes,stop_reason=reason,
            training_elapsed_seconds=total_elapsed+time.monotonic()-start)
        atomic_save(latest,lambda f:torch.save(saved,f))
        if step in cfg['weight_steps']:
            weights={k:saved[k] for k in ['model','step','cfg','config_hash','implementation_hashes',
                'source_contract_hash','completed_sequences','completed_batches','plan_sha256']}
            weights['schema']='shared-base-campaign-weights-v1'
            atomic_save(out/f'model_{step}.pt',lambda f:torch.save(weights,f))
        write_json(out/'status.json',dict(schema='shared-base-campaign-status-v1',step=step,
            production_steps=cfg['steps'],complete=step==cfg['steps'],reason=reason,
            completed_sequences=saved['completed_sequences'],config_hash=config_hash,plan_sha256=plan_hash,
            training_elapsed_seconds=saved['training_elapsed_seconds']))
    try:
        if checkpoint is None:
            save('initialized')
        dataset=IndexedSharedBase(cfg['seed'],cfg['batch'],cfg['n_ctx'],step,stop_step)
        loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,
            generator=torch.Generator().manual_seed(171717),persistent_workers=False,
            worker_init_fn=initialize_worker,multiprocessing_context='spawn' if workers else None,
            timeout=120 if workers else 0,pin_memory=device=='cuda')
        model.train(); running_loss=0.; running_steps=0
        with (out/'log.jsonl').open('a',buffering=1) as log:
            for xb in loader_iterator(loader):
                if stopped['signal']:
                    break
                learning_rate=cfg['continuation_lr'] if cfg['arm']=='continue_large' else lr_at(cfg,step)
                for group in optimizer.param_groups:
                    group['lr']=learning_rate
                xb=xb.to(target,non_blocking=device=='cuda')
                logits=model(xb[:,:-1])
                loss=F.cross_entropy(logits.reshape(-1,64),xb[:,1:].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite training loss at batch{step}')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)
                optimizer.step(); step+=1
                running_loss+=float(loss.detach()); running_steps+=1
                if step%cfg['log_every']==0:
                    record=dict(event='train',step=step,loss_bits=running_loss/running_steps/math.log(2),
                        lr=learning_rate,completed_sequences=step*cfg['batch'],run_elapsed_seconds=time.monotonic()-start)
                    log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                    running_loss=0.; running_steps=0
                if not stopped['signal'] and (step%cfg['val_every']==0 or step==cfg['steps']):
                    record=dict(event='validation',step=step,val=evaluate(model,validation,target,cfg['val_chunk']))
                    log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                if step%cfg['ckpt_every']==0 or step in cfg['weight_steps']:
                    save('periodic')
                if stopped['signal']:
                    break
        reason='completed' if step==cfg['steps'] else stopped['signal'] or 'max_steps'
        save(reason)
        return 0 if step==cfg['steps'] else 75
    finally:
        for sig,handler in previous.items():
            signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=PLAN); p.add_argument('--arm',choices=['continue_large','xlarge'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1],required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent-root',type=Path); p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--workers',type=int,default=6)
    p.add_argument('--max-steps',type=int); a=p.parse_args()
    return train(a.plan,a.arm,a.seed,a.out,parent_root=a.parent_root,resume=a.resume,device=a.device,workers=a.workers,max_steps=a.max_steps)

if __name__=='__main__':
    raise SystemExit(main())

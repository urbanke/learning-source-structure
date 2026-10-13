"""Fresh larger-model training on the unchanged shared-base LSA prior.

The immutable plan fixes the production schedule. --max-steps only caps the
current invocation; exit75 means an atomic resumable checkpoint, never completed
production. Indexed source batches make worker count/prefetching immaterial.
CPU continuation is bitwise tested; CUDA kernels may still be nondeterministic.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT
from source import Source, seed_of
from train_mixture import (atomic_save, digest_json, evaluate, initialize_worker,
    loader_iterator, lr_at, restore_rng, rng_state, runtime_contract, write_json)

SOURCE = dict(d=64, n_ctx=1024, alpha=[.5,1.5], ls=[0,8], permute=True)
PRODUCTION = dict(batch=64, steps=100000, warmup=1000, lr=3e-4, wd=.1, clip=1.,
    n_layer=4, n_head=8, d_model=256, val_n=64,
    val_settings=[[1,2],[1,4],[.75,2],[1.25,3]], val_chunk=32,
    log_every=100, val_every=1000, ckpt_every=5000, weight_steps=[50000,100000])
SOURCE_CONTRACT = dict(version='shared-base-scaling-source-v1', alphabet=64,
    alpha_uniform=[.5,1.5], depth_uniform_inclusive=[0,8], alphabet_permutation=True,
    parameters='Fresh exponent, common depth, rank permutation and transition matrix per sequence.',
    initial_token='Drawn from the same base law q.', model_input='Tokens only.',
    sampling='indexed-batch-seedsequence-v1')
IMPLEMENTATIONS = ('source.py','families.py','model.py','mixture_source.py',
                   'train_mixture.py','train_shared_base_scaling.py')
CHECKPOINT_SCHEMA = 'shared-base-scaling-checkpoint-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    return {name:file_hash(Path(__file__).parent/name) for name in IMPLEMENTATIONS}


def planned_config(plan, seed):
    require(type(seed) is int and seed in [0,1], 'seed must be0 or1')
    return dict(schema='shared-base-scaling-training-v1', **plan['config'], **plan['source'], seed=seed)


def read_plan(path):
    raw=Path(path).read_bytes(); plan=json.loads(raw)
    require(plan['schema']=='shared-base-scaling-plan-v1' and plan['seeds']==[0,1]
            and all(type(s) is int for s in plan['seeds']), 'unexpected scaling plan')
    require(plan['source']==SOURCE, 'source prior or context length changed')
    require(plan['config']==PRODUCTION, 'production architecture, schedule or monitoring settings changed')
    require(all(type(plan['config'][k]) is int for k in ['batch','steps','warmup','n_layer','n_head',
            'd_model','val_n','val_chunk','log_every','val_every','ckpt_every'])
            and all(type(s) is int for s in plan['config']['weight_steps'])
            and type(plan['source']['d']) is int and type(plan['source']['n_ctx']) is int
            and plan['source']['permute'] is True, 'configuration field types changed')
    require(type(plan['n_params']) is int and plan['n_params']==3454464, 'unexpected production parameter count')
    require(plan['implementation_sha256']==implementation_hashes(), 'planned implementation changed')
    return plan,hashlib.sha256(raw).hexdigest()


def batch_rng(seed, index):
    require(type(seed) is int and type(index) is int and 0<=seed<2**64 and 0<=index<2**64,
            'invalid indexed source seed')
    return np.random.default_rng(np.random.SeedSequence(
        [0x53424C53,seed & 0xffffffff,seed>>32,index & 0xffffffff,index>>32]))


def draw_batch(seed, index, batch, n_ctx):
    source=Source(64,n_ctx+1,alpha=(.5,1.5),ls=(0,8),permute=True)
    return source.batch(batch,batch_rng(seed,index))[0]


class IndexedSharedBase(torch.utils.data.Dataset):
    def __init__(self, seed, batch, n_ctx, start, stop):
        self.seed,self.batch,self.n_ctx,self.start,self.stop=seed,batch,n_ctx,start,stop
    def __len__(self):
        return self.stop-self.start
    def __getitem__(self, index):
        if not 0<=index<len(self):
            raise IndexError(index)
        return torch.from_numpy(draw_batch(self.seed,self.start+index,self.batch,self.n_ctx))


def draw_validation(cfg):
    """Same fixed source settings and exact monitoring RNGs as historical train.py."""
    values={}
    for alpha,depth in cfg['val_settings']:
        alpha=float(alpha); depth=int(depth)
        source=Source(64,cfg['n_ctx']+1,alpha=alpha,ls=depth,permute=True)
        tokens,_=source.batch(cfg['val_n'],np.random.default_rng(seed_of('val',64,alpha,depth)))
        values[f'a{alpha:g}_L{depth}']=torch.from_numpy(tokens)
    return values


def train(plan_path, seed, out, *, resume=False, device='cpu', workers=0, max_steps=None):
    plan,plan_hash=read_plan(plan_path)
    cfg=planned_config(plan,seed)
    require(type(workers) is int and workers>=0, 'workers must be nonnegative')
    require(max_steps is None or (type(max_steps) is int and 0<=max_steps<=cfg['steps']), 'invalid absolute step cap')
    require(device in ['cpu','cuda'], 'device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out=Path(out)
    require(not out.is_symlink(), 'output directory must not be a symlink')
    if resume:
        require((out/'latest.pt').is_file(), 'resume requires latest.pt; will not restart')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another trainer holds this output directory') from exc
        try:
            return _train_locked(cfg,plan_hash,out,resume=resume,device=device,workers=workers,max_steps=max_steps)
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def _train_locked(cfg, plan_hash, out, *, resume, device, workers, max_steps):
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
        step=0
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
    validation=draw_validation(cfg)
    validation_hashes={name:hashlib.sha256(x.numpy().tobytes()).hexdigest() for name,x in validation.items()}
    if checkpoint is not None:
        require(checkpoint['validation_hashes']==validation_hashes, 'resume refused: monitoring data changed')
    receipt=dict(config=cfg,config_hash=config_hash,plan_sha256=plan_hash,
        family='lsa',d=64,n_ctx=cfg['n_ctx'],alpha=[.5,1.5],ls=[0,8],no_permute=False,seed=cfg['seed'],
        model=model.cfg,n_params=model.n_params(),source_contract=SOURCE_CONTRACT,
        source_contract_hash=source_hash,implementation_hashes=hashes,validation_hashes=validation_hashes,
        precision='float32',runtime=runtime,device=device)
    if checkpoint is not None:
        require(receipt==checkpoint['cfg'], 'resume refused: regenerated configuration differs')
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
            weights['schema']='shared-base-scaling-weights-v1'
            atomic_save(out/f'model_{step}.pt',lambda f:torch.save(weights,f))
        write_json(out/'status.json',dict(schema='shared-base-scaling-status-v1',step=step,
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
                learning_rate=lr_at(cfg,step)
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv); torch.set_num_threads(1)
    if args.device=='cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    code=train(args.plan,args.seed,args.out,resume=args.resume,device=args.device,workers=args.workers,max_steps=args.max_steps)
    status=json.loads((args.out/'status.json').read_text())
    runtime=dict(schema='shared-base-scaling-runtime-v1',plan_sha256=file_hash(args.plan),seed=args.seed,
        step=status['step'],device=args.device,return_code=code,cuda_device=None,
        max_memory_allocated_bytes=None,max_memory_reserved_bytes=None,total_memory_bytes=None)
    if args.device=='cuda':
        runtime.update(cuda_device=torch.cuda.get_device_name(),max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    write_json(args.out/'scaling_runtime.json',runtime)
    return code


if __name__=='__main__':
    sys.exit(main())

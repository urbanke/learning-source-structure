"""Further hidden-encoding scaling using immutable, tested training engines.

Continue the verified four-layer models at their existing learning-rate floor,
or train the predeclared eight-layer models on the original indexed streams.
Campaign provenance distinguishes these checkpoints from earlier experiments.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch

import train_hidden_factor as original
import train_hidden_factor_scaling as previous
from train_mixture import write_json

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'code/transformer/configs/hidden_factor_campaign_plan_2026-09-18.json'
SCHEMA = 'hidden-factor-campaign-plan-v1'
require = previous.require
sha = previous.file_hash


def parameter_count(config, dimensions):
    d = config['d_model']
    return config['n_layer']*(12*d*d+13*d) + d*(config['n_ctx']+2*math.prod(dimensions)+2)


def read_plan(path):
    path = Path(path); plan = json.loads(path.read_text())
    require(plan['schema'] == SCHEMA and plan['seeds'] == [0, 1], 'Unexpected campaign plan or seeds')
    require(plan['implementation_sha256'] == original.implementation_hashes(), 'Immutable numerical modules changed')
    for name, digest in plan['adapter_dependencies_sha256'].items():
        require(sha(Path(__file__).parent/name) == digest, 'Immutable adapter dependency changed: '+name)
    require(sha(ROOT/plan['parent']['plan_path']) == plan['parent']['plan_sha256'], 'Original scaling plan changed')
    require(set(plan['arms']) == {'continue_large', 'xlarge'}, 'Unexpected campaign arm')
    parent = plan['parent']['config']; continuation = plan['arms']['continue_large']; larger = plan['arms']['xlarge']
    require(continuation['mode'] == 'continuation' and larger['mode'] == 'fresh', 'Invalid arm modes')
    require(parent['steps'] == plan['parent']['final_step'] == continuation['parent_step']
            and plan['parent']['completed_sequences'] == parent['steps']*parent['batch'], 'Parent budget mismatch')
    require({k:v for k,v in parent.items() if k not in ('steps', 'weight_steps')}
            == {k:v for k,v in continuation['config'].items() if k not in ('steps', 'weight_steps')},
            'Continuation changes undeclared settings')
    require(math.isclose(continuation['continuation_lr'], .1*parent['lr'], rel_tol=1e-14), 'Continuation changed LR floor')
    require(continuation['additional_sequences'] == (continuation['config']['steps']-parent['steps'])*parent['batch'],
            'Continuation sample budget mismatch')
    require(larger['total_sequences'] == larger['config']['steps']*larger['config']['batch'], 'New-model sample budget mismatch')
    for arm, spec in plan['arms'].items():
        for seed in plan['seeds']:
            cfg = original.validate_config(dict(spec['config'], source_dimensions=plan['source_dimensions'], seed=seed))
            require(parameter_count(cfg, plan['source_dimensions']) == spec['n_params'], 'Planned parameter count differs')
            require(spec['milestones'] == cfg['weight_steps'] and spec['milestones'][-1] == plan['primary']['final_steps'][arm],
                    'Milestones or final checkpoint changed')
    require(plan['benchmark']['arm'] == 'xlarge' and plan['benchmark']['seed'] == 0
            and plan['benchmark']['steps'] == 1000, 'Benchmark must be the declared 1000-step largest model')
    return plan, sha(path)


def parent_adapter(plan, seed):
    """Describe the verified previous large checkpoint to its unchanged checker."""
    parent_cfg = dict(plan['parent']['config'], scaling_provenance=dict(
        plan_sha256=plan['parent']['plan_sha256'], arm='large', seed=seed,
        trainer_sha256=plan['adapter_dependencies_sha256']['train_hidden_factor_scaling.py'], parent=None))
    return dict(baseline=dict(config=parent_cfg, checkpoint_sha256=plan['parent']['checkpoint_sha256']),
                source_dimensions=plan['source_dimensions'], implementation_sha256=plan['implementation_sha256'])


def train(plan_path, arm, seed, out, *, parent_checkpoint=None, parent_weights=None,
          resume=False, device='cpu', workers=0, max_steps=None):
    plan, digest = read_plan(plan_path)
    require(arm in plan['arms'] and seed in plan['seeds'], 'Unplanned arm or seed')
    require(device in ('cpu', 'cuda') and workers >= 0 and (max_steps is None or max_steps >= 0), 'Invalid execution settings')
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    out = Path(out)
    cfg = original.validate_config(dict(plan['arms'][arm]['config'], seed=seed, source_dimensions=plan['source_dimensions']))
    provenance = dict(plan_sha256=digest, arm=arm, seed=seed, trainer_sha256=sha(__file__), parent=None)
    parent = None
    if arm == 'continue_large':
        require(parent_checkpoint is not None, 'Continuation requires the completed four-layer optimizer checkpoint')
        require(out.resolve() != Path(parent_checkpoint).parent.resolve()
                and out.resolve() not in Path(parent_checkpoint).parents
                and Path(parent_checkpoint).parent.resolve() not in out.resolve().parents, 'Output overlaps the parent run')
        parent, ancestry, validation = previous._parent_checkpoint(parent_adapter(plan, seed), seed,
            parent_checkpoint, torch.device(device), parent_weights)
        provenance['parent'] = ancestry
        # The unchanged continuation engine binds configuration, optimizer/RNG,
        # source batches, runtime and parent tensors. Its internal legacy label
        # is retained in checkpoints; campaign_provenance supplies this arm name.
        cfg['scaling_provenance'] = provenance
    else:
        require(parent_checkpoint is None and parent_weights is None, 'Fresh arm cannot inherit a parent')
        cfg['campaign_provenance'] = provenance
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if arm == 'xlarge':
        code = original.train(cfg, out, resume=resume, device=device, workers=workers, max_steps=max_steps)
    else:
        require(not resume or (out/'latest.pt').is_file(), 'Resume requires latest.pt')
        out.mkdir(parents=True, exist_ok=True)
        with (out/'.training.lock').open('a+') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another trainer holds the campaign output') from error
            adapter = dict(arms={'longsmall': plan['arms'][arm]})
            code = previous._continue_locked(adapter, cfg, out, parent, validation, resume=resume,
                                            device=torch.device(device), workers=workers, max_steps=max_steps)
    state = json.loads((out/'status.json').read_text())
    state.update(campaign_arm=arm, campaign_plan_sha256=digest)
    write_json(out/'status.json', state)
    if device == 'cuda':
        torch.cuda.synchronize()
    write_json(out/'campaign_runtime.json', dict(schema='hidden-factor-campaign-runtime-v1',
        plan_sha256=digest, trainer_sha256=sha(__file__), arm=arm, seed=seed, device=device,
        cuda_device=torch.cuda.get_device_name() if device == 'cuda' else None,
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else None,
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        max_memory_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None,
        step=state['step'], reason=state['reason'], complete=state['complete'], max_steps=max_steps,
        return_code=code, training_elapsed_seconds=state['training_elapsed_seconds']))
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=PLAN); p.add_argument('--arm', choices=('continue_large', 'xlarge'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1), required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent-checkpoint', type=Path); p.add_argument('--parent-weights', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda'); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max-steps', type=int); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    return train(a.plan, a.arm, a.seed, a.out, parent_checkpoint=a.parent_checkpoint, parent_weights=a.parent_weights,
                 resume=a.resume, device=a.device, workers=a.workers, max_steps=a.max_steps)


if __name__ == '__main__':
    raise SystemExit(main())

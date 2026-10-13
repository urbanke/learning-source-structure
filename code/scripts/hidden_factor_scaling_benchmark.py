"""Verify the capped H100 run and report measured larger-model throughput."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def collect(plan_path, run):
    plan_path, run = Path(plan_path), Path(run)
    plan = json.loads(plan_path.read_text())
    require(plan['schema'] == 'hidden-factor-scaling-plan-v1', 'Unexpected plan')
    state = json.loads((run/'status.json').read_text())
    receipt = json.loads((run/'config.json').read_text())
    config = receipt['config']
    cap = plan['benchmark']['steps']
    require(state['step'] == cap and state['reason'] == 'max_steps'
            and state['complete'] is False, 'Benchmark did not reach its intended cap')
    require(receipt['device'] == 'cuda', 'A CUDA throughput measurement is required')
    expected = dict(plan['arms']['large']['config'], seed=0,
                    source_dimensions=plan['source_dimensions'])
    provenance = config.get('scaling_provenance', {})
    require({k: v for k, v in config.items() if k != 'scaling_provenance'} == expected,
            'Benchmark model or training configuration differs from the plan')
    require(provenance.get('plan_sha256') == sha(plan_path), 'Benchmark plan changed')
    require(state['completed_sequences'] == cap*expected['batch']
            and state['production_steps'] == expected['steps'], 'Benchmark counts differ')
    require(receipt['n_params'] == plan['arms']['large']['n_params'], 'Model size differs')
    model = dict(vocab=math.prod(plan['source_dimensions']), n_ctx=expected['n_ctx'],
                 n_layer=expected['n_layer'], n_head=expected['n_head'],
                 d_model=expected['d_model'], mlp=True, dropout=0.0)
    require(receipt['model'] == model, 'Recorded model architecture differs')
    require(receipt['implementation_hashes'] == plan['implementation_sha256'],
            'Benchmark numerical implementation differs')
    require(state['config_hash'] == receipt['config_hash'], 'Configuration identity differs')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
    require(digest == receipt['config_hash'], 'Configuration checksum differs')
    events = [json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
    records = [e for e in events if e.get('event') == 'train']
    require(len(records) >= 2 and records[-1]['step'] == cap, 'Incomplete throughput log')
    require(all(b['step'] > a['step'] and b['run_elapsed_seconds'] > a['run_elapsed_seconds']
                for a, b in zip(records, records[1:])),
            'Benchmark was interrupted; use a fresh benchmark output for a reliable timing')
    first, last = records[0], records[-1]
    elapsed = last['run_elapsed_seconds'] - first['run_elapsed_seconds']
    updates = last['step'] - first['step']
    require(math.isfinite(elapsed) and elapsed > 0 and updates > 0, 'Invalid measured interval')
    require(all(math.isfinite(e['loss_bits']) for e in records), 'Nonfinite training loss')
    seconds_per_step = elapsed/updates
    runtime = json.loads((run/'scaling_runtime.json').read_text())
    require(runtime['plan_sha256'] == sha(plan_path) and runtime['arm'] == 'large'
            and runtime['seed'] == 0 and runtime['step'] == cap
            and runtime['device'] == 'cuda' and runtime['return_code'] == 75,
            'Missing matching GPU runtime receipt')
    require(0 < runtime['max_memory_allocated_bytes'] <= runtime['max_memory_reserved_bytes']
            <= runtime['total_memory_bytes'], 'Invalid GPU memory measurement')
    return dict(schema='hidden-factor-scaling-benchmark-v1', passed=True,
        plan_sha256=sha(plan_path), completed_steps=cap, model_parameters=receipt['n_params'],
        seconds_per_step=seconds_per_step,
        projected_training_hours=seconds_per_step*expected['steps']/3600,
        cuda_device=runtime['cuda_device'],
        peak_allocated_gib=runtime['max_memory_allocated_bytes']/2**30,
        peak_reserved_gib=runtime['max_memory_reserved_bytes']/2**30,
        projection_scope='Steady training throughput only; validation, checkpointing and evaluation add overhead.',
        inputs={name:sha(run/name) for name in ['config.json','status.json','log.jsonl','scaling_runtime.json']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = collect(args.plan, args.run)
    saved = args.run/'benchmark.json'
    if args.check_only:
        require(json.loads(saved.read_text()) == result, 'Benchmark receipt differs')
    else:
        saved.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Review the largest-model 1000-step benchmark before its production jobs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)


def collect(plan_path,run):
    run=Path(run);plan=json.loads(Path(plan_path).read_text());state=json.loads((run/'status.json').read_text())
    receipt=json.loads((run/'config.json').read_text());runtime=json.loads((run/'campaign_runtime.json').read_text())
    cfg=receipt['config'];provenance=cfg['campaign_provenance'];cap=plan['benchmark']['steps']
    require(plan['schema']=='hidden-factor-campaign-plan-v1' and cap==1000,'Unexpected benchmark plan')
    require(state['step']==cap and state['reason']=='max_steps' and state['complete'] is False,'Benchmark missed 1000-step cap')
    expected=dict(plan['arms']['xlarge']['config'],source_dimensions=plan['source_dimensions'],seed=0)
    require({k:v for k,v in cfg.items() if k!='campaign_provenance'}==expected,'Benchmark training configuration differs')
    require(provenance['plan_sha256']==sha(plan_path) and provenance['arm']=='xlarge' and provenance['seed']==0,'Benchmark provenance differs')
    require(receipt['n_params']==plan['arms']['xlarge']['n_params'] and receipt['device']=='cuda','Wrong model or execution device')
    require(runtime['plan_sha256']==sha(plan_path) and runtime['arm']=='xlarge' and runtime['seed']==0
            and runtime['step']==cap and runtime['return_code']==75 and runtime['reason']=='max_steps','GPU runtime receipt differs')
    require(0<runtime['max_memory_allocated_bytes']<=runtime['max_memory_reserved_bytes']<=runtime['total_memory_bytes'], 'Invalid GPU memory values')
    require(runtime['max_memory_reserved_bytes']/runtime['total_memory_bytes']<=plan['benchmark']['maximum_reserved_memory_fraction'],
            'Benchmark leaves insufficient GPU memory headroom')
    records=[r for r in map(json.loads,(run/'log.jsonl').read_text().splitlines()) if r.get('event')=='train']
    require(len(records)>=2 and records[-1]['step']==cap,'Incomplete benchmark timing records')
    require(all(b['step']>a['step'] and b['run_elapsed_seconds']>a['run_elapsed_seconds'] for a,b in zip(records,records[1:])),
            'Benchmark was restarted; use a fresh uninterrupted benchmark output')
    require(all(math.isfinite(r['loss_bits']) for r in records),'Nonfinite training loss')
    speed=(records[-1]['run_elapsed_seconds']-records[0]['run_elapsed_seconds'])/(records[-1]['step']-records[0]['step'])
    hours=speed*expected['steps']/3600
    require(0<hours<=plan['benchmark']['maximum_projected_training_hours'],'Largest model exceeds planned one-week runtime budget')
    return dict(schema='hidden-factor-campaign-benchmark-v1',passed=True,plan_sha256=sha(plan_path),
                completed_steps=cap,model_parameters=receipt['n_params'],seconds_per_step=speed,
                projected_training_hours=hours,projected_milestone_hours={str(s):speed*s/3600 for s in plan['arms']['xlarge']['milestones']},
                projected_24h_chunks=math.ceil(hours/24),peak_allocated_gib=runtime['max_memory_allocated_bytes']/2**30,
                peak_reserved_gib=runtime['max_memory_reserved_bytes']/2**30,cuda_device=runtime['cuda_device'],
                scope='Steady training only; evaluation/checkpointing/queue overhead excluded. Explicit resumes after 24h allocations.',
                inputs={n:sha(run/n) for n in ('config.json','status.json','log.jsonl','campaign_runtime.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args();r=collect(a.plan,a.run);path=a.run/'benchmark.json'
    if a.check_only: require(json.loads(path.read_text())==r,'Saved benchmark receipt changed')
    else: path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()

"""Verify the isolated1000-update GPU benchmark before larger production."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(plan_path, run):
    plan_path=Path(plan_path); run=Path(run); plan=json.loads(plan_path.read_text())
    digest=sha(plan_path); cfg=json.loads((run/'config.json').read_text()); state=json.loads((run/'status.json').read_text())
    runtime=json.loads((run/'campaign_runtime.json').read_text())
    def require(ok,message):
        if not ok: raise ValueError(message)
    require(plan['schema']=='shared-base-campaign-plan-v1','invalid plan')
    require(state['step']==1000 and state['reason']=='max_steps' and state['complete'] is False,'benchmark cap not reached normally')
    require(state['completed_sequences']==64000 and state['production_steps']==400000,'wrong benchmark budget')
    require(all(x['plan_sha256']==digest for x in (cfg,state,runtime)),'benchmark plan differs')
    actual=cfg['config']; expected=dict(plan['arms']['xlarge']['config'],**plan['source'],seed=0,
        schema='shared-base-campaign-training-v1',arm='xlarge',parent=None,continuation_lr=None)
    require(actual==expected,'benchmark configuration differs')
    canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(cfg['config_hash']==state['config_hash']==canonical(expected),'configuration digest differs')
    require(cfg['implementation_hashes']==plan['implementation_sha256'],'implementation differs')
    root=Path(__file__).resolve().parents[2]
    require(all(sha(root/'code/transformer'/n)==v for n,v in plan['implementation_sha256'].items()),'training code changed')
    r=cfg['runtime']
    require(cfg['device']==runtime['device']=='cuda' and runtime['arm']=='xlarge' and runtime['seed']==0
            and runtime['step']==1000 and runtime['return_code']==75,'not a completed GPU throughput benchmark')
    require(cfg['precision']=='float32' and r['device']=='cuda' and r['float32_matmul_precision']=='highest'
            and not r['cuda_matmul_allow_tf32'] and not r['cudnn_allow_tf32'],'wrong precision')
    require(cfg['n_params']==plan['arms']['xlarge']['n_params'],'wrong parameter count')
    lines=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
    records=[x for x in lines if x.get('event')=='train']
    require([x['step'] for x in records]==list(range(100,1001,100)),'benchmark interrupted or incomplete')
    require(all(math.isfinite(x['loss_bits']) and math.isfinite(x['run_elapsed_seconds']) for x in records),
            'nonfinite benchmark loss or time')
    require(all(b['run_elapsed_seconds']>a['run_elapsed_seconds'] for a,b in zip(records,records[1:])),'nonmonotone timings')
    seconds=(records[-1]['run_elapsed_seconds']-records[0]['run_elapsed_seconds'])/900
    hours=seconds*400000/3600
    allocated=runtime['max_memory_allocated_bytes']; reserved=runtime['max_memory_reserved_bytes']; total=runtime['total_memory_bytes']
    require(all(isinstance(v,(int,float)) and math.isfinite(v) for v in (seconds,hours,allocated,reserved,total)),'invalid memory/timing')
    require(seconds>0 and 0<allocated<=reserved<=total,'invalid memory/timing')
    require(reserved<=total*plan['benchmark']['maximum_memory_fraction'],'benchmark leaves less than8percent GPU memory headroom')
    require(hours<=plan['benchmark']['maximum_projected_hours'],'projected training exceeds72hours; replan before production')
    return dict(schema='shared-base-campaign-benchmark-v1',passed=True,plan_sha256=digest,
        seconds_per_step=seconds,projected_training_hours=hours,n_params=cfg['n_params'],cuda_device=runtime['cuda_device'],
        peak_allocated_gib=allocated/2**30,peak_reserved_gib=reserved/2**30,
        projection_scope='400000 training updates only; validation, checkpointing and evaluation add overhead.',
        inputs={name:sha(run/name) for name in ('config.json','status.json','campaign_runtime.json','log.jsonl')})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True); p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    result=collect(a.plan,a.run)
    if a.check_only:
        if json.loads((a.run/'benchmark.json').read_text())!=result:
            raise ValueError('saved benchmark receipt differs from recomputed validation')
    else: (a.run/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__': main()

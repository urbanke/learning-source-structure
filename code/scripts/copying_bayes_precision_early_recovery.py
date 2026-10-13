"""One bounded Linux continuation for eight early references with corrected time bounds.

The original shard is verified terminal; its last checkpoint is retained.
No prior, draw target, audit, or numerical gate changes. No automatic retry.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import copying_bayes_precision_chains as original
r=original.r;b=original.b
SCHEMA='copying-precision-early8-recovery-v1'

def codes():return {str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__),**original.codes()}

def prepare(out,parent,terminal):
    old,_=original.load(parent);receipt=b.read(terminal)
    b.require(receipt['terminal_verified'] and receipt['active_jobs']==[],'Affected shard is not terminal')
    b.require(not (out/'plan.json').exists(),'No repeated preparation');specs=[]
    for old_index in old['shards'][322]['tasks']:
        source=old['tasks'][old_index];t=source['task'];paths=b.accuracy.paths(parent,t)
        b.require(source['early_fresh'] and source['target_draws']==1048576,'Recovery scope changed')
        b.require(not paths['report'].exists(),'Completed inference must be retained, not repeated')
        files={};counts=[0]*4
        if paths['checkpoint'].exists():
            b.require(paths['identity'].exists(),'Checkpoint identity missing')
            identity=b.read(paths['identity'])
            b.require(identity['plan_id']==old['plan_id'] and identity['task']==source and identity['runtime']==old['runtime'],'Checkpoint identity differs')
            files={k:r.bind(paths[k]) for k in ('identity','checkpoint')};counts=r.checkpoint_counts(paths['checkpoint'])
        spec=dict(index=len(specs),task=t,target_draws=source['target_draws'],parent_files=files,parent_draw_counts=counts,
          original_index=old_index,reason='Correct fresh Linux run time; source timing had no sampling or came from laptop')
        specs.append(spec)
    b.require(len(specs)==8,'Need all eight early references')
    p=dict(schema=SCHEMA,parent=str(parent),parent_plan=r.bind(parent/'plan.json'),terminal=r.bind(terminal),
      code_sha256=codes(),runtime=old['runtime'],tasks=specs,
      resources=dict(cpus=6,memory_gib=36,task_seconds=8700,wall_seconds=9000,
        maximum_cpu_hours=120,maximum_sampling_cost_chf=.66,automatic_retries=False))
    p['plan_id']=b.fp(p);b.write(out/'plan.json',p,immutable=True)
    print(json.dumps(dict(event='early_recovery_prepared',tasks=8,retained_draw_counts=[x['parent_draw_counts'] for x in specs],max_chf=.66)),flush=True)

def load(out):
    p=b.read(out/'plan.json');b.require(p['schema']==SCHEMA and p['plan_id']==b.fp({k:v for k,v in p.items() if k!='plan_id'}),'Plan changed')
    b.require(p['code_sha256']==codes(),'Code changed');r.verify(p['parent_plan']);r.verify(p['terminal'])
    old,parent=original.load(Path(p['parent']));b.require(p['runtime']==old['runtime'],'Runtime changed')
    return p,parent

def worker(out,index):
    p,old=load(out);previous=r.load;r.load=lambda folder:(p,old)
    try:return r.worker(out,index)
    finally:r.load=previous

def dispatch(out,index):
    p,_=load(out);b.require(0<=index<8,'Invalid recovery index')
    folder=out/'attempts';folder.mkdir(exist_ok=True)
    with (folder/f'{index}.lock').open('x'):
        child=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(index)],start_new_session=True)
        try:status=child.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:b.frequency.stop(child);status=75
    b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,returncode=status),immutable=True)
    return status

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','dispatch','worker'))
    a.add_argument('--out',type=Path,required=True);a.add_argument('--parent',type=Path);a.add_argument('--terminal',type=Path);a.add_argument('--index',type=int)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.terminal);return 0
    return globals()[v.stage](v.out,v.index)
if __name__=='__main__':raise SystemExit(main())

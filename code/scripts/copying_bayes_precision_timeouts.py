"""One frozen continuation of terminal copying timeouts, with unchanged targets.

Other tasks in the original shards remain live. A completed per-task dispatcher
receipt proves the selected child has stopped before its checkpoint is copied.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import copying_bayes_precision_chains as original
r=original.r;b=original.b
SCHEMA='copying-precision-terminal-timeouts-v1'

def codes():return {str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__),**original.codes()}

def prepare(out,parent,candidates):
    p,_=original.load(parent);items=b.read(candidates)
    b.require(not (out/'plan.json').exists() and len(items)==40,'Need new fixed40-case continuation')
    specs=[];retained={};snapshots=[]
    b.require(len({x['index'] for x in items})==40,'Repeated candidate')
    for x in items:
        i=x['index'];source=p['tasks'][i];t=source['task']
        b.require(x['spec']==source and x['returncode']==75,'Candidate identity differs')
        shard=p['shards'][int(x['shard'])];b.require(i in shard['tasks'],'Candidate belongs to another shard')
        status=b.read(parent/'shards'/str(x['shard'])/'status.json')
        attempt=[a for a in status['attempts'] if a['index']==i]
        b.require(status['plan_id']==p['plan_id'] and attempt==[dict(index=i,returncode=75)],'Original task has not terminated')
        snapshots.append(dict(index=i,shard=x['shard'],terminal_attempt=attempt[0]))
        paths=b.accuracy.paths(parent,t)
        if paths['report'].exists():
            report=b.read(paths['report'])
            b.require(report['plan_id']==p['plan_id'] and report['task']==source and report['complete'] and report['audit_passed'],'Unexpected numerical failure requires inspection')
            retained[t['id']]=r.bind(paths['report']);continue
        b.require(paths['checkpoint'].exists() and paths['identity'].exists(),'Timeout lacks retained checkpoint')
        ident=b.read(paths['identity']);b.require(ident['plan_id']==p['plan_id'] and ident['task']==source and ident['runtime']==p['runtime'],'Checkpoint identity differs')
        counts=r.checkpoint_counts(paths['checkpoint']);b.require(len(counts)==4 and max(counts)<=source['target_draws'],'Wrong retained draw count')
        specs.append(dict(index=len(specs),task=t,target_draws=source['target_draws'],parent_files={k:r.bind(paths[k]) for k in ('identity','checkpoint')},
            parent_draw_counts=counts,original_index=i,reason='Per-task timeout; unchanged numerical target'))
    plan=dict(schema=SCHEMA,parent=str(parent),parent_plan=r.bind(parent/'plan.json'),candidates=r.bind(candidates),
      code_sha256=codes(),runtime=p['runtime'],tasks=specs,retained=retained,terminal_snapshots=snapshots,
      resources=dict(cpus=2,memory_gib=12,task_seconds=4350,wall_seconds=4500,maximum_tasks=40,
        maximum_cpu_hours=100,maximum_sampling_cost_chf=.55,automatic_retries=False))
    plan['plan_id']=b.fp(plan);b.write(out/'plan.json',plan,immutable=True)
    print(json.dumps(dict(event='terminal_recovery_prepared',tasks=len(specs),retained=len(retained),retained_draws=sum(sum(s['parent_draw_counts']) for s in specs))),flush=True)

def load(out):
    p=b.read(out/'plan.json');b.require(p['schema']==SCHEMA and p['plan_id']==b.fp({k:v for k,v in p.items() if k!='plan_id'}),'Plan changed')
    b.require(p['code_sha256']==codes(),'Code changed');r.verify(p['parent_plan']);r.verify(p['candidates'])
    old,parent=original.load(Path(p['parent']));b.require(p['runtime']==old['runtime'],'Runtime changed')
    return p,parent

def worker(out,index):
    p,old=load(out);previous=r.load;r.load=lambda folder:(p,old)
    try:return r.worker(out,index)
    finally:r.load=previous

def dispatch(out,index):
    p,_=load(out);b.require(0<=index<40,'Invalid recovery index');folder=out/'attempts';folder.mkdir(exist_ok=True)
    with (folder/f'{index}.lock').open('x'):
        if index>=len(p['tasks']):
            b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,skipped=True,reason='Original task completed and passed'),immutable=True);return 0
        child=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(index)],start_new_session=True)
        try:status=child.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:b.frequency.stop(child);status=75
    b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,returncode=status),immutable=True)
    return status

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','dispatch','worker'))
    a.add_argument('--out',type=Path,required=True);a.add_argument('--parent',type=Path);a.add_argument('--candidates',type=Path);a.add_argument('--index',type=int)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.candidates);return 0
    return globals()[v.stage](v.out,v.index)
if __name__=='__main__':raise SystemExit(main())

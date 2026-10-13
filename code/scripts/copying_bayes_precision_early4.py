"""One conditional recovery of four known audit-only timing allocations.

Run only after their original mixed shards finish. Passed originals are kept;
otherwise continue the saved checkpoint. No duplicate live work or retries.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import copying_bayes_precision_chains as original
r=original.r;b=original.b
SCHEMA='copying-precision-early4-recovery-v1'
INDICES=(10804,10920,11115,11210)

def codes():return {str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__),**original.codes()}

def prepare(out,parent,receipt_path):
    old,_=original.load(parent);receipt=b.read(receipt_path)
    b.require(receipt['original_indices']==list(INDICES),'Declared four cases changed')
    active=subprocess.check_output(['squeue','-j',','.join(receipt['parent_jobs']),'-h','-o','%i|%T'],text=True).strip()
    b.require(not active,'Original mixed shards remain active')
    b.require(not (out/'plan.json').exists(),'No repeated preparation');specs=[];retained={}
    for i in INDICES:
        source=old['tasks'][i];t=source['task'];paths=b.accuracy.paths(parent,t)
        b.require(source['early_fresh'] and source['target_draws'] in (131072,262144),'Recovery scope changed')
        target=source['target_draws'];files={};counts=[0]*4
        if paths['report'].exists():
            report=b.read(paths['report']);b.require(report['complete'] and report['plan_id']==old['plan_id'] and report['task']==source,'Original report identity differs')
            if report['audit_passed']:
                retained[t['id']]=r.bind(paths['report']);continue
            target*=2
        if paths['checkpoint'].exists():
            ident=b.read(paths['identity']);b.require(ident['plan_id']==old['plan_id'] and ident['task']==source and ident['runtime']==old['runtime'],'Checkpoint identity differs')
            files={k:r.bind(paths[k]) for k in ('identity','checkpoint')};counts=r.checkpoint_counts(paths['checkpoint'])
        specs.append(dict(index=len(specs),task=t,target_draws=target,parent_files=files,parent_draw_counts=counts,
          original_index=i,reason='Conditional one-shot correction of audit-only runtime estimate'))
    p=dict(schema=SCHEMA,parent=str(parent),parent_plan=r.bind(parent/'plan.json'),submission=r.bind(receipt_path),
      code_sha256=codes(),runtime=old['runtime'],tasks=specs,retained=retained,terminal_verified=True,
      resources=dict(cpus=4,memory_gib=24,task_seconds=10500,wall_seconds=10800,
        maximum_cpu_hours=48,maximum_sampling_cost_chf=.264,automatic_retries=False))
    p['plan_id']=b.fp(p);b.write(out/'plan.json',p,immutable=True)
    print(json.dumps(dict(event='early4_prepared',tasks=len(specs),retained=list(retained),max_chf=.264)),flush=True)

def load(out):
    p=b.read(out/'plan.json');b.require(p['schema']==SCHEMA and p['plan_id']==b.fp({k:v for k,v in p.items() if k!='plan_id'}),'Plan changed')
    b.require(p['code_sha256']==codes(),'Code changed');r.verify(p['parent_plan']);r.verify(p['submission'])
    old,parent=original.load(Path(p['parent']));b.require(p['runtime']==old['runtime'],'Runtime changed')
    return p,parent

def worker(out,index):
    p,old=load(out);previous=r.load;r.load=lambda folder:(p,old)
    try:return r.worker(out,index)
    finally:r.load=previous

def dispatch(out,index):
    p,_=load(out);b.require(0<=index<4,'Invalid recovery index')
    folder=out/'attempts';folder.mkdir(exist_ok=True)
    with (folder/f'{index}.lock').open('x'):
        if index>=len(p['tasks']):
            b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,skipped=True,reason='Original result passed; no new sampling'),immutable=True);return 0
        child=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(index)],start_new_session=True)
        try:status=child.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:b.frequency.stop(child);status=75
    b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,returncode=status),immutable=True)
    return status

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','dispatch','worker'))
    a.add_argument('--out',type=Path,required=True);a.add_argument('--parent',type=Path);a.add_argument('--receipt',type=Path);a.add_argument('--index',type=int)
    v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.parent,v.receipt);return 0
    return globals()[v.stage](v.out,v.index)
if __name__=='__main__':raise SystemExit(main())

"""One bounded same-target continuation of the frozen evening copying scope."""
from pathlib import Path
import argparse,json,subprocess,sys
import copying_bayes_precision_chains as original
r=original.r;b=original.b
SCHEMA='copying-precision-evening-continuation-v1'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}

def codes():return {str(Path(__file__).relative_to(b.ROOT)):b.sha(__file__),**original.codes()}

def request_scope(request,old):
    ids=request['original_indices']
    b.require(len(ids)==len(set(ids))==request['maximum_tasks'] and len(ids) in (67,19),'Evening scope differs')
    b.require(request['mode'] in ('terminal','conditional') and len(ids)==(67 if request['mode']=='terminal' else 19),'Wrong continuation layer')
    r.verify(request['original_plan'])
    b.require(request['original_plan']['sha256']==b.sha(Path(request['parent'])/'plan.json'),'Wrong original plan')
    protected=set(old['shards'][322]['tasks'])|{10804,10920,11115,11210}
    timeout=b.read(request['timeout40_plan']['path']);r.verify(request['timeout40_plan'])
    b.require(timeout['parent_plan']==request['original_plan'],'Prior timeout layer has another parent')
    protected|={s['original_index'] for s in timeout['tasks']}
    b.require(len(protected)==52 and not protected.intersection(ids),'Evening scope overlaps prior repairs')
    b.require(all(0<=i<len(old['tasks']) and old['tasks'][i]['target_draws'] in (131072,262144) for i in ids),'Unexpected evening draw target')
    b.require(request['code_sha256']==codes(),'Evening implementation changed')
    return ids

def prepare(out,request_path):
    request=b.read(request_path);parent=Path(request['parent']);old,_=original.load(parent)
    ids=request_scope(request,old);b.require(not (out/'plan.json').exists(),'No repeated preparation')
    a,z=out.resolve(),parent.resolve();b.require(a!=z and a not in z.parents and z not in a.parents,'Recovery overlaps original output')
    terminal_receipt=None
    if request['mode']=='conditional':
        jobs=request['parent_jobs'];b.require(len(jobs)==9 and len(set(jobs))==9,'Expected nine original jobs')
        active=subprocess.check_output(['squeue','-j',','.join(jobs),'-h','-o','%i|%T'],text=True).strip()
        b.require(not active,'Original jobs remain active')
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobID,State'],text=True)
        states={line.split('|')[0]:line.split('|')[1].split()[0].rstrip('+') for line in raw.splitlines() if '|' in line}
        b.require(all(states.get(j) in TERMINAL for j in jobs),'Original termination not confirmed')
        terminal_receipt=dict(jobs=jobs,accounting=raw)
    specs=[];retained={};snapshots=[]
    for i in ids:
        source=old['tasks'][i];t=source['task'];paths=b.accuracy.paths(parent,t)
        shard=next(s for s in old['shards'] if i in s['tasks'])
        state=b.read(parent/'shards'/str(shard['index'])/'status.json')
        attempts=[a for a in state['attempts'] if a['index']==i]
        b.require(state['plan_id']==old['plan_id'] and len(attempts)<=1,'Original dispatcher identity differs')
        if request['mode']=='terminal':b.require(attempts==[dict(index=i,returncode=75)],'Original child is not a terminal timeout')
        elif attempts:b.require(attempts[0]['returncode'] in (0,75),'Unexpected original numerical failure')
        snapshots.append(dict(index=i,shard=shard['index'],attempts=attempts))
        if paths['report'].exists():
            report=b.read(paths['report']);b.require(report['plan_id']==old['plan_id'] and report['task']==source and report['complete'] and report['audit_passed'],'Original report requires scientific review')
            retained[t['id']]=r.bind(paths['report']);continue
        if paths['checkpoint'].exists():
            ident=b.read(paths['identity']);b.require(ident['plan_id']==old['plan_id'] and ident['task']==source and ident['runtime']==old['runtime'],'Checkpoint identity differs')
            files={k:r.bind(paths[k]) for k in ('identity','checkpoint')};counts=r.checkpoint_counts(paths['checkpoint'])
        else:
            b.require(not paths['identity'].exists(),'Partial original identity without checkpoint')
            files=source['parent_files'];counts=source['parent_draw_counts']
            for item in files.values():r.verify(item)
        b.require(len(counts)==4 and max(counts)<=source['target_draws'],'Invalid retained draw count')
        specs.append(dict(index=len(specs),task=t,target_draws=source['target_draws'],parent_files=files,parent_draw_counts=counts,original_index=i,reason='One evening continuation at unchanged target'))
    p=dict(schema=SCHEMA,parent=str(parent),parent_plan=r.bind(parent/'plan.json'),request=r.bind(request_path),code_sha256=codes(),runtime=old['runtime'],tasks=specs,retained=retained,terminal_snapshots=snapshots,terminal_receipt=terminal_receipt,resources=dict(cpus=2,memory_gib=12,task_seconds=7050,wall_seconds=7200,maximum_tasks=len(ids),maximum_cpu_hours=4*len(ids),maximum_sampling_cost_chf=.022*len(ids),automatic_retries=False))
    p['plan_id']=b.fp(p);b.write(out/'plan.json',p,immutable=True)
    print(json.dumps(dict(event='evening_prepared',tasks=len(specs),retained=len(retained),scope=len(ids),maximum_cost_chf=p['resources']['maximum_sampling_cost_chf'])),flush=True)

def load(out):
    p=b.read(out/'plan.json');b.require(p['schema']==SCHEMA and p['plan_id']==b.fp({k:v for k,v in p.items() if k!='plan_id'}) and p['code_sha256']==codes(),'Evening plan/code changed')
    r.verify(p['parent_plan']);r.verify(p['request']);old,parent=original.load(Path(p['parent']));ids=request_scope(b.read(p['request']['path']),old)
    byid={old['tasks'][i]['task']['id']:i for i in ids};tasks={s['task']['id']:s for s in p['tasks']}
    b.require(p['runtime']==old['runtime'] and len(tasks)==len(p['tasks']) and not set(tasks).intersection(p['retained']) and set(tasks)|set(p['retained'])==set(byid),'Evening coverage differs')
    for key,s in tasks.items():
        source=old['tasks'][byid[key]];b.require(s['original_index']==byid[key] and s['task']==source['task'] and s['target_draws']==source['target_draws'],'Evening source/seed/target changed')
    for key,binding in p['retained'].items():
        r.verify(binding);source=old['tasks'][byid[key]];b.require(Path(binding['path'])==b.accuracy.paths(Path(p['parent']),source['task'])['report'],'Unrelated retained reference')
        report=b.read(binding['path']);b.require(report['complete'] and report['audit_passed'] and report['task']==source and report['plan_id']==old['plan_id'],'Retained reference failed')
    b.require({s['index'] for s in p['terminal_snapshots']}==set(ids),'Incomplete terminal evidence')
    request=b.read(p['request']['path'])
    if request['mode']=='terminal':b.require(all(s['attempts']==[dict(index=s['index'],returncode=75)] for s in p['terminal_snapshots']),'Missing terminal child receipt')
    else:
        proof=p['terminal_receipt'];b.require(proof is not None and proof['jobs']==request['parent_jobs'],'Missing terminal scheduler receipt')
        states={line.split('|')[0]:line.split('|')[1].split()[0].rstrip('+') for line in proof['accounting'].splitlines() if '|' in line}
        b.require(all(states.get(j) in TERMINAL for j in proof['jobs']),'Original job was not terminal')
    b.require(p['resources']['maximum_tasks']==len(ids) and p['resources']['task_seconds']==7050 and p['resources']['wall_seconds']==7200 and not p['resources']['automatic_retries'],'Runtime bound changed')
    return p,parent

def worker(out,index):
    p,old=load(out);previous=r.load;r.load=lambda folder:(p,old)
    try:return r.worker(out,index)
    finally:r.load=previous

def dispatch(out,index):
    p,_=load(out);b.require(0<=index<p['resources']['maximum_tasks'],'Invalid task');folder=out/'attempts';folder.mkdir(exist_ok=True)
    with (folder/f'{index}.lock').open('x'):
        if index>=len(p['tasks']):b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,skipped=True,reason='Original passed'),immutable=True);return 0
        child=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out),'--index',str(index)],start_new_session=True)
        try:status=child.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:b.frequency.stop(child);status=75
    b.write(folder/f'{index}.json',dict(plan_id=p['plan_id'],index=index,returncode=status),immutable=True);return status

def main():
    print('Bounded copying evening continuation; fixed cohort, unchanged targets.',flush=True)
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('prepare','dispatch','worker'));a.add_argument('--out',type=Path,required=True);a.add_argument('--request',type=Path);a.add_argument('--index',type=int);v=a.parse_args()
    if v.stage=='prepare':prepare(v.out,v.request);return 0
    return globals()[v.stage](v.out,v.index)
if __name__=='__main__':raise SystemExit(main())

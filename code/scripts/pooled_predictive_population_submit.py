#!/usr/bin/env python3
"""Lightweight scheduler dispatcher; numerical work runs only in Slurm jobs."""
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]);p=argparse.ArgumentParser(description='Explicit bounded stage; no automatic successor or population launch.')
p.add_argument('--stage',choices=('timing','refinement_timing','refine','population'),default='timing')
p.add_argument('--plan',type=Path,default=root/'code/transformer/configs/pooled_predictive_population_plan_2026-09-19.json')
p.add_argument('--raw',type=Path,default=Path.home()/'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
p.add_argument('--dependencies',type=Path,default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
p.add_argument('--out',type=Path,default=root/'output/pooled_predictive_population_2026-09-19')
p.add_argument('--calibration',type=Path,default=root/'output/pooled_predictive_calibration_2026-09-18/stronger')
p.add_argument('--observed',type=int,choices=(16,64,256,1024),action='append');p.add_argument('--only-task',type=int,action='append');p.add_argument('--merge',action='store_true')
p.add_argument('--after-job');p.add_argument('--submit',action='store_true');p.add_argument('--retry',action='store_true',help='Explicit second attempt, only after every earlier submission is terminal');a=p.parse_args(sys.argv[2:])
if a.after_job and not a.after_job.isdigit():p.error('dependency must be one numeric job ID')
if a.merge and a.only_task:p.error('merge must cover the complete declared stage')
for name in ('plan','raw','dependencies','out','calibration'):
 value=getattr(a,name).expanduser().resolve()
 if any(c in str(value) for c in '\n\r\t'):p.error('paths cannot contain control characters')
 setattr(a,name,value)
protected=[a.raw,a.dependencies,a.calibration,*[root/q for q in ('code','results','notes','figures','.git')]]
if a.out==root or a.out in root.parents or any(a.out==q or a.out in q.parents or q in a.out.parents for q in protected):
 p.error('output must be separate from original data and sources')
plan=json.loads(a.plan.read_text())
if plan.get('schema')!='pooled-predictive-population-plan-v1':p.error('wrong plan')
counts=dict(timing=36,refinement_timing=36,refine=192,population=2304)
count=counts[a.stage];times=(16,64,256,1024);requested=set(a.observed or times)
eligible=[i for i in range(count) if times[i//(count//4)] in requested]
indices=sorted(set(a.only_task)) if a.only_task else eligible
if not indices or any(i not in eligible for i in indices):p.error('selected task is outside requested contexts')
if any(i<0 or i>=count for i in indices):p.error('task outside declared stage')
chunks=[indices[i:i+768] for i in range(0,len(indices),768)]
settings=plan['stages'][a.stage];concurrency=max(1,settings['concurrency']//len(chunks))
script=root/'code/transformer/scitas/pooled_predictive_population.sbatch'
def command(mode,dependency=None,chunk=None):
 cmd=['sbatch','--parsable','--chdir='+str(root),'--job-name=ppop-'+a.stage+'-'+mode,
      '--time='+('00:30:00' if mode!='run' else f'{settings["allocation_minutes"]//60:02d}:{settings["allocation_minutes"]%60:02d}:00'),
      '--output='+str(a.out/a.stage/'logs'/f'{mode}-%A_%a.out')]
 if dependency:cmd+=['--dependency='+('afterany:' if mode=='merge' else 'afterok:')+dependency,'--kill-on-invalid-dep=yes']
 if chunk is not None:
  selection=f'0-{len(chunk)-1}' if len(chunk)>1 else '0'
  cmd+=['--array='+selection+'%'+str(concurrency)]
 return cmd+[str(script),str(a.plan),str(a.raw),str(a.dependencies),str(a.out),str(a.calibration),a.stage,mode,','.join(map(str,sorted(requested))),','.join(map(str,chunk)) if chunk is not None else '-']
receipt=None;attempt=None;receipt_path=None

def now():return datetime.now(timezone.utc).isoformat()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save_receipt():
 if receipt is None:return
 receipt['updated_utc']=now()
 temporary=receipt_path.with_name(receipt_path.name+'.tmp.'+str(os.getpid()))
 with temporary.open('w') as stream:
  json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temporary,receipt_path)

@contextmanager
def submission_lock(directory):
 directory.mkdir(parents=True,exist_ok=True)
 with (directory/'.submission.lock').open('a+') as stream:
  try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ValueError('Another launch is being recorded. Wait for it to finish; do not submit twice.')
  try:yield
  finally:fcntl.flock(stream,fcntl.LOCK_UN)

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED'}
def require_terminal(saved):
 jobs=[]
 for trial in saved['attempts']:
  if trial.get('submitting') is not None or trial['status']!='submitted':
   raise ValueError('An earlier dispatch is incomplete or uncertain; inspect its receipt and scheduler jobs before retrying: '+saved['key'])
  jobs.extend(x['job_id'] for x in trial['jobs'])
 if not jobs:raise ValueError('A prior receipt has no scheduler jobs; inspect it before retrying')
 # Fail closed if accounting is delayed or a scheduler query fails.
 queued=subprocess.run(['squeue','--noheader','--jobs',','.join(jobs),'--format=%i|%T'],check=True,text=True,stdout=subprocess.PIPE)
 if queued.stdout.strip():raise ValueError('An earlier campaign allocation is still active; no overlapping or extra concurrent wave will be submitted')
 rows=subprocess.run(['sacct','-X','--noheader','--parsable2','--jobs',','.join(jobs),'--format=JobID,State'],check=True,text=True,stdout=subprocess.PIPE)
 seen=set()
 for line in rows.stdout.splitlines():
  fields=line.strip().split('|')
  if len(fields)<2:continue
  job=fields[0].split('_')[0].split('.')[0];state=fields[1].split()[0].rstrip('+')
  if job not in jobs:continue
  if state not in TERMINAL:raise ValueError('Earlier allocation is not terminal: '+fields[0]+' '+fields[1])
  seen.add(job)
 if seen!=set(jobs):raise ValueError('Scheduler accounting is incomplete; retry only once all earlier jobs have terminal records')


def begin_submission():
 global receipt,attempt,receipt_path
 directory=a.out/'submissions'
 identity=dict(stage=a.stage,mode='merge' if a.merge else 'wave',indices=[] if a.merge else indices,
  observed=sorted(requested),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
  raw=str(a.raw),dependencies=str(a.dependencies),calibration=str(a.calibration),plan=str(a.plan),out=str(a.out),
  code_sha256={str(q.relative_to(root)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
   (root/'code/scripts/pooled_predictive_population.py',Path(__file__).resolve(),script,root/'code/transformer/scitas/submit_pooled_predictive_population.sh')})
 key=fingerprint(identity);receipt_path=directory/(key+'.json')
 saved=[json.loads(q.read_text()) for q in sorted(directory.glob('*.json'))]
 exact=next((r for r in saved if r['key']==key),None)
 if exact and not a.retry:
  if any(t.get('submitting') is not None or t['status']!='submitted' for t in exact['attempts']):
   raise ValueError('The matching launch receipt is incomplete or uncertain. Inspect it before any further submission: '+str(receipt_path))
  print(json.dumps(dict(action='already_submitted',receipt=str(receipt_path),attempts=exact['attempts'])),flush=True)
  return False
 # One explicit wave at a time in this campaign prevents aggregate concurrency
 # exceeding its bound, including different selections and duplicate pastes.
 for r in saved:require_terminal(r)
 reserved_hours=sum(job['maximum_cpu_hours'] for r in saved for trial in r['attempts'] for job in trial['jobs'])
 next_hours=.5 if a.merge else 1+len(indices)*settings['allocation_minutes']/60
 if reserved_hours+next_hours>plan['resource_limits']['maximum_total_projected_cpu_hours']:
  raise ValueError('The campaign allocation-hour cap would be exceeded, including all gates, merges and retries')
 prior_counts={i:0 for i in indices}
 for r in saved:
  previous=r['identity']
  if previous['stage']!=a.stage or previous['mode']!='wave':continue
  overlap=set(indices)&set(previous['indices'])
  if overlap and not a.merge:
   if not a.retry:raise ValueError('This selection contains previously submitted tasks; use an explicit --retry only for unfinished tasks')
   if any(previous[k]!=identity[k] for k in ('plan_sha256','raw','dependencies','calibration','code_sha256')):
    raise ValueError('Previously submitted tasks have a different scientific or execution identity')
   for trial in r['attempts']:
    for job in trial['jobs']:
     if job['mode']=='run':
      for i in overlap & set(job['indices']):prior_counts[i]+=1
 if not a.merge and any(n>=2 for n in prior_counts.values()):raise ValueError('The two-attempt allocation cap has been reached for a selected task')
 if a.merge and exact and len(exact['attempts'])>=2:raise ValueError('The two-attempt merge allocation cap has been reached')
 receipt=exact or dict(schema='pooled-predictive-population-submission-v1',key=key,identity=identity,attempts=[],created_utc=now())
 attempt=dict(attempt=len(receipt['attempts'])+1,status='preparing',jobs=[],submitting=None,created_utc=now())
 receipt['attempts'].append(attempt);save_receipt()
 return True


def dispatch(cmd,mode,chunk=None):
 print(shlex.join(cmd),flush=True)
 if not a.submit:return 'DRY_RUN_JOB'
 attempt['status']='submitting';attempt['submitting']=dict(mode=mode,indices=chunk or [],command=cmd,started_utc=now());save_receipt()
 result=subprocess.run(cmd,check=True,cwd=root,text=True,stdout=subprocess.PIPE)
 print(result.stdout,end='',flush=True)
 job=result.stdout.strip().split(';')[0]
 if not job.isdigit():raise RuntimeError('Scheduler did not return a numeric job ID; inspect before retrying')
 allocation_hours=.5 if mode!='run' else len(chunk)*settings['allocation_minutes']/60
 attempt['jobs'].append(dict(mode=mode,indices=chunk or [],job_id=job,maximum_cpu_hours=allocation_hours,submitted_utc=now()))
 attempt['submitting']=None;attempt['status']='preparing';save_receipt()
 return job
print(json.dumps(dict(stage=a.stage,tasks=0 if a.merge else len(indices),arrays=0 if a.merge else len(chunks),
 concurrency=0 if a.merge else concurrency*len(chunks),maximum_first_attempt_worker_cpu_hours=0 if a.merge else len(indices)*settings['allocation_minutes']/60,
 setup_and_merge_cpu_hour_cap=.5 if a.merge else 1.,automatic_successor_submission=False,scientific_reference_validated=False)),flush=True)
if a.submit:
 if not (a.raw/'heldout/manifest.json').is_file() or not (a.dependencies/'tables/anchors_prod/manifest.json').is_file():
  p.error('frozen data/dependency paths missing')
 (a.out/a.stage/'logs').mkdir(parents=True,exist_ok=True)
def launch():
 if a.merge:
  dispatch(command('merge',a.after_job),'merge')
 else:
  gate=dispatch(command('gate',a.after_job),'gate')
  jobs=[dispatch(command('run',gate,chunk),'run',chunk) for chunk in chunks]
  review=dispatch(command('merge',':'.join(jobs)),'merge')
  print(json.dumps(dict(stage=a.stage,gate_job=gate,worker_jobs=jobs,merge_job=review,population_submitted=a.stage=='population' and a.submit)),flush=True)
 if a.submit:
  attempt['status']='submitted';save_receipt()
  print(json.dumps(dict(receipt=str(receipt_path),attempt=attempt['attempt'],status='submitted')),flush=True)
if a.submit:
 with submission_lock(a.out/'submissions'):
  if begin_submission():launch()
else:launch()

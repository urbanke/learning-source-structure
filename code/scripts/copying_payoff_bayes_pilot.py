"""Twelve bounded canonical-prefix Bayes timing tasks; no population approval.

The established full-target sampler and checkpoint implementation are reused
without modification. Each task has one allocation and a fixed draw budget.
Canonical prefixes, original prior, implementation hashes and runtime are bound
before sampling. A separate explicit run command starts selected tasks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/transformer'))
INVENTORY = ROOT/'output/copying_payoff_reference_readiness_2026-09-19.json'
DEFAULT_OUT = ROOT/'output/copying_payoff_canonical_bayes_pilot_2026-09-19'
DEFAULT_PYTHON = ROOT/'output/unseen_context_refinement_runtime_2026-09-19/bin/python'
SCHEMA = 'copying-payoff-canonical-bayes-pilot-v1'
NUMERICAL_FILES = (
    'shared_base_curve_runner.py','shared_base_surrogate.py','shared_base_efficiency.py',
    'surrogate_chain_runner.py','stable_chain_statistics.py','shared_base_chain_runner.py',
    'shared_base_mcmc.py','ranked_lsa.py','weighted_lsa.py','shared_depth.py','lsa_seq.py',
    'phi_source.py','source.py','eval_shared_base_reference.py','full_context_statistics.py',
    'eval_shared_base_models.py','validate_shared_base_mcmc.py')


def require(value, message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value,immutable=False):
    path=Path(path)
    if immutable and path.exists():
        require(read(path)==value,'Frozen file changed: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def code_hashes():
    paths=[Path(__file__),ROOT/'code/scripts/prepare_copying_payoff_references.py']
    paths += [ROOT/'code/transformer'/name for name in NUMERICAL_FILES]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def runtime():
    import numpy as np
    import scipy
    return dict(python=list(sys.version_info[:3]),numpy=np.__version__,scipy=scipy.__version__,
                longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant+1),platform=platform.system(),
                machine=platform.machine())


def environment(plan):
    env=dict(os.environ,**plan['environment'])
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[name]='1'
    for name in ('PMM_PHI_BIAS','PMM_PHI_WAVE'):
        require(float(env.get(name,'0'))==0,'Numerical perturbations are forbidden')
    return env


def ensure_output(out):
    out=Path(out).absolute()
    require(out.resolve().is_relative_to(ROOT/'output') and out.resolve()!=ROOT/'output'
            and not out.is_symlink(),'Pilot output must be an isolated local output directory')
    for path in out.parents:
        if path==ROOT:break
        require(not path.is_symlink(),'Symlink in output path')
    return out


def tasks_from_inventory(inventory):
    from prepare_copying_payoff_references import canonicalize,reference_key
    require(inventory['schema']=='copying-payoff-canonical-reference-readiness-v1'
            and inventory['histories_total']==768 and inventory['depths']==[0,2,3],
            'Unexpected original cohort inventory')
    selected=inventory['pilot']['representative_patterns']
    require(len(selected)==6 and sorted(p['time'] for p in selected)==[5,5,8,8,16,16],
            'Need both fixed structural representatives at t5,8,16')
    tasks=[]
    for pattern in selected:
        tokens,labels=canonicalize(pattern['tokens'])
        require(list(tokens)==pattern['tokens'] and len(tokens)==pattern['time']
                and len(labels)==pattern['n_seen'] and reference_key(tokens)==pattern['reference_key'],
                'Representative canonical prefix identity differs')
        for order in (16,32):
            task=dict(task_index=len(tasks),id=pattern['reference_key']+f'_a{order}',
                reference_key=pattern['reference_key'],tokens=pattern['tokens'],time=pattern['time'],
                n_seen=pattern['n_seen'],selection=pattern['selection'],order=order)
            task['seed']=int(fingerprint(['copying-canonical-pilot',20260919,task])[:8],16)
            tasks.append(task)
    return tasks


def prepare(inventory_path,out,python):
    out=ensure_output(out)
    require(not (out/'plan.json').exists(),'Pilot already prepared; inspect or run its frozen plan')
    require(not out.exists() or not any(out.iterdir()),'New pilot output directory must be empty')
    inventory=read(inventory_path)
    require(inventory['script_sha256']==sha(ROOT/'code/scripts/prepare_copying_payoff_references.py'),
            'Inventory builder changed')
    require(sha(ROOT/'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json')
            ==inventory['manifest_sha256'],'Source cohort manifest changed')
    protected=[Path(item['path']).resolve().parent for item in inventory['source_bindings'].values()]
    protected += [Path(path).resolve() for path in inventory['candidate_scan']['roots']]
    require(all(out.resolve()!=path and out.resolve() not in path.parents and path not in out.resolve().parents
                for path in protected),'Pilot output overlaps original source or reference trees')
    tasks=tasks_from_inventory(inventory)
    original=read(ROOT/'output/shared_base_curve_campaign_2026-09-15/plan.json')
    plan=dict(schema=SCHEMA,output=str(out.resolve()),inventory=str(Path(inventory_path).resolve()),
        inventory_sha256=sha(inventory_path),tasks=tasks,python=str(Path(python).absolute()),
        environment={k:v for k,v in original['worker']['environment'].items() if v is not None},
        prior=dict(alphabet=64,alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation',initial_token='q'),
        sampler=dict(proposal='shared_base_surrogate.SharedBaseSurrogateMCMC',surrogate_steps=64,
                     chains=4,burn=16384,thin=1,draws_per_chain=4096),
        resources=dict(max_workers=1,task_wall_seconds=600,total_worker_seconds=7200,
                       maximum_attempts_per_task=1,checkpoint_seconds=120),
        code_sha256=code_hashes(),scientific_timing_validated=False,population_reference_approved=False)
    store=Path(plan['environment']['LSA_STORE']);src=Path(plan['environment']['PMM_SRC'])/'product_model_with_memory'
    dependencies=[store/name for name in ('manifest.json','anchors.json')]
    dependencies += [src/name for name in ('universal_tables.py','production_coding.py','mellin.py','kernel.py')]
    plan['dependency_sha256']={str(path):sha(path) for path in dependencies}
    command=[plan['python'],str(Path(__file__).resolve()),'runtime']
    process=subprocess.run(command,env=environment(plan),cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
    plan['runtime']=json.loads(process.stdout)
    plan['plan_id']=fingerprint(plan)
    write(out/'plan.json',plan,immutable=True)
    return plan


def load_plan(out):
    out=ensure_output(out);plan=read(out/'plan.json')
    require(plan['schema']==SCHEMA and plan['plan_id']==fingerprint({k:v for k,v in plan.items() if k!='plan_id'})
            and plan['output']==str(out.resolve()),'Pilot identity differs')
    require(plan['code_sha256']==code_hashes() and sha(plan['inventory'])==plan['inventory_sha256'],
            'Pilot code or inventory changed')
    require(all(sha(path)==digest for path,digest in plan['dependency_sha256'].items()),
            'Pilot numerical dependency changed')
    require(plan['tasks']==tasks_from_inventory(read(plan['inventory'])),'Pilot task selection changed')
    require(plan['sampler']==dict(proposal='shared_base_surrogate.SharedBaseSurrogateMCMC',surrogate_steps=64,
            chains=4,burn=16384,thin=1,draws_per_chain=4096),'Sampling budget or proposal changed')
    require(plan['resources']==dict(max_workers=1,task_wall_seconds=600,total_worker_seconds=7200,
            maximum_attempts_per_task=1,checkpoint_seconds=120),'Pilot limits changed')
    return plan


def task_paths(out,task):
    folder=Path(out)/'tasks'/task['id']
    require(not folder.is_symlink(),'Task directory is a symlink')
    return {name:folder/file for name,file in dict(checkpoint='chains.npz',report='report.json',
        moments='moments.npz',identity='identity.json',attempt='attempt.json',log='worker.log').items()}


def worker(out,index):
    plan=load_plan(out);require(0<=index<len(plan['tasks']),'Task index outside frozen pilot')
    os.environ.update(environment(plan));require(runtime()==plan['runtime'],'Pilot runtime differs')
    task=plan['tasks'][index];paths=task_paths(out,task);settings=plan['sampler']
    attempt=read(paths['attempt'])
    require(attempt['status']=='running' and attempt['plan_id']==plan['plan_id'] and attempt['task']==task
            and attempt['dispatcher_pid']==os.getppid(), 'Worker requires its live bounded dispatcher allocation')
    import numpy as np
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_surrogate import SharedBaseSurrogateMCMC
    from shared_base_curve_runner import CurveChainRun
    from eval_shared_base_reference import selected_grid_checks
    from eval_shared_base_models import atomic_npz
    from full_context_statistics import diagnostic_gates
    from stable_chain_statistics import stable_predictive_moments
    from validate_shared_base_mcmc import json_safe
    with lock(Path(out)/'worker.lock'):
        require(not paths['checkpoint'].exists() and not paths['report'].exists(),
                'Fixed pilot task already has sampling outputs; no silent continuation or restart')
        nodes,weights=np.polynomial.legendre.leggauss(task['order'])
        kernel=RankedLSARowEvidence(64,1+nodes/2,term_cache_size=16384)
        sampler=SharedBaseSurrogateMCMC(64,kernel,weights/2,surrogate_steps=64)
        run=CurveChainRun(sampler,np.asarray(task['tokens'],dtype=int),chains=4,burn=16384,thin=1,seed=task['seed'])
        identity=dict(plan_id=plan['plan_id'],task=task,target=run.fingerprint,proposal=run.proposal_descriptor,
                      runtime=runtime(),sampling=settings)
        write(paths['identity'],identity,immutable=True)
        saved=time.monotonic();began=saved
        def progress(event):
            nonlocal saved
            now=time.monotonic()
            if event['phase']=='burn' or event['draws']==event['requested_draws'] or now-saved>=120:
                run.save(paths['checkpoint']);saved=now
            print(json.dumps(dict(task=task['id'],**event)),flush=True)
        estimate=run.extend(4096,block_draws=2048,progress=progress)
        run.save(paths['checkpoint'])
        moments=stable_predictive_moments(run.samples['predictive_probs'])
        require(np.allclose(moments['mean'],estimate['predictive_probs'],rtol=0,atol=2e-14),
                'Predictive summaries disagree')
        atomic_npz(paths['moments'],**moments)
        grids=selected_grid_checks(sampler,np.asarray(task['tokens'],dtype=int),estimate)
        result=json_safe(dict(schema=SCHEMA,plan_id=plan['plan_id'],task=task,identity_sha256=sha(paths['identity']),
            complete=True,estimate=estimate,classic_diagnostic_gates=diagnostic_gates(estimate),
            final_state_grid_checks=grids,elapsed_seconds=time.monotonic()-began,
            checkpoint_sha256=sha(paths['checkpoint']),moments_sha256=sha(paths['moments']),
            scientific_timing_validated=False,population_reference_approved=False,
            interpretation='Fixed-budget feasibility record only; no source-cohort regret gate, independent raw audit or exact-t4 reproduction has been passed by this pilot.'))
        write(paths['report'],result,immutable=True)
        return result


def summarize(out):
    plan=load_plan(out);reports=[];attempts=[]
    for task in plan['tasks']:
        paths=task_paths(out,task)
        if paths['attempt'].exists():attempts.append(read(paths['attempt']))
        if not paths['report'].exists():continue
        r=read(paths['report'])
        require(r['plan_id']==plan['plan_id'] and r['task']==task and r['complete']
                and r['identity_sha256']==sha(paths['identity'])
                and r['checkpoint_sha256']==sha(paths['checkpoint'])
                and r['moments_sha256']==sha(paths['moments']),'Pilot report or arrays changed')
        reports.append(r)
    paired=[]
    for reference in sorted({t['reference_key'] for t in plan['tasks']}):
        rows={r['task']['order']:r for r in reports if r['task']['reference_key']==reference}
        if set(rows)!={16,32}:continue
        import numpy as np
        left,right=(np.asarray(rows[o]['estimate']['predictive_probs']) for o in (16,32))
        errors=[np.asarray(rows[o]['estimate']['mcse_probs']) for o in (16,32)]
        paired.append(dict(reference_key=reference,max_absolute_probability_difference=float(np.max(abs(left-right))),
            maximum_joint_probability_mcse=float(np.max(np.sqrt(errors[0]**2+errors[1]**2))),
            numerical_acceptance_established=False))
    result=dict(schema=SCHEMA,plan_id=plan['plan_id'],expected_tasks=12,completed_tasks=len(reports),
        complete=len(reports)==12,attempts=attempts,paired_order_diagnostics=paired,
        scientific_timing_validated=False,population_reference_approved=False,
        interpretation='Performance and fixed-budget diagnostics only. No automatic production or draw extension.')
    write(Path(out)/'summary.json',result)
    return result


def dispatch(out,only_tasks=None):
    plan=load_plan(out);selected=sorted(set(range(12) if only_tasks is None else only_tasks))
    require(selected and all(type(i)is int and 0<=i<12 for i in selected),'Need declared task indices0..11')
    with lock(Path(out)/'dispatcher.lock'):
        for index in selected:
            task=plan['tasks'][index];paths=task_paths(out,task)
            if paths['attempt'].exists():continue
            spent=sum(read(task_paths(out,t)['attempt'])['reserved_seconds'] for t in plan['tasks']
                      if task_paths(out,t)['attempt'].exists())
            require(spent+600<=7200,'Total pilot allocation exhausted')
            attempt=dict(plan_id=plan['plan_id'],task=task,reserved_seconds=600,
                         status='running',started_unix=time.time(),dispatcher_pid=os.getpid())
            write(paths['attempt'],attempt,immutable=True);began=time.monotonic()
            process=None
            try:
                with paths['log'].open('x') as stream:
                    process=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'worker',
                        '--out',str(out),'--index',str(index)],cwd=ROOT,env=environment(plan),
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        exit_code=process.wait(timeout=600)
                        attempt.update(status='complete' if exit_code==0 else 'failed',exit_code=exit_code)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                        attempt.update(status='time_limit',exit_code=75)
            except BaseException:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                attempt['status']='interrupted';raise
            finally:
                attempt.update(elapsed_seconds=time.monotonic()-began,finished_unix=time.time())
                write(paths['attempt'],attempt)
            summarize(out)
    return summarize(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','summary','runtime'))
    parser.add_argument('--inventory',type=Path,default=INVENTORY)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--python',type=Path,default=DEFAULT_PYTHON)
    parser.add_argument('--only-task',type=int,action='append');parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.action=='runtime':print(json.dumps(runtime()));return
    if args.action=='prepare':result=prepare(args.inventory,args.out,args.python)
    elif args.action=='run':result=dispatch(args.out,args.only_task)
    elif args.action=='worker':
        require(args.index is not None,'Worker needs a declared task index');result=worker(args.out,args.index)
    else:result=summarize(args.out)
    print(json.dumps({k:v for k,v in result.items() if k in ('plan_id','complete','completed_tasks','expected_tasks')}),flush=True)


if __name__=='__main__':main()

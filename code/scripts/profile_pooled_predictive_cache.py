"""Two bounded, identically seeded cache diagnostics; never an accuracy approval.

Each fresh process profiles initialization and SMC for at most900seconds.
The saved plan binds the existing development history and unchanged numerics.
Only cache capacity differs; completed estimator outputs must match exactly.
"""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
SETTINGS = dict(family='product', index=1, observed=256, development=True,
    particles=32, temperatures=64, moves=4, seed=2026092001,
    alpha_order=16, row_bounds=[-32.,32.], row_step=1/32,
    resample_fraction=.75, term_caches=[1024,4096], seconds_per_variant=900)
FILES = ('code/scripts/profile_pooled_predictive_cache.py',
    'code/transformer/scitas/pooled_predictive_cache_profile.sbatch',
    'code/transformer/scitas/submit_pooled_predictive_cache_profile.sh',
    'code/scripts/pooled_predictive_population.py',
    'code/transformer/configs/pooled_predictive_population_plan_2026-09-19.json',
    'code/transformer/configs/pooled_predictive_calibration_plan_2026-09-18.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def require(condition, message):
    if not condition: raise ValueError(message)


def protected_output(out, raw, dependencies):
    protected=[raw,dependencies,*(ROOT/n for n in ('code','results','notes','figures','.git'))]
    require(out!=ROOT and out not in ROOT.parents and not any(
        out==p or out in p.parents or p in out.parents for p in protected),
        'Output overlaps frozen inputs or repository sources')


def submit(a):
    """Standard-library-only submission; no numerical input reads on login nodes."""
    protected_output(a.out,a.raw,a.dependencies)
    command=['sbatch','--parsable','--chdir='+str(ROOT),'--job-name=pooled-cache-profile',
        '--output='+str(a.out/'profile-%j.out'),str(ROOT/FILES[1]),str(a.out)]
    print(shlex.join(command),flush=True)
    if not a.submit: return 0
    require((a.raw/'heldout/manifest.json').is_file() and
        (a.dependencies/'tables/anchors_prod/manifest.json').is_file(), 'Frozen inputs missing')
    # Exclusive directory creation also blocks retries after an ambiguous sbatch response.
    a.out.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='pooled-cache-profile-submission-v1',status='submitting',
        settings=SETTINGS,raw=str(a.raw),dependencies=str(a.dependencies),
        code_sha256={f:sha(ROOT/f) for f in FILES},scientific_reference_validated=False)
    save(a.out/'submission.json',receipt)
    result=subprocess.run(command,text=True,capture_output=True,cwd=ROOT)
    print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr)
    receipt.update(status='submitted' if result.returncode==0 else 'submission_error',
        scheduler_stdout=result.stdout,scheduler_stderr=result.stderr,returncode=result.returncode)
    save(a.out/'submission.json',receipt)
    return result.returncode


def prepare(out):
    import pooled_predictive_population as m
    receipt=json.loads((out/'submission.json').read_text())
    require(receipt['settings']==SETTINGS,'Diagnostic settings changed')
    for f,h in receipt['code_sha256'].items(): require(sha(ROOT/f)==h,'Diagnostic code changed: '+f)
    parent,parent_hash=m.read_plan(m.PLAN)
    calibration,_=m.calibration.read_plan(m.calibration.PLAN)
    dep=Path(receipt['dependencies']); store=dep/'tables/anchors_prod'; src=dep/'src'
    hashes=m.calibration.dependencies(calibration,store,src)
    environment=m.calibration.numerical_environment(store,src)
    tokens,_,_,inputs=m.load_case(Path(receipt['raw']),parent,SETTINGS)
    plan=dict(schema='pooled-cache-profile-plan-v1',settings=SETTINGS,
        parent_plan_sha256=parent_hash,code_sha256={**parent['code_sha256'],**receipt['code_sha256']},
        dependency_sha256=hashes,input_sha256=inputs,tokens=tokens.tolist(),
        store=str(store),src=str(src),numerical_environment=environment,
        runtime=dict(python=sys.version,numpy=m.np.__version__,scipy=m.calibration.scipy.__version__),
        scientific_reference_validated=False,
        scope='Performance diagnostic only. Profile overhead affects measured runtimes; no population submission.')
    require(not (out/'plan.json').exists(),'Diagnostic already prepared; use its existing reports')
    save(out/'plan.json',plan)
    return plan


def deadline(signum, frame):
    raise TimeoutError('Diagnostic sampling deadline or allocation signal')


def profile_variant(out, capacity, make_kernel, compute, seconds):
    """Persist counters/profile even when the estimator or hard timer interrupts."""
    profiler=cProfile.Profile(); kernel=None; started=time.monotonic()
    result=dict(cache_capacity=capacity,complete=False,scientific_reference_validated=False)
    old={s:signal.signal(s,deadline) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGUSR1)}
    try:
        signal.setitimer(signal.ITIMER_REAL,seconds)
        profiler.enable()
        kernel=make_kernel(capacity)
        result['estimate']=compute(kernel)
        result.update(complete=True,status='completed')
    except TimeoutError as error: result.update(status='time_limit',error=str(error))
    except Exception as error:
        result.update(status='error',error=repr(error),error_traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); profiler.disable()
        for sig,handler in old.items(): signal.signal(sig,handler)
        result.update(elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            cache_info=kernel.cache_info() if kernel is not None else {})
        profiler.dump_stats(str(out/f'cache{capacity}.prof'))
        stats=pstats.Stats(profiler)
        result['profile_top_cumulative']=[dict(file=f,line=line,function=name,
            primitive_calls=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3])
            for (f,line,name),v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:50]]
        save(out/f'cache{capacity}.json',result)
    return result


def worker(out, capacity):
    import pooled_predictive_population as m
    plan=json.loads((out/'plan.json').read_text()); require(plan['settings']==SETTINGS,'Plan changed')
    require(capacity in SETTINGS['term_caches'],'Unplanned cache capacity')
    for f,h in plan['code_sha256'].items(): require(sha(ROOT/f)==h,'Code changed: '+f)
    require(not (out/f'cache{capacity}.json').exists(),'Variant already attempted')
    store,src=Path(plan['store']),Path(plan['src'])
    print(f'Checking frozen dependencies for cache{capacity}; sampling has not started.',flush=True)
    calibration,_=m.calibration.read_plan(m.calibration.PLAN)
    require(m.calibration.dependencies(calibration,store,src)==plan['dependency_sha256'],
        'Sealed dependencies changed')
    environment=m.calibration.numerical_environment(store,src)
    require(environment==plan['numerical_environment'],'Numerical environment changed')
    nodes,weights=m.leggauss(SETTINGS['alpha_order'])
    def kernel(c):
        return m.RankedLSARowEvidence(64,1+nodes/2,depths=range(9),store=store,src=src,
            u_lo=-32.,u_hi=32.,step=1/32,term_cache_size=c)
    def compute(k):
        return m.estimate(m.SharedBaseMCMC(64,k,weights/2),plan['tokens'],
            **{k:SETTINGS[k] for k in ('particles','temperatures','moves','seed','resample_fraction')},
            seconds_limit=SETTINGS['seconds_per_variant'])
    print(f'Starting product history1, context256, cache{capacity}; hard sampling limit900seconds.',flush=True)
    result=profile_variant(out,capacity,kernel,compute,SETTINGS['seconds_per_variant'])
    result['plan_sha256']=sha(out/'plan.json'); save(out/f'cache{capacity}.json',result)
    return 0 if result['status'] in ('completed','time_limit') else 1


def parity(reports):
    if not all(r.get('complete') for r in reports): return None
    def numerical(r):
        return {k:v for k,v in r['estimate'].items() if k not in
            ('sampler_seconds','prediction_seconds','elapsed_seconds')}
    return numerical(reports[0])==numerical(reports[1])


def run(out):
    # An atomic claim prevents manual/direct duplicate compute runs.
    (out/'.running').mkdir()
    try:
        print('Checking frozen inputs on a compute node before two cache diagnostics; no accuracy result will be approved.',flush=True)
        plan=prepare(out); reports=[]
        for capacity in plan['settings']['term_caches']:
            command=[sys.executable,__file__,'worker','--out',str(out),'--capacity',str(capacity)]
            try:
                exit_code=subprocess.run(command,timeout=960).returncode
            except subprocess.TimeoutExpired: exit_code=124
            path=out/f'cache{capacity}.json'
            report=json.loads(path.read_text()) if path.exists() else dict(complete=False,status='worker_error')
            report['worker_exit_code']=exit_code; reports.append(report)
            print(json.dumps(dict(cache_capacity=capacity,status=report['status'])),flush=True)
        matched=parity(reports)
        summary=dict(schema='pooled-cache-profile-summary-v1',plan_sha256=sha(out/'plan.json'),
            reports=reports,exact_seeded_parity=matched,scientific_reference_validated=False,
            diagnostic_complete=all(r['worker_exit_code']==0 and
                r['status'] in ('completed','time_limit') for r in reports))
        save(out/'summary.json',summary)
        return 0 if summary['diagnostic_complete'] and matched is not False else 1
    finally: (out/'.running').rmdir()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('submit','run','worker'))
    p.add_argument('--raw',type=Path,default=Path.home()/'Projects/layered_architectures_multifamily/output/multifamily_2026-09-16')
    p.add_argument('--dependencies',type=Path,default=Path.home()/'Projects/shared_base_dependencies_2026-09-16')
    p.add_argument('--out',type=Path,default=ROOT/'output/pooled_predictive_cache_profile_2026-09-20')
    p.add_argument('--capacity',type=int); p.add_argument('--submit',action='store_true'); a=p.parse_args()
    a.out,a.raw,a.dependencies=(v.expanduser().resolve() for v in (a.out,a.raw,a.dependencies))
    if a.mode=='submit': return submit(a)
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'),
        'Numerical run/worker modes require a Slurm compute-node step')
    if a.mode=='worker': return worker(a.out,a.capacity)
    return run(a.out)


if __name__=='__main__': raise SystemExit(main())

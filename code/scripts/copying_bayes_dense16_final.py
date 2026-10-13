"""One immutable final continuation, composing two audited reference layers.

The frozen recovery worker and complete-cohort merge are reused unchanged.
Only the merge's parent-record resolver is supplied an explicit, hash-bound
first-recovery layer; no prediction or numerical acceptance rule is altered.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import copying_bayes_dense16_recovery as r

CASE='t0016_711138aabc9f84b7b35b_a32'
WRAPPER='code/transformer/scitas/copying_bayes_dense16_final.sbatch'
ORIGINAL_READER=r.parent_record


def own_codes():
    return {str(Path(__file__).relative_to(r.ROOT)):r.sha(__file__),WRAPPER:r.sha(r.ROOT/WRAPPER)}


def check_first(spec,plan,folder):
    paths=r.base.accuracy.paths(folder,spec['task'])
    report=r.read(paths['report']);ident=r.read(paths['identity']);audit=r.read(paths['audit'])
    r.require(report['complete'] and report['task']==ident['task']==spec and report['plan_id']==ident['plan_id']==plan['plan_id']
              and ident['runtime']==plan['runtime'],'First-recovery identity changed')
    for key in ('audit','identity','checkpoint','moments'):
        r.require(r.sha(paths[key])==report[key+'_sha256'],'First-recovery artifact changed')
    r.require(report['audit_passed']==bool(audit['passed'] and all(x['passed'] for x in audit['final_state_grid_checks'])),'First-recovery audit flag differs')
    moments=r.base.archive(paths['moments']);r.moments_checked(moments,report['estimate'],len(set(spec['task']['tokens'])),spec['target_draws'])
    return report,moments,{k:r.bind(paths[k]) for k in ('report','audit','identity','checkpoint','moments')}


def prepare(out,first,terminal):
    p,_=r.load(first);r.require(not (out/'plan.json').exists(),'Final continuation already prepared')
    receipt=r.read(terminal);r.require(receipt['terminal_verified'] and not receipt['active_jobs'],'First recovery still active')
    refs=dict(p['retained_report_bindings']);failed=[];selected=None
    for spec in p['tasks']:
        report,m,files=check_first(spec,p,first)
        if report['audit_passed']:refs[spec['task']['id']]=files['report']
        else:failed.append(spec['task']['id']);selected=(spec,files)
    r.require(failed==[CASE],'Final selection changed; inspect new failures explicitly')
    spec,files=selected;r.require(spec['target_draws']==65536 and r.checkpoint_counts(files['checkpoint']['path'])==[65536]*4,'Saved one-case budget differs')
    task=dict(index=0,task=spec['task'],reasons=['continued_ESS_below_200'],target_draws=262144,parent_files=files,parent_draw_counts=[65536]*4)
    plan={k:p[k] for k in ('schema','parent','parent_plan','parent_prepared','parent_code_sha256','code_sha256','runtime','original_tasks','bank_files','automatic_retries','acceptance')}
    plan.update(terminal_receipt=r.bind(terminal),tasks=[task],retained_report_bindings=refs,paired_failures=[],
        first_recovery_plan=r.bind(first/'plan.json'),first_recovery=str(first),final_code_sha256=own_codes(),
        resources=dict(task_seconds=13800,maximum_concurrent=1,allocated_cpus_per_task=2,memory_gib=12,
                       maximum_cpu_hours=12,maximum_cost_chf=.066))
    plan['plan_id']=r.fp(plan);r.write(out/'plan.json',plan,immutable=True)
    print(dict(event='prepared',case=CASE,draws=262144,maximum_cost_chf=.066),flush=True)


def load(out):
    p,old=r.load(out);r.require(p['final_code_sha256']==own_codes(),'Final wrapper changed')
    r.verify(p['first_recovery_plan']);first=Path(p['first_recovery']);fp,_=r.load(first)
    r.require(fp['plan_id']==r.read(p['first_recovery_plan']['path'])['plan_id'],'First plan identity differs')
    return p,first,fp


def worker(out):
    load(out);return r.worker(out,0)


def dispatch(out):
    p,_,_=load(out)
    with (out/'attempt.lock').open('x'):
        process=subprocess.Popen([sys.executable,__file__,'worker','--out',str(out)],start_new_session=True)
        try:code=process.wait(timeout=p['resources']['task_seconds'])
        except subprocess.TimeoutExpired:r.base.frequency.stop(process);code=75
    r.write(out/'attempt.json',dict(plan_id=p['plan_id'],returncode=code),immutable=True);return code


def merge(out):
    p,first,fp=load(out);specs={x['task']['id']:x for x in fp['tasks']}
    def overlay_parent(parent,original_plan,task):
        if task['id'] not in specs:return ORIGINAL_READER(parent,original_plan,task)
        report,moments,files=check_first(specs[task['id']],fp,first)
        r.require(report['audit_passed'],'Unpassed earlier recovery cannot be retained')
        spec=specs[task['id']]
        if 'checkpoint' in spec['parent_files']:
            r.preserve(spec['parent_files']['checkpoint']['path'],files['checkpoint']['path'],spec['target_draws'])
        return report,moments,files
    previous_reader=r.parent_record
    r.parent_record=overlay_parent
    try:code=r.merge(out)
    finally:r.parent_record=previous_reader
    if code==0:
        summary=r.read(out/'summary.json');arrays=r.base.archive(out/'arrays.npz')
        old=r.base.archive(Path(p['parent'])/'input.npz');exact=True
        for g in r.base.CONDITIONS:
            for new,name in (('bayes_loss_a32','bayes_loss'),('bayes_loss_a16','bayes_loss_a16'),
                             ('stationary_loss','stationary_loss'),('stationary_coarse_loss','stationary_coarse_loss')):
                exact=exact and r.np.array_equal(arrays[g+'_'+new][:,:8],old[g+'_'+name])
            for role in ('ordered','shuffled'):
                for seed in (0,1):exact=exact and r.np.array_equal(arrays[g+f'_{role}_seed{seed}'],old[g+f'_{role}_seed{seed}'])
        r.require(exact,'Retained source-score pairing changed')
        original=r.read(Path(p['parent'])/'input_receipt.json')['old_validation']['artifact_sha256']['arrays.npz']
        receipt=dict(schema='copying-dense16-release-validation-v1',passed=True,summary_sha256=r.sha(out/'summary.json'),
          arrays_sha256=r.sha(out/'arrays.npz'),input_bindings_sha256=r.sha(out/'input_bindings.json'),old_arrays_sha256=original,
          complete_original_cohorts=True,retained_early_scores_exact=True,neural_pairing_exact=True,
          plan_sha256=r.sha(out/'plan.json'),first_recovery_plan_sha256=p['first_recovery_plan']['sha256'],
          validation_scope='Full original768 histories; unchanged instantaneous and cumulative numerical gates; exact retained first8 scores.')
        r.write(out/'validation.json',receipt,immutable=True)
    return code


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('stage',choices=('prepare','worker','dispatch','merge'))
    q.add_argument('--out',type=Path,required=True);q.add_argument('--first',type=Path);q.add_argument('--terminal',type=Path)
    a=q.parse_args()
    if a.stage=='prepare':prepare(a.out,a.first,a.terminal);return 0
    return dict(worker=worker,dispatch=dispatch,merge=merge)[a.stage](a.out)
if __name__=='__main__':raise SystemExit(main())

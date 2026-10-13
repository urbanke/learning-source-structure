"""Exercise shared-base orchestration without Slurm, GPUs or numerical work."""
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
LAUNCHER=Path('code/transformer/kuma/submit_shared_base_shuffle.sh')
BATCH=Path('code/transformer/kuma/shared_base_shuffle_train.sbatch')
HELPER=Path('code/scripts/shared_base_shuffle_benchmark.py')
PLAN=Path('code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json')


@pytest.fixture
def repo(tmp_path):
    root=tmp_path/'isolated repository'
    for relative in (LAUNCHER,BATCH):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,path)
    path=root/PLAN;path.parent.mkdir(parents=True);path.write_text('{"launcher_test_fixture":true}\n')
    helper=root/HELPER;helper.parent.mkdir(parents=True,exist_ok=True)
    # The real benchmark helper has separate scientific tests. This stub only
    # exposes its documented CLI and acceptance/failure to the shell scripts.
    helper.write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
if a.check_only:
    path=a.run/'benchmark.json'
    if not path.exists() or not json.loads(path.read_text()).get('passed'):raise SystemExit('Passed benchmark required')
else:
    a.run.mkdir(parents=True,exist_ok=True);(a.run/'benchmark.json').write_text('{"passed":true}')
''')
    return root


def fake_runtime(tmp_path,repo,train_exit=0,array_id='0'):
    binary=tmp_path/'fakebin';binary.mkdir(exist_ok=True);calls=tmp_path/'fake_calls.jsonl'
    for name in ('sbatch','srun','python','python3'):
        path=binary/name
        path.write_text(f'#!{sys.executable}\n'+r'''
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name;args=sys.argv[1:]
with open(os.environ['FAKE_CALLS'],'a') as stream:
    stream.write(json.dumps(dict(program=name,args=args,cwd=os.getcwd()))+'\n')
if name=='sbatch':print('FAKE_JOB_ID')
elif name=='srun':
    if not any(a.endswith('train_shared_base_shuffle.py') for a in args):raise SystemExit('Unexpected numerical process')
    sys.exit(int(os.environ['FAKE_TRAIN_EXIT']))
elif args and (args[0]=='-' or args[0].endswith('shared_base_shuffle_benchmark.py')):
    os.execv(sys.executable,[sys.executable,*args])
elif args[:2]==['-m','pip']:print('fake-runtime==0')
elif args[:1]==['-c']:print('FAKE CUDA; no real device inspected')
else:raise SystemExit('Unexpected fake Python invocation')
''')
        path.chmod(0o755)
    venv=tmp_path/'venv with spaces';(venv/'bin').mkdir(parents=True,exist_ok=True)
    (venv/'bin/activate').write_text(f'export PATH={shlex.quote(str(binary))}:"$PATH"\n')
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],SHARED_BASE_SHUFFLE_VENV=str(venv),
        FAKE_CALLS=str(calls),FAKE_TRAIN_EXIT=str(train_exit),SLURM_SUBMIT_DIR=str(repo),
        SLURM_JOB_ID='test-job',SLURM_ARRAY_TASK_ID=str(array_id))
    return env,calls


def invoke(repo,*args,env=None):
    return subprocess.run(['bash',str(repo/LAUNCHER),*map(str,args)],cwd=repo.parent,env=env,capture_output=True,text=True)


def batch(repo,out,kind,env):
    return subprocess.run(['bash',str(repo/BATCH),str(out),kind,str(repo/PLAN),str(repo/'ordered train')],cwd=repo.parent,env=env,capture_output=True,text=True)


def tree(root):return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
def read_calls(path):return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]


def passed_benchmark(out,passed=True):
    path=out/'benchmark/seed0/benchmark.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(passed=passed)))


@pytest.mark.parametrize('flags',[(),('--benchmark',),('--dry-run',),('--benchmark','--dry-run')])
def test_dry_run_never_submits_or_mutates(repo,tmp_path,flags):
    env,calls=fake_runtime(tmp_path,repo);before=tree(repo)
    result=invoke(repo,*flags,env=env)
    assert result.returncode==0,result.stderr
    assert tree(repo)==before and not any(c['program']=='sbatch' for c in read_calls(calls))
    command=shlex.split(result.stdout)
    assert command[-2]==str(repo/PLAN) and command[-4]==str(repo/'output/shared_base_shuffle_2026-09-17')
    assert command[-1]==str(repo/'output/shared_base_scaling_2026-09-17/train')
    assert f'--chdir={repo}' in command
    if '--benchmark' in flags:
        assert '--qos=debug' in command and '--time=00:30:00' in command
        assert not any(s.startswith('--array=') for s in command)
    else:assert '--array=0-1%2' in command


@pytest.mark.parametrize('flags',[('--unknown',),('--out',),('--benchmark','--benchmark'),('--plan','other.json'),
    ('--arm','longsmall'),('--summary',)])
def test_unplanned_modes_rejected_without_mutation(repo,flags):
    before=tree(repo);result=invoke(repo,*flags)
    assert result.returncode!=0 and tree(repo)==before


def test_production_requires_passed_benchmark_before_writing_or_submission(repo,tmp_path):
    env,calls=fake_runtime(tmp_path,repo);out=tmp_path/'new result directory'
    result=invoke(repo,'--submit','--out',out,env=env)
    assert result.returncode!=0 and not out.exists()
    assert not any(c['program']=='sbatch' for c in read_calls(calls))
    passed_benchmark(out,False);before=tree(out)
    assert invoke(repo,'--submit','--out',out,env=env).returncode!=0 and tree(out)==before
    passed_benchmark(out);result=invoke(repo,'--submit','--out',out,env=env)
    assert result.returncode==0,result.stderr
    submissions=[c for c in read_calls(calls) if c['program']=='sbatch']
    assert len(submissions)==1 and submissions[0]['cwd']==str(repo)
    assert submissions[0]['args'][-4:]==[str(out),'train',str(repo/PLAN),str(repo/'output/shared_base_scaling_2026-09-17/train')]


def test_benchmark_submission_and_account_quoting(repo,tmp_path):
    env,calls=fake_runtime(tmp_path,repo);env['SHARED_BASE_SHUFFLE_ACCOUNT']='account with spaces'
    out=tmp_path/'new result directory';result=invoke(repo,'--benchmark','--submit','--out',out,env=env)
    assert result.returncode==0,result.stderr
    calls=read_calls(calls);submitted=[c for c in calls if c['program']=='sbatch']
    assert len(submitted)==1 and '--account=account with spaces' in submitted[0]['args']
    assert not any(c['args'] and c['args'][0].endswith('shared_base_shuffle_benchmark.py') for c in calls)


@pytest.mark.parametrize('location',['results','code','repo','ancestor','old_outputs','alias'])
def test_existing_science_and_aliases_are_protected(repo,tmp_path,location):
    if location in ('results','code'):out=repo/location/'some experiment'
    elif location=='repo':out=repo
    elif location=='ancestor':out=repo.parent
    else:
        old=tmp_path/'old run';old.mkdir();(old/'model_100000.pt').write_bytes(b'unchanged checkpoint')
        out=old
        if location=='alias':out=tmp_path/'alias';out.symlink_to(old,target_is_directory=True)
    env,calls=fake_runtime(tmp_path,repo);before=tree(repo)
    result=invoke(repo,'--benchmark','--submit','--out',out,env=env)
    assert result.returncode!=0 and tree(repo)==before
    assert not any(c['program']=='sbatch' for c in read_calls(calls))


@pytest.mark.parametrize('seed',[0,1])
def test_batch_only_runs_one_planned_seed_without_inference(repo,tmp_path,seed):
    out=tmp_path/'outputs with spaces';passed_benchmark(out)
    env,calls=fake_runtime(tmp_path,repo,array_id=seed)
    result=batch(repo,out,'train',env);assert result.returncode==0,result.stderr
    runs=[c['args'] for c in read_calls(calls) if c['program']=='srun'];assert len(runs)==1
    command=runs[0];assert command[command.index('--seed')+1]==str(seed)
    assert command[command.index('--out')+1]==str(out/'train'/f'seed{seed}')
    assert command[command.index('--device')+1]=='cuda' and command[command.index('--workers')+1]=='6'
    assert '--resume' not in command and '--max-steps' not in command


def test_batch_rechecks_gate_and_rejects_unplanned_seed(repo,tmp_path):
    env,calls=fake_runtime(tmp_path,repo);out=tmp_path/'outputs'
    result=batch(repo,out,'train',env);assert result.returncode!=0 and not out.exists()
    assert not any(c['program']=='srun' for c in read_calls(calls))
    env['SLURM_ARRAY_TASK_ID']='2';assert batch(repo,out,'train',env).returncode!=0


@pytest.mark.parametrize('status',[0,1,75])
def test_benchmark_accepts_only_the_intended_cap(repo,tmp_path,status):
    env,calls=fake_runtime(tmp_path,repo,train_exit=status);out=tmp_path/'outputs'
    result=batch(repo,out,'benchmark',env)
    assert (result.returncode==0)==(status==75)
    command=next(c['args'] for c in read_calls(calls) if c['program']=='srun')
    assert command[command.index('--max-steps')+1]=='1000'
    assert command[command.index('--out')+1]==str(out/'benchmark/seed0')
    assert (out/'benchmark/seed0/benchmark.json').exists()==(status==75)


def test_resume_preserves_checkpoint_and_propagates_incomplete_exit(repo,tmp_path):
    out=tmp_path/'outputs';passed_benchmark(out);checkpoint=out/'train/seed1/latest.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'original resume state')
    env,calls=fake_runtime(tmp_path,repo,array_id=1,train_exit=75)
    result=batch(repo,out,'train',env);assert result.returncode==75
    command=next(c['args'] for c in read_calls(calls) if c['program']=='srun')
    assert '--resume' in command and checkpoint.read_bytes()==b'original resume state'
    assert '--max-steps' not in command


def test_allocations_and_checkpoint_signal_are_explicit():
    text=(ROOT/BATCH).read_text()
    for directive in ('--partition=h100','--gres=gpu:1','--cpus-per-task=8','--mem=32G',
                      '--time=06:00:00','--signal=USR1@180','--no-requeue'):
        assert '#SBATCH '+directive in text
    assert 'OPENBLAS_NUM_THREADS=1' in text and 'OMP_NUM_THREADS=1' in text


def test_ordered_root_cannot_be_overwritten(repo,tmp_path):
    env,calls=fake_runtime(tmp_path,repo)
    original=tmp_path/'original weights';original.mkdir()
    for output in [original,original/'nested',original.parent]:
        result=invoke(repo,'--benchmark','--submit','--ordered-root',original,'--out',output,env=env)
        assert result.returncode!=0
    assert not any(c['program']=='sbatch' for c in read_calls(calls))

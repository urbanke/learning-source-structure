"""Verify confirmation submissions without a cluster, inference or reference IO."""
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path('code/transformer/kuma/submit_hidden_factor_confirmation.sh')
BATCH = Path('code/transformer/kuma/hidden_factor_confirmation.sbatch')
PLAN = Path('code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json')


@pytest.fixture
def setup(tmp_path):
    root = tmp_path/'repository with spaces'
    for rel in (LAUNCHER, BATCH):
        dst = root/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dst)
    plan = root/PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text('{}')
    baseline = root/'output/hidden_factor_2026-09-16/2x8'
    scaling = root/'output/hidden_factor_scaling_2026-09-16'
    paths = [baseline/'eval'/name for name in ('cohort.json', 'cohort.npz', 'references/manifest.json')]
    for seed in (0, 1):
        paths.extend([baseline/f'train/seed{seed}/model_100000.pt',
                      scaling/f'train/longsmall/seed{seed}/model_400000.pt',
                      scaling/f'train/large/seed{seed}/model_100000.pt'])
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not real scientific data; never read on the login node')
    binary = tmp_path/'bin'
    binary.mkdir()
    calls = tmp_path/'calls.jsonl'
    for name in ('sbatch', 'srun', 'python'):
        path = binary/name
        path.write_text(f'#!{sys.executable}\n'+'''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['FAKE_CALLS'], 'a') as f:
    f.write(json.dumps({'program': name, 'args': sys.argv[1:]})+'\\n')
if name == 'sbatch': print('12345')
elif name == 'srun': sys.exit(int(os.environ.get('FAKE_EVAL_EXIT', '0')))
elif sys.argv[1:2] == ['-c']: print('fake CUDA')
else: raise SystemExit('Unexpected numerical process outside srun')
''')
        path.chmod(0o755)
    venv = tmp_path/'venv with spaces'
    (venv/'bin').mkdir(parents=True)
    (venv/'bin/activate').write_text('export PATH='+shlex.quote(str(binary))+':"$PATH"\n')
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], FAKE_CALLS=str(calls),
               HIDDEN_FACTOR_VENV=str(venv), SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='12345')
    return root, baseline, scaling, env, calls


def invoke(setup, *args):
    root, _, _, env, _ = setup
    return subprocess.run(['bash', str(root/LAUNCHER), *map(str, args)], env=env, capture_output=True, text=True)


def recorded(path):
    return [] if not path.exists() else [json.loads(s) for s in path.read_text().splitlines()]


def test_print_only_and_submission_never_read_scientific_inputs(setup):
    root, baseline, scaling, env, calls = setup
    out = root/'output/hidden_factor_confirmation_2026-09-18'
    result = invoke(setup)
    assert result.returncode == 0, result.stderr
    assert not out.exists() and recorded(calls) == []
    command = shlex.split(result.stdout)
    assert command[-5:] == [str(out), str(baseline), str(scaling), str(root/PLAN), '16']
    result = invoke(setup, '--submit')
    assert result.returncode == 0, result.stderr
    assert [c['program'] for c in recorded(calls)] == ['sbatch']
    assert (out/'logs').is_dir()


def test_submission_quotes_paths_account_and_dependency(setup, tmp_path):
    root, _, _, env, calls = setup
    env['HIDDEN_FACTOR_ACCOUNT'] = 'account with spaces'
    out = tmp_path/'confirmation with spaces'
    result = invoke(setup, '--submit', '--out', out, '--after-job', '456', '--batch-size', '8')
    assert result.returncode == 0, result.stderr
    command = recorded(calls)[0]['args']
    assert '--account=account with spaces' in command
    assert '--dependency=afterok:456' in command and '--kill-on-invalid-dep=yes' in command
    assert command[-5] == str(out) and command[-1] == '8'


@pytest.mark.parametrize('bad', [(' --submit',), ('--after-job', '1; bad'), ('--batch-size', '0'),
                               ('--batch-size', '-1'), ('--out',), ('--summary',)])
def test_invalid_arguments_cannot_submit(setup, bad):
    result = invoke(setup, *bad)
    assert result.returncode != 0
    assert recorded(setup[-1]) == []


def test_missing_checkpoint_prevents_submission_and_output_creation(setup):
    root, _, scaling, _, calls = setup
    (scaling/'train/large/seed1/model_100000.pt').unlink()
    result = invoke(setup, '--submit')
    assert result.returncode != 0 and 'Missing final checkpoint' in result.stderr
    assert not (root/'output/hidden_factor_confirmation_2026-09-18').exists()
    assert recorded(calls) == []


def test_input_and_source_paths_cannot_be_overwritten_including_aliases(setup, tmp_path):
    root, baseline, scaling, _, calls = setup
    alias = tmp_path/'alias'
    alias.symlink_to(scaling, target_is_directory=True)
    for out in (baseline, baseline/'new', baseline.parent, scaling, alias, root, root.parent, root/'code/new'):
        result = invoke(setup, '--submit', '--out', out)
        assert result.returncode != 0, str(out)
    assert recorded(calls) == []


@pytest.mark.parametrize('exit_code', [0, 1, 75])
def test_allocated_job_runs_all_evaluation_and_propagates_failure(setup, tmp_path, exit_code):
    root, baseline, scaling, env, calls = setup
    env['FAKE_EVAL_EXIT'] = str(exit_code)
    out = tmp_path/'confirmation output'
    result = subprocess.run(['bash', str(root/BATCH), str(out), str(baseline), str(scaling), str(root/PLAN), '16'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    runs = [c['args'] for c in recorded(calls) if c['program'] == 'srun']
    assert len(runs) == 1
    command = runs[0]
    assert command[:3] == ['--unbuffered', 'python', 'code/transformer/hidden_factor_confirmation_eval.py']
    assert command[command.index('--stage')+1] == 'all'
    assert command[command.index('--device')+1] == 'cuda'
    assert command[command.index('--baseline-root')+1] == str(baseline)
    assert command[command.index('--scaling-root')+1] == str(scaling)


def test_allocation_is_suitable_for_kuma_memory_limit():
    script = (ROOT/BATCH).read_text()
    for setting in ('--partition=h100', '--gres=gpu:1', '--cpus-per-task=8', '--mem=32G',
                    '--time=02:00:00', '--no-requeue'):
        assert '#SBATCH '+setting in script
    assert 32 * 1024 / 8 < 5900  # Site's known MaxMemPerCPU limit in MiB.

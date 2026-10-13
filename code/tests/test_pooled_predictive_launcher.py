"""The cluster smoke launcher must remain bounded and avoid login-node inference."""
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'code/transformer/scitas/submit_pooled_predictive_pilot.sh'


def test_dry_run_has_three_bounded_cases_and_creates_nothing(tmp_path):
    out=tmp_path/'new output'
    r=subprocess.run(['bash',str(SCRIPT),'--out',str(out)],capture_output=True,text=True)
    assert r.returncode==0 and not out.exists()
    cmd=shlex.split(r.stdout)
    assert '--array=0-2%3' in cmd and cmd[-1]=='16'
    assert cmd[-2]==str(out)
    batch=(ROOT/'code/transformer/scitas/pooled_predictive_pilot.sbatch').read_text()
    assert '#SBATCH --time=01:00:00' in batch
    assert '--index 1 --development' in batch and '--particles 32' in batch
    assert '--replicates 2' in batch and '--seconds-per-replica 300' in batch
    assert 'srun --ntasks=1 --unbuffered python' in batch


def test_protected_output_and_unplanned_context_are_rejected(tmp_path):
    for args in (['--out',str(ROOT/'notes/new')],['--observed','1024'],
                 ['--raw',str(tmp_path/'input'),'--out',str(tmp_path/'input/child')]):
        r=subprocess.run(['bash',str(SCRIPT),*args],capture_output=True,text=True)
        assert r.returncode!=0

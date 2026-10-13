"""Cache parity and interrupted diagnostics, without production sampling."""
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'))
sys.path.insert(0,str(ROOT/'code/transformer'))
import profile_pooled_predictive_cache as cli


def test_real_eviction_preserves_seeded_estimator(tmp_path):
    from ranked_lsa import RankedLSARowEvidence
    from shared_base_mcmc import SharedBaseMCMC
    from shared_base_predictive_smc import estimate
    def kernel(capacity):
        return RankedLSARowEvidence(3,[.6,1.3],[0,1],term_cache_size=capacity)
    def compute(k):
        return estimate(SharedBaseMCMC(3,k,[.3,.7]),[0,1,0,2,0,1],
            particles=8,temperatures=4,moves=2,seed=301,seconds_limit=20)
    rows=[cli.profile_variant(tmp_path,c,kernel,compute,20) for c in (1,64)]
    assert cli.parity(rows) is True
    assert rows[0]['cache_info']['phi_terms_computed']>rows[1]['cache_info']['phi_terms_computed']
    assert all(not r['scientific_reference_validated'] for r in rows)
    rows[1]['estimate']['log_evidence']+=1e-12
    assert cli.parity(rows) is False


@pytest.mark.parametrize('failure', ['timer','error'])
def test_interruption_retains_profile_and_counters(tmp_path,failure):
    def compute(k):
        if failure=='timer': time.sleep(.2)
        raise ValueError('deliberate error')
    old=signal.getsignal(signal.SIGALRM)
    row=cli.profile_variant(tmp_path,1024,lambda c:SimpleNamespace(cache_info=lambda:{'calls':7}),
        compute,.03)
    assert row['status']==('time_limit' if failure=='timer' else 'error')
    assert not row['complete'] and row['cache_info']=={'calls':7}
    assert (tmp_path/'cache1024.prof').stat().st_size>0
    assert json.loads((tmp_path/'cache1024.json').read_text())==row
    assert signal.getsignal(signal.SIGALRM)==old
    assert cli.parity([row,row]) is None


def test_prepare_uses_real_plan_api_and_only_tokens(tmp_path,monkeypatch):
    import pooled_predictive_population as m
    # Real fixed plan loaders retain their hash checks; mock only large local data.
    monkeypatch.setattr(m.calibration,'dependencies',lambda *a:{'sealed':'hash'})
    monkeypatch.setattr(m.calibration,'numerical_environment',lambda *a:{'mode':'production'})
    def load(raw,plan,spec):
        assert spec==cli.SETTINGS and spec['family']=='product' and spec['observed']==256
        return np.arange(256)%64,object(),object(),{'cohort':'hash'}
    monkeypatch.setattr(m,'load_case',load)
    receipt=dict(settings=cli.SETTINGS,raw=str(tmp_path/'raw'),dependencies=str(tmp_path/'dep'),
        code_sha256={f:cli.sha(ROOT/f) for f in cli.FILES})
    cli.save(tmp_path/'submission.json',receipt)
    plan=cli.prepare(tmp_path)
    assert plan['tokens']==(np.arange(256)%64).tolist()
    assert plan['dependency_sha256']=={'sealed':'hash'}
    assert not plan['scientific_reference_validated']
    with pytest.raises(ValueError,match='already prepared'): cli.prepare(tmp_path)


def test_dry_run_is_small_and_does_not_write(tmp_path):
    out=tmp_path/'absent'
    result=subprocess.run([sys.executable,str(ROOT/cli.FILES[0]),'submit','--out',str(out)],
        check=True,capture_output=True,text=True)
    assert 'sbatch --parsable' in result.stdout and not out.exists()
    batch=(ROOT/cli.FILES[1]).read_text()
    for wanted in ('--cpus-per-task=1','--mem=6G','--time=00:40:00'): assert wanted in batch
    assert cli.SETTINGS['seconds_per_variant']*len(cli.SETTINGS['term_caches'])==1800
    with pytest.raises(ValueError): cli.protected_output(ROOT/'code/bad',tmp_path/'raw',tmp_path/'dep')


def test_real_local_frozen_cohort_loader():
    import pooled_predictive_population as m
    raw=ROOT/'output/multifamily_results_2026-09-16'
    if not (raw/'heldout/manifest.json').exists(): pytest.skip('Optional local frozen cohort absent')
    plan,_=m.read_plan(m.PLAN)
    tokens,oracle,scores,hashes=m.load_case(raw,plan,cli.SETTINGS)
    assert tokens.shape==(256,) and oracle.shape==(64,) and len(scores)==2 and len(hashes)==7


def test_local_sealed_backend_and_production_constructor(tmp_path,monkeypatch):
    import pooled_predictive_population as m
    base=Path.home()/'Projects/product_model_with_memory'
    store,src=base/'tables/anchors_prod',base/'src'
    if not (store/'manifest.json').exists(): pytest.skip('Optional sealed local store absent')
    plan,_=m.calibration.read_plan(m.calibration.PLAN)
    assert m.calibration.dependencies(plan,store,src)
    # The backend builds a tiny C helper on first import; preserve the source checkout.
    copied=tmp_path/'src'
    manifest=json.loads((ROOT/plan['dependencies']['manifest']).read_text())
    for name in manifest['files']:
        if name.startswith('src/'):
            relative=Path(name).relative_to('src'); target=copied/relative
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src/relative,target)
    src=copied
    with monkeypatch.context() as patch:
        patch.setattr(m.calibration.os,'environ',dict(m.calibration.os.environ))
        m.calibration.numerical_environment(store,src)
        nodes,weights=m.leggauss(16)
        def kernel(c):
            return m.RankedLSARowEvidence(64,1+nodes/2,range(9),store=store,src=src,term_cache_size=c)
        def compute(k):
            assert np.isfinite(k._term(0,0,1)).all() # Exercise the sealed depth2..8 backend.
            return m.estimate(m.SharedBaseMCMC(64,k,weights/2),[0],particles=2,
                temperatures=1,moves=1,seed=cli.SETTINGS['seed'],seconds_limit=20)
        reports=[cli.profile_variant(tmp_path,c,kernel,compute,20) for c in (1024,4096)]
    assert cli.parity(reports) is True


def test_ambiguous_submission_is_not_repeated(tmp_path,monkeypatch):
    raw,dep,out=(tmp_path/n for n in ('raw','dep','out'))
    for path in (raw/'heldout/manifest.json',dep/'tables/anchors_prod/manifest.json'):
        path.parent.mkdir(parents=True); path.write_text('{}')
    calls=[]
    def submit(*a,**kw):
        calls.append(a); return SimpleNamespace(returncode=1,stdout='maybe accepted',stderr='connection lost')
    monkeypatch.setattr(cli.subprocess,'run',submit)
    args=SimpleNamespace(out=out,raw=raw,dependencies=dep,submit=True)
    assert cli.submit(args)==1
    with pytest.raises(FileExistsError): cli.submit(args)
    assert len(calls)==1
    assert json.loads((out/'submission.json').read_text())['status']=='submission_error'

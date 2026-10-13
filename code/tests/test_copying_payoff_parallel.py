"""Independent lifecycle checks without running any numerical sampler."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'))
import copying_payoff_parallel as parallel


def test_reserved_memory_and_startup_admission():
    assert parallel.available_slots(0,0)==2
    assert parallel.available_slots(1,1)==1
    assert parallel.available_slots(0,2)==4
    assert parallel.available_slots(4,200)==0
    assert parallel.RESOURCES['reserved_memory_bytes_per_worker']*4==parallel.RESOURCES['total_memory_bytes']


@pytest.fixture
def setup(tmp_path,monkeypatch):
    tasks=[dict(task_index=i,id=f'task{i}',time=6,order=16 if i%2==0 else 32,kind='fresh') for i in range(8)]
    plan=dict(plan_id='science',tasks=tasks,python=sys.executable,environment={})
    campaign=dict(campaign_id='allocation',times=[6])
    monkeypatch.setattr(parallel,'load_campaign',lambda out:campaign)
    monkeypatch.setattr(parallel.dense,'load_plan',lambda out:plan)
    monkeypatch.setattr(parallel.dense.pilot,'environment',lambda plan:{})
    monkeypatch.setattr(parallel.dense,'summary',lambda *a,**kw:dict(complete=True,all_individual_audits_passed=True))
    monkeypatch.setattr(parallel,'campaign_summary',lambda out:dict(complete=True))
    monkeypatch.setattr(parallel,'merge_wave',lambda *a:dict(complete=True))
    monkeypatch.setattr(parallel,'resident_bytes',lambda processes:0)
    clock=[0.];monkeypatch.setattr(parallel.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(parallel.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    held=[];events=[];processes=[];settings=dict(duration={},exit={},audit={})
    @contextmanager
    def lock(path):
        held.append(str(path));events.append(('lock',str(path)))
        try:yield
        finally:events.append(('unlock',str(path)));held.remove(str(path))
    monkeypatch.setattr(parallel.dense,'lock',lock)
    class Process:
        def __init__(self,index):
            self.index=index;self.pid=10000+index;self.end=clock[0]+settings['duration'].get(index,2)
            self.returncode=None;self.started=clock[0];self.code=settings['exit'].get(index,0)
            task=tasks[index];paths=parallel.dense.paths(tmp_path/'t6',task)
            parallel.write(paths['report'],dict(campaign_id='allocation',plan_id='science',task=task,
                complete=True,audit_passed=settings['audit'].get(index,True),peak_rss_bytes=100))
        def poll(self):
            if self.returncode is None and clock[0]>=self.end:self.returncode=self.code
            return self.returncode
    def popen(command,**kwargs):
        process=Process(int(command[-1]));processes.append(process);events.append(('start',process.index,clock[0]))
        assert len(held)==2
        assert sum(p.poll() is None for p in processes)<=4
        return process
    def terminate(process):
        if process.poll() is None:
            assert len(held)==2,'Dispatcher released global locks before stopping children'
            events.append(('terminate',process.index));process.returncode=-15
    monkeypatch.setattr(parallel.subprocess,'Popen',popen)
    monkeypatch.setattr(parallel,'terminate',terminate)
    return dict(out=tmp_path,tasks=tasks,settings=settings,processes=processes,clock=clock,events=events)


def test_two_completed_startup_references_before_four_workers(setup):
    s=setup;s['settings']['duration'][1]=4
    assert parallel.dispatch(s['out'])['complete']
    starts={p.index:p.started for p in s['processes']}
    assert starts[0]==starts[1]==0
    assert min(starts[i] for i in range(2,6))>=4
    assert len({starts[i] for i in range(2,6)})==1
    status=parallel.read(s['out']/'status.json')
    assert status['status']=='complete' and status['completed_tasks']==8 and status['active_workers']==[]
    with pytest.raises(ValueError,match='already used'):parallel.dispatch(s['out'])


def test_numerical_unresolved_is_kept_without_stopping_independent_tasks(setup):
    s=setup;s['settings']['audit'][2]=False
    parallel.dispatch(s['out'])
    status=parallel.read(s['out']/'status.json')
    assert status['unresolved_task_ids']==['task2'] and status['completed_tasks']==8
    attempt=parallel.read(parallel.dense.paths(s['out']/'t6',s['tasks'][2])['attempt'])
    assert attempt['status']=='unresolved'


def test_technical_failure_stops_and_cleans_up_with_locks_held(setup):
    s=setup;s['settings']['exit'][0]=7;s['settings']['duration'][1]=100
    with pytest.raises(RuntimeError,match='Technical worker failure'):parallel.dispatch(s['out'])
    assert len(s['processes'])==2 and s['processes'][1].returncode==-15
    status=parallel.read(s['out']/'status.json')
    assert status['status']=='stopped' and 'exit7' in status['reason'] and not status['active_workers']


def test_near_deadline_stops_admission_but_drains_existing_workers(setup,monkeypatch):
    s=setup;s['settings']['duration'].update({0:2,1:5})
    resources=dict(parallel.RESOURCES,wall_seconds=901)
    monkeypatch.setattr(parallel,'RESOURCES',resources)
    with pytest.raises(RuntimeError,match='No full900'):parallel.dispatch(s['out'])
    assert s['clock'][0]==5 and len(s['processes'])==2
    assert all(p.returncode==0 for p in s['processes'])
    assert not any(e[0]=='terminate' for e in s['events'])


def test_signal_style_interruption_preserves_checkpoints_and_stops_children(setup,monkeypatch):
    s=setup
    def interrupted(seconds):raise InterruptedError('SIGTERM test')
    monkeypatch.setattr(parallel.time,'sleep',interrupted)
    with pytest.raises(InterruptedError):parallel.dispatch(s['out'])
    assert len(s['processes'])==2 and all(p.returncode==-15 for p in s['processes'])
    assert parallel.read(s['out']/'allocation.json')['reason']=='SIGTERM test'


def test_memory_guard_stops_before_more_admissions(setup,monkeypatch):
    s=setup
    monkeypatch.setattr(parallel,'resident_bytes',lambda processes:11*2**30 if processes else 0)
    with pytest.raises(ValueError,match='RSS'):parallel.dispatch(s['out'])
    assert len(s['processes'])==2 and all(p.returncode==-15 for p in s['processes'])


def test_competing_dispatcher_cannot_overwrite_winners_receipt(setup,monkeypatch):
    s=setup;winner={'dispatcher_pid':999,'status':'running'}
    @contextmanager
    def raced_lock(path):
        parallel.write(s['out']/'allocation.json',winner)
        raise BlockingIOError('another dispatcher won')
        yield
    monkeypatch.setattr(parallel.dense,'lock',raced_lock)
    with pytest.raises(BlockingIOError):parallel.dispatch(s['out'])
    assert parallel.read(s['out']/'allocation.json')==winner
    assert not (s['out']/'status.json').exists()


def test_manifest_rejects_changed_resource_policy_even_if_resigned(tmp_path,monkeypatch):
    monkeypatch.setattr(parallel.dense.pilot,'ensure_output',lambda out:Path(out))
    monkeypatch.setattr(parallel,'code_hashes',lambda:{'frozen':'hash'})
    value=dict(schema=parallel.SCHEMA,output=str(tmp_path),resources=dict(parallel.RESOURCES,max_workers=12),
        code_sha256={'frozen':'hash'})
    value['campaign_id']=parallel.dense.fingerprint(value)
    parallel.write(tmp_path/'campaign.json',value)
    with pytest.raises(ValueError,match='resource policy'):parallel.load_campaign(tmp_path)


def test_both_mergers_and_original_numerical_implementation_are_frozen():
    hashes=parallel.code_hashes()
    for path in ('code/scripts/merge_copying_payoff_parallel.py','code/scripts/merge_copying_payoff_dense.py',
                 'code/scripts/copying_payoff_dense.py','code/transformer/ranked_lsa.py',
                 'code/transformer/shared_base_curve_runner.py'):
        assert hashes[path]==parallel.sha(ROOT/path)

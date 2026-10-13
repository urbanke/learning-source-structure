"""No-sampling tests for the small canonical full-Bayes pilot plan."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import copying_payoff_bayes_pilot as pilot
from prepare_copying_payoff_references import reference_key


def inventory():
    selected=[]
    for n in (5,8,16):
        for values,name in [([0]*(n-1)+[1],'fewest_seen'),(list(range(n)),'most_seen')]:
            selected.append(dict(reference_key=reference_key(values),tokens=values,time=n,
                                 n_seen=len(set(values)),selection=name))
    return dict(schema='copying-payoff-canonical-reference-readiness-v1',histories_total=768,
                depths=[0,2,3],pilot={'representative_patterns':selected})


def test_fixed_tasks_have_both_orders_and_unique_seeds():
    tasks=pilot.tasks_from_inventory(inventory())
    assert len(tasks)==12
    assert [t['task_index'] for t in tasks]==list(range(12))
    assert len({t['id'] for t in tasks})==len({t['seed'] for t in tasks})==12
    for left,right in zip(tasks[::2],tasks[1::2]):
        assert left['reference_key']==right['reference_key']
        assert [left['order'],right['order']]==[16,32]
    assert [tasks[i]['time'] for i in (0,2)]==[5,5]
    assert [tasks[i]['order'] for i in (0,2)]==[16,16]


def test_task_seeds_deterministic_and_outcomes_do_not_select():
    saved=inventory();before=pilot.tasks_from_inventory(saved)
    for item in saved['pilot']['representative_patterns']:item['reported_loss_gap']=100.
    assert before==pilot.tasks_from_inventory(saved)


@pytest.mark.parametrize('mutation',['missing','length','labels','hash'])
def test_changed_representative_rejected(mutation):
    saved=deepcopy(inventory());p=saved['pilot']['representative_patterns']
    if mutation=='missing':p.pop()
    elif mutation=='length':p[0]['tokens'].append(0)
    elif mutation=='labels':p[0]['tokens'][0]=63
    else:p[0]['reference_key']='t0005_forged'
    with pytest.raises(ValueError):pilot.tasks_from_inventory(saved)


def test_output_cannot_alias_repository_or_source_tree():
    with pytest.raises(ValueError):pilot.ensure_output(pilot.ROOT)
    with pytest.raises(ValueError):pilot.ensure_output(pilot.ROOT/'code')
    with pytest.raises(ValueError):pilot.ensure_output(pilot.ROOT/'output')

"""Full selected cohorts, immutable comparators and unchanged numerical gates."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_unseen_context_refinement as p
from shared_base_precision_cohort import deterministic_precision


@pytest.fixture
def data():
    groups, table, specs, audits, sources = [], [], [], [], {}
    q = np.ones(64)/64
    for depth in (0, 1, 4, 8):
        for time in (4, 16, 64):
            n = p.TARGETS.get((depth, time), 2)
            selected = list(range(n))
            group = dict(depth=depth, time=time, n_selected=n, n_original=256,
                         selected_indices=selected, predictors={}, paired_loss_bits={})
            groups.append(group)
            for i in selected:
                row = dict(depth=depth, time=time, index=i, query=1, source_id=f'{depth}:{i}',
                           entropy_bits=6., bayes_released=False)
                row.update({name+'_loss_bits': 6.1+j*.01+i*.0001
                            for j, name in enumerate(p.original.PREDICTORS)})
                table.append(row)
                if (depth, time) not in p.TARGETS:
                    continue
                sources[depth, i] = dict(source_id=row['source_id'], tokens=[0]*63+[1],
                    identity=dict(source_depth=depth, source_index=i),
                    oracle=dict(transition_probs=np.tile(q, (64, 1)).tolist()))
                prob = np.full(64, (1-(2+i/100)/64)/63)
                prob[0] = (2+i/100)/64
                for order in (16, 32):
                    task = dict(id=f'L{depth}_i{i}_t64_a{order}', depth=depth, index=i,
                                time=64, order=order, setting=f'a1_L{depth}')
                    spec = dict(task=task, source_id=row['source_id'])
                    specs.append(spec)
                    record = dict(case_id=i, sampling_seed=None, precision=deterministic_precision(prob, q),
                        full_prior_valid=True, provenance_valid=True, mixing_passed=True,
                        row_grid_passed=True, per_sequence_precision_passed=False)
                    audit = dict(index=i, time=64, order=order, status='ready', record=record)
                    audits.append(dict(task=dict(task, key=p.refinement.key(spec)), corrected=dict(audit=audit)))
    baseline = dict(groups=groups, script_sha256='historical', arithmetic_checks={}, input_sha256={})
    plan = dict(groups=[dict(depth=d, time=t, selected_indices=list(range(n)))
                        for (d, t), n in p.TARGETS.items()], tasks=specs)
    return baseline, table, plan, audits, sources


def test_only_123_bayes_rows_change_and_all_paired_means_recomputed(data):
    original = copy.deepcopy(data)
    summary, rows = p.refine_compact(*data)
    assert summary['refinement']['all_groups_passed']
    assert summary['arithmetic_checks']['replaced_bayesian_rows'] == 123
    assert data == original
    changed = 0
    for old, new in zip(data[1], rows):
        target = (old['depth'], old['time']) in p.TARGETS
        if not target:
            assert new == old
        else:
            changed += 1
            assert new['bayes_released']
            assert all(new[k] == old[k] for k in old if k not in ('bayes_loss_bits', 'bayes_released'))
    assert changed == 123
    group = next(g for g in summary['groups'] if (g['depth'], g['time']) == (4, 64))
    selected = [r for r in rows if (r['depth'], r['time']) == (4, 64)]
    for name in ('larger_seed0', 'larger_seed1', 'transformer_seed0', 'transformer_seed1'):
        values = np.array([r[name+'_loss_bits']-r['bayes_loss_bits'] for r in selected])
        saved = group['paired_loss_bits'][name+'_minus_bayes']
        assert saved['mean'] == pytest.approx(values.mean())
        assert saved['standard_error'] == pytest.approx(values.std(ddof=1)/np.sqrt(66))


def test_failed_original_gate_withholds_whole_group_without_losing_histories(data):
    data[3][0]['corrected']['audit']['record']['mixing_passed'] = False
    summary, rows = p.refine_compact(*data)
    assert not summary['refinement']['all_groups_passed']
    selected = [r for r in rows if (r['depth'], r['time']) == (4, 64)]
    assert len(selected) == 66 and all(not r['bayes_released'] for r in selected)
    assert all(r['bayes_released'] for r in rows if (r['depth'], r['time']) == (8, 64))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'selection', 'source', 'oracle', 'regret'])
def test_rejects_coverage_identity_or_arithmetic_changes(data, change):
    if change == 'missing':
        data[3].pop()
    elif change == 'duplicate':
        data[3][-1] = copy.deepcopy(data[3][0])
    elif change == 'selection':
        data[2]['groups'][0]['selected_indices'][-1] = 250
    elif change == 'source':
        data[4][4, 0]['source_id'] = 'wrong'
    elif change == 'oracle':
        data[4][4, 0]['tokens'][63] = 0
    else:
        data[3][1]['corrected']['audit']['record']['precision']['bayes_regret_bits'] += .1
    with pytest.raises(ValueError):
        p.refine_compact(*data)


def test_baseline_loading_validates_saved_producer_and_preserves_all_rows():
    if not (p.BASE/'summary.json').exists():
        pytest.skip('Historical local export unavailable')
    summary, rows, bindings = p.load_baseline(p.BASE)
    assert len(summary['groups']) == 12
    assert len(rows) == sum(g['n_selected'] for g in summary['groups'])
    assert len(bindings) == 4


def test_recovery_cannot_replace_different_parent_with_same_coordinates(data, tmp_path, monkeypatch):
    row = next(r for r in data[3] if (r['task']['depth'], r['task']['index'], r['task']['order']) == (4, 58, 32))
    old = row['task']
    old.update(reference='old/r', checkpoint='old/c', moments='old/m',
               file_sha256={'old/r': 'bound', 'old/c': 'bound', 'old/m': 'bound'})
    fake = SimpleNamespace(validated_overlay=lambda *a, **k: copy.deepcopy(row))
    monkeypatch.setitem(sys.modules, 'unseen_context_recovery', fake)
    (tmp_path/'plan.json').write_text(json.dumps(dict(parent_files={'old/r': 'other'})))
    with pytest.raises(ValueError, match='parent files'):
        p.apply_recovery(data[2], data[3], tmp_path)


def test_runtime_changes_are_rejected_before_raw_replay():
    with pytest.raises(ValueError, match='frozen local'):
        p.check_runtime(dict(runtime=dict(python=[0, 0, 0], numpy='different', scipy='different')))

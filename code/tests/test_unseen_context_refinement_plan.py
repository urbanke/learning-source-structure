"""Independent rejection checks for the frozen unseen-context continuation plan."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_unseen_context_refinement as plan


def synthetic_plan():
    groups = [dict(depth=4, time=64, selected_indices=list(range(66))),
              dict(depth=8, time=64, selected_indices=list(range(57)))]
    result = dict(groups=groups, tasks=[], proposal={'fixed': True},
                  runtime={'python': [3, 14, 6]}, worker={'seed': 20261006, 'burn': 16384})
    for group in groups:
        for index in group['selected_indices']:
            for order in (16, 32):
                depth = group['depth']
                identity = dict(inputs={'index': index, 'depth': depth},
                                target={'runtime': result['runtime'], 'order': order},
                                proposal=result['proposal'], master_sampling_seed=20261006, burn=16384)
                spec = dict(task=dict(id=f'L{depth}_i{index}_t64_a{order}', depth=depth,
                                     index=index, time=64, order=order, setting=f'a1_L{depth}'),
                            baseline_draws=4096, target_draws=16384, parent_identity=identity,
                            parent_reference_id=plan.fingerprint(identity),
                            estimated_peak_bytes=plan.slot_bytes(16384, order),
                            runtime_identity_sha256=plan.fingerprint(result['runtime']))
                for prefix, field in [('source', 'inputs'), ('target', 'target'), ('proposal', 'proposal')]:
                    spec[prefix + '_identity_sha256'] = plan.fingerprint(identity[field])
                result['tasks'].append(spec)
    return result


def test_complete_fixed_selection_accepts():
    plan.validate_tasks(synthetic_plan())


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'one_order', 'selected_ids', 'budget', 'memory'])
def test_selection_and_budget_mutations_rejected(mutation):
    frozen = synthetic_plan()
    if mutation == 'missing':
        frozen['tasks'].pop()
    elif mutation == 'duplicate':
        frozen['tasks'][-1] = deepcopy(frozen['tasks'][0])
    elif mutation == 'one_order':
        frozen['tasks'][0]['task']['order'] = 32
    elif mutation == 'selected_ids':
        frozen['groups'][0]['selected_indices'][-1] = 200
    elif mutation == 'budget':
        frozen['tasks'][0]['target_draws'] *= 2
    else:
        frozen['tasks'][0]['estimated_peak_bytes'] = 1024
    with pytest.raises(ValueError):
        plan.validate_tasks(frozen)


def test_runtime_and_identity_are_bound():
    frozen = synthetic_plan()
    frozen['tasks'][0]['parent_identity']['target'] = {'runtime': {'numpy': 'different'}}
    with pytest.raises(ValueError, match='identity mismatch'):
        plan.validate_tasks(frozen)


def parent_fixture(tmp_path):
    files = {'source': dict(source_id='fixed', tokens=list(range(64)),
                            identity={'source_depth': 4, 'source_index': 5}),
             'reference': dict(identity={'inputs': {'model_files': {}}}, reference_id='reference',
                               record={'precision': {'draws_per_chain': 4096}}),
             'checkpoint': b'raw chain', 'moments': b'raw moment'}
    specs = {}
    for key in ('checkpoint', 'moments', 'source', 'reference'):
        value = files[key]
        if key == 'reference':
            value.update(checkpoint_sha256=specs['checkpoint']['sha256'],
                         moments_sha256=specs['moments']['sha256'])
        path = tmp_path / key
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        specs[key] = dict(path=key, sha256=plan.sha(path), size_bytes=path.stat().st_size)
    return dict(source_files={'source': specs['source']},
                parent_files={k: specs[k] for k in ('reference', 'checkpoint', 'moments')},
                parent_identity=files['reference']['identity'], parent_reference_id='reference',
                baseline_draws=4096, source_id='fixed', task={'depth': 4, 'index': 5})


def test_raw_hash_check_is_explicit_and_detects_same_size_corruption(tmp_path):
    spec = parent_fixture(tmp_path)
    plan.verify_task_inputs({}, spec, root=tmp_path, verify_raw=True)
    (tmp_path / 'checkpoint').write_bytes(b'bad chain')
    plan.verify_task_inputs({}, spec, root=tmp_path, verify_raw=False)
    with pytest.raises(ValueError, match='checksum changed'):
        plan.verify_task_inputs({}, spec, root=tmp_path, verify_raw=True)


def test_unseen_event_rechecked_even_if_file_binding_is_consistent(tmp_path):
    spec = parent_fixture(tmp_path)
    path = tmp_path / 'source'
    source = json.loads(path.read_text())
    source['tokens'][63] = 62
    path.write_text(json.dumps(source))
    spec['source_files']['source'].update(sha256=plan.sha(path), size_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='selection changed'):
        plan.verify_task_inputs({}, spec, root=tmp_path)


def test_manifest_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match='within the repository'):
        plan.safe_path(tmp_path, '../parent.json')
    with pytest.raises(ValueError, match='within the repository'):
        plan.safe_path(tmp_path, '/tmp/parent.json')


def test_largest_task_memory_reservation():
    assert plan.slot_bytes(1048576, 32) == int(13.5 * 1024**3)

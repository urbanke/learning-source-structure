"""The reference driver must condition on exactly the frozen model prefix."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transformer'))
from eval_shared_base_models import (atomic_json, atomic_npz, fingerprint,
                                     sha_array, sha_file, sources)
from eval_shared_base_reference import load_inputs


@pytest.fixture
def cohort(tmp_path):
    item = sources(tmp_path, 1, 20260922, create=True)[0]
    r = item['record']; x = np.asarray(r['tokens'])
    atomic_json(tmp_path/'cohort.json', dict(n_sequences=1, master_seed=20260922,
        prefixes=[1024], records=[dict(source_id=r['source_id'],
                                       file_sha256=item['file_sha256'])]))
    oracle = np.asarray(r['oracle']['transition_probs'])[x[1023]]
    for i,name in enumerate(('d64','d64_seed1')):
        path = tmp_path/'models'/name/'seq_000000.json'
        arrays = path.with_suffix('.npz')
        p = np.random.default_rng(i).dirichlet(np.ones(64))
        atomic_npz(arrays, predictive_probs=p[None,:], oracle_probs=oracle[None,:])
        identity = dict(source_id=r['source_id'], model=name)
        atomic_json(path, dict(cache_id=fingerprint(identity), identity=identity,
            source_id=r['source_id'], tokens_sha256=r['tokens_sha256'],
            prefix_tokens_sha256={'1024':sha_array(x[:1024],'<i2')},
            prefixes=[1024], target_indices=[1024], targets=[int(x[1024])],
            predictive_probs=p[None,:].tolist(), arrays_file=arrays.name,
            arrays_sha256=sha_file(arrays)))
    return tmp_path, r


def test_only_observed_prefix_enters_inference_and_oracle_row_is_current_label(cohort):
    folder, source = cohort
    _, x, oracle, models, _ = load_inputs(folder, 0, 1024)
    assert len(x) == 1024
    np.testing.assert_array_equal(x, source['tokens'][:1024])
    np.testing.assert_array_equal(oracle, source['oracle']['transition_probs'][x[-1]])
    assert set(models) == {'d64','d64_seed1'}
    assert not np.array_equal(models['d64']['prediction'], models['d64_seed1']['prediction'])


def test_wrong_target_alignment_is_rejected_even_with_correct_prefix(cohort):
    folder, _ = cohort
    path = folder/'models/d64/seq_000000.json'
    record = json.loads(path.read_text()); record['target_indices'] = [1023]
    atomic_json(path, record)
    with pytest.raises(ValueError, match='target alignment'):
        load_inputs(folder, 0, 1024)


def test_changed_source_or_model_cache_cannot_be_paired(cohort):
    folder, _ = cohort
    path = folder/'models/d64/seq_000000.json'
    record = json.loads(path.read_text()); record['prefix_tokens_sha256']['1024'] = 'wrong'
    atomic_json(path, record)
    with pytest.raises(ValueError, match='tokens differ'):
        load_inputs(folder, 0, 1024)
    with pytest.raises(ValueError, match='outside declared cohort'):
        load_inputs(folder, 1, 1024)


def test_array_corruption_is_not_used_as_a_new_prediction(cohort):
    folder, _ = cohort
    path = folder/'models/d64/seq_000000.npz'
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='arrays checksum'):
        load_inputs(folder, 0, 1024)

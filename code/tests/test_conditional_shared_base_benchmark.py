"""Conditional source generation and strict predictor-input checks."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('conditional_shared_base_benchmark',ROOT/'code/scripts/conditional_shared_base_benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

@pytest.mark.parametrize('depth',[0,1,4,8])
def test_conditional_draw_mirrors_training_generator(depth):
    seed=91821
    tokens,q,t,permutation=b.draw(1.0,depth,seed)
    expected,meta=b.Source(64,1025,alpha=1.0,ls=depth,permute=True).batch(1,np.random.default_rng(seed))
    assert np.array_equal(tokens,expected[0])
    assert np.array_equal(permutation,meta['perm'][0])
    assert np.max(np.abs(t.sum(1)-1))<1e-12
    if depth==0:assert np.array_equal(t,np.broadcast_to(q,t.shape))

def test_alpha_depth_and_index_change_source_identity():
    original=b.identity(1.0,4,0,192)
    for args in [(1.25,4,0,192),(1.0,8,0,192),(1.0,4,1,192)]:
        assert b.identity(*args)['sequence_seed']!=original['sequence_seed']
        assert b.fingerprint(b.identity(*args))!=b.fingerprint(original)

def test_wrong_condition_rejected_by_frozen_source_validation():
    import json
    path=ROOT/'output/conditional_shared_base_benchmark_2026-09-15/a1_L4/data/seq_000000.json'
    if not path.exists():pytest.skip('benchmark output absent')
    record=json.loads(path.read_text())
    wrong=dict(record['identity'],source_depth=8)
    with pytest.raises(ValueError,match='cache inputs changed'):b.validate_source_record(record,wrong)

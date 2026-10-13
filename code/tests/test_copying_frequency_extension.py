from pathlib import Path
import sys

import numpy as np
import pytest
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from copying_frequency_extension import bank_weights,all_types
import copying_frequency_extension as extension
from stationary_frequency_moments import frequency_type


def test_original_quadrature_and_depth_weights():
    banks,weights=bank_weights((16,16,16))
    assert banks==16 and len(weights)==2064
    nodes=weights.reshape(16,129)
    assert np.isclose(nodes[:,0].sum(),1/9)
    for depth in range(8):
        assert np.isclose(nodes[:,1+16*depth:1+16*(depth+1)].sum(),1/9)
    assert np.isclose(weights.sum(),1)


def test_needed_joints_cover_every_next_label():
    tokens=np.array([[4,2,4,7,8,2,9,10,11]])
    kinds=set(all_types([dict(tokens=tokens)]))
    for n in range(1,9):
        prefix=tokens[0,:n].tolist()
        assert frequency_type(prefix) in kinds
        assert all(frequency_type(prefix+[label]) in kinds for label in range(64))


def test_one_attempt_and_timeout_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(extension,'OUT',tmp_path)
    def fail(stage):raise TimeoutError('test deadline')
    monkeypatch.setattr(extension,'run',fail)
    assert extension.bounded_run('refined')==75
    path=tmp_path/'refined/status.json';before=path.read_bytes()
    assert json.loads(before)['status']=='time_limit'
    with pytest.raises(ValueError,match='One attempt'):
        extension.bounded_run('refined')
    assert path.read_bytes()==before

"""Independent record-identity and distance checks for the product pilot.

These tests require neither a transformer checkpoint nor LSA tables.
"""
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from eval_product_components import SCHEMA, checked_load, distances
from eval_conditional_crossing import array_hash, fingerprint
from source import seed_of


def write_record(path, arrays, metadata):
    np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)


def example_record(index=0):
    spec = dict(schema=SCHEMA, master_seed=20260928, n_seq=2, n_ctx=2)
    arrays = dict(tokens=np.array([0, 1, 2], dtype=np.int16),
                  transformer_probs=np.array([[0.25, 0.75], [0.5, 0.5]]))
    metadata = dict(spec=spec, spec_id=fingerprint(spec), index=index,
                    sequence_seed=seed_of(SCHEMA, spec["master_seed"], index),
                    array_hashes={name: array_hash(value) for name, value in arrays.items()})
    return arrays, metadata


def test_valid_record_roundtrip_and_swapped_draw_rejected(tmp_path):
    arrays, meta = example_record()
    first = tmp_path / "seq_000000.npz"
    second = tmp_path / "seq_000001.npz"
    assert checked_load(first, meta["spec_id"]) is None
    write_record(first, arrays, meta)
    loaded = checked_load(first, meta["spec_id"])
    assert set(loaded) == set(arrays)
    for name, value in arrays.items():
        np.testing.assert_array_equal(loaded[name], value)
    # A perfectly intact file cannot masquerade as a different source draw.
    shutil.copyfile(first, second)
    with pytest.raises(ValueError, match="Changed inputs"):
        checked_load(second, meta["spec_id"])


@pytest.mark.parametrize("change", ["array", "embedded_spec", "seed", "extra_array"])
def test_record_tampering_is_detected(tmp_path, change):
    arrays, meta = example_record()
    original_spec_id = meta["spec_id"]
    if change == "array":
        arrays["tokens"][1] = 7
    elif change == "embedded_spec":
        meta["spec"]["master_seed"] += 1
    elif change == "seed":
        meta["sequence_seed"] = seed_of(SCHEMA, meta["spec"]["master_seed"], 1)
    else:
        arrays["unexpected"] = np.array([123])
    path = tmp_path / "seq_000000.npz"
    write_record(path, arrays, meta)
    with pytest.raises(ValueError):
        checked_load(path, original_spec_id)


def test_probability_distances_have_correct_direction_units_and_zero_support():
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
    q = np.array([[0.5, 0.5], [0.25, 0.75], [0.0, 1.0], [0.0, 1.0]])
    kl, tv, l2 = distances(p, q)
    np.testing.assert_allclose(kl[:3], [1.0, 0.5 * math.log2(4 / 3), 0.0], atol=1e-15)
    assert np.isposinf(kl[3])
    np.testing.assert_allclose(tv, [0.5, 0.25, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(l2, [1 / math.sqrt(2), 1 / math.sqrt(8), 0.0, math.sqrt(2)], atol=1e-15)

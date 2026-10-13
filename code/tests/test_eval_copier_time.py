"""Likelihood alignment and immutable scientific-record checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from eval_copier_time import SCHEMA, checked_load, check_bayes, publish_protocol, publish_record
from eval_conditional_crossing import fingerprint
from source import seed_of


def test_uniform_likelihood_alignment_and_corruption():
    tokens = np.array([4, 9, 12, 9, 12])
    result = dict(positions=np.arange(1, 5), probs=np.full((4, 64), 1/64),
                  loss=np.full(5, 6.), log_evidence=-np.arange(1, 6)*np.log(64),
                  depth_posterior=np.full((5, 9), 1/9))
    assert max(check_bayes(result, tokens).values()) < 1e-13
    shifted = {**result, "positions": np.arange(4)}
    with pytest.raises(ValueError, match="misaligned"):
        check_bayes(shifted, tokens)
    wrong = {**result, "loss": np.array([6., 6., 6., 5., 6.])}
    with pytest.raises(ValueError, match="likelihood alignment"):
        check_bayes(wrong, tokens)
    unnormalized = {**result, "probs": result["probs"]*1.001}
    with pytest.raises(ValueError, match="normalization"):
        check_bayes(unnormalized, tokens)
    for name in ("loss", "log_evidence", "depth_posterior"):
        values = result[name].copy()
        values.flat[-1] = np.nan
        with pytest.raises(ValueError, match="Invalid Bayes"):
            check_bayes({**result, name: values}, tokens)


def test_immutable_records_detect_changes_and_reject_changed_protocol(tmp_path):
    spec = dict(schema=SCHEMA, master_seed=713, n_seq=4)
    identifier = fingerprint(spec)
    metadata = dict(spec=spec, spec_id=identifier, index=2,
                    sequence_seed=seed_of(SCHEMA, spec["master_seed"], 2))
    arrays = dict(tokens=np.array([1, 2, 1]), probs=np.full((2, 3), 1/3))
    publish_protocol(tmp_path, spec)
    publish_protocol(tmp_path, spec)
    path = tmp_path / "seq_000002.npz"
    publish_record(path, arrays, metadata)
    before = path.read_bytes()
    publish_record(path, arrays, metadata)
    assert path.read_bytes() == before
    loaded, saved = checked_load(path, identifier)
    assert np.array_equal(loaded["tokens"], arrays["tokens"])
    assert saved["sequence_seed"] == metadata["sequence_seed"]
    with pytest.raises(ValueError, match="different protocol"):
        publish_protocol(tmp_path, {**spec, "master_seed": 714})
    with pytest.raises(ValueError, match="Concurrent"):
        publish_record(path, {**arrays, "tokens": np.array([1, 2, 2])}, metadata)
    assert path.read_bytes() == before
    # The byte-level container still opens, but an altered scientific value
    # must fail its retained array digest on reuse.
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata_json=json.dumps(saved),
                            **{**arrays, "tokens": np.array([1, 2, 2])})
    with pytest.raises(ValueError, match="checksum"):
        checked_load(path, identifier)

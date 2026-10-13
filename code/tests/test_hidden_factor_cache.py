"""Small complete-cache tests; no large hypothesis files are generated."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import hidden_factor_cache as cache
from hidden_factor import encodings


def test_complete_atomic_cache_reuses_source_contract_and_read_only_chunks(tmp_path):
    path = tmp_path/"encodings.npy"
    metadata = cache.prepare_encoding_cache(path, (2, 3), chunk_size=7)
    expected = encodings((2, 3))
    assert metadata["complete"] and metadata["written_rows"] == len(expected)
    assert metadata["sha256_payload"] == hashlib.sha256(expected.tobytes()).hexdigest()
    actual = cache.open_encoding_cache(path, (2, 3), verify_hash=True)
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)
    chunks = list(cache.iter_cached_encodings(path, (2, 3), 11))
    assert all(len(chunk) <= 11 and not chunk.flags.writeable for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    stat = path.stat()
    assert cache.prepare_encoding_cache(path, (2, 3)) == metadata
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert not list(tmp_path.glob("*.partial"))


def test_existing_wrong_dimensions_or_source_contract_rejected(tmp_path, monkeypatch):
    path = tmp_path/"encodings.npy"
    cache.prepare_encoding_cache(path, (2, 3))
    with pytest.raises(ValueError, match="contract mismatch"):
        cache.prepare_encoding_cache(path, (3, 3))
    monkeypatch.setattr(cache, "_source_sha256", lambda: "changed")
    with pytest.raises(ValueError, match="source_sha256"):
        cache.open_encoding_cache(path, (2, 3))


def test_payload_integrity_check_detects_corruption(tmp_path):
    path = tmp_path/"encodings.npy"
    cache.prepare_encoding_cache(path, (2, 3))
    array = np.load(path, mmap_mode="r+")
    array[0, 0] = 1
    array.flush()
    del array
    # Even if a modified file's identity is recorded, payload hash must match.
    receipt = Path(str(path)+".json")
    metadata = json.loads(receipt.read_text())
    metadata["mtime_ns"] = path.stat().st_mtime_ns
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="payload hash"):
        cache.open_encoding_cache(path, (2, 3), verify_hash=True)


def test_size_guard_creates_no_cache(tmp_path):
    path = tmp_path/"encodings.npy"
    with pytest.raises(MemoryError):
        cache.prepare_encoding_cache(path, (2, 8), max_bytes=1024)
    assert not path.exists()


def test_timed_out_build_is_not_published(tmp_path, monkeypatch):
    path = tmp_path/"encodings.npy"
    def delayed(*args, **kwargs):
        time.sleep(.02)
        yield encodings((2, 3))
    monkeypatch.setattr(cache.hidden_factor, "iter_encodings", delayed)
    with pytest.raises(TimeoutError):
        cache.prepare_encoding_cache(path, (2, 3), timeout_seconds=.01)
    assert not path.exists() and not Path(str(path)+".json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_incomplete_enumeration_is_not_published(tmp_path, monkeypatch):
    path = tmp_path/"encodings.npy"
    expected = encodings((2, 3))
    monkeypatch.setattr(cache.hidden_factor, "iter_encodings", lambda *args, **kwargs: iter([expected[:-1]]))
    with pytest.raises(ValueError, match="incomplete enumeration"):
        cache.prepare_encoding_cache(path, (2, 3))
    assert not path.exists() and not list(tmp_path.glob("*.partial"))

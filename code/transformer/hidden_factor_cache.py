"""Reusable, complete canonical-encoding cache for exact hidden-factor Bayes.

The .npy payload is a read-only uint8 array of pair->token permutations.
Generation happens once under a process lock, in bounded-memory chunks.
The receipt records the source-code contract, full hypothesis count, and
SHA256 of every payload byte. It never describes a sampled subset as exact.
No cache is created on import.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import hidden_factor

CONTRACT = "hidden-factor-canonical-encodings-v1"
DEFAULT_MAX_BYTES = 8*1024**3


def _source_sha256():
    return hashlib.sha256(Path(hidden_factor.__file__).read_bytes()).hexdigest()


def _paths(path):
    path = Path(path)
    if path.suffix != ".npy":
        raise ValueError("encoding cache path must end in .npy")
    return path, Path(str(path)+".json"), Path(str(path)+".lock")


def _contract(dimensions):
    dimensions = hidden_factor._dimensions(dimensions)
    return {"contract": CONTRACT, "dimensions": list(dimensions),
            "encoding_count": hidden_factor.encoding_count(dimensions),
            "shape": [hidden_factor.encoding_count(dimensions), int(np.prod(dimensions))],
            "dtype": "uint8", "order": "C", "direction": "pair-index-to-token",
            "prior": "uniform-over-all-canonical-encodings",
            "source_sha256": _source_sha256()}


def _check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("encoding-cache preparation exceeded its time budget")


@contextmanager
def _locked(path, deadline):
    with path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                _check_deadline(deadline)
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _payload_hash(array, chunk_size=65536, deadline=None):
    digest = hashlib.sha256()
    for start in range(0, len(array), chunk_size):
        if deadline is not None:
            _check_deadline(deadline)
        digest.update(memoryview(array[start:start+chunk_size]).cast("B"))
    return digest.hexdigest()


def open_encoding_cache(path, dimensions, *, verify_hash=False, deadline=None):
    """Open complete read-only cache, rejecting incompatible or changed files.

    By default validate the receipt, complete shape, dtype and recorded file
    identity. prepare_encoding_cache verifies the full payload hash once per
    call. Repeated evaluation reads may omit rehashing gigabytes; set
    verify_hash=True to repeat the full integrity check explicitly.
    """
    path, metadata_path, _ = _paths(path)
    expected = _contract(dimensions)
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"encoding cache contract mismatch: {key}")
    if metadata.get("complete") is not True or metadata.get("written_rows") != expected["encoding_count"]:
        raise ValueError("encoding cache is incomplete")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != expected["shape"] or array.dtype != np.uint8 or not array.flags.c_contiguous:
        raise ValueError("encoding cache shape or dtype mismatch")
    stat = path.stat()
    if metadata.get("file_bytes") != stat.st_size or metadata.get("mtime_ns") != stat.st_mtime_ns:
        raise ValueError("encoding cache file changed after generation")
    if stat.st_size != array.offset+array.nbytes:
        raise ValueError("encoding cache has truncated or extra payload")
    if verify_hash and _payload_hash(array, deadline=deadline) != metadata.get("sha256_payload"):
        raise ValueError("encoding cache payload hash mismatch")
    return array


def iter_cached_encodings(path, dimensions, chunk_size=65536):
    """Yield read-only views of the complete cached hypothesis list."""
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    array = open_encoding_cache(path, dimensions)
    for start in range(0, len(array), chunk_size):
        yield array[start:start+chunk_size]


def prepare_encoding_cache(path, dimensions, *, chunk_size=65536,
                           max_bytes=DEFAULT_MAX_BYTES, timeout_seconds=300):
    """Create or fully verify a reusable complete cache and return its receipt.

    Refuse a cache larger than max_bytes (default 8 GiB), insufficient disk,
    or an expired preparation budget. Interrupted builds leave no published
    partial cache. A crash between publishing data and receipt is recovered
    by rebuilding under the same lock. Existing incompatible complete caches
    are rejected, not overwritten.
    """
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(max_bytes, (int, np.integer)) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    deadline = time.monotonic()+timeout_seconds
    path, metadata_path, lock_path = _paths(path)
    metadata = _contract(dimensions)
    data_bytes = int(np.prod(metadata["shape"]))
    # NPY header is small; reserve a conservative page in all size checks.
    required_bytes = data_bytes+4096
    if required_bytes > max_bytes:
        raise MemoryError(f"complete encoding cache needs approximately {required_bytes} bytes, exceeds limit {max_bytes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(lock_path, deadline):
        if path.exists() and metadata_path.exists():
            open_encoding_cache(path, dimensions, verify_hash=True, deadline=deadline)
            with metadata_path.open() as handle:
                return json.load(handle)
        if shutil.disk_usage(path.parent).free < required_bytes+16*1024**2:
            raise OSError("insufficient free disk space for the complete encoding cache")
        temporary_paths = []
        array = None
        started = time.monotonic()
        try:
            with tempfile.NamedTemporaryFile(prefix=path.name+".", suffix=".partial", dir=path.parent, delete=False) as handle:
                data_temp = Path(handle.name)
            temporary_paths.append(data_temp)
            array = np.lib.format.open_memmap(data_temp, mode="w+", dtype=np.uint8,
                                               shape=tuple(metadata["shape"]), fortran_order=False)
            written = 0
            digest = hashlib.sha256()
            for chunk in hidden_factor.iter_encodings(tuple(dimensions), chunk_size):
                _check_deadline(deadline)
                if (chunk.ndim != 2 or chunk.shape[1] != metadata["shape"][1]
                        or chunk.dtype != np.uint8 or not len(chunk)
                        or written+len(chunk) > metadata["encoding_count"]):
                    raise ValueError("encoding generator violated its complete-cache contract")
                array[written:written+len(chunk)] = chunk
                digest.update(memoryview(chunk).cast("B"))
                written += len(chunk)
            if written != metadata["encoding_count"]:
                raise ValueError(f"incomplete enumeration: {written} of {metadata['encoding_count']} encodings")
            array.flush()
            del array
            array = None
            with data_temp.open("rb") as handle:
                os.fsync(handle.fileno())
            _check_deadline(deadline)
            stat = data_temp.stat()
            metadata.update({"complete": True, "written_rows": written,
                             "sha256_payload": digest.hexdigest(), "payload_bytes": data_bytes,
                             "file_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "generation_seconds": time.monotonic()-started})
            with tempfile.NamedTemporaryFile(prefix=metadata_path.name+".", suffix=".partial", dir=path.parent,
                                             mode="w", delete=False) as handle:
                metadata_temp = Path(handle.name)
                temporary_paths.append(metadata_temp)
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(data_temp, path)
            os.replace(metadata_temp, metadata_path)
            return metadata
        finally:
            if array is not None:
                del array
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)

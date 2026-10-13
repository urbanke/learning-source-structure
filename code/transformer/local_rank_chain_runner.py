"""Proposal-bound checkpoints for the separately evaluated local-rank sampler.

The frozen base runner still owns scheduling, RNG state and sample storage.
Its target fingerprint is retained unchanged. An additional checksummed
metadata field identifies this proposal and both new implementation files.
Use this wrapper, never the original runner, to resume these checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from shared_base_chain_runner import ChainRun
from shared_base_local_rank import MOVE_NAMES, PROPOSAL_SPEC, SharedBaseLocalRankMCMC


HERE = Path(__file__).resolve().parent


def proposal_descriptor():
    """Return the explicit proposal specification and current code hashes."""
    return dict(
        schema="shared-base-local-rank-proposal-v1",
        sampler_class="shared_base_local_rank.SharedBaseLocalRankMCMC",
        proposal_spec=deepcopy(PROPOSAL_SPEC),
        code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("shared_base_local_rank.py", "local_rank_chain_runner.py")})


class LocalRankChainRun(ChainRun):
    """Resumable local-rank chains with a separately validated algorithm tag."""

    def __init__(self, sampler, x, **kwargs):
        if type(sampler) is not SharedBaseLocalRankMCMC:
            raise TypeError("LocalRankChainRun requires SharedBaseLocalRankMCMC")
        self.proposal_descriptor = proposal_descriptor()
        super().__init__(sampler, x, **kwargs)

    def report(self):
        result = super().report()
        result["diagnostics"]["move_names"] = list(MOVE_NAMES)
        result["diagnostics"]["proposal_spec"] = deepcopy(PROPOSAL_SPEC)
        return result

    def save(self, path):
        """Atomically add a proposal marker without changing saved array hashes."""
        if self.proposal_descriptor != proposal_descriptor():
            raise ValueError("proposal implementation changed during this run")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The untagged intermediate checkpoint is private and never exposed
        # at the requested checkpoint path.
        with tempfile.TemporaryDirectory(prefix=".local-rank-checkpoint-", dir=path.parent) as directory:
            temporary = Path(directory) / "checkpoint.npz"
            super().save(temporary)
            with np.load(temporary, allow_pickle=False) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            metadata["proposal_descriptor"] = deepcopy(self.proposal_descriptor)
            serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            arrays["metadata"] = np.asarray(serialized)
            arrays["metadata_sha256"] = np.asarray(hashlib.sha256(serialized.encode()).hexdigest())
            with temporary.open("wb") as target:
                np.savez(target, **arrays)
            os.replace(temporary, path)

    @classmethod
    def load(cls, sampler, x, path):
        """Reject original, untagged, or differently implemented proposals."""
        with np.load(path, allow_pickle=False) as stored:
            serialized = str(stored["metadata"].item())
            if hashlib.sha256(serialized.encode()).hexdigest() != str(stored["metadata_sha256"].item()):
                raise ValueError("checkpoint metadata checksum mismatch")
            metadata = json.loads(serialized)
            if metadata.get("proposal_descriptor") != proposal_descriptor():
                raise ValueError("checkpoint proposal descriptor is missing or changed")
        return super().load(sampler, x, path)

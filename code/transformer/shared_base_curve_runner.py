"""Resumable, losslessly compressed checkpoints for full-target proposals.

The numerical posterior and sampling loop remain in the existing modules.
The proposal descriptor prevents resuming with a different Markov kernel.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from shared_base_chain_runner import ChainRun
from shared_base_surrogate import SharedBaseSurrogateMCMC
from shared_base_efficiency import SharedBaseEfficiencyMCMC, SharedBaseEfficiencyRowMCMC
from surrogate_chain_runner import SurrogateChainRun
from stable_chain_statistics import SUMMARY_SPEC


def descriptor(sampler):
    if type(sampler) not in (SharedBaseSurrogateMCMC, SharedBaseEfficiencyMCMC,
                             SharedBaseEfficiencyRowMCMC):
        raise TypeError("unsupported full-target proposal")
    # Proposal specifications are fixed at construction, while these public
    # attributes drive the actual transition. Refuse a stale specification
    # rather than silently fingerprinting a different Markov kernel.
    live_settings = {"surrogate_steps": sampler.surrogate_steps}
    if type(sampler) is SharedBaseEfficiencyMCMC:
        live_settings["auxiliary_power"] = sampler.auxiliary_power
    elif type(sampler) is SharedBaseEfficiencyRowMCMC:
        live_settings["concentration"] = sampler.concentration
    for name, value in live_settings.items():
        if sampler.proposal_spec.get(name) != value:
            raise ValueError(f"live proposal setting differs from construction descriptor: {name}")
    here = Path(__file__).resolve().parent
    names = ["shared_base_curve_runner.py", "shared_base_surrogate.py",
             "shared_base_efficiency.py", "surrogate_chain_runner.py",
             "stable_chain_statistics.py"]
    return dict(schema="shared-base-curve-proposal-v1",
                sampler_class=type(sampler).__module__ + "." + type(sampler).__name__,
                proposal_spec=deepcopy(sampler.proposal_spec),
                summary_spec=deepcopy(SUMMARY_SPEC),
                storage="lossless ZIP deflate; original float64 arrays and checksums",
                code_sha256={name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                             for name in names})


class CurveChainRun(ChainRun):
    def __init__(self, sampler, x, **kwargs):
        self.proposal_descriptor = descriptor(sampler)
        super().__init__(sampler, x, **kwargs)

    report = SurrogateChainRun.report

    def save(self, path):
        if self.proposal_descriptor != descriptor(self.sampler):
            raise ValueError("proposal implementation changed during run")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".curve-checkpoint-", dir=path.parent) as temporary:
            raw = Path(temporary)/"raw.npz"
            super().save(raw)
            with np.load(raw, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata"].item()))
            metadata["proposal_descriptor"] = deepcopy(self.proposal_descriptor)
            serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            extra = Path(temporary)/"metadata.npz"
            np.savez(extra, metadata=np.asarray(serialized),
                     metadata_sha256=np.asarray(hashlib.sha256(serialized.encode()).hexdigest()))
            compressed = Path(temporary)/"compressed.npz"
            with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as destination:
                for source in (raw, extra):
                    with zipfile.ZipFile(source) as archive:
                        for name in archive.namelist():
                            if source == raw and name in ("metadata.npy", "metadata_sha256.npy"):
                                continue
                            with archive.open(name) as reader, destination.open(name, "w", force_zip64=True) as writer:
                                shutil.copyfileobj(reader, writer, length=1024*1024)
            os.replace(compressed, path)

    @classmethod
    def load(cls, sampler, x, path):
        with np.load(path, allow_pickle=False) as stored:
            serialized = str(stored["metadata"].item())
            if hashlib.sha256(serialized.encode()).hexdigest() != str(stored["metadata_sha256"].item()):
                raise ValueError("checkpoint metadata checksum failed")
            metadata = json.loads(serialized)
            if metadata.get("proposal_descriptor") != descriptor(sampler):
                raise ValueError("checkpoint proposal or implementation differs")
        return super().load(sampler, x, path)

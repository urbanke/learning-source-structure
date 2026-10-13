"""Proposal-bound checkpoints for the separately evaluated surrogate-block sampler.

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
from shared_base_surrogate import MOVE_NAMES, PROPOSAL_SPEC, SharedBaseSurrogateMCMC
from stable_chain_statistics import stable_chain_summary, SUMMARY_SPEC


HERE = Path(__file__).resolve().parent


def proposal_descriptor(sampler):
    """Return the explicit proposal specification and current code hashes."""
    return dict(
        schema="shared-base-surrogate-block-proposal-v1",
        sampler_class="shared_base_surrogate.SharedBaseSurrogateMCMC",
        proposal_spec=dict(deepcopy(PROPOSAL_SPEC), surrogate_steps=sampler.surrogate_steps),
        summary_spec=deepcopy(SUMMARY_SPEC),
        code_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("shared_base_surrogate.py", "surrogate_chain_runner.py", "stable_chain_statistics.py")})


class SurrogateChainRun(ChainRun):
    """Resumable surrogate-block chains with a separately validated algorithm tag."""

    def __init__(self, sampler, x, **kwargs):
        if type(sampler) is not SharedBaseSurrogateMCMC:
            raise TypeError("SurrogateChainRun requires SharedBaseSurrogateMCMC")
        self.proposal_descriptor = proposal_descriptor(sampler)
        super().__init__(sampler, x, **kwargs)

    def report(self):
        """Return the same numerical report and diagnostics as SharedBaseMCMC.sample."""
        samples = self.samples
        draws = samples["predictive_probs"].shape[1]
        if draws < 16:
            raise ValueError("at least 16 draws per chain are needed for diagnostics")
        alphas = np.eye(len(self.sampler.alpha_weights))[samples["alpha_indices"]]
        pred, depth, alpha, scalars = map(stable_chain_summary,
            (samples["predictive_probs"], samples["posterior_depth"], alphas, samples["alpha_and_loglik"]))

        def finite_list(a):
            return [float(v) if np.isfinite(v) else None for v in a]

        diagnostic = dict(
            chains=self.chains, burn=self.burn, draws=draws, thin=self.thin, seed=self.seed,
            move_names=list(MOVE_NAMES),
            proposal_spec=self.proposal_descriptor["proposal_spec"],
            summary_spec=deepcopy(SUMMARY_SPEC),
            move_attempts=self._attempts.tolist(), move_accepts=self._accepts.tolist(),
            initializations=self.initializations,
            final_states=[dict(alpha_index=s.alpha_index, ranks=s.ranks.tolist()) for s in self._states],
            max_conditional_probability_sum_error=float(self._max_errors.max()),
            probability_sum_error=float(pred["mean"].sum() - 1),
            rhat_probs=finite_list(pred["rhat"]), ess_probs=pred["ess"].tolist(),
            rhat_depth=finite_list(depth["rhat"]), rhat_alpha_and_loglik=finite_list(scalars["rhat"]),
            chain_mean_probs=pred["chain_means"].tolist(), batch_length=pred["batch_length"],
            note="Classic split Rhat and conservative batch/between-chain MCSE are diagnostics, not a convergence certificate; exponent and row quadrature error are separate.")
        return dict(predictive_probs=pred["mean"], mcse_probs=pred["mcse"],
                    posterior_depth=depth["mean"], mcse_depth=depth["mcse"],
                    posterior_alpha=alpha["mean"], mcse_alpha=alpha["mcse"],
                    diagnostics=diagnostic, elapsed_seconds=self.elapsed_seconds)

    def save(self, path):
        """Atomically add a proposal marker without changing saved array hashes."""
        if self.proposal_descriptor != proposal_descriptor(self.sampler):
            raise ValueError("proposal implementation changed during this run")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The untagged intermediate checkpoint is private and never exposed
        # at the requested checkpoint path.
        with tempfile.TemporaryDirectory(prefix=".surrogate-block-checkpoint-", dir=path.parent) as directory:
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
            if metadata.get("proposal_descriptor") != proposal_descriptor(sampler):
                raise ValueError("checkpoint proposal descriptor is missing or changed")
        return super().load(sampler, x, path)

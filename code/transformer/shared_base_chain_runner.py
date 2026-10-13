"""Resumable independent chains using the established shared-base sampler.

Only chain storage and scheduling live here. Transitions, conditional
predictions and Monte Carlo summaries come from shared_base_mcmc.py.
Checkpoints preserve every chain's RNG and current state, including chains
whose retained draw counts differ while an extension is in progress.
Immutable moment/evidence caches are reconstructed rather than serialized.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import scipy

from shared_base_mcmc import PrefixPosterior, chain_summary
from shared_depth import _tokens


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _hash_array(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _fingerprint(sampler, x):
    directory = Path(__file__).resolve().parent
    local = ["shared_base_chain_runner.py", "shared_base_mcmc.py", "ranked_lsa.py",
             "weighted_lsa.py", "shared_depth.py", "lsa_seq.py", "phi_source.py", "source.py"]
    files = {name: directory / name for name in local}
    kernel = sampler.kernel
    if np.any(sampler.depths >= 2):
        if not kernel._store or not kernel._src:
            raise ValueError("checkpoint provenance needs the configured production store and source")
        store = Path(kernel._store).expanduser().resolve()
        source = Path(kernel._src).expanduser().resolve() / "product_model_with_memory"
        files.update({f"store/{name}": store / name for name in ("manifest.json", "anchors.json")})
        files.update({f"pmm/{name}": source / name for name in
                      ("universal_tables.py", "production_coding.py", "mellin.py", "kernel.py")})
    hashes = {}
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"checkpoint provenance file is missing: {path}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        d=sampler.d, token_hash=_hash_array(np.asarray(x, dtype="<i8")), prefix_length=len(x),
        alpha_nodes=np.asarray(kernel.alpha_nodes).tolist(), alpha_weights=sampler.alpha_weights.tolist(),
        depths=sampler.depths.tolist(), depth_prior=sampler.depth_prior.tolist(),
        rank_proposal_weights=sampler.rank_proposal_weights.tolist(), prob_tol=sampler.prob_tol,
        kernel_grid=dict(lower=kernel.u_lo, upper=kernel.u_hi, step=kernel.step,
                         u_hash=_hash_array(kernel.u)),
        base_laws_hash=_hash_array(kernel.base_laws), numerical_code_hashes=hashes,
        runtime=dict(python=list(sys.version_info[:3]), numpy=np.__version__, scipy=scipy.__version__))


class ChainRun:
    """Extend an existing run without repeating burn-in or discarding draws.

    ``extend(total_draws, block_draws=2048, progress=callback)`` calls the
    callback after each completed burn and retained block. A callback may
    call ``save``; those checkpoints may contain different draw counts per
    chain, and resume safely. The callback receives a dictionary, never the
    numerical samples. Without a callback, retained-block progress prints.

    ``report`` and ``samples`` require equal retained counts across chains.
    ``samples`` keys are predictive_probs, posterior_depth, alpha_indices,
    and alpha_and_loglik. Arrays are returned as new stacked arrays.
    """

    def __init__(self, sampler, x, *, chains=4, burn=8192, thin=4, seed=20260914):
        start = time.monotonic()
        self.sampler = sampler
        self.x = _tokens(x, sampler.d).copy()
        if not len(self.x):
            raise ValueError("a resumable run needs a nonempty observed prefix")
        self.chains = _integer(chains, "chains", 2)
        self.burn = _integer(burn, "burn", 0)
        self.thin = _integer(thin, "thin", 1)
        self.seed = _integer(seed, "seed", 0)
        self.fingerprint = _fingerprint(sampler, self.x)
        self.target = PrefixPosterior(sampler, self.x)
        self._rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(chains)]
        self._states = []
        self.initializations = []
        self._burn_done = np.zeros(chains, dtype=np.int64)
        self._attempts = np.zeros((chains, 4), dtype=np.int64)
        self._accepts = np.zeros_like(self._attempts)
        self._predictions = [np.empty((0, sampler.d)) for _ in range(chains)]
        self._depths = [np.empty((0, len(sampler.depths))) for _ in range(chains)]
        self._alpha_indices = [np.empty(0, dtype=np.int64) for _ in range(chains)]
        self._scalars = [np.empty((0, 2)) for _ in range(chains)]
        self._max_errors = np.zeros(chains)
        frequency_order = np.argsort(-np.bincount(self.x, minlength=sampler.d), kind="stable")
        for chain, rng in enumerate(self._rngs):
            ranks = rng.permutation(sampler.d)
            if chain % 4 == 1:
                ranks[frequency_order] = np.arange(sampler.d)
            elif chain % 4 == 2:
                ranks[frequency_order] = np.arange(sampler.d - 1, -1, -1)
            index = (0 if chain % 4 == 1 else len(sampler.alpha_weights) - 1
                     if chain % 4 == 2 else int(rng.choice(len(sampler.alpha_weights), p=sampler.alpha_weights)))
            state = self.target.evaluate(index, ranks)
            self._states.append(state)
            self.initializations.append(dict(alpha_index=index, ranks=ranks.tolist(),
                                             log_likelihood=state.log_likelihood))
        self._elapsed = time.monotonic() - start
        self._active_started = None

    @property
    def draw_counts(self):
        return np.array([len(a) for a in self._alpha_indices], dtype=np.int64)

    @property
    def elapsed_seconds(self):
        return self._elapsed + (0.0 if self._active_started is None
                                else time.monotonic() - self._active_started)

    def _notify(self, progress, phase, chain, requested):
        event = dict(phase=phase, chain=chain, chains=self.chains,
                     draws=int(self.draw_counts[chain]), requested_draws=requested,
                     draw_counts=self.draw_counts.tolist(), elapsed_seconds=self.elapsed_seconds)
        if progress is not None:
            progress(event)
        elif phase == "draws":
            print(f"chain {chain + 1}/{self.chains}: {event['draws']}/{requested} draws", flush=True)

    def extend(self, total_draws, *, block_draws=2048, progress=None):
        """Reach total_draws in every chain, preserving all previous draws."""
        total_draws = _integer(total_draws, "total_draws", 16)
        block_draws = _integer(block_draws, "block_draws", 1)
        if np.any(self.draw_counts > total_draws):
            raise ValueError("total_draws cannot discard already retained samples")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable or None")
        if self._active_started is not None:
            raise RuntimeError("a run cannot be extended recursively")
        self._active_started = time.monotonic()
        try:
            for chain, rng in enumerate(self._rngs):
                state = self._states[chain]
                if self._burn_done[chain] < self.burn:
                    before = (state, deepcopy(rng.bit_generator.state), int(self._burn_done[chain]),
                              self._attempts[chain].copy(), self._accepts[chain].copy())
                    try:
                        for _ in range(int(self._burn_done[chain]), self.burn):
                            state = self.sampler._step(self.target, state, rng,
                                                       self._attempts[chain], self._accepts[chain])
                            self._burn_done[chain] += 1
                        self._states[chain] = state
                    except BaseException:
                        state, rng.bit_generator.state, self._burn_done[chain], attempts, accepts = before
                        self._states[chain] = state
                        self._attempts[chain], self._accepts[chain] = attempts, accepts
                        raise
                    self._attempts[chain] = self._accepts[chain] = 0
                    self._notify(progress, "burn", chain, total_draws)
                while len(self._alpha_indices[chain]) < total_draws:
                    length = min(block_draws, total_draws - len(self._alpha_indices[chain]))
                    predictions = np.empty((length, self.sampler.d))
                    depths = np.empty((length, len(self.sampler.depths)))
                    indices = np.empty(length, dtype=np.int64)
                    scalars = np.empty((length, 2))
                    before = (state, deepcopy(rng.bit_generator.state), self._attempts[chain].copy(),
                              self._accepts[chain].copy(), float(self._max_errors[chain]))
                    try:
                        for draw in range(length):
                            for _ in range(self.thin):
                                state = self.sampler._step(self.target, state, rng,
                                                           self._attempts[chain], self._accepts[chain])
                            predictions[draw], depths[draw], error = self.target.predict(state)
                            self._max_errors[chain] = max(self._max_errors[chain], error)
                            indices[draw] = state.alpha_index
                            scalars[draw] = self.sampler.kernel.alpha_nodes[state.alpha_index], state.log_likelihood
                    except BaseException:
                        state, rng.bit_generator.state, attempts, accepts, self._max_errors[chain] = before
                        self._states[chain] = state
                        self._attempts[chain], self._accepts[chain] = attempts, accepts
                        raise
                    self._states[chain] = state
                    self._predictions[chain] = np.concatenate([self._predictions[chain], predictions])
                    self._depths[chain] = np.concatenate([self._depths[chain], depths])
                    self._alpha_indices[chain] = np.concatenate([self._alpha_indices[chain], indices])
                    self._scalars[chain] = np.concatenate([self._scalars[chain], scalars])
                    self._notify(progress, "draws", chain, total_draws)
        finally:
            self._elapsed += time.monotonic() - self._active_started
            self._active_started = None
        return self.report()

    @property
    def samples(self):
        if len(set(self.draw_counts)) != 1:
            raise ValueError("sample arrays need equal retained counts across chains")
        return dict(predictive_probs=np.stack(self._predictions), posterior_depth=np.stack(self._depths),
                    alpha_indices=np.stack(self._alpha_indices), alpha_and_loglik=np.stack(self._scalars))

    def report(self):
        """Return the same numerical report and diagnostics as SharedBaseMCMC.sample."""
        samples = self.samples
        draws = samples["predictive_probs"].shape[1]
        if draws < 16:
            raise ValueError("at least 16 draws per chain are needed for diagnostics")
        alphas = np.eye(len(self.sampler.alpha_weights))[samples["alpha_indices"]]
        pred, depth, alpha, scalars = map(chain_summary,
            (samples["predictive_probs"], samples["posterior_depth"], alphas, samples["alpha_and_loglik"]))

        def finite_list(a):
            return [float(v) if np.isfinite(v) else None for v in a]

        diagnostic = dict(
            chains=self.chains, burn=self.burn, draws=draws, thin=self.thin, seed=self.seed,
            move_names=["seen_label_swap", "head_weighted_rank_swap", "alpha_prior_proposal", "alpha_adjacent_proposal"],
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
        """Atomically save metadata, chain states, RNG states and retained samples."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(burn_done=self._burn_done, attempts=self._attempts, accepts=self._accepts,
                      max_errors=self._max_errors)
        for chain in range(self.chains):
            arrays.update({f"predictions_{chain}": self._predictions[chain],
                           f"depths_{chain}": self._depths[chain],
                           f"alpha_indices_{chain}": self._alpha_indices[chain],
                           f"scalars_{chain}": self._scalars[chain]})
        metadata = dict(
            format="shared_base_chain_run_v1", fingerprint=self.fingerprint,
            config=dict(chains=self.chains, burn=self.burn, thin=self.thin, seed=self.seed),
            elapsed_seconds=self.elapsed_seconds, initializations=self.initializations,
            rng_states=[rng.bit_generator.state for rng in self._rngs],
            states=[dict(alpha_index=s.alpha_index, ranks=s.ranks.tolist(),
                         log_likelihood=s.log_likelihood, log_target=s.log_target) for s in self._states],
            array_hashes={name: _hash_array(value) for name, value in arrays.items()})
        serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        arrays["metadata"] = np.asarray(serialized)
        arrays["metadata_sha256"] = np.asarray(hashlib.sha256(serialized.encode()).hexdigest())
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as target:
            np.savez(target, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load(cls, sampler, x, path):
        """Resume only when tokens, model, numerical code and stored states agree."""
        x = _tokens(x, sampler.d)
        with np.load(path, allow_pickle=False) as stored:
            serialized = str(stored["metadata"].item())
            if hashlib.sha256(serialized.encode()).hexdigest() != str(stored["metadata_sha256"].item()):
                raise ValueError("checkpoint metadata checksum mismatch")
            metadata = json.loads(serialized)
            if metadata.get("format") != "shared_base_chain_run_v1":
                raise ValueError("unsupported chain checkpoint format")
            if metadata.get("fingerprint") != _fingerprint(sampler, x):
                raise ValueError("checkpoint fingerprint differs: tokens, model, grid, code, store or runtime changed")
            arrays = {}
            for name, expected in metadata["array_hashes"].items():
                arrays[name] = stored[name].copy()
                if _hash_array(arrays[name]) != expected:
                    raise ValueError(f"checkpoint array checksum mismatch: {name}")
        run = cls(sampler, x, **metadata["config"])
        if len(metadata["states"]) != run.chains or len(metadata["rng_states"]) != run.chains:
            raise ValueError("checkpoint chain-state count mismatch")
        run._states = []
        for chain, saved in enumerate(metadata["states"]):
            state = run.target.evaluate(saved["alpha_index"], saved["ranks"])
            if state.log_likelihood != saved["log_likelihood"] or state.log_target != saved["log_target"]:
                raise ValueError("checkpoint state likelihood does not reproduce exactly")
            run._states.append(state)
            run._rngs[chain].bit_generator.state = metadata["rng_states"][chain]
            run._predictions[chain] = arrays[f"predictions_{chain}"]
            run._depths[chain] = arrays[f"depths_{chain}"]
            run._alpha_indices[chain] = arrays[f"alpha_indices_{chain}"]
            run._scalars[chain] = arrays[f"scalars_{chain}"]
            n = len(run._alpha_indices[chain])
            if (run._predictions[chain].shape != (n, sampler.d)
                    or run._depths[chain].shape != (n, len(sampler.depths))
                    or run._scalars[chain].shape != (n, 2)
                    or run._alpha_indices[chain].dtype.kind not in "iu"
                    or np.any(run._alpha_indices[chain] < 0)
                    or np.any(run._alpha_indices[chain] >= len(sampler.alpha_weights))):
                raise ValueError("checkpoint sample shapes or exponent indices are invalid")
            if n and (run._alpha_indices[chain][-1] != state.alpha_index
                      or run._scalars[chain][-1, 1] != state.log_likelihood):
                raise ValueError("checkpoint state disagrees with its last retained draw")
        run._burn_done, run._attempts = arrays["burn_done"], arrays["attempts"]
        run._accepts, run._max_errors = arrays["accepts"], arrays["max_errors"]
        if (run._burn_done.shape != (run.chains,)
                or np.any(run._burn_done < 0) or np.any(run._burn_done > run.burn)
                or run._attempts.shape != (run.chains, 4) or run._accepts.shape != (run.chains, 4)
                or run._max_errors.shape != (run.chains,)
                or not np.isfinite(run._max_errors).all() or np.any(run._max_errors < 0)
                or np.any(run._accepts < 0) or np.any(run._accepts > run._attempts)):
            raise ValueError("checkpoint counters are invalid")
        for chain, n in enumerate(run.draw_counts):
            if n and (run._burn_done[chain] != run.burn
                      or run._attempts[chain].sum() != n * run.thin):
                raise ValueError("checkpoint retained draws disagree with transition counters")
        run.initializations = metadata["initializations"]
        run._elapsed = float(metadata["elapsed_seconds"])
        return run

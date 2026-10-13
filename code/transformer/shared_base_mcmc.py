"""Fixed-prefix posterior sampling for the original shared-base source.

Exponent quadrature replaces only its continuous integral. Rank assignments
are sampled, weighted row laws are integrated numerically, and the one shared
depth is marginalized exactly. Independent chains target the observed prefix
only. This produces selected-prefix predictions, not a marginal-likelihood
or sequential codelength estimator. Diagnostics are not convergence proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.special import logsumexp

from shared_depth import _tokens


def _positive_weights(values, size, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must have {size} finite positive entries")
    return np.exp(np.log(values) - logsumexp(np.log(values)))


def chain_summary(samples):
    """Means, conservative batch/independent-chain SE, and classic split Rhat.

    samples has shape (chains, draws, coordinates). The reported SE is the
    larger of batch-means and between-chain estimates. Split Rhat is not
    rank-normalized. Neither statistic bounds undiscovered-mode bias.
    """
    values = np.asarray(samples, dtype=float)
    chains, draws, _ = values.shape
    mean = values.mean(axis=(0, 1))
    chain_means = values.mean(axis=1)
    between_se = chain_means.std(axis=0, ddof=1) / np.sqrt(chains)
    batch_length = max(2, int(np.sqrt(draws)))
    n_batches = draws // batch_length
    batches = values[:, :n_batches * batch_length].reshape(chains, n_batches, batch_length, -1).mean(axis=2)
    batch_se = batches.reshape(chains * n_batches, -1).std(axis=0, ddof=1) / np.sqrt(chains * n_batches)
    mcse = np.maximum(batch_se, between_se)
    half = draws // 2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * split.mean(axis=1).var(axis=0, ddof=1)
    variance = (half - 1) / half * within + between / half
    rhat = np.ones_like(within)
    np.divide(variance, within, out=rhat, where=within > 1e-30)
    rhat = np.sqrt(np.maximum(rhat, 0))
    rhat[(within <= 1e-30) & (between > 1e-30)] = np.inf
    # A batch-means effective sample size, bounded by the actual draw count.
    marginal_variance = values.reshape(chains * draws, -1).var(axis=0, ddof=1)
    ess = np.full_like(mean, chains * draws, dtype=float)
    np.divide(marginal_variance, mcse**2, out=ess, where=mcse > 1e-30)
    ess = np.minimum(ess, chains * draws)
    return dict(mean=mean, mcse=mcse, rhat=rhat, ess=ess,
                chain_means=chain_means, batch_length=batch_length)


@dataclass
class PosteriorState:
    alpha_index: int
    ranks: np.ndarray  # zero-based rank assigned to each observed label
    rows: np.ndarray   # active-row log evidence, in depth order
    log_likelihood: float
    log_target: float


class PrefixPosterior:
    """Sufficient statistics and collapsed target for one observed prefix."""

    def __init__(self, sampler, x):
        self.sampler = sampler
        self.d = sampler.d
        self.x = _tokens(x, self.d)
        if not len(self.x):
            raise ValueError("PrefixPosterior needs at least one observed token")
        counts = np.zeros((self.d, self.d), dtype=np.int64)
        np.add.at(counts, (self.x[:-1], self.x[1:]), 1)
        self.contexts = np.flatnonzero(counts.sum(axis=1))
        self.counts = counts[self.contexts]
        self.current_row = counts[self.x[-1]]
        self.seen = np.unique(self.x)
        self.unseen = np.setdiff1d(np.arange(self.d), self.seen)

    def _finish(self, alpha_index, ranks, rows):
        depth_terms = self.sampler.log_depth_prior + rows.sum(axis=0)
        first = self.sampler.kernel.log_base_laws[alpha_index, ranks[self.x[0]]]
        likelihood = float(first + logsumexp(depth_terms))
        return PosteriorState(alpha_index, ranks.copy(), rows, likelihood,
                              likelihood + self.sampler.log_alpha_weights[alpha_index])

    def evaluate(self, alpha_index, ranks):
        ranks = np.asarray(ranks)
        if (ranks.shape != (self.d,) or ranks.dtype.kind not in "iu"
                or not np.array_equal(np.sort(ranks), np.arange(self.d))):
            raise ValueError("ranks must be a permutation of 0,...,d-1")
        if (isinstance(alpha_index, (bool, np.bool_))
                or not isinstance(alpha_index, (int, np.integer))
                or not 0 <= alpha_index < len(self.sampler.alpha_weights)):
            raise ValueError("invalid alpha index")
        rank_order = np.argsort(ranks)
        rows = np.array([self.sampler.kernel.log_evidence(row[rank_order], alpha_index)
                         for row in self.counts]).reshape(len(self.counts), -1) if len(self.counts) else np.empty((0, len(self.sampler.depths)))
        return self._finish(alpha_index, ranks, rows)

    def swap(self, state, i, j):
        if i == j:
            return state
        ranks = state.ranks.copy()
        ranks[i], ranks[j] = ranks[j], ranks[i]
        rows = state.rows.copy()
        affected = np.flatnonzero(self.counts[:, i] != self.counts[:, j])
        rank_order = np.argsort(ranks)
        for row in affected:
            rows[row] = self.sampler.kernel.log_evidence(self.counts[row, rank_order], state.alpha_index)
        return self._finish(state.alpha_index, ranks, rows)

    def predict(self, state):
        depth_terms = self.sampler.log_depth_prior + state.rows.sum(axis=0)
        log_depth = depth_terms - logsumexp(depth_terms)
        rank_counts = self.current_row[np.argsort(state.ranks)]
        row_log_probs = self.sampler.kernel.log_predictive(rank_counts, state.alpha_index)
        error = float(np.max(np.abs(np.exp(row_log_probs).sum(axis=1) - 1)))
        if error > self.sampler.prob_tol or not np.isfinite(row_log_probs).all():
            raise FloatingPointError(f"conditional row normalization error {error:g}")
        rank_probs = np.exp(logsumexp(log_depth[:, None] + row_log_probs, axis=0))
        probs = rank_probs[state.ranks].copy()
        # Conditional on all seen assignments, unseen rank assignments are
        # uniformly distributed. Average them analytically, without sampling.
        if len(self.unseen):
            probs[self.unseen] = probs[self.unseen].mean()
        return probs, np.exp(log_depth), error


class SharedBaseMCMC:
    """Metropolis sampling over exponent nodes and full rank permutations."""

    def __init__(self, d, row_kernel, alpha_weights, depth_prior=None, prob_tol=1e-7):
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 2:
            raise ValueError("d must be an integer at least two")
        if row_kernel.d != d:
            raise ValueError("kernel alphabet does not match d")
        self.d, self.kernel = int(d), row_kernel
        self.depths = np.asarray(row_kernel.depths)
        self.alpha_weights = _positive_weights(alpha_weights, len(row_kernel.alpha_nodes), "alpha_weights")
        self.log_alpha_weights = np.log(self.alpha_weights)
        depth_weights = np.ones(len(self.depths)) if depth_prior is None else depth_prior
        self.depth_prior = _positive_weights(depth_weights, len(self.depths), "depth_prior")
        self.log_depth_prior = np.log(self.depth_prior)
        rank_weights = 1 / np.arange(1, self.d + 1, dtype=float)
        self.rank_proposal_weights = rank_weights / rank_weights.sum()
        if not np.isfinite(prob_tol) or prob_tol <= 0:
            raise ValueError("prob_tol must be finite and positive")
        self.prob_tol = float(prob_tol)

    def _step(self, target, state, rng, attempts, accepts):
        move = rng.choice(4, p=[0.4, 0.4, 0.1, 0.1])
        attempts[move] += 1
        correction = 0.0
        if move == 0:
            # P(i,j) is symmetric even when exactly one of the labels is seen.
            i, j = int(rng.choice(target.seen)), int(rng.integers(self.d))
            proposed = target.swap(state, i, j)
        elif move == 1:
            # Visit the statistically influential small ranks more often.
            # An unordered pair {r,s} has probability (v_r+v_s)/d in
            # either direction: swapping their labels leaves it unchanged.
            r = int(rng.choice(self.d, p=self.rank_proposal_weights))
            s = int(rng.integers(self.d))
            labels = np.argsort(state.ranks)
            proposed = target.swap(state, int(labels[r]), int(labels[s]))
        elif move == 2:
            index = int(rng.choice(len(self.alpha_weights), p=self.alpha_weights))
            proposed = target.evaluate(index, state.ranks) if index != state.alpha_index else state
            correction = self.log_alpha_weights[state.alpha_index] - self.log_alpha_weights[index]
        else:
            index = state.alpha_index + int(rng.choice([-1, 1]))
            proposed = target.evaluate(index, state.ranks) if 0 <= index < len(self.alpha_weights) else state
        if np.log(rng.random()) < min(0.0, proposed.log_target - state.log_target + correction):
            accepts[move] += 1
            return proposed
        return state

    def sample(self, x, *, chains=4, burn=512, draws=1024, thin=2,
               seed=20260914, retain_samples=False):
        for value, name, minimum in ((chains, "chains", 2), (burn, "burn", 0),
                                     (draws, "draws", 16), (thin, "thin", 1)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        x = _tokens(x, self.d)
        if not len(x):
            return dict(predictive_probs=np.full(self.d, 1 / self.d),
                        mcse_probs=np.zeros(self.d), posterior_depth=self.depth_prior.copy(),
                        mcse_depth=np.zeros(len(self.depths)),
                        posterior_alpha=self.alpha_weights.copy(),
                        mcse_alpha=np.zeros(len(self.alpha_weights)), elapsed_seconds=0.0,
                        diagnostics=dict(exact_empty_prefix=True))
        target = PrefixPosterior(self, x)
        start = time.monotonic()
        predictions = np.empty((chains, draws, self.d))
        depths = np.empty((chains, draws, len(self.depths)))
        alphas = np.zeros((chains, draws, len(self.alpha_weights)))
        scalar = np.empty((chains, draws, 2))
        attempts = np.zeros((chains, 4), dtype=np.int64)
        accepts = np.zeros_like(attempts)
        initializations = []
        final_states = []
        max_error = 0.0
        seeds = np.random.SeedSequence(seed).spawn(chains)
        frequency_order = np.argsort(-np.bincount(x, minlength=self.d), kind="stable")
        for chain in range(chains):
            rng = np.random.default_rng(seeds[chain])
            ranks = rng.permutation(self.d)
            if chain % 4 == 1:
                ranks[frequency_order] = np.arange(self.d)
            elif chain % 4 == 2:
                ranks[frequency_order] = np.arange(self.d - 1, -1, -1)
            index = 0 if chain % 4 == 1 else len(self.alpha_weights) - 1 if chain % 4 == 2 else int(rng.choice(len(self.alpha_weights), p=self.alpha_weights))
            state = target.evaluate(index, ranks)
            initializations.append(dict(alpha_index=index, ranks=ranks.tolist(), log_likelihood=state.log_likelihood))
            for _ in range(burn):
                state = self._step(target, state, rng, attempts[chain], accepts[chain])
            # Acceptance summaries refer to retained-sampling transitions only.
            attempts[chain] = accepts[chain] = 0
            for draw in range(draws):
                for _ in range(thin):
                    state = self._step(target, state, rng, attempts[chain], accepts[chain])
                predictions[chain, draw], depths[chain, draw], error = target.predict(state)
                max_error = max(max_error, error)
                alphas[chain, draw, state.alpha_index] = 1
                scalar[chain, draw] = self.kernel.alpha_nodes[state.alpha_index], state.log_likelihood
            final_states.append(dict(alpha_index=state.alpha_index, ranks=state.ranks.tolist()))
        pred, depth, alpha, scalars = map(chain_summary, (predictions, depths, alphas, scalar))
        def finite_list(a):
            return [float(v) if np.isfinite(v) else None for v in a]
        diagnostic = dict(
            chains=chains, burn=burn, draws=draws, thin=thin, seed=seed,
            move_names=["seen_label_swap", "head_weighted_rank_swap", "alpha_prior_proposal", "alpha_adjacent_proposal"],
            move_attempts=attempts.tolist(), move_accepts=accepts.tolist(),
            initializations=initializations, final_states=final_states,
            max_conditional_probability_sum_error=max_error,
            probability_sum_error=float(pred["mean"].sum() - 1),
            rhat_probs=finite_list(pred["rhat"]), ess_probs=pred["ess"].tolist(),
            rhat_depth=finite_list(depth["rhat"]), rhat_alpha_and_loglik=finite_list(scalars["rhat"]),
            chain_mean_probs=pred["chain_means"].tolist(),
            batch_length=pred["batch_length"],
            note="Classic split Rhat and conservative batch/between-chain MCSE are diagnostics, not a convergence certificate; exponent and row quadrature error are separate.")
        out = dict(predictive_probs=pred["mean"], mcse_probs=pred["mcse"],
                   posterior_depth=depth["mean"], mcse_depth=depth["mcse"],
                   posterior_alpha=alpha["mean"], mcse_alpha=alpha["mcse"],
                   diagnostics=diagnostic, elapsed_seconds=time.monotonic() - start)
        if retain_samples:
            out["samples"] = dict(predictive_probs=predictions, posterior_depth=depths,
                                  alpha_index=alphas.argmax(axis=2), alpha_and_loglik=scalar)
        return out

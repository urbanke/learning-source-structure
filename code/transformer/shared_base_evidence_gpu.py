"""Experimental float64, batched SMC for selected shared-base short strings.

This is a whole-sampler GPU feasibility benchmark, NOT a long-context
implementation or a validated scientific reference. It reuses the sealed
weighted-LSA quadrature through the existing short-prefix moment helper.
Only strings of at most five tokens using already available m1..m4, M11,
and M21 row moments are accepted. Other moments fail explicitly.

The four proposals and their probabilities match SharedBaseMCMC. The
uniform rank-permutation prior is implicit. Initial particles follow the
normalized prior conditional on the first token; its marginal 1/d is
restored in log Z. Only the collapsed transition likelihood is tempered.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time

import numpy as np
import torch


MOVE_NAMES = ["seen_label_swap", "head_weighted_rank_swap",
              "alpha_prior_proposal", "alpha_adjacent_proposal"]
MOVE_PROBABILITIES = (.4, .4, .1, .1)


def checked_device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this float64 benchmark supports cpu or cuda only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    return device


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class MomentTables:
    base_laws: np.ndarray
    alpha_weights: np.ndarray
    depth_weights: np.ndarray
    moments: dict
    validation: dict

    @classmethod
    def from_sampler(cls, sampler):
        # This exact helper is also covered by the saved reference receipt.
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from shortprefix_shared_base_exact import row_moments
        values, checks = [], []
        for ai in range(len(sampler.alpha_weights)):
            moment, error = row_moments(sampler.kernel, ai)
            values.append(moment)
            checks.append(error)
        return cls(sampler.kernel.base_laws.copy(), sampler.alpha_weights.copy(),
                   sampler.depth_prior.copy(),
                   {key: np.stack([v[key] for v in values]) for key in values[0]},
                   dict(row_moment_checks=checks, renormalized=False))


def row_descriptors(tokens, d):
    x = np.asarray(tokens)
    if (x.ndim != 1 or x.dtype.kind not in "iu" or not 1 <= len(x) <= 5
            or np.any(x < 0) or np.any(x >= d)):
        raise ValueError("requires one through five integer tokens in the alphabet")
    counts = np.zeros((d, d), dtype=np.int64)
    np.add.at(counts, (x[:-1], x[1:]), 1)
    result = []
    for row in counts:
        labels = np.flatnonzero(row)
        if not len(labels):
            continue
        if len(labels) == 1:
            result.append((f"m{row[labels[0]]}", (int(labels[0]),)))
        elif len(labels) == 2 and tuple(row[labels]) == (1, 1):
            result.append(("M11", tuple(map(int, labels))))
        elif len(labels) == 2 and sorted(row[labels]) == [1, 2]:
            order = sorted(labels, key=lambda label: -row[label])
            result.append(("M21", tuple(map(int, order))))
        else:
            raise ValueError(f"unsupported short row moment {row[labels].tolist()}; "
                             "benchmark is limited to existing m1..m4/M11/M21 tables")
    return tuple(result)


class ShortTarget:
    """Batched log-target; the one shared depth is summed after all rows."""

    def __init__(self, tables, tokens, device="cpu"):
        self.device = checked_device(device)
        q = np.asarray(tables.base_laws, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] < 2:
            raise ValueError("invalid base-law array")
        self.n_alpha, self.d = q.shape
        self.rows = row_descriptors(tokens, self.d)
        self.tokens = tuple(map(int, tokens))
        self.first = self.tokens[0]
        self.seen = torch.tensor(np.unique(tokens), dtype=torch.long, device=self.device)
        aw = np.asarray(tables.alpha_weights, dtype=np.float64)
        dw = np.asarray(tables.depth_weights, dtype=np.float64)
        for value, shape in ((q, q.shape), (aw, (self.n_alpha,)), (dw, (len(dw),))):
            if value.shape != shape or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError("invalid positive prior array")
        if (np.max(abs(q.sum(axis=1)-1)) > 1e-12 or abs(aw.sum()-1) > 1e-12
                or abs(dw.sum()-1) > 1e-12):
            raise ValueError("prior arrays must already be normalized")
        def tensor(value):
            return torch.tensor(value, dtype=torch.float64, device=self.device)
        self.base = tensor(q)
        self.log_base = self.base.log()
        self.alpha_weights = tensor(aw)
        self.log_alpha = self.alpha_weights.log()
        self.log_depth = tensor(dw).log()
        self.log_moments = {}
        for key in {row[0] for row in self.rows}:
            moment = np.asarray(tables.moments[key], dtype=np.float64)
            arity = 1 if key.startswith("m") else 2
            if (moment.shape != (self.n_alpha, len(dw)) + (self.d,)*arity
                    or not np.isfinite(moment).all() or np.any(moment <= 0)):
                raise ValueError(f"invalid {key} moment table")
            self.log_moments[key] = tensor(moment).log()
        self.rank_weights = 1 / torch.arange(1, self.d+1, dtype=torch.float64, device=self.device)
        self.rank_weights /= self.rank_weights.sum()
        self.move_probabilities = tensor(MOVE_PROBABILITIES)

    def terms(self, alpha, ranks):
        """Return untempered first-token/prior term and transition log likelihood."""
        n = len(alpha)
        depth = self.log_depth.expand(n, -1).clone()
        for key, labels in self.rows:
            # Explicit depth index keeps advanced-index dimensions N × L.
            index = (alpha[:, None], torch.arange(len(self.log_depth), device=self.device)[None, :])
            index += tuple(ranks[:, label, None] for label in labels)
            depth = depth + self.log_moments[key][index]
        prior = self.log_alpha[alpha] + self.log_base[alpha, ranks[:, self.first]]
        return prior, torch.logsumexp(depth, dim=1)

    def propose_from_draws(self, alpha, ranks, move, seen_index, uniform_label,
                           head_rank, uniform_rank, prior_alpha, direction):
        """Pure batched proposal map, exposed for finite-state balance tests.

        Seen-label and head-rank swaps are symmetric. The independence alpha
        move needs log prior(old)-log prior(new). Adjacent proposals outside
        the node range remain at the current node (not reflected/clipped).
        """
        n = len(alpha)
        particle = torch.arange(n, device=self.device)
        inverse = torch.argsort(ranks, dim=1)
        i = torch.where(move == 1, inverse[particle, head_rank], self.seen[seen_index])
        j = torch.where(move == 1, inverse[particle, uniform_rank], uniform_label)
        i = torch.where(move < 2, i, torch.zeros_like(i))
        j = torch.where(move < 2, j, torch.zeros_like(j))
        proposed_ranks = ranks.clone()
        proposed_ranks[particle, i] = ranks[particle, j]
        proposed_ranks[particle, j] = ranks[particle, i]
        adjacent = alpha + direction
        adjacent = torch.where((adjacent >= 0) & (adjacent < self.n_alpha), adjacent, alpha)
        proposed_alpha = torch.where(move == 2, prior_alpha,
                                     torch.where(move == 3, adjacent, alpha))
        correction = torch.where(move == 2, self.log_alpha[alpha]-self.log_alpha[proposed_alpha],
                                 torch.zeros(n, dtype=torch.float64, device=self.device))
        return proposed_alpha, proposed_ranks, correction

    def mh_sweep(self, alpha, ranks, prior, likelihood, beta, generator):
        """One independently chosen original proposal and MH test per particle."""
        n, device = len(alpha), self.device
        move = torch.multinomial(self.move_probabilities, n, replacement=True, generator=generator)
        def randint(high):
            return torch.randint(high, (n,), device=device, generator=generator)
        proposal = self.propose_from_draws(alpha, ranks, move, randint(len(self.seen)), randint(self.d),
            torch.multinomial(self.rank_weights, n, replacement=True, generator=generator), randint(self.d),
            torch.multinomial(self.alpha_weights, n, replacement=True, generator=generator), 2*randint(2)-1)
        changed = self.accept_proposal(alpha, ranks, prior, likelihood, beta, proposal,
            torch.rand(n, dtype=torch.float64, device=device, generator=generator).log())
        return (*changed[:4], torch.bincount(move, minlength=4),
                torch.bincount(move, weights=changed[4].to(torch.float64), minlength=4))

    def accept_proposal(self, alpha, ranks, prior, likelihood, beta, proposal, log_uniform):
        """Actual MH acceptance map, also used by exhaustive finite-state tests."""
        new_alpha, new_ranks, correction = proposal
        new_prior, new_likelihood = self.terms(new_alpha, new_ranks)
        log_acceptance = new_prior-prior + beta*(new_likelihood-likelihood) + correction
        accepted = log_uniform < log_acceptance.clamp(max=0)
        return (torch.where(accepted, new_alpha, alpha), torch.where(accepted[:, None], new_ranks, ranks),
                torch.where(accepted, new_prior, prior), torch.where(accepted, new_likelihood, likelihood),
                accepted)


def systematic_indices(weights, generator):
    n = len(weights)
    cumulative = weights.cumsum(0)
    cumulative[-1] = 1.  # normalized CDF endpoint, only roundoff
    points = (torch.rand((), dtype=torch.float64, device=weights.device, generator=generator)
              + torch.arange(n, dtype=torch.float64, device=weights.device)) / n
    return torch.searchsorted(cumulative, points, right=True)


@torch.no_grad()
def estimate_log_evidence(target, *, particles=256, temperatures=64, moves=4,
                          seed=2026091603, resample_fraction=.75):
    """One independent whole-SMC repetition; time includes initialization/MH."""
    for name, value, minimum in (("particles", particles, 2), ("temperatures", temperatures, 1),
                                 ("moves", moves, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 < resample_fraction <= 1:
        raise ValueError("invalid resampling fraction")
    device = target.device
    generator = torch.Generator(device=device).manual_seed(seed)
    synchronize(device)
    started = time.monotonic()
    alpha = torch.multinomial(target.alpha_weights, particles, replacement=True, generator=generator)
    first_rank = torch.multinomial(target.base[alpha], 1, generator=generator).squeeze(1)
    ranks = torch.multinomial(torch.ones((particles, target.d), dtype=torch.float64, device=device),
                              target.d, replacement=False, generator=generator)
    other = (ranks == first_rank[:, None]).to(torch.long).argmax(1)
    index = torch.arange(particles, device=device)
    old_first = ranks[:, target.first].clone()
    ranks[:, target.first] = first_rank
    ranks[index, other] = old_first
    prior, likelihood = target.terms(alpha, ranks)
    log_weights = torch.full((particles,), -math.log(particles), dtype=torch.float64, device=device)
    log_evidence = torch.tensor(-math.log(target.d), dtype=torch.float64, device=device)
    betas = np.linspace(0., 1., temperatures+1)**3
    attempts = torch.zeros(4, dtype=torch.long, device=device)
    accepts = torch.zeros(4, dtype=torch.float64, device=device)
    ancestry = torch.arange(particles, device=device)
    history, resampled = [], []
    for previous, beta in zip(betas[:-1], betas[1:]):
        incremental = log_weights + float(beta-previous)*likelihood
        normalizer = torch.logsumexp(incremental, dim=0)
        log_evidence += normalizer
        log_weights = incremental-normalizer
        weights = log_weights.exp()
        ess = 1 / weights.square().sum()
        history.append(ess)
        should_resample = bool(ess < resample_fraction*particles)
        resampled.append(should_resample)
        if should_resample:
            selected = systematic_indices(weights, generator)
            alpha, ranks = alpha[selected], ranks[selected]
            prior, likelihood = prior[selected], likelihood[selected]
            ancestry = ancestry[selected]
            log_weights.fill_(-math.log(particles))
        for _ in range(moves):
            alpha, ranks, prior, likelihood, attempted, accepted = target.mh_sweep(
                alpha, ranks, prior, likelihood, float(beta), generator)
            attempts += attempted
            accepts += accepted
    # Include synchronization in wall time; table preparation/transfer is separate.
    result = dict(log_evidence=float(log_evidence), effective_sample_sizes=torch.stack(history).cpu().tolist(),
                  resampled=resampled, move_attempts=attempts.cpu().tolist(),
                  move_accepts=accepts.to(torch.long).cpu().tolist(),
                  final_distinct_initial_ancestors=int(torch.unique(ancestry).numel()))
    synchronize(device)
    result.update(elapsed_seconds=time.monotonic()-started, particles=particles,
                  temperatures=temperatures, moves=moves, seed=seed, device=str(device), dtype="float64",
                  n_tokens=len(target.tokens), beta_schedule="(k/K)^3", move_names=MOVE_NAMES,
                  resample_fraction=resample_fraction, scientific_reference_validated=False,
                  scope="selected short strings only; no long-context or population validation")
    return result

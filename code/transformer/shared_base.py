"""Small-alphabet Bayes reference for the shared-base LSA source.

The exponent is integrated by Gauss--Legendre quadrature, all rank
assignments are enumerated, and one latent depth is shared by every
transition row. These three latent variables are marginalized jointly.
Only the row-evidence kernel controls inner numerical integration.

The first token is drawn from q; an empty transition row instead predicts
the weighted-LSA prior mean, which generally differs from q. No shortcut
equating these distributions is used. Calls to run/log_joint start fresh
sequence state; only pure row integrals may be cached by the supplied
kernel. Probabilities are returned without renormalizing numerical errors.
"""
from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from shared_depth import _integer_vector, _positions, _tokens

LN2 = np.log(2.0)


class SharedBaseLSA:
    """Enumerated base hypotheses and a shared-depth posterior.

    row_kernel.depths gives the depth order. Its method
    log_evidence(counts, q) returns natural-log ordered-observation row
    evidence in that order. The default exponent prior is uniform on
    alpha_bounds, approximated with alpha_order Gauss--Legendre nodes.

    Diagnostics can fix alpha, retain known rank ordering, or supply the
    true known_q. known_q replaces both exponent and rank uncertainty and
    cannot be combined with fixed_alpha or known_order.
    """

    def __init__(self, d, row_kernel, *, alpha_order=16, alpha_bounds=(0.5, 1.5),
                 depth_prior=None, known_order=False, fixed_alpha=None, known_q=None):
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 2:
            raise ValueError("d must be an integer of at least two")
        depths = _integer_vector(row_kernel.depths, "depths", allow_empty=False)
        if np.any(depths < 0) or len(np.unique(depths)) != len(depths):
            raise ValueError("depths must be distinct nonnegative integers")
        depth_weights = np.ones(len(depths)) if depth_prior is None else np.asarray(depth_prior, dtype=float)
        if depth_weights.shape != (len(depths),) or not np.all(np.isfinite(depth_weights)) or np.any(depth_weights <= 0):
            raise ValueError("depth_prior must contain one finite positive weight per depth")
        if not isinstance(known_order, (bool, np.bool_)):
            raise ValueError("known_order must be boolean")
        self.d = int(d)
        self.row_kernel = row_kernel
        self.depths = depths.astype(np.int64, copy=True)
        self.log_depth_prior = np.log(depth_weights) - logsumexp(np.log(depth_weights))
        self.depth_prior = np.exp(self.log_depth_prior)
        self.known_order = bool(known_order)
        self.fixed_alpha = fixed_alpha
        self.known_q = known_q is not None

        if known_q is not None:
            if known_order or fixed_alpha is not None:
                raise ValueError("known_q replaces fixed_alpha and known_order; specify only known_q")
            q = np.asarray(known_q, dtype=float)
            if q.shape != (self.d,) or not np.all(np.isfinite(q)) or np.any(q <= 0):
                raise ValueError("known_q must be a positive finite probability vector of length d")
            if not np.isclose(q.sum(), 1.0, rtol=1e-12, atol=1e-14):
                raise ValueError("known_q must sum to one")
            self.base_laws = (q / q.sum())[None, :]
            self.alpha_nodes = None
            self.alpha_weights = None
            self.alphas = None
            self.perms = None
            self.hyper_prior = np.ones(1)
        else:
            if fixed_alpha is not None:
                if isinstance(fixed_alpha, (bool, np.bool_)) or not np.isscalar(fixed_alpha) or not np.isfinite(fixed_alpha) or fixed_alpha < 0:
                    raise ValueError("fixed_alpha must be a finite nonnegative number")
                nodes = np.array([float(fixed_alpha)])
                weights = np.ones(1)
            else:
                bounds = np.asarray(alpha_bounds, dtype=float)
                if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] < 0 or bounds[1] <= bounds[0]:
                    raise ValueError("alpha_bounds must be increasing finite nonnegative endpoints")
                if isinstance(alpha_order, (bool, np.bool_)) or not isinstance(alpha_order, (int, np.integer)) or alpha_order < 1:
                    raise ValueError("alpha_order must be a positive integer")
                nodes, weights = leggauss(int(alpha_order))
                nodes = (bounds[0] + bounds[1]) / 2 + (bounds[1] - bounds[0]) * nodes / 2
                weights /= weights.sum()
            if known_order:
                assignments = np.arange(1, self.d + 1)[None, :]
            else:
                if factorial(self.d) > 100000:
                    raise ValueError("too many rank permutations to enumerate; use a small alphabet")
                assignments = np.array(list(permutations(range(1, self.d + 1))), dtype=int)
            count = len(assignments)
            self.alpha_nodes = nodes
            self.alpha_weights = weights
            self.alphas = np.repeat(nodes, count)
            # perms[h,b] is the one-based latent rank of observed symbol b,
            # i.e. the inverse source permutation, not a token relabel map.
            self.perms = np.tile(assignments, (len(nodes), 1))
            logq = -self.alphas[:, None] * np.log(self.perms)
            self.base_laws = np.exp(logq - logsumexp(logq, axis=1, keepdims=True))
            self.hyper_prior = np.repeat(weights / count, count)
        self.n_hypotheses = len(self.base_laws)
        self.log_prior = np.log(self.hyper_prior)[:, None] + self.log_depth_prior[None, :]

    def _row_evidence(self, counts):
        if not np.any(counts):
            return np.zeros((self.n_hypotheses, len(self.depths)))
        evidence = np.asarray([self.row_kernel.log_evidence(counts, q) for q in self.base_laws], dtype=float)
        expected = (self.n_hypotheses, len(self.depths))
        if evidence.shape != expected:
            raise ValueError(f"row kernel evidence must have shape {expected}, got {evidence.shape}")
        if not np.all(np.isfinite(evidence)):
            raise FloatingPointError("nonfinite weighted row evidence")
        return evidence

    def log_joint(self, x):
        """Natural-log joint probability, rebuilt from the batch row counts."""
        x = _tokens(x, self.d)
        if not len(x):
            return 0.0
        counts = np.zeros((self.d, self.d), dtype=np.int64)
        np.add.at(counts, (x[:-1], x[1:]), 1)
        log_terms = self.log_prior + np.log(self.base_laws[:, x[0]])[:, None]
        for row in counts:
            if np.any(row):
                log_terms = log_terms + self._row_evidence(row)
        return float(logsumexp(log_terms))

    def joint(self, x):
        """Joint probability (use log_joint to avoid long-sequence underflow)."""
        return float(np.exp(self.log_joint(x)))

    def run(self, x, full_probs=False, positions=None, posteriors=False):
        """Predict before every token, including the initial q-distributed one.

        loss is in bits; log_evidence is the cumulative natural-log joint
        probability. With full_probs, probs contains the sorted selected
        zero-based token positions, and prob_sum_error is its row sum minus
        one. With posteriors, posterior[n,H,K], base_posterior[n,H], and
        depth_posterior[n,K] refer to the state BEFORE each token. Supplying
        x[0] free means scoring loss[1:], while retaining its posterior update.
        """
        x = _tokens(x, self.d)
        n = len(x)
        selected = _positions(n, full_probs, positions)
        want = np.zeros(n, dtype=bool)
        want[selected] = True
        loss = np.empty(n)
        evidence = np.empty(n)
        if full_probs:
            probability = np.empty((len(selected), self.d))
        if posteriors:
            posterior = np.empty((n, self.n_hypotheses, len(self.depths)))
        counts = np.zeros((self.d, self.d), dtype=np.int64)
        current_evidence = np.zeros((self.d, self.n_hypotheses, len(self.depths)))
        log_weights = self.log_prior.copy()
        cumulative = 0.0
        saved = 0
        for t, symbol in enumerate(x):
            if posteriors:
                posterior[t] = np.exp(log_weights)
            if t == 0:
                if want[t]:
                    for b in range(self.d):
                        probability[saved, b] = np.exp(logsumexp(log_weights + np.log(self.base_laws[:, b])[:, None]))
                increment = np.log(self.base_laws[:, symbol])[:, None]
            else:
                context = int(x[t - 1])
                before = current_evidence[context]
                if want[t]:
                    observed_evidence = None
                    for b in range(self.d):
                        candidate = counts[context].copy()
                        candidate[b] += 1
                        after = self._row_evidence(candidate)
                        probability[saved, b] = np.exp(logsumexp(log_weights + after - before))
                        if b == symbol:
                            observed_evidence = after
                else:
                    candidate = counts[context].copy()
                    candidate[symbol] += 1
                    observed_evidence = self._row_evidence(candidate)
                increment = observed_evidence - before
                counts[context, symbol] += 1
                current_evidence[context] = observed_evidence
            log_prediction = logsumexp(log_weights + increment)
            log_weights = log_weights + increment - log_prediction
            loss[t] = -log_prediction / LN2
            cumulative += log_prediction
            evidence[t] = cumulative
            if want[t]:
                saved += 1
        out = {"loss": loss, "log_evidence": evidence}
        if full_probs:
            out.update(probs=probability, positions=selected,
                       prob_sum_error=probability.sum(axis=1) - 1.0)
        if posteriors:
            out.update(posterior=posterior, base_posterior=posterior.sum(axis=2),
                       depth_posterior=posterior.sum(axis=1))
        return out

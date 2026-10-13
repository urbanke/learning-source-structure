"""Uniform-base LSA chains with one latent depth shared by all rows.

``SharedDepthLSA(SeqLSA(...), prior=...)`` reuses the numerical per-depth
row evidence of lsa_seq, leaving its legacy row-local mixture unchanged.
The joint evidence is d**-1 * sum_L prior[L] * product_a Q_L(M_a).
Depth zero and depth one use their exact uniform/Dirichlet evidence;
empty and single-observation rows are exact at every depth. Higher-depth
accuracy is inherited from the supplied SeqLSA kernel's quadrature.

All positions index tokens from zero. Loss and posterior at position t
are computed BEFORE observing x[t]. The first token is included, with
uniform probability; callers supplying it free should slice loss[1:].
Returned probabilities are evidence ratios, never silently renormalized.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

LN2 = np.log(2.0)


def _integer_vector(values, name, allow_empty=True):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    if not allow_empty and not array.size:
        raise ValueError(f"{name} must not be empty")
    return array


def _tokens(x, d):
    x = _integer_vector(x, "x")
    if np.any(x < 0) or np.any(x >= d):
        raise ValueError(f"tokens must lie in 0..{d - 1}")
    return x.astype(np.int64, copy=False)


def _positions(n, full_probs, positions):
    if positions is not None and not full_probs:
        raise ValueError("positions requires full_probs=True")
    if not full_probs:
        return np.empty(0, dtype=np.int64)
    if positions is None:
        return np.arange(n, dtype=np.int64)
    positions = _integer_vector(positions, "positions")
    if np.any(positions < 0) or np.any(positions >= n):
        raise ValueError("positions must index tokens in x")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("positions must be distinct")
    return np.sort(positions.astype(np.int64, copy=False))


class SharedDepthLSA:
    """One independent sequence per run; the immutable kernel may be reused.

    kernel is a SeqLSA instance exposing d, depths and new_table(). prior
    consists of finite, strictly positive weights in kernel.depths order;
    omitted weights are uniform. No source-depth range is inferred here.
    """

    def __init__(self, kernel, prior=None):
        if isinstance(kernel.d, (bool, np.bool_)) or not isinstance(kernel.d, (int, np.integer)) or kernel.d < 2:
            raise ValueError("kernel.d must be an integer of at least two")
        depths = _integer_vector(kernel.depths, "depths", allow_empty=False)
        if np.any(depths < 0) or len(np.unique(depths)) != len(depths):
            raise ValueError("depths must be distinct nonnegative integers")
        weights = np.ones(len(depths)) if prior is None else np.asarray(prior, dtype=float)
        if weights.shape != (len(depths),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("prior must contain one finite positive weight per depth")
        self.kernel = kernel
        self.d = int(kernel.d)
        self.depths = depths.astype(np.int64, copy=True)
        self.log_prior = np.log(weights) - logsumexp(np.log(weights))
        self.prior = np.exp(self.log_prior)
        self._depth0 = self.depths == 0
        self._depth1 = self.depths == 1
        self._logd = np.log(self.d)

    def _correct_evidence(self, raw, counts, k):
        """Anchor exact cases; numerical quadrature is needed only for L>=2."""
        result = np.asarray(raw, dtype=float).copy()
        if result.shape != self.depths.shape:
            raise ValueError("row evidence shape does not match kernel.depths")
        if k == 0:
            result.fill(0.0)
        elif k == 1:
            result.fill(-self._logd)
        else:
            result[self._depth0] = -k * self._logd
            if np.any(self._depth1):
                result[self._depth1] = gammaln(self.d) - gammaln(self.d + k) + gammaln(counts + 1).sum()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite per-depth row evidence")
        return result

    def _predictive(self, table, log_weights):
        if table.k == 0:
            return np.full(self.d, 1.0 / self.d)
        probability = np.empty(self.d)
        for count in np.unique(table.m):
            mask = table.m == count
            symbol = int(np.flatnonzero(mask)[0])
            candidate_counts = table.m.copy()
            candidate_counts[symbol] += 1
            after = self._correct_evidence(table._logQ_after(int(count)), candidate_counts, table.k + 1)
            probability[mask] = np.exp(logsumexp(log_weights + after - table.logQ))
        return probability

    def run(self, x, full_probs=False, positions=None, depth_posteriors=False):
        """Return loss (bits) and cumulative log_evidence (natural logs).

        With full_probs, probs has shape (len(positions), d), positions
        is sorted, and prob_sum_error reports each raw row sum minus one.
        With depth_posteriors, depth_posterior has shape (len(x), K) and
        holds depth probabilities BEFORE each token, including x[0].
        """
        x = _tokens(x, self.d)
        n = len(x)
        if n > 1 and hasattr(self.kernel, "rmax"):
            visits = np.bincount(x[:-1], minlength=self.d)
            if int(visits.max()) > self.kernel.rmax:
                raise ValueError("kernel.rmax must cover the total number of transitions from each context "
                                 f"(need {int(visits.max())}, have {self.kernel.rmax})")
        selected = _positions(n, full_probs, positions)
        want = np.zeros(n, dtype=bool)
        want[selected] = True
        loss = np.empty(n)
        evidence = np.empty(n)
        if full_probs:
            probabilities = np.empty((len(selected), self.d))
        if depth_posteriors:
            posterior = np.empty((n, len(self.depths)))
        tables = [None] * self.d
        log_weights = self.log_prior.copy()
        cumulative = 0.0
        saved = 0
        for t, symbol in enumerate(x):
            if depth_posteriors:
                posterior[t] = np.exp(log_weights)
            if t == 0:
                log_prediction = -self._logd
                if want[t]:
                    probabilities[saved] = 1.0 / self.d
            else:
                context = int(x[t - 1])
                if tables[context] is None:
                    tables[context] = self.kernel.new_table()
                table = tables[context]
                if want[t]:
                    probabilities[saved] = self._predictive(table, log_weights)
                before = np.zeros(len(self.depths)) if table.k == 0 else table.logQ.copy()
                # The legacy return value mixes this row's depths locally;
                # only its updated per-depth evidence is used here.
                table.observe(int(symbol))
                table.logQ = self._correct_evidence(table.logQ, table.m, table.k)
                increment = table.logQ - before
                log_prediction = logsumexp(log_weights + increment)
                log_weights += increment - log_prediction
            loss[t] = -log_prediction / LN2
            cumulative += log_prediction
            evidence[t] = cumulative
            if want[t]:
                saved += 1
        out = {"loss": loss, "log_evidence": evidence}
        if full_probs:
            out.update(probs=probabilities, positions=selected,
                       prob_sum_error=probabilities.sum(axis=1) - 1.0)
        if depth_posteriors:
            out["depth_posterior"] = posterior
        return out


class ProductSharedDepthLSA:
    """Independent coordinate chains, encoded as x = u * d2 + v."""

    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.d1 = first.d
        self.d2 = second.d
        self.d = self.d1 * self.d2

    def run(self, x, full_probs=False, positions=None, depth_posteriors=False):
        x = _tokens(x, self.d)
        selected = _positions(len(x), full_probs, positions)
        args = dict(full_probs=full_probs, positions=selected if full_probs else None,
                    depth_posteriors=depth_posteriors)
        first = self.first.run(x // self.d2, **args)
        second = self.second.run(x % self.d2, **args)
        out = {"loss": first["loss"] + second["loss"],
               "log_evidence": first["log_evidence"] + second["log_evidence"]}
        if full_probs:
            probability = (first["probs"][:, :, None] * second["probs"][:, None, :]).reshape(len(selected), self.d)
            out.update(probs=probability, positions=selected,
                       prob_sum_error=probability.sum(axis=1) - 1.0)
        if depth_posteriors:
            out["depth_posterior1"] = first["depth_posterior"]
            out["depth_posterior2"] = second["depth_posterior"]
        return out

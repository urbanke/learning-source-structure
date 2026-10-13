"""Float64 weighted-LSA row evidence for small-alphabet Bayesian references.

For counts m, k=sum(m)>0, and positive weights q:
  Z_L(m;q) = 1/Gamma(k) int exp(k*u)
             prod_b q_b**m_b phi_{L,m_b}(q_b*exp(u)) du.

The result is the probability of ordered observations, without a multinomial
coefficient. Depth zero and empty rows are exact; uniform depth-one rows use
the Dirichlet formula. Nonuniform first observations require the integral:
their probabilities are generally NOT the base weights q.

Higher-depth phi values are queried directly at shifted points from the
read-only anchor store. No float32 tables, linear shift interpolation,
post-hoc probability normalization or inferred base law is used.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import os

import numpy as np
from scipy.special import gammaln, logsumexp


class WeightedLSARowEvidence:
    """Cached row-evidence kernel; shared safely across independent sequences.

    q may be proportional positive weights; common scale is removed.
    Simultaneously permuting q and counts leaves the cache key unchanged.
    Numerical accuracy must be established for the requested bounds, spacing,
    counts and depths, e.g. with validate_shared_base.py.
    """

    def __init__(self, depths=range(9), *, u_lo=-32.0, u_hi=32.0,
                 step=1 / 32, store=None, src=None,
                 term_cache_size=1024, evidence_cache_size=100000):
        depths = np.asarray(list(depths))
        if (depths.ndim != 1 or depths.size == 0 or depths.dtype.kind not in "iu"
                or np.any(depths < 0) or len(np.unique(depths)) != len(depths)):
            raise ValueError("depths must be distinct nonnegative integers")
        if (not np.isfinite([u_lo, u_hi, step]).all()
                or not u_lo < u_hi or step <= 0):
            raise ValueError("require finite u_lo < u_hi and positive step")
        cells = (u_hi - u_lo) / step
        if not np.isclose(cells, round(cells), rtol=0, atol=1e-8):
            raise ValueError("the integration interval must contain an integer number of steps")
        if term_cache_size < 1 or evidence_cache_size < 1:
            raise ValueError("cache capacities must be positive")
        self.depths = depths.astype(np.int64)
        self.u_lo, self.u_hi, self.step = float(u_lo), float(u_hi), float(step)
        self.u = self.u_lo + self.step * np.arange(int(round(cells)) + 1)
        self._positive = np.flatnonzero(self.depths > 0)
        self._zero = self.depths == 0
        self._one = self.depths == 1
        self._store = store or os.environ.get("LSA_STORE")
        self._src = src or os.environ.get("PMM_SRC")
        self._backend = None
        self._terms = OrderedDict()
        self._evidence = OrderedDict()
        self.term_cache_size = int(term_cache_size)
        self.evidence_cache_size = int(evidence_cache_size)
        self.row_calls = 0
        self.row_cache_hits = 0
        self.phi_terms_computed = 0

    @staticmethod
    def _canonical(counts, q):
        counts = np.asarray(counts)
        q = np.asarray(q, dtype=np.float64)
        if (counts.ndim != 1 or len(counts) < 2 or counts.dtype.kind not in "iu"
                or np.any(counts < 0)):
            raise ValueError("counts must be a nonnegative integer vector of length at least two")
        if q.shape != counts.shape or not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError("q must have one finite strictly positive weight per symbol")
        # Sort before summation so floating-point normalization and keys
        # are identical for all rank assignments of the same base law.
        order = np.lexsort((counts, -q))
        q = q[order]
        q = q / q[0]  # avoid overflow when normalizing large proportional weights
        q = q / math.fsum(q.tolist())
        if np.any(q <= 0):
            raise ValueError("weight ratios underflowed; use less extreme positive weights")
        return counts[order].astype(np.int64), q

    def _get_backend(self):
        if self._backend is None:
            if not self._store or not self._src:
                raise ValueError("depths >=2 require LSA_STORE and PMM_SRC (or store/src arguments)")
            from phi_source import StoreTables
            tables = StoreTables(int(self.depths.max()), os.path.expanduser(self._store),
                                 os.path.expanduser(self._src),
                                 u_lo=self.u_lo, u_hi=self.u_hi)
            self._backend = tables.ut
        return self._backend

    def _term(self, logq, count):
        key = (float(logq), int(count))
        if key in self._terms:
            self._terms.move_to_end(key)
            return self._terms[key]
        shifted = self.u + logq
        values = np.empty((len(self._positive), len(self.u)), dtype=np.float64)
        for j, i in enumerate(self._positive):
            depth = int(self.depths[i])
            if depth == 1:
                values[j] = gammaln(count + 1) - (count + 1) * np.logaddexp(0, shifted)
            else:
                values[j] = self._get_backend().log_phi_matrix(depth, [count], shifted)[0]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite shifted phi values")
        self._terms[key] = values
        self.phi_terms_computed += 1
        if len(self._terms) > self.term_cache_size:
            self._terms.popitem(last=False)
        return values

    def log_evidence(self, counts, q):
        """Return natural-log row evidence, in self.depths order."""
        counts, q = self._canonical(counts, q)
        key = (tuple(counts), tuple(q))
        self.row_calls += 1
        if key in self._evidence:
            self.row_cache_hits += 1
            self._evidence.move_to_end(key)
            return self._evidence[key].copy()
        k = int(counts.sum())
        result = np.zeros(len(self.depths), dtype=np.float64)
        if k:
            logq = np.log(q)
            weight_term = float(counts @ logq)
            result[self._zero] = weight_term
            if len(self._positive):
                integrand = np.broadcast_to(k * self.u, (len(self._positive), len(self.u))).copy()
                for count, weight in zip(counts, logq):
                    integrand += self._term(float(weight), int(count))
                result[self._positive] = (logsumexp(integrand, axis=1)
                                          + np.log(self.step) - gammaln(k) + weight_term)
            if np.all(q == q[0]):
                if k == 1:
                    result.fill(-np.log(len(q)))
                if np.any(self._one):
                    result[self._one] = (gammaln(len(q)) - gammaln(len(q) + k)
                                         + gammaln(counts + 1).sum())
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite weighted row evidence")
        self._evidence[key] = result.copy()
        if len(self._evidence) > self.evidence_cache_size:
            self._evidence.popitem(last=False)
        return result

    def cache_info(self):
        return dict(row_calls=self.row_calls, row_cache_hits=self.row_cache_hits,
                    phi_terms_computed=self.phi_terms_computed,
                    stored_rows=len(self._evidence), stored_phi_terms=len(self._terms),
                    phi_cache_bytes=sum(value.nbytes for value in self._terms.values()))

    def clear_cache(self):
        self._terms.clear()
        self._evidence.clear()

"""Weighted-LSA evidence with cached rank terms for a fixed exponent grid.

Each exponent specifies q(r) proportional to r**(-alpha), r=1,...,d.
The all-zero-count log-phi sum is retained once per exponent. A row then
adds corrections only at ranks with positive counts, rather than summing
all d shifted phi arrays for every evaluation. Continuous-u integration
uses the same float64 shifted anchor queries as weighted_lsa.py; this
cache rearrangement introduces no interpolation or posterior approximation.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln

from weighted_lsa import WeightedLSARowEvidence
from lsa_seq import _lse_rows


class RankedLSARowEvidence(WeightedLSARowEvidence):
    """Rank-ordered row evidence, with bounded term and row caches.

    ``log_evidence(counts, alpha_index)`` takes counts in increasing latent
    rank order, not observed-symbol order. ``base_laws[a, r]`` gives q at
    zero-based rank r for exponent index a. Both array properties are
    read-only. Empty rows and depth zero are exact. A uniform exponent-zero
    base also has exact singleton and depth-one shortcuts.

    Baseline arrays cost A*K*U*8 bytes at most, where K counts positive
    depths and U integration nodes. A cached term costs K*U*8 bytes. The
    default 1024-term capacity is approximately 128 MiB at the default
    integration grid and depths. Baselines are prepared lazily; their
    number is bounded by the supplied exponent grid.
    """

    def __init__(self, d, alpha_nodes, depths=range(9), *, u_lo=-32.0,
                 u_hi=32.0, step=1 / 32, store=None, src=None,
                 term_cache_size=1024, evidence_cache_size=100000,
                 predictive_cache_size=10000):
        if (isinstance(d, (bool, np.bool_))
                or not isinstance(d, (int, np.integer)) or d < 2):
            raise ValueError("d must be an integer of at least two")
        nodes = np.asarray(alpha_nodes, dtype=np.float64)
        if (nodes.ndim != 1 or not nodes.size or not np.isfinite(nodes).all()
                or np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes)):
            raise ValueError("alpha_nodes must be distinct finite nonnegative numbers")
        for capacity in (term_cache_size, evidence_cache_size, predictive_cache_size):
            if (isinstance(capacity, (bool, np.bool_))
                    or not isinstance(capacity, (int, np.integer)) or capacity < 1):
                raise ValueError("cache capacities must be positive integers")
        super().__init__(depths, u_lo=u_lo, u_hi=u_hi, step=step,
                         store=store, src=src, term_cache_size=term_cache_size,
                         evidence_cache_size=evidence_cache_size)
        self.d = int(d)
        self.alpha_nodes = nodes.copy()
        weights = np.exp(-nodes[:, None] * np.log(np.arange(1, d + 1)))
        self.base_laws = np.asarray([q / math.fsum(q.tolist()) for q in weights])
        if np.any(self.base_laws <= 0):
            raise ValueError("rank weights underflowed; use less extreme exponents")
        self.log_base_laws = np.log(self.base_laws)
        self.alpha_nodes.setflags(write=False)
        self.base_laws.setflags(write=False)
        self.log_base_laws.setflags(write=False)
        self._baselines = {}
        self._predictive = OrderedDict()
        self.predictive_cache_size = int(predictive_cache_size)
        self.predictive_calls = 0
        self.predictive_cache_hits = 0

    def _index(self, alpha_index):
        if (isinstance(alpha_index, (bool, np.bool_))
                or not isinstance(alpha_index, (int, np.integer))
                or alpha_index < 0 or alpha_index >= len(self.alpha_nodes)):
            raise ValueError("alpha_index must index the supplied exponent grid")
        return int(alpha_index)

    def _counts(self, counts):
        counts = np.asarray(counts)
        if (counts.shape != (self.d,) or counts.dtype.kind not in "iu"
                or np.any(counts < 0)):
            raise ValueError("counts must be a nonnegative integer vector of length d")
        if np.any(counts > np.iinfo(np.int64).max):
            raise ValueError("counts exceed the supported integer range")
        counts = counts.astype(np.int64, copy=False)
        if sum(map(int, counts)) > np.iinfo(np.int64).max:
            raise ValueError("total count exceeds the supported integer range")
        return counts

    @staticmethod
    def _sparse_key(counts):
        # Unobserved ranks contribute identically to every row and need no
        # per-key storage. Keeping explicit rank/count pairs preserves all
        # label information for nonuniform base laws.
        return tuple((int(rank), int(counts[rank])) for rank in np.flatnonzero(counts))

    def _get_backend(self):
        """Reuse the production store, refusing open caches or stale modes."""
        if self._backend is None:
            if not self._store or not self._src:
                raise ValueError("depths >=2 require LSA_STORE and PMM_SRC (or store/src arguments)")
            store = Path(self._store).expanduser().resolve()
            if not (store / "anchors.json").is_file() or not (store / "manifest.json").is_file():
                raise RuntimeError("a preexisting sealed production anchor store is required; refusing an open cache")

            def check_imported_mode():
                module = sys.modules.get("product_model_with_memory.universal_tables")
                if module is not None:
                    for name, expected in (("_LADDER_EVERY", 1), ("_LADDER_DEGREE", 11),
                                           ("_SADDLE_MIN_L", 54)):
                        if getattr(module, name, None) != expected:
                            raise RuntimeError("universal_tables was already imported outside production mode; "
                                               "configure the sealed store in a fresh process")

            check_imported_mode()
            src = str(Path(self._src).expanduser().resolve())
            if src not in sys.path:
                sys.path.insert(0, src)
            # This is the existing production guard and configuration,
            # including all four required PMM settings. It reads no columns.
            from product_model_with_memory.production_coding import configure_production_tables
            configure_production_tables(store)
            check_imported_mode()
            self._store, self._src = str(store), src
        return super()._get_backend()

    def _raw_phi(self, alpha_index, rank, count):
        shifted = self.u + self.log_base_laws[alpha_index, rank]
        values = np.empty((len(self._positive), len(self.u)), dtype=np.float64)
        for j, i in enumerate(self._positive):
            depth = int(self.depths[i])
            if depth == 1:
                values[j] = gammaln(count + 1) - (count + 1) * np.logaddexp(0, shifted)
            else:
                values[j] = self._get_backend().log_phi_matrix(depth, [count], shifted)[0]
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite shifted phi values")
        self.phi_terms_computed += 1
        return values

    def _term(self, alpha_index, rank, count):
        # Count zero stores log phi_0; positive counts store log phi_m -
        # log phi_0. Both share one bounded LRU cache.
        key = (alpha_index, int(rank), int(count))
        if key in self._terms:
            self._terms.move_to_end(key)
            return self._terms[key]
        values = self._raw_phi(alpha_index, rank, count)
        if count:
            values -= self._term(alpha_index, rank, 0)
        values.setflags(write=False)
        self._terms[key] = values
        if len(self._terms) > self.term_cache_size:
            self._terms.popitem(last=False)
        return values

    def _baseline(self, alpha_index):
        if alpha_index not in self._baselines:
            baseline = np.zeros((len(self._positive), len(self.u)), dtype=np.float64)
            for rank in range(self.d):
                baseline += self._term(alpha_index, rank, 0)
            baseline.setflags(write=False)
            self._baselines[alpha_index] = baseline
        return self._baselines[alpha_index]

    def log_evidence(self, counts_in_rank_order, alpha_index):
        """Natural-log ordered-row evidence, in ``self.depths`` order.

        Numerical integration errors are returned as computed, without any
        probability renormalization. The caller must validate the requested
        count range, bounds and spacing as for WeightedLSARowEvidence.
        """
        alpha_index = self._index(alpha_index)
        counts = self._counts(counts_in_rank_order)
        uniform = self.alpha_nodes[alpha_index] == 0
        if uniform:
            counts = np.sort(counts)
        key = (alpha_index, self._sparse_key(counts))
        self.row_calls += 1
        if key in self._evidence:
            self.row_cache_hits += 1
            self._evidence.move_to_end(key)
            return self._evidence[key].copy()
        k = int(counts.sum())
        result = np.zeros(len(self.depths), dtype=np.float64)
        if k:
            weight_term = float(counts @ self.log_base_laws[alpha_index])
            result[self._zero] = weight_term
            if uniform and k == 1:
                result.fill(-np.log(self.d))
            elif len(self._positive):
                if not (uniform and np.all(self.depths[self._positive] == 1)):
                    integrand = self._baseline(alpha_index) + k * self.u
                    for rank in np.flatnonzero(counts):
                        integrand += self._term(alpha_index, int(rank), int(counts[rank]))
                    # Reuse the established vectorized reduction. At most
                    # U*exp(-50) of each relative row sum is omitted.
                    result[self._positive] = (_lse_rows(integrand, drop=50.0)
                                              + np.log(self.step) - gammaln(k) + weight_term)
                if uniform and np.any(self._one):
                    result[self._one] = (gammaln(self.d) - gammaln(self.d + k)
                                        + gammaln(counts + 1).sum())
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite weighted row evidence")
        self._evidence[key] = result.copy()
        if len(self._evidence) > self.evidence_cache_size:
            self._evidence.popitem(last=False)
        return result

    def log_evidence_many(self, counts_matrix, alpha_index):
        """Evaluate rows of a rank-ordered count matrix with shared caches."""
        alpha_index = self._index(alpha_index)
        counts = np.asarray(counts_matrix)
        if counts.ndim != 2 or counts.shape[1] != self.d:
            raise ValueError("counts_matrix must have shape (number_of_rows, d)")
        if not len(counts):
            return np.empty((0, len(self.depths)), dtype=np.float64)
        return np.stack([self.log_evidence(row, alpha_index) for row in counts])

    def log_predictive(self, counts_in_rank_order, alpha_index):
        """Natural-log next-rank probabilities of shape (number_of_depths, d).

        Each column is one row-evidence increment. The sum over next ranks
        is not corrected for numerical integration error. The separate
        bounded cache makes reused sparse row profiles and empty-row prior
        means inexpensive across posterior samples.
        """
        alpha_index = self._index(alpha_index)
        counts = self._counts(counts_in_rank_order)
        if int(counts.sum()) == np.iinfo(np.int64).max:
            raise ValueError("the count cannot be incremented within the supported integer range")
        key = (alpha_index, self._sparse_key(counts))
        self.predictive_calls += 1
        if key in self._predictive:
            self.predictive_cache_hits += 1
            self._predictive.move_to_end(key)
            return self._predictive[key].copy()
        before = self.log_evidence(counts, alpha_index)
        candidate = counts.copy()
        result = np.empty((len(self.depths), self.d), dtype=np.float64)
        for rank in range(self.d):
            candidate[rank] += 1
            result[:, rank] = self.log_evidence(candidate, alpha_index) - before
            candidate[rank] -= 1
        self._predictive[key] = result.copy()
        if len(self._predictive) > self.predictive_cache_size:
            self._predictive.popitem(last=False)
        return result

    def cache_info(self):
        info = super().cache_info()
        info.update(stored_baselines=len(self._baselines),
                    baseline_cache_bytes=sum(a.nbytes for a in self._baselines.values()),
                    predictive_calls=self.predictive_calls,
                    predictive_cache_hits=self.predictive_cache_hits,
                    stored_predictives=len(self._predictive),
                    predictive_cache_bytes=sum(a.nbytes for a in self._predictive.values()))
        return info

    def clear_cache(self):
        super().clear_cache()
        self._baselines.clear()
        self._predictive.clear()

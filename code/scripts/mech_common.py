"""Shared machinery for the two-strategy (frequency vs copier) experiments.

Sources, and a profile ledger for exact code-length evaluation through
the LSA package's intended mechanism.  Every code length here is the
depth-averaged codelength of a COUNT PROFILE (the sorted nonzero counts;
by exchangeability the profile is sufficient).  Profiles are collected,
deduplicated, and evaluated in batches by
``lsa.codelength.depth_averaged_codelength_profiles``, whose moment
tables come from the permanent universal store: certified values,
computed once ever, grown on demand, shared by every experiment and
machine.  The counting and evaluation pattern follows the production
scripts of the reference projects (``bible_online_states_experiment.py``
in the LSA repository; ``calibrate_markov1_sampling.py`` and
``context_profiles_at_stops`` in product_model_with_memory).  Nothing
in this module builds tables.

Scale notes.  The chain sampler materializes one rule row only when its
context is first visited, stores just the row's cumulative sums (for
O(log d) inverse-CDF steps), and frees everything once the chain is
generated; per-context successor counts are sparse Counters, and stage
counting is vectorized.  The ledger resolves checkpoints to plain
floats as soon as their profiles are evaluated, so nothing large is
retained per checkpoint.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from lsa.codelength import profile_of

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def zipf(d: int, alpha: float) -> np.ndarray:
    p = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return p / p.sum()


class ChainSampler:
    """Markov chain from tilted layered rules, memory-frugal.

    Row a of the transition matrix is q * V_a renormalized, with
    V_a ~ layered simplex distribution of depth L (L = 0: V_a uniform,
    every row equals q).  A row is sampled the first time context a is
    visited (lazily, via the LSA package's sampler, consuming the
    stream's rng), reduced at once to its cumulative sums, and each
    chain step is one binary search.  ``release_rules()`` frees the
    rows; the generated prefix is kept.
    """

    _UBLOCK = 65536  # uniforms are drawn in blocks (cuts call overhead)

    def __init__(self, d: int, alpha: float, L: int,
                 rng: np.random.Generator):
        self.d = d
        self.L = L
        self.q = zipf(d, alpha)
        self.q_cum = np.cumsum(self.q)
        self.rng = rng
        self.rows: dict[int, np.ndarray] | None = {}  # context -> cumsum
        self._u = np.empty(0)
        self._ui = 0
        self.x = np.empty(1, dtype=np.int32)
        self.x[0] = min(int(np.searchsorted(
            self.q_cum, self._next_u(), side="right")), d - 1)
        self.n = 1

    def _next_u(self) -> float:
        if self._ui >= len(self._u):
            self._u = self.rng.random(self._UBLOCK)
            self._ui = 0
        u = self._u[self._ui]
        self._ui += 1
        return u

    def _row_cum(self, a: int) -> np.ndarray:
        row = self.rows.get(a)
        if row is None:
            if self.L == 0:
                row = self.q_cum
            else:
                from lsa.product_simplex import (
                    sample_product_simplex_distribution)

                v = sample_product_simplex_distribution(self.d, self.L,
                                                        self.rng)
                v *= self.q
                v /= v.sum()
                row = np.cumsum(v)
            self.rows[a] = row
        return row

    def extend_to(self, n: int) -> np.ndarray:
        if n > self.n and self.rows is None:
            raise RuntimeError("rules released; cannot extend further")
        if n > len(self.x):
            grown = np.empty(n, dtype=np.int32)
            grown[: self.n] = self.x[: self.n]
            self.x = grown
        searchsorted, next_u, row_cum = (np.searchsorted, self._next_u,
                                         self._row_cum)
        x, dm1 = self.x, self.d - 1
        for t in range(self.n, n):
            cum = row_cum(int(x[t - 1]))
            tok = int(searchsorted(cum, next_u(), side="right"))
            x[t] = tok if tok < dm1 else dm1
        self.n = max(self.n, n)
        return self.x[:n]

    def release_rules(self) -> None:
        """Free the rule rows (call once the chain is fully generated)."""

        self.rows = None


# ---------------------------------------------------------------------------
# the profile ledger
# ---------------------------------------------------------------------------

class ProfileTrajectory:
    """Per-checkpoint exact code lengths along one sequence, staged.

    ``advance(x, checkpoints)`` counts transitions (vectorized per
    checkpoint interval; per-context successor counts are sparse
    Counters, as in product_model_with_memory's
    ``context_profiles_at_stops``) and appends PENDING checkpoint
    entries ``(t, F_profile, states)``, where ``states`` maps each
    distinct per-context profile to its multiplicity.  After the
    pending profiles are evaluated, ``resolve(ev)`` converts them to
    plain float rows (t, l_F, l_C) in ``losses`` and drops the profile
    structures:

        l_F(t) = bits(F_profile)
        l_C(t) = log2(d) + sum_p states[p] * bits(p)

    with bits(p) = -log2 q_avg(p); the log2(d) term is the uniform code
    for the first token, which has no context (equal to the
    depth-averaged code of a single symbol, exactly).
    """

    def __init__(self, d: int):
        self.d = d
        self.uni = np.zeros(d, dtype=np.int64)
        self.succ: dict[int, Counter] = {}
        self.prof: dict[int, tuple] = {}   # context -> its current profile
        self.mult: Counter = Counter()     # profile -> number of contexts
        self.dirty: set[int] = set()
        self.pos = 0
        self.pending: list[tuple[int, tuple, Counter]] = []
        self.losses: list[tuple[int, float, float]] = []

    def _count_interval(self, x: np.ndarray, lo: int, hi: int) -> None:
        """Add tokens x[lo:hi] (positions lo..hi-1) to the counts."""

        if hi <= lo:
            return
        self.uni += np.bincount(x[lo:hi], minlength=self.d)
        p0 = max(lo, 1)
        if hi > p0:
            codes = x[p0 - 1:hi - 1].astype(np.int64) * self.d + x[p0:hi]
            uniq, cnt = np.unique(codes, return_counts=True)
            for code, c in zip(uniq.tolist(), cnt.tolist()):
                a, b = divmod(code, self.d)
                row = self.succ.get(a)
                if row is None:
                    row = self.succ[a] = Counter()
                row[b] += c
                self.dirty.add(a)

    def advance(self, x: np.ndarray, checkpoints: list[int]) -> None:
        """Consume x up to its end, recording at the given checkpoints.
        Checkpoints at or before the last recorded one are skipped, so
        passing the full grid on every stage is fine."""

        last = (self.pending[-1][0] if self.pending
                else (self.losses[-1][0] if self.losses else 0))
        cps = [c for c in sorted(set(checkpoints))
               if last < c <= len(x)]
        for c in cps:
            self._count_interval(x, self.pos, c)
            self.pos = c
            self._record(c)
        if self.pos < len(x):
            self._count_interval(x, self.pos, len(x))
            self.pos = len(x)

    def _record(self, t: int) -> None:
        for a in self.dirty:
            new = profile_of(self.succ[a].values())
            old = self.prof.get(a)
            if old is not None:
                self.mult[old] -= 1
                if not self.mult[old]:
                    del self.mult[old]
            self.prof[a] = new
            self.mult[new] += 1
        self.dirty.clear()
        self.pending.append(
            (t, profile_of(self.uni.tolist()), self.mult.copy()))

    def profiles_needed(self) -> set:
        """Every profile the pending checkpoints refer to."""

        need: set = set()
        for (_, fp, states) in self.pending:
            need.add(fp)
            need.update(states.keys())
        return need

    def resolve(self, ev) -> None:
        """Turn pending checkpoints into float rows using ``ev``
        (profile -> DepthAveragedCodelength) and drop their profiles."""

        for (t, fp, states) in self.pending:
            l_F = -ev[fp].log2_q_avg
            l_C = math.log2(self.d) + sum(
                m * (-ev[p].log2_q_avg) for p, m in states.items())
            self.losses.append((t, l_F, l_C))
        self.pending.clear()

    def lam(self) -> list[tuple[int, float]]:
        return [(t, lF - lC) for (t, lF, lC) in self.losses]


def geometric_grid(lo: int, hi: int, ratio: float = 1.12) -> list[int]:
    """Strictly increasing integer grid from lo to hi, ratio-spaced."""

    out, v = [], float(lo)
    while v < hi:
        t = int(round(v))
        if not out or t > out[-1]:
            out.append(t)
        v *= ratio
    if not out or out[-1] != hi:
        out.append(hi)
    return out

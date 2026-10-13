"""Paired product-source histories for encoding mismatch and parameter changes.

The source prior is the known-encoding 8 x 8 product family from ``families``:
each coordinate has eight independent uniform-base LSA transition rows, with
a fixed source depth for that coordinate. This module samples the actual law;
it does not change the prior supplied to either tested predictor.

``observed=n`` means predicting ``tokens[n]`` from ``tokens[:n]``. A change
after 256 observed tokens therefore first affects the transition from
``tokens[255]`` to ``tokens[256]``. All conditions share initial coordinates
and one pair of independent uniform innovations per subsequent transition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from source import Source


FACTOR_SIZE = 8
VOCAB_SIZE = FACTOR_SIZE ** 2
CONDITIONS = ("stationary", "relabeled", "switch_one", "switch_both")
DEFAULT_SEED = 2026091809


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result != value or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class ProductShiftCase:
    """One history and its exact generating distributions.

    ``pre_matrices`` and ``post_matrices`` have shape (2, 8, 8), in the
    canonical coordinate labels. ``permutation[a]`` is the observed label of
    canonical joint state ``a``; it is identity outside the relabeled case.
    No change of encoding takes place at the transition-matrix change.
    """

    condition: str
    index: int
    seed: int
    depths: tuple[int, int]
    switch_observed: int
    tokens: np.ndarray
    pre_matrices: np.ndarray
    post_matrices: np.ndarray
    permutation: np.ndarray

    def true_probs(self, observed: int | np.ndarray) -> np.ndarray:
        """True next-token law, in observed labels, at selected prefix lengths.

        A scalar returns shape (64,); a one-dimensional array returns (N, 64).
        Only predictions with a stored target, 1 <= observed < len(tokens),
        are accepted. The distribution is never inferred from sampled counts.
        """
        raw = np.asarray(observed)
        scalar = raw.ndim == 0
        if raw.ndim > 1 or raw.dtype.kind not in "iu":
            raise ValueError("observed must contain integer prefix lengths")
        points = np.atleast_1d(raw).astype(np.int64, copy=False)
        if np.any(points < 1) or np.any(points >= len(self.tokens)):
            raise ValueError("observed must be between 1 and len(tokens)-1")
        inverse = np.argsort(self.permutation)
        states = inverse[self.tokens[points - 1]]
        use_post = points >= self.switch_observed
        p1 = self.pre_matrices[0, states // FACTOR_SIZE].copy()
        p2 = self.pre_matrices[1, states % FACTOR_SIZE].copy()
        p1[use_post] = self.post_matrices[0, states[use_post] // FACTOR_SIZE]
        p2[use_post] = self.post_matrices[1, states[use_post] % FACTOR_SIZE]
        canonical = (p1[:, :, None] * p2[:, None, :]).reshape(-1, VOCAB_SIZE)
        result = np.empty_like(canonical)
        result[:, self.permutation] = canonical
        return result[0] if scalar else result


def _draw_matrix(depth: int, rng: np.random.Generator) -> np.ndarray:
    # At alpha=0 the row prior is invariant to a simultaneous coordinate
    # relabeling; sampling in the visible coordinate labels has exactly the
    # same law as ProductFamily's separate coordinate permutations.
    source = Source(FACTOR_SIZE, 2, alpha=0.0, ls=depth, permute=False)
    _, matrix = source.matrices(np.array([0.0]), np.array([depth]), rng)
    return matrix[0]


def _sample_history(pre: np.ndarray, post: np.ndarray, uniforms: np.ndarray,
                    switch_observed: int) -> np.ndarray:
    n_tokens = len(uniforms)
    coordinates = np.empty((n_tokens, 2), dtype=np.int64)
    coordinates[0] = np.floor(FACTOR_SIZE * uniforms[0]).astype(np.int64)
    cumulative = np.cumsum(np.stack((pre, post)), axis=-1)
    cumulative[..., -1] = 1.0
    for position in range(1, n_tokens):
        phase = int(position >= switch_observed)
        for factor in range(2):
            row = cumulative[phase, factor, coordinates[position - 1, factor]]
            coordinates[position, factor] = np.searchsorted(
                row, uniforms[position, factor], side="left")
    return coordinates[:, 0] * FACTOR_SIZE + coordinates[:, 1]


def generate_product_shift_cases(
    index: int,
    *,
    depths: tuple[int, int] = (4, 4),
    seed: int = DEFAULT_SEED,
    n_tokens: int = 1025,
    switch_observed: int = 256,
) -> dict[str, ProductShiftCase]:
    """Generate all four paired conditions for one independent source draw.

    Component-specific seeds depend only on the master seed, depth pair and
    source index, not on invocation order or which histories will be scored.
    The one-factor and two-factor interventions use the same replacement for
    coordinate one. Coordinate two retains its original matrix in switch_one.
    The relabeled condition is exactly a permutation of the stationary path.
    """
    if isinstance(index, (bool, np.bool_)) or int(index) != index or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(depths) != 2:
        raise ValueError("depths must contain one depth per coordinate")
    depths = tuple(_positive_integer(value, "depth") for value in depths)
    n_tokens = _positive_integer(n_tokens, "n_tokens")
    switch_observed = _positive_integer(switch_observed, "switch_observed")
    if not 1 <= switch_observed < n_tokens:
        raise ValueError("switch_observed must be between 1 and n_tokens-1")
    index, seed = int(index), int(seed)

    # Separate SeedSequence inputs avoid collapsing all source/component
    # identities into the legacy 32-bit seed_of hash.
    component_ids = {name: i for i, name in enumerate(
        ("pre-0", "pre-1", "post-0", "post-1", "path-uniforms", "joint-permutation"))}

    def rng(component: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(
            [seed, index, depths[0], depths[1], component_ids[component], 1]))

    pre = _readonly(np.stack([
        _draw_matrix(depth, rng(f"pre-{factor}"))
        for factor, depth in enumerate(depths)]))
    post_both = _readonly(np.stack([
        _draw_matrix(depth, rng(f"post-{factor}"))
        for factor, depth in enumerate(depths)]))
    post_one = _readonly(np.stack((post_both[0], pre[1])))
    uniforms = rng("path-uniforms").random((n_tokens, 2))
    permutation = _readonly(rng("joint-permutation").permutation(VOCAB_SIZE))
    identity = _readonly(np.arange(VOCAB_SIZE, dtype=np.int64))
    stationary = _sample_history(pre, pre, uniforms, switch_observed)
    histories = {
        "stationary": stationary,
        "relabeled": permutation[stationary],
        "switch_one": _sample_history(pre, post_one, uniforms, switch_observed),
        "switch_both": _sample_history(pre, post_both, uniforms, switch_observed),
    }
    posts = {"stationary": pre, "relabeled": pre,
             "switch_one": post_one, "switch_both": post_both}
    return {
        condition: ProductShiftCase(
            condition=condition, index=index, seed=seed, depths=depths,
            switch_observed=switch_observed, tokens=_readonly(histories[condition]),
            pre_matrices=pre, post_matrices=posts[condition],
            permutation=permutation if condition == "relabeled" else identity,
        )
        for condition in CONDITIONS
    }

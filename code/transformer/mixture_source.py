"""The three existing d=64 priors, pooled without an input family label.

Batch ``i`` is a pure function of (seed, i, n_ctx, batch). Family counts are
balanced over each three consecutive batches; random within-batch order and
all source draws use independent SeedSequence namespaces. Worker count and
prefetching therefore cannot change the training sequence on restart.
"""
from __future__ import annotations

import numpy as np
import torch

from families import ProductFamily
from source import Source

FAMILIES = ("product", "shared_base", "uniform_base")
SOURCE_CONTRACT = {
    "version": "three-family-priors-v1",
    "alphabet": 64,
    "product": {"dimensions": [8, 8], "alpha": 0.0,
                "independent_coordinate_depths": [1, 8], "coordinate_permutations": True},
    "shared_base": {"alpha_uniform": [0.5, 1.5], "depth_uniform_inclusive": [0, 8],
                    "alphabet_permutation": True},
    "uniform_base": {"alpha": 0.0, "depth_uniform_inclusive": [0, 8],
                     "alphabet_permutation": True},
    "family_weights": [1 / 3, 1 / 3, 1 / 3],
    "batch_schedule": "balanced-three-batch-cycle-random-order-v1",
    "family_input": False,
}


def stream_rng(seed: int, index: int, purpose: int, family: int = 0):
    """No intermediate 32-bit seed hash: retain all seed and index bits."""
    if min(seed, index, purpose, family) < 0 or seed >= 2**64 or index >= 2**64:
        raise ValueError("seed/index must be unsigned 64-bit integers")
    words = [0x4D495831, seed & 0xFFFFFFFF, seed >> 32,
             index & 0xFFFFFFFF, index >> 32, purpose, family]
    return np.random.default_rng(np.random.SeedSequence(words))


def family_sources(n_ctx: int):
    return (ProductFamily(8, 8, n_ctx + 1, ls=(1, 8), alpha=0.0),
            Source(64, n_ctx + 1, alpha=(0.5, 1.5), ls=(0, 8), permute=True),
            Source(64, n_ctx + 1, alpha=0.0, ls=(0, 8), permute=True))


def family_counts(batch_size: int, batch_index: int):
    if batch_size < 3 or batch_index < 0:
        raise ValueError("a mixture batch needs at least three sequences")
    counts = np.full(3, batch_size // 3, dtype=np.int64)
    for j in range(batch_size % 3):
        counts[(batch_index + j) % 3] += 1
    return counts


def completed_counts(batch_size: int, completed_batches: int):
    cycles, remainder = divmod(completed_batches, 3)
    counts = np.full(3, cycles * batch_size, dtype=np.int64)
    for i in range(remainder):
        counts += family_counts(batch_size, i)
    return dict(zip(FAMILIES, map(int, counts)))


def draw_training_batch(seed: int, batch_index: int, batch_size: int,
                        n_ctx: int, *, return_family_ids=False):
    counts = family_counts(batch_size, batch_index)
    xs = [src.batch(int(count), stream_rng(seed, batch_index, 1, k))[0]
          for k, (src, count) in enumerate(zip(family_sources(n_ctx), counts))]
    order = stream_rng(seed, batch_index, 2).permutation(batch_size)
    tokens = np.concatenate(xs, axis=0)[order]
    if return_family_ids:  # audit only; the training Dataset never exposes this
        ids = np.repeat(np.arange(3), counts)[order]
        return tokens, ids
    return tokens


def draw_validation(n_ctx: int, n_sequences: int, validation_seed: int):
    """Fixed full-prior monitoring sets; namespace 3 is disjoint from training."""
    return {name: torch.from_numpy(src.batch(n_sequences,
                    stream_rng(validation_seed, 0, 3, k))[0])
            for k, (name, src) in enumerate(zip(FAMILIES, family_sources(n_ctx)))}


class IndexedMixture(torch.utils.data.Dataset):
    """Map-style batches preserve order across worker/prefetch changes."""
    def __init__(self, seed, batch_size, n_ctx, start, stop):
        self.seed = seed
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.start = start
        self.stop = stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return torch.from_numpy(draw_training_batch(self.seed, self.start + index,
                                                   self.batch_size, self.n_ctx))

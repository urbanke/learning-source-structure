"""The source of Section 2.1, vectorised over a batch of sequences, for training and evaluating transformers.

One sequence: base law q_b proportional to b^(-alpha); one rule vector per context, the normalised product of L_s
independent uniform draws from the simplex (equivalently of L_s independent unit exponentials per entry); the row of
context a is q times its rule vector, normalised; X_1 ~ q, X_{t+1} ~ T(. | X_t). At L_s = 0 the rows all equal q.
This matches draw_table / simulate_path in code/scripts/building_block/chain_sim.py.

Two things are added for the transformer. First, alpha and L_s may be drawn afresh for every sequence from a range,
so that a model trained on the stream does not know them (as the two mechanisms of Section 2.2 do not). Second, the
alphabet may be relabelled by a random permutation for every sequence, so that the model cannot learn the base law
in its weights. Neither changes the crossing n*_d of Section 2.3: the mechanisms know only d and are invariant to
relabelling, and each evaluation set fixes alpha and L_s.
"""
from __future__ import annotations
import numpy as np


def zipf(d, alpha):
    b = np.arange(1, d + 1, dtype=float)
    q = b ** (-alpha)
    return q / q.sum()


class Source:
    def __init__(self, d, n_tokens, alpha=(0.5, 1.5), ls=(0, 8), permute=True):
        """alpha: a number (fixed) or a pair (lo, hi) for a uniform draw per sequence; ls: an int (fixed) or a pair
        (lo, hi) for a uniform integer draw per sequence, inclusive."""
        self.d = int(d); self.n = int(n_tokens)
        self.alpha = alpha; self.ls = ls; self.permute = bool(permute)

    def draw_params(self, B, rng):
        if np.isscalar(self.alpha):
            alpha = np.full(B, float(self.alpha))
        else:
            alpha = rng.uniform(self.alpha[0], self.alpha[1], size=B)
        if np.isscalar(self.ls):
            ls = np.full(B, int(self.ls))
        else:
            ls = rng.integers(self.ls[0], self.ls[1] + 1, size=B)
        return alpha, ls

    def matrices(self, alpha, ls, rng):
        """Transition matrices for the given per-sequence parameters: array (B, d, d)."""
        d = self.d; B = len(alpha)
        b = np.arange(1, d + 1, dtype=float)
        q = b[None, :] ** (-alpha[:, None]); q /= q.sum(axis=1, keepdims=True)              # (B, d)
        T = np.repeat(q[:, None, :], d, axis=1)                                             # rows = q
        for L in np.unique(ls):
            if L == 0:
                continue
            sel = np.flatnonzero(ls == L)
            # product of L unit exponentials per entry, as the log of the product to keep it in range
            logW = np.log(rng.exponential(size=(L, len(sel), d, d))).sum(axis=0)
            W = np.exp(logW - logW.max(axis=2, keepdims=True))
            T[sel] = T[sel] * W
            T[sel] /= T[sel].sum(axis=2, keepdims=True)
        return q, T

    def sequences(self, q, T, rng):
        """Sample one sequence of n tokens from each matrix: array (B, n) of ints."""
        B, d, _ = T.shape; n = self.n
        x = np.empty((B, n), dtype=np.int64)
        cq = np.cumsum(q, axis=1); cq[:, -1] = 1.0
        x[:, 0] = np.minimum((rng.random(B)[:, None] > cq).sum(axis=1), d - 1)
        cT = np.cumsum(T, axis=2); cT[:, :, -1] = 1.0
        rows = np.arange(B)
        U = rng.random((B, n))
        for t in range(1, n):
            c = cT[rows, x[:, t - 1], :]                                                   # (B, d)
            x[:, t] = np.minimum((U[:, t][:, None] > c).sum(axis=1), d - 1)
        return x

    def batch(self, B, rng):
        """A batch: tokens (B, n) after relabelling, and the per-sequence alpha, L_s and permutation."""
        alpha, ls = self.draw_params(B, rng)
        q, T = self.matrices(alpha, ls, rng)
        x = self.sequences(q, T, rng)
        if self.permute:
            perm = np.stack([rng.permutation(self.d) for _ in range(B)])                  # (B, d)
            x = np.take_along_axis(perm, x, axis=1)
        else:
            perm = np.repeat(np.arange(self.d)[None, :], B, axis=0)
        return x, dict(alpha=alpha, ls=ls, perm=perm)


def seed_of(*parts):
    """A reproducible 32-bit seed from a few integers or strings."""
    h = 1469598103934665603
    for p in parts:
        for c in str(p).encode():
            h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h % (2 ** 32 - 1))

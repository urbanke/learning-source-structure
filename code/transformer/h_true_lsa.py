"""The per-position entropy of the true next-token distribution for an evaluation set of the paper's source
(eval_sets.eval_set), regenerated from the set's seed: the sequences are drawn again in the same order, the
transition matrices are kept, and the md5 of the tokens is checked against the stored one.
    h = h_true_lsa(d, alpha, ls, n_ctx, n_eval, md5)    # (n_eval, n_ctx) in bits
"""
from __future__ import annotations
import numpy as np
from source import Source, seed_of
from eval_sets import md5_of
from families import chain_h_true


def h_true_lsa(d, alpha, ls, n_ctx, n_eval, md5=None, permute=True):
    src = Source(d, n_ctx + 1, alpha=float(alpha), ls=int(ls), permute=permute)
    rng = np.random.default_rng(seed_of("eval", d, f"{alpha:g}", ls, n_ctx, int(permute)))
    a, L = src.draw_params(n_eval, rng); q, T = src.matrices(a, L, rng); x0 = src.sequences(q, T, rng)
    if permute:
        perm = np.stack([rng.permutation(d) for _ in range(n_eval)]); x = np.take_along_axis(perm, x0, axis=1)
    else:
        x = x0
    if md5 is not None:
        assert md5_of(x.astype(np.int16)) == md5, "the regenerated set differs from the stored one"
    return chain_h_true(T, x0)

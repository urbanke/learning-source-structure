"""The evaluation sets: for a setting (d, alpha, L_s) and a context length, N sequences from the source with a seed
fixed by the setting, relabelled per sequence like the training data. Generated identically on every machine (the
tokens' md5 is stored with every result so that the transformer's outputs and the mechanisms' references can be
checked to refer to the same sequences)."""
from __future__ import annotations
import hashlib
import numpy as np
from source import Source, seed_of

GRID_ALPHA = (0.5, 0.75, 1.0, 1.25, 1.5)
GRID_LS = (1, 2, 3, 4, 6)


def tag_of(d, alpha, ls):
    return f"d{d}_a{alpha:g}_L{ls}"


def eval_set(d, alpha, ls, n_ctx, n_eval, permute=True):
    src = Source(d, n_ctx + 1, alpha=float(alpha), ls=int(ls), permute=permute)
    rng = np.random.default_rng(seed_of("eval", d, f"{alpha:g}", ls, n_ctx, int(permute)))
    x, meta = src.batch(n_eval, rng)
    return x.astype(np.int16), meta


def md5_of(x):
    return hashlib.md5(np.ascontiguousarray(x, dtype=np.int16).tobytes()).hexdigest()


def parse_settings(s):
    """'0.5,1.5x1,2,3' -> all pairs; 'grid' -> the default grid; '1,2;1,4' -> explicit pairs."""
    if s == "grid":
        return [(a, L) for a in GRID_ALPHA for L in GRID_LS]
    if "x" in s:
        a, L = s.split("x")
        return [(float(u), int(v)) for u in a.split(",") for v in L.split(",")]
    return [(float(p.split(",")[0]), int(p.split(",")[1])) for p in s.split(";")]

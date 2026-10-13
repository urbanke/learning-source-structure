"""The source families of notes/families_plan.md, with one interface for training, evaluation sets and references.

A family object has .d (alphabet size), .n (tokens per sequence) and .batch(B, rng) -> (x, meta), where x is an int64
array (B, n) of tokens (after any relabelling) and meta holds the per-sequence parameters and 'h_true', the entropy
in bits of the true next-token distribution at every position t = 1..n-1 (the prediction of x_{t+1} given x_1..x_t),
which is the loss of the ideal predictor that knows the source; the regret of a predictor is its loss minus h_true.

Specs (strings, used by train.py --family, eval_sets.py and refs_family.py):
    lsa                       the source of the paper (Source in source.py); alpha, L_s, permute from the arguments
    binary:twopoint           d = 2, iid, theta in {0.2, 0.8} with probability 1/2 each               (F1a)
    binary:eightpoint         theta uniform on {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9}                (F1c)
    binary:uniform            theta uniform on [0, 1]; the Bayes predictor is add-one (Laplace)       (F1b)
    binary:fixed=0.5          theta fixed (test sets for cross-training)
    expfam:k=1                iid over d symbols, p(b) ~ exp(-theta1 ln b), theta1 uniform in [0.5, 1.5], labels known (F2)
    expfam:k=2                p(b) ~ exp(-theta1 ln b - theta2 (ln b)^2), theta2 uniform in [-0.15, 0.15]
    expfam:k=1:perm           the same with the alphabet relabelled per sequence (the labels unknown)
    expfam:k=1:theta=1        fixed parameters (test sets)
    product:8x8               tokens (u, v), two independent LSA chains with uniform base laws, depths from the
                              --ls range (default 1..8), coordinates relabelled separately; d = d1 d2             (F3)
    product:8x8:L=2,4         fixed depths (test sets)
"""
from __future__ import annotations
import numpy as np
from source import Source, seed_of

LN2 = np.log(2.0)


def h_bits(p, axis=-1):
    p = np.clip(p, 1e-300, None)
    return -(p * np.log(p)).sum(axis=axis) / LN2


def parse_spec(spec):
    """'name:key=val:flag' -> (name, {key: val, flag: True})."""
    parts = spec.split(":"); name = parts[0]; opts = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1); opts[k] = v
        else:
            opts[p] = True
    return name, opts


# ---- F1: binary iid with a prior on theta --------------------------------------------------------------------------
class BinaryFamily:
    PRIORS = {"twopoint": np.array([0.2, 0.8]), "eightpoint": np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])}

    def __init__(self, n_tokens, prior="twopoint", theta=None):
        self.d = 2; self.n = int(n_tokens); self.prior = prior; self.theta_fixed = theta

    def draw_theta(self, B, rng):
        if self.theta_fixed is not None:
            return np.full(B, float(self.theta_fixed))
        if self.prior == "uniform":
            return rng.uniform(0.0, 1.0, size=B)
        return rng.choice(self.PRIORS[self.prior], size=B)

    def batch(self, B, rng):
        theta = self.draw_theta(B, rng)
        x = (rng.random((B, self.n)) < theta[:, None]).astype(np.int64)
        h = h_bits(np.stack([1 - theta, theta], 1))
        return x, dict(theta=theta, h_true=np.repeat(h[:, None], self.n - 1, axis=1).astype(np.float32))


# ---- F2: iid exponential family over d symbols -----------------------------------------------------------------------
class ExpFamily:
    def __init__(self, d, n_tokens, k=1, permute=False, theta1=(0.5, 1.5), theta2=(-0.15, 0.15), fixed=None):
        self.d = int(d); self.n = int(n_tokens); self.k = int(k); self.permute = bool(permute)
        self.theta1 = theta1; self.theta2 = theta2; self.fixed = fixed
        b = np.arange(1, self.d + 1, dtype=float)
        self.F = np.stack([-np.log(b), -np.log(b) ** 2])[: self.k]                  # features (k, d)

    def probs(self, theta):
        """theta: (B, k) -> (B, d)."""
        logits = theta @ self.F
        p = np.exp(logits - logits.max(axis=1, keepdims=True)); return p / p.sum(axis=1, keepdims=True)

    def draw_theta(self, B, rng):
        if self.fixed is not None:
            return np.repeat(np.asarray(self.fixed, dtype=float)[None, : self.k], B, axis=0)
        th = np.empty((B, self.k)); th[:, 0] = rng.uniform(*self.theta1, size=B)
        if self.k > 1:
            th[:, 1] = rng.uniform(*self.theta2, size=B)
        return th

    def batch(self, B, rng):
        theta = self.draw_theta(B, rng); p = self.probs(theta)
        cp = np.cumsum(p, axis=1); cp[:, -1] = 1.0
        x = np.minimum((rng.random((B, self.n))[:, :, None] > cp[:, None, :]).sum(axis=2), self.d - 1)
        if self.permute:
            perm = np.stack([rng.permutation(self.d) for _ in range(B)]); x = np.take_along_axis(perm, x, axis=1)
        else:
            perm = np.repeat(np.arange(self.d)[None, :], B, axis=0)
        h = h_bits(p)
        return x.astype(np.int64), dict(theta=theta, perm=perm, p=p, h_true=np.repeat(h[:, None], self.n - 1, axis=1).astype(np.float32))


# ---- chains: per-position true entropies -----------------------------------------------------------------------------
def chain_h_true(T, x_unperm):
    """T: (B, d, d) in the unpermuted labels, x_unperm: (B, n) unpermuted tokens -> h_true (B, n-1): the entropy of
    T(. | x_t) for t = 1..n-1."""
    H = h_bits(T, axis=2)                                                            # (B, d) row entropies
    return np.take_along_axis(H, x_unperm[:, :-1], axis=1).astype(np.float32)


class LSAFamily:
    """The source of the paper, with h_true added to the meta."""
    def __init__(self, d, n_tokens, alpha=(0.5, 1.5), ls=(0, 8), permute=True):
        self.src = Source(d, n_tokens, alpha=alpha, ls=ls, permute=permute); self.d = int(d); self.n = int(n_tokens)

    def batch(self, B, rng):
        s = self.src; alpha, ls = s.draw_params(B, rng); q, T = s.matrices(alpha, ls, rng); x0 = s.sequences(q, T, rng)
        if s.permute:
            perm = np.stack([rng.permutation(s.d) for _ in range(B)]); x = np.take_along_axis(perm, x0, axis=1)
        else:
            perm = np.repeat(np.arange(s.d)[None, :], B, axis=0); x = x0
        return x, dict(alpha=alpha, ls=ls, perm=perm, h_true=chain_h_true(T, x0))


# ---- F3: the product chain -------------------------------------------------------------------------------------------
class ProductFamily:
    def __init__(self, d1, d2, n_tokens, ls=(1, 8), alpha=0.0, fixed_ls=None):
        self.d1 = int(d1); self.d2 = int(d2); self.d = self.d1 * self.d2; self.n = int(n_tokens)
        l1, l2 = (fixed_ls if fixed_ls is not None else (ls, ls))
        self.s1 = Source(self.d1, self.n, alpha=alpha, ls=l1, permute=True)
        self.s2 = Source(self.d2, self.n, alpha=alpha, ls=l2, permute=True)

    @staticmethod
    def _draw(s, B, rng):
        alpha, ls = s.draw_params(B, rng); q, T = s.matrices(alpha, ls, rng); x0 = s.sequences(q, T, rng)
        perm = np.stack([rng.permutation(s.d) for _ in range(B)]); x = np.take_along_axis(perm, x0, axis=1)
        return x, x0, T, perm, ls

    def batch(self, B, rng):
        u, u0, T1, p1, l1 = self._draw(self.s1, B, rng); v, v0, T2, p2, l2 = self._draw(self.s2, B, rng)
        x = u * self.d2 + v
        h = chain_h_true(T1, u0) + chain_h_true(T2, v0)
        return x.astype(np.int64), dict(ls1=l1, ls2=l2, perm1=p1, perm2=p2, h_true=h)

    def split(self, x):
        """Joint tokens -> (u, v)."""
        return x // self.d2, x % self.d2


# ---- the registry ----------------------------------------------------------------------------------------------------
def make_family(spec, n_tokens, d=None, alpha=(0.5, 1.5), ls=(0, 8), permute=True):
    """A family object from a spec; d, alpha, ls, permute are the defaults for the 'lsa' family (and ls for 'product')."""
    name, o = parse_spec(spec)
    if name == "lsa":
        return LSAFamily(d, n_tokens, alpha=alpha, ls=ls, permute=permute)
    if name == "binary":
        prior = next((k for k in ("twopoint", "eightpoint", "uniform") if k in o), None)
        theta = float(o["fixed"]) if "fixed" in o else None
        if prior is None and theta is None:
            raise ValueError(f"binary family needs a prior or fixed=theta: {spec}")
        return BinaryFamily(n_tokens, prior=prior or "uniform", theta=theta)
    if name == "expfam":
        k = int(o.get("k", 1)); dd = int(o.get("d", d if d is not None else 32))
        fixed = [float(t) for t in o["theta"].split(",")] if "theta" in o else None
        return ExpFamily(dd, n_tokens, k=k, permute=("perm" in o), fixed=fixed)
    if name == "product":
        size = next(k for k in o if "x" in k and k.replace("x", "").isdigit()); d1, d2 = (int(s) for s in size.split("x"))
        fixed = tuple(int(t) for t in o["L"].split(",")) if "L" in o else None
        return ProductFamily(d1, d2, n_tokens, ls=ls if ls != (0, 8) else (1, 8), fixed_ls=fixed)
    raise ValueError(f"unknown family {spec}")


def tag_of_spec(spec):
    return spec.replace(":", "_").replace("=", "").replace(",", "-")


def family_eval_set(spec, n_ctx, n_eval, d=None):
    """A fixed evaluation set of the family: n_eval sequences of n_ctx + 1 tokens, seeded by the spec."""
    fam = make_family(spec, n_ctx + 1, d=d)
    rng = np.random.default_rng(seed_of("eval", spec, n_ctx, n_eval))
    x, meta = fam.batch(n_eval, rng)
    return x.astype(np.int16), meta, fam

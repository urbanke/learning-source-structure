"""Short, frequency-based references with three explicitly different laws.

``count_only_original_probs`` conditions the original ordered source law on
its distinguished first token and the multiset of its remaining observed
prefix. It averages *evidence*, not normalized ordered predictions. This is
an information restriction of the original law, not the law of finite-tail
shuffling used in the neural control's training.

``stationary_iid_joint`` instead defines a surrogate: draw the first token
from q, then draw independent tokens from the source's stationary law nu,
with one common uniformly random label permutation. A bank of latent (q,nu)
pairs can be integrated by averaging these joints before taking predictive
ratios. Neither construction claims to be exact finite-shuffle Bayes.

The existing short catalog supplies original-law joints through length five,
so it supports count-only predictions only through four observations. No
sampler, source generator, catalog, or active numerical workflow is modified.
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import math

import numpy as np


LAWS = dict(
    count_only_original='Original ordered law conditioned on first token and observed tail counts.',
    stationary_iid='First token from q; subsequent iid tokens from stationary nu; shared random labels.',
    finite_shuffle='Actual finite-length training-tail permutation; not implemented by these references.',
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _tokens(values, d, maximum):
    array = np.asarray(values)
    _require(array.ndim == 1 and (array.size == 0 or array.dtype.kind in 'iu'),
             'Tokens must be a one-dimensional integer sequence')
    _require(0 <= len(array) <= maximum, f'At most {maximum} tokens are supported')
    _require(np.all((array >= 0) & (array < d)), 'Token is outside the alphabet')
    return tuple(map(int, array))


def _logsum(values):
    values = list(map(float, values))
    _require(bool(values) and all(math.isfinite(v) or v == -math.inf for v in values),
             'Joint provider returned invalid natural-log evidence')
    largest = max(values)
    if largest == -math.inf:
        return largest
    return largest + math.log(math.fsum(math.exp(v-largest) for v in values))


def count_only_original_probs(prefix, log_joint, *, d=64, max_observed=4,
                              normalization_tolerance=1e-9):
    """P(next | first, multiset(observed tail)) under the *original* law.

    ``log_joint`` is a callable accepting a tuple of tokens and returning the
    natural logarithm of its original ordered-sequence probability. Repeated
    tail labels need only unique permutations: every unique arrangement has
    the same multiplicity. The candidate next token is never permuted into
    that observed tail. Return raw predictive probabilities, unrenormalized.
    """
    _require(type(d) is int and d >= 2 and type(max_observed) is int
             and 0 <= max_observed <= 4, 'Need alphabet >=2 and at most four observations')
    _require(normalization_tolerance > 0, 'Normalization tolerance must be positive')
    tokens = _tokens(prefix, d, max_observed)
    histories = [()] if not tokens else [(tokens[0],) + tail
                                          for tail in sorted(set(itertools.permutations(tokens[1:])))]
    denominator = _logsum(log_joint(history) for history in histories)
    _require(math.isfinite(denominator), 'Conditioning event has zero probability')
    numerators = [_logsum(log_joint(history + (y,)) for history in histories) for y in range(d)]
    result = np.exp(np.asarray(numerators)-denominator)
    _require(np.isfinite(result).all() and np.all(result >= 0)
             and abs(math.fsum(map(float, result))-1) <= normalization_tolerance,
             'Predictive joints fail normalization; no clipping or renormalization is applied')
    return result


@lru_cache(maxsize=6)
def set_partitions(size):
    """Disjoint set partitions, each block internally ordered; no duplicates."""
    _require(type(size) is int and 0 <= size <= 5, 'At most five distinct labels are supported')
    if size == 0:
        return ((),)
    result = []
    for partition in set_partitions(size-1):
        result.append(partition + ((size-1,),))
        for i in range(len(partition)):
            result.append(partition[:i] + (partition[i] + (size-1,),) + partition[i+1:])
    return tuple(result)


def _laws(q, nu):
    q, nu = np.broadcast_arrays(np.asarray(q, dtype=np.longdouble), np.asarray(nu, dtype=np.longdouble))
    _require(q.ndim >= 1 and q.shape[-1] >= 2 and np.isfinite(q).all() and np.isfinite(nu).all()
             and np.all(q >= 0) and np.all(nu >= 0), 'q and nu must be finite nonnegative probability vectors')
    _require(np.all(np.abs(q.sum(-1)-1) <= 1e-10) and np.all(np.abs(nu.sum(-1)-1) <= 1e-10),
             'q and nu must each normalize; no renormalization is applied')
    return q, nu


def _stable_injective_sum(q, nu, powers):
    """Nonnegative subset DP used only when signed contractions cancel.

    After each rank, state[mask] sums assignments of exactly those labels to
    distinct ranks already visited. Updating from the old state cannot reuse
    a rank. This is independent of the partition formula and uses no clipping.
    """
    k = len(powers); state = np.zeros(q.shape[:-1] + (1 << k,), dtype=np.longdouble)
    state[..., 0] = 1
    for rank in range(q.shape[-1]):
        previous = state.copy()
        for mask in range(1, 1 << k):
            for label, (a, b) in enumerate(powers):
                if mask & (1 << label):
                    state[..., mask] += previous[..., mask ^ (1 << label)] * q[..., rank]**a * nu[..., rank]**b
    return state[..., -1]


def injective_rank_average(q, nu, exponents, *, max_degree=5):
    """Average product(q[r_j]**a_j * nu[r_j]**b_j) over distinct ranks.

    q/nu may have leading bank axes and must share the last alphabet axis.
    ``exponents`` gives one nonzero (a,b) pair per distinct observed label.
    Its total degree is at most five; at most one q factor is allowed because
    only the distinguished first token comes from q. An empty product is one.

    Möbius inversion over set partitions enforces injectivity. For block B,
    use moment sum_r q[r]**sum_B(a) nu[r]**sum_B(b) with coefficient
    (-1)**(|B|-1) (|B|-1)!. Divide the final sum by (d)_k, not d**k.
    Nearly cancelling rows are recomputed by a nonnegative subset recurrence.
    """
    _require(type(max_degree) is int and 0 <= max_degree <= 5, 'Maximum total degree is five')
    powers = tuple(tuple(p) for p in exponents)
    _require(all(len(p) == 2 and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                                    and v >= 0 for v in p) and sum(p) > 0 for p in powers),
             'Each label needs nonnegative integer powers with positive total')
    _require(sum(a for a, _ in powers) <= 1 and sum(a+b for a, b in powers) <= max_degree,
             'Only the first token has a q factor; total degree exceeds the short-word limit')
    # Label order has no meaning after uniform injective averaging. Canonical
    # factor order also makes tail-permuted evaluations bitwise identical.
    powers = tuple(sorted(powers))
    q, nu = _laws(q, nu); d = q.shape[-1]; k = len(powers)
    _require(k <= d, 'More distinct labels than ranks')
    if not powers:
        return np.ones(q.shape[:-1], dtype=float)
    moments = {}; terms = []
    for partition in set_partitions(k):
        term = np.ones(q.shape[:-1], dtype=np.longdouble)
        for block in partition:
            a, b = (sum(powers[j][axis] for j in block) for axis in (0, 1))
            if (a, b) not in moments:
                moments[a, b] = np.sum(q**a * nu**b, axis=-1, dtype=np.longdouble)
            coefficient = (-1)**(len(block)-1) * math.factorial(len(block)-1)
            term = term * coefficient * moments[a, b]
        terms.append(term)
    stacked = np.stack(terms)
    value = np.sum(stacked, axis=0, dtype=np.longdouble)
    roundoff = (32 * len(terms) * np.finfo(np.longdouble).eps
                * np.sum(np.abs(stacked), axis=0, dtype=np.longdouble))
    uncertain = value <= roundoff
    if np.any(uncertain):
        # Only affected bank rows use the more expensive nonnegative recurrence.
        flat_q, flat_nu = q.reshape(-1, d), nu.reshape(-1, d)
        flat_value = value.reshape(-1).copy(); mask = np.asarray(uncertain).reshape(-1)
        flat_value[mask] = _stable_injective_sum(flat_q[mask], flat_nu[mask], powers)
        value = flat_value.reshape(value.shape)
    result = value / math.prod(range(d-k+1, d+1))
    _require(np.isfinite(result).all() and np.all(result >= 0), 'Invalid injective rank average')
    return np.asarray(result, dtype=float)


def stationary_iid_joint(tokens, q, nu):
    """Per-bank ordered-word evidence, with q for X1 and nu for X2 onward.

    Available word lengths are 0..5. A predictive ratio at t observations
    needs joints of lengths t and t+1 and must average over the latent bank
    *before* dividing. Token relabeling and permutations of the observed tail
    preserve this surrogate's evidence; moving the first token need not.
    """
    q_array = np.asarray(q)
    _require(q_array.ndim >= 1, 'q must have an alphabet axis')
    values = _tokens(tokens, q_array.shape[-1], 5)
    labels = list(dict.fromkeys(values)); powers = []
    for label in labels:
        powers.append((int(bool(values) and label == values[0]), values[1:].count(label)))
    return injective_rank_average(q, nu, powers)

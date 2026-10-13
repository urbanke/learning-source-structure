"""Optimal learning curves for two priors where the mixture is
computable directly, for the examples part of the appendix.

Uniform prior on the simplex: the mixture is the add-one
(Laplace) law, in closed form from the counts,
  log2 m(x^k) = log2 [ (d-1)! prod_a m_a! / (k+d-1)! ].
We draw p ~ Dirichlet(1,...,1), draw k samples, and record the
realized redundancy log2 p(x^k) - log2 m(x^k) on a k grid.

Zipf profile prior: the probability multiset is fixed (Zipf alpha),
only the assignment sigma of probabilities to symbols is random.
The mixture is the average over all d! assignments, evaluated by
a subset recursion for its permanent (d small). The expected curve
approaches the assignment entropy log2 d! as sample size increases.

    python code/scripts/necessary_examples.py --out results/mechanism/necessary_examples.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

LOG2 = math.log(2.0)


def kgrid(kmax: int, per_decade: int = 8) -> np.ndarray:
    g = np.unique(np.round(np.logspace(0, math.log10(kmax),
                                       int(per_decade * math.log10(kmax)) + 1)
                           ).astype(int))
    return g[g >= 1]


def uniform_prior_curve(d: int, kmax: int, draws: int, rng) -> dict:
    ks = kgrid(kmax)
    rho = np.zeros((draws, len(ks)))
    for r in range(draws):
        p = rng.dirichlet(np.ones(d))
        x = rng.choice(d, size=kmax, p=p)
        logp = np.log(p)
        cum_logp = np.concatenate([[0.0], np.cumsum(logp[x])])
        counts = np.zeros(d)
        prev = 0
        for j, k in enumerate(ks):
            counts += np.bincount(x[prev:k], minlength=d)
            prev = k
            # log m(x^k) for the Dirichlet(1) mixture
            logm = (gammaln(d) + gammaln(counts + 1).sum()
                    - gammaln(k + d))
            rho[r, j] = (cum_logp[k] - logm) / LOG2
    return {"d": d, "k": ks.tolist(),
            "rho_mean": rho.mean(0).tolist(),
            "rho_se": (rho.std(0, ddof=1) / math.sqrt(draws)).tolist(),
            "draws": draws, "se_ddof": 1}


def log_permutation_mixture(counts: np.ndarray, q: np.ndarray,
                            reference_logp: np.ndarray | None = None) -> np.ndarray:
    """Log sequence mixture over all assignments, by subset dynamic programming.

    Each entry is a permanent divided by d!. The subset recursion is exact
    algebraically and needs O(d 2**d) work per count vector, rather than d!.
    Rows of ``counts`` may contain any nonnegative integer counts. With
    ``reference_logp``, return log(mixture/reference) directly. This avoids
    subtracting two large sequence log probabilities near the entropy limit.
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    d = len(q)
    if counts.shape[1] != d:
        raise ValueError("one count per source probability is required")
    logq = np.log(q)
    reference = (np.zeros_like(counts) if reference_logp is None else
                 np.broadcast_to(reference_logp, counts.shape))
    dp = np.full((len(counts), 1 << d), -np.inf)
    dp[:, 0] = 0.0
    for mask in range(1, 1 << d):
        row = mask.bit_count() - 1
        choices = [j for j in range(d) if mask & (1 << j)]
        terms = np.stack([dp[:, mask ^ (1 << j)]
                          + counts[:, row] * (logq[j] - reference[:, row])
                          for j in choices])
        dp[:, mask] = logsumexp(terms, axis=0)
    return dp[:, -1] - math.lgamma(d + 1)


def profile_prior_curve(d: int, alpha: float, kmax: int, draws: int,
                        rng, per_decade: int = 8) -> dict:
    q = 1.0 / np.arange(1, d + 1) ** alpha
    q /= q.sum()
    ks = kgrid(kmax, per_decade)
    rho = np.zeros((draws, len(ks)))
    all_counts = np.zeros((draws, len(ks), d), dtype=np.int64)
    source_logp = np.zeros((draws, d))
    for r in range(draws):
        sigma = rng.permutation(d)
        p = q[sigma]                                   # true source
        x = rng.choice(d, size=kmax, p=p)
        source_logp[r] = np.log(p)
        counts = np.zeros(d, dtype=np.int64)
        prev = 0
        for j, k in enumerate(ks):
            counts += np.bincount(x[prev:k], minlength=d)
            prev = k
            all_counts[r, j] = counts
    # Keep the original source/sample RNG protocol, then evaluate in small
    # batches so the dynamic-programming table has bounded memory use.
    for start in range(0, draws, 32):
        stop = min(start + 32, draws)
        batch = all_counts[start:stop].reshape(-1, d)
        reference = np.repeat(source_logp[start:stop],len(ks),axis=0)
        logratio = log_permutation_mixture(batch, q, reference).reshape(stop-start, -1)
        rho[start:stop] = -logratio / LOG2
    return {"d": d, "alpha": alpha, "k": ks.tolist(),
            "rho_mean": rho.mean(0).tolist(),
            "rho_se": (rho.std(0, ddof=1) / math.sqrt(draws)).tolist(),
            "draws": draws,
            "se_ddof": 1,
            "plateau_log2_dfact": math.lgamma(d + 1) / LOG2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    out = {"uniform": [], "profile": [], "seed": 7,
           "provenance": "Regenerated with the original source/sample RNG protocol; permutation mixture evaluated by subset dynamic programming.",
           "se_ddof": 1}
    for d in ((4, 16) if args.smoke else (64, 1024, 16384)):
        print(f"uniform d={d}", flush=True)
        out["uniform"].append(uniform_prior_curve(
            d, 200 if args.smoke else 10 ** 6, 8 if args.smoke else 32, rng))
    for d in ((4, 5) if args.smoke else (6, 8)):
        print(f"profile d={d}", flush=True)
        out["profile"].append(
            profile_prior_curve(d, 1.0, 200 if args.smoke else 10 ** 5,
                                16 if args.smoke else 2048, rng,
                                per_decade=16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

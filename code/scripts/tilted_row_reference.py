"""Fixed-depth Bayes sequence probabilities for normalized tilted iid rows.

For known q and L, the row prior is P_b=q_b W_{L,b}/sum_c q_c W_{L,c}.
This evaluates its exact one-dimensional integral numerically. It does not
average or optimize over depths. For L>=2 it only reads a sealed LSA moment
store; it never provisions or builds columns. Outer quadrature and the q
shifts are implemented here, separately from the symmetric LSA evaluator.

Run --validate for bounded comparisons with the independent depth-one
adaptive integral and the production fixed-depth uniform-q code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp


LN2 = math.log(2.0)
DEFAULT_STORE = Path("/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod")
DEFAULT_LSA = Path("/Users/ruediger/Projects/layered-simplex-architecture/src")
MAX_ZERO_CACHE_BYTES = 64 * 1024**2


def configure_store(store: Path, lsa_source: Path = DEFAULT_LSA) -> None:
    """Set the existing production interpolation mode before importing LSA."""
    if not (store / "anchors.json").is_file() or not (store / "manifest.json").is_file():
        raise ValueError("A preexisting sealed anchor store is required; no table builds are permitted")
    expected = {
        "LSA_UNIVERSAL_TABLES": str(store), "LSA_PHI_LADDER_EVERY": "1",
        "LSA_PHI_LADDER_DEGREE": "11", "LSA_PHI_SADDLE_MIN_L": "54",
    }
    for name, value in expected.items():
        if "lsa.universal_tables" in sys.modules and os.environ.get(name) != value:
            raise RuntimeError("Configure the production store before importing lsa.universal_tables")
        os.environ[name] = value
    if str(lsa_source) not in sys.path:
        sys.path.insert(0, str(lsa_source))


class TiltedRowReference:
    """Numerical log Bayes probability for one ordered iid sequence.

    The input counts retain their symbol labels. The prior q and depth L
    are fixed for the object. The table and grid are reused across samples.
    Zero-count terms are cached only when they fit in the memory budget;
    larger alphabets use their sum and recompute occupied terms in batches.
    Diagnostics are numerical comparisons, not rigorous integration bounds.
    """

    def __init__(self, q, L: int, *, store: Path = DEFAULT_STORE,
                 step: float = 1 / 64, lower: float = -60., upper: float = 30.,
                 tail_drop: float = 45.,
                 zero_cache_bytes: int = MAX_ZERO_CACHE_BYTES,
                 batch_size: int = 32):
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or len(q) < 2 or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError("q must be a strictly positive probability vector")
        if not np.isclose(q.sum(), 1., rtol=1e-12, atol=1e-14):
            raise ValueError("q must sum to one")
        if not isinstance(L, int) or not 1 <= L < 54:
            raise ValueError("This reference supports fixed integer depths 1..53")
        if not isinstance(zero_cache_bytes, int) or not 0 <= zero_cache_bytes <= MAX_ZERO_CACHE_BYTES:
            raise ValueError("zero_cache_bytes must be between zero and 64 MiB")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be between one and 32")
        self.q = q / q.sum()
        self.logq = np.log(self.q)
        self.L, self.d = L, len(q)
        self.step, self.tail_drop = step, tail_drop
        self.batch_size = batch_size
        self.u = np.arange(lower, upper + step / 2, step)
        self.store = Path(store)
        self.tables = None
        if L >= 2:
            configure_store(self.store)
            from lsa.universal_tables import UniversalTables
            self.tables = UniversalTables(self.store, read_only=True)
        # Keep the whole matrix only for small alphabets. No d-by-grid
        # allocation is made once it exceeds the cache budget.
        cache_bytes = self.d * len(self.u) * np.dtype(float).itemsize
        self.zero = (np.empty((self.d, len(self.u)))
                     if cache_bytes <= zero_cache_bytes else None)
        self.zero_sum = np.zeros(len(self.u))
        for start in range(0, self.d, self.batch_size):
            stop = min(start + self.batch_size, self.d)
            grid = self.u[None, :] + self.logq[start:stop, None]
            values = self._log_phi(0, grid)
            if self.zero is not None:
                self.zero[start:stop] = values
            self.zero_sum += values.sum(axis=0)

    def _log_phi(self, r: int, u):
        u = np.asarray(u, dtype=float)
        if self.L == 1:
            return gammaln(r + 1.) - (r + 1.) * np.logaddexp(0., u)
        return self.tables.log_phi(self.L, int(r), u.ravel()).reshape(u.shape)

    def _kernel(self, counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
        u = self.u[indices]
        result = counts.sum() * u + self.zero_sum[indices]
        for r in np.unique(counts[counts > 0]):
            labels = np.flatnonzero(counts == r)
            for start in range(0, len(labels), self.batch_size):
                batch = labels[start:start + self.batch_size]
                shifted = u[None, :] + self.logq[batch, None]
                zero = (self._log_phi(0, shifted) if self.zero is None
                        else self.zero[np.ix_(batch, indices)])
                result += (self._log_phi(int(r), shifted) - zero).sum(axis=0)
        return result

    @staticmethod
    def _log_trapezoid(values: np.ndarray, step: float) -> float:
        weighted = values.copy()
        weighted[[0, -1]] -= LN2
        return float(logsumexp(weighted) + math.log(step))

    def log_probability(self, counts) -> tuple[float, dict]:
        """Return log sequence probability in nats and quadrature diagnostics."""
        counts = np.asarray(counts)
        if counts.shape != (self.d,) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("counts must be a nonnegative integer vector matching q")
        counts = counts.astype(np.int64)
        k = int(counts.sum())
        if k == 0:
            return 0., {"k": 0, "coarse_fine_difference_bits": 0.}
        stride = max(1, int(round(0.5 / self.step)))
        coarse_idx = np.unique(np.r_[np.arange(0, len(self.u), stride), len(self.u) - 1])
        coarse = self._kernel(counts, coarse_idx)
        peak = float(coarse.max())
        if max(coarse[0], coarse[-1]) > peak - self.tail_drop:
            raise RuntimeError("Outer integration grid is too short; increase its explicit limits")
        active = np.flatnonzero(coarse >= peak - self.tail_drop)
        left = coarse_idx[max(0, int(active[0]) - 2)]
        right = coarse_idx[min(len(coarse_idx) - 1, int(active[-1]) + 2)]
        # An even number of intervals allows a nested two-step comparison.
        right = min(right + (right - left) % 2, len(self.u) - 1)
        if (right - left) % 2:
            left -= 1
        indices = np.arange(left, right + 1)
        values = self._kernel(counts, indices)
        fine = self._log_trapezoid(values, self.step)
        coarse_integral = self._log_trapezoid(values[::2], 2 * self.step)
        result = float(counts @ self.logq - gammaln(k) + fine)
        diagnostics = {
            "k": k, "nonzero_counts": int(np.count_nonzero(counts)),
            "step": self.step, "integration_lower": float(self.u[left]),
            "integration_upper": float(self.u[right]),
            "mode_grid": float(self.u[indices[int(np.argmax(values))]]),
            "coarse_fine_difference_bits": abs(fine - coarse_integral) / LN2,
            "left_endpoint_drop_nats": float(values.max() - values[0]),
            "right_endpoint_drop_nats": float(values.max() - values[-1]),
            "zero_count_cache_bytes": 0 if self.zero is None else self.zero.nbytes,
            "batch_size": self.batch_size,
            "qualification": "Grid refinement and endpoint drops are checks, not certified total error bounds.",
        }
        if diagnostics["coarse_fine_difference_bits"] > 2e-5:
            raise RuntimeError(f"Outer quadrature needs a smaller step: {diagnostics}")
        return result, diagnostics

    def close(self) -> None:
        if self.tables is not None:
            self.tables.close()


def zipf(d: int, alpha: float) -> np.ndarray:
    q = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return q / q.sum()


def validate(store: Path) -> dict:
    configure_store(store)
    os.environ["LSA_NO_TRUNCATE"] = "1"
    from context_row_check import log_mixture_adaptive
    from lsa.codelength import depth_averaged_codelength_profiles
    from lsa.codelength import profile_of

    start = time.monotonic()
    rng = np.random.default_rng(20260906)
    depth_one = []
    for d, alpha in [(3, .75), (16, 1.), (64, 1.5)]:
        q = zipf(d, alpha)
        reference = TiltedRowReference(q, 1, store=store)
        for k in [1, 16, 256]:
            counts = rng.multinomial(k, q)
            got, diagnostics = reference.log_probability(counts)
            expected, adaptive_diagnostics = log_mixture_adaptive(counts, q)
            depth_one.append({"d": d, "alpha": alpha, "k": k,
                              "difference_bits": (got - expected) / LN2,
                              "diagnostics": diagnostics,
                              "adaptive_diagnostics": adaptive_diagnostics})
        reference.close()

    uniform_three = []
    for d in [16, 64, 256]:
        q = np.full(d, 1 / d)
        reference = TiltedRowReference(q, 3, store=store)
        cases = []
        for k in [1, 16, 256, 4096]:
            weights = rng.exponential(size=(3, d)).prod(axis=0)
            counts = rng.multinomial(k, weights / weights.sum())
            got, diagnostics = reference.log_probability(counts)
            cases.append((counts, got, diagnostics))
        profiles = {tuple(profile_of(counts)): tuple(profile_of(counts))
                    for counts, _, _ in cases}
        evaluated = depth_averaged_codelength_profiles(
            profiles, d=d, l_max=3, jobs=1, tables_source="universal", universal_path=str(store))
        for counts, got, diagnostics in cases:
            expected = evaluated[tuple(profile_of(counts))].log2_q_by_depth[2] * LN2
            uniform_three.append({"d": d, "k": int(counts.sum()),
                                  "reference_log_probability_bits": got / LN2,
                                  "production_log_probability_bits": expected / LN2,
                                  "difference_bits": (got - expected) / LN2,
                                  "diagnostics": diagnostics})
        reference.close()

    tilted_three = []
    q = np.array([.6, .3, .1])
    reference = TiltedRowReference(q, 3, store=store)
    finer = TiltedRowReference(q, 3, store=store, step=1 / 128, lower=-68., upper=34.)
    for k in [1, 2, 4]:
        mass, mass_finer = 0., 0.
        differences = []
        for i in range(k + 1):
            for j in range(k - i + 1):
                counts = np.array([i, j, k - i - j])
                value, _ = reference.log_probability(counts)
                refined, _ = finer.log_probability(counts)
                log_multiplicity = gammaln(k + 1) - gammaln(counts + 1).sum()
                mass += math.exp(log_multiplicity + value)
                mass_finer += math.exp(log_multiplicity + refined)
                differences.append(abs(value - refined) / LN2)
        tilted_three.append({"d": 3, "q": q.tolist(), "L": 3, "k": k,
                             "total_count_probability": mass,
                             "refined_total_count_probability": mass_finer,
                             "maximum_refinement_difference_bits": max(differences),
                             "refinement": "step1/64 to1/128 and outer limits[-60,30] to[-68,34]"})
    reference.close()
    finer.close()

    assert max(abs(row["difference_bits"]) for row in depth_one) < 1e-7
    assert max(abs(row["difference_bits"]) for row in uniform_three) < 1e-7
    assert max(abs(row["refined_total_count_probability"] - 1.) for row in tilted_three) < 1e-7
    assert max(row["maximum_refinement_difference_bits"] for row in tilted_three) < 1e-7

    return {"seed": 20260906, "code": "Fixed-depth normalized tilted Bayes row code; q and L known.",
            "store": str(store),
            "anchor_manifest_sha256": hashlib.sha256((store / "anchors.json").read_bytes()).hexdigest(),
            "depth_one_adaptive_comparison": depth_one,
            "depth_three_uniform_comparison": uniform_three,
            "depth_three_tilted_normalization": tilted_three,
            "elapsed_seconds": time.monotonic() - start,
            "scope": "Bounded evaluator validation; no chain or crossing simulation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path,
                        default=Path("results/mechanism/tilted_row_reference_validation.json"))
    args = parser.parse_args()
    if not args.validate:
        parser.error("Use --validate, or import TiltedRowReference for a bounded row experiment")
    result = validate(args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"depth_one_max_difference_bits": max(abs(x["difference_bits"]) for x in result["depth_one_adaptive_comparison"]),
                      "depth_three_uniform_max_difference_bits": max(abs(x["difference_bits"]) for x in result["depth_three_uniform_comparison"]),
                      "depth_three_tilted_max_mass_error": max(abs(x["refined_total_count_probability"] - 1.) for x in result["depth_three_tilted_normalization"]),
                      "depth_three_tilted_max_refinement_bits": max(x["maximum_refinement_difference_bits"] for x in result["depth_three_tilted_normalization"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

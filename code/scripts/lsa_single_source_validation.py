"""Compare deterministic single-distribution LSA costs with saved rho curves.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/lsa_single_source_validation.py

Recreates the saved 32 source distributions, but draws no tokens or counts.
The saved August 27 curves used c-kH(p) and ddof=0 standard errors. Their
one-token means and standard errors independently check the recreated sources.
The full depth mixture is evaluated by lsa_expected_profile.fractional_cost;
mean multiplicities and their covariance use exact finite count formulas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from lsa_expected_profile import fractional_cost
from lsa_single_covariance import count_multiplicity_moments
from tilted_row_reference import DEFAULT_LSA, DEFAULT_STORE, configure_store

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_COMMIT = "66121b4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CachedMoments:
    """Reuse the same quadrature kernels across fixed-d,k source rows."""

    def __init__(self, tables):
        self.tables = tables
        self.values = {}

    def log_phi_matrix(self, depth, counts, grid):
        key = (depth, tuple(counts), grid.shape, grid.tobytes())
        if key not in self.values:
            self.values[key] = self.tables.log_phi_matrix(depth, counts, grid)
        return self.values[key]


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()),
                source_only_SE=float(values.std(ddof=1)/math.sqrt(len(values))),
                source_only_SE_ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=[.75, 1., 1.25])
    parser.add_argument("--ks", type=int, nargs="+", default=[67, 112])
    parser.add_argument("--step", type=float, default=.025)
    parser.add_argument("--stop-after-seconds", type=float, default=900.,
                        help="Stop after completing an alpha if this runtime is exceeded")
    parser.add_argument("--out", type=Path,
                        default=ROOT/"results/mechanism/lsa_single_source_validation.json")
    args = parser.parse_args()
    configure_store(DEFAULT_STORE)
    from lsa.codelength import default_l_max
    from lsa.universal_tables import UniversalTables
    from mechanism_quantities import entropy_bits, sample_row
    from mech_common import zipf

    started = time.monotonic()
    tables = UniversalTables(DEFAULT_STORE, read_only=True)
    result = dict(schema_version=1, complete=False,
        target="Actual depth-mixture LSA learning one iid distribution",
        requested_alphas=args.alphas, requested_sample_sizes=args.ks,
        source_protocol="numpy.default_rng([row_index,d,L,int(alpha*100),23])",
        historical_producer_commit=HISTORICAL_COMMIT,
        historical_script="code/scripts/mechanism_quantities.py",
        saved_SE_ddof=0,
        saved_SE_provenance="Historical producer at66121b4 uses R.std(0)/sqrt(rows); saved JSON omits ddof.",
        producer_sha256=sha(__file__),
        profile_helper_sha256=sha(Path(__file__).with_name("lsa_expected_profile.py")),
        covariance_helper_sha256=sha(Path(__file__).with_name("lsa_single_covariance.py")),
        source_sampler_sha256=sha(DEFAULT_LSA/"lsa/product_simplex.py"),
        current_source_driver_sha256=sha(Path(__file__).with_name("mechanism_quantities.py")),
        moment_anchor_sha256=sha(DEFAULT_STORE/"anchors.json"),
        command_argv=sys.argv, sources=[], comparisons=[],
        qualifications=[
            "No new token or count simulations; the 32 original sources are recreated from seeds.",
            "Full normalized source probabilities are retained; no replacement by q or an effective alphabet.",
            "The approximation includes the complete finite coding-depth mixture inside the logarithm.",
            "The curvature correction is not a proved error bound.",
            "Prediction SE measures variation across source distributions only.",
            "Saved rho SE includes source and token-sample variation and originally used ddof=0.",
            "Saved per-source simulated costs are absent; no paired difference SE can be computed.",
            "Nested quadrature comparisons are numerical diagnostics, not certified integration error bounds.",
        ])

    def save():
        result["elapsed_seconds"] = time.monotonic()-started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    for alpha in args.alphas:
        path = ROOT/f"results/mechanism/deep/rho_d256_a{str(alpha)}_L3_deep.json"
        saved = json.loads(path.read_text())
        d, source_depth, rows = saved["d"], saved["L"], saved["rows"]
        assert d == 256 and source_depth == 3 and rows == 32
        coding_depth = default_l_max(d)
        assert coding_depth == saved["l_max"]
        q = zipf(d, alpha)
        probabilities = [sample_row(d, source_depth,
            np.random.default_rng([row, d, source_depth, int(alpha*100), 23]), q)
            for row in range(rows)]
        entropies = np.asarray([entropy_bits(p) for p in probabilities])
        first_index = saved["k"].index(1)
        one_token = math.log2(d)-entropies
        fingerprint = dict(mean_error=float(one_token.mean()-saved["rho_mean"][first_index]),
            SE_error=float(one_token.std(ddof=0)/math.sqrt(rows)-saved["rho_se"][first_index]))
        assert max(abs(value) for value in fingerprint.values()) < 1e-10, fingerprint
        result["sources"].append(dict(d=d, alpha=alpha, source_depth=source_depth,
            coding_depth=coding_depth, rows=rows,
            saved_path=str(path.relative_to(ROOT)), saved_sha256=sha(path),
            one_token_fingerprint=fingerprint,
            source_probabilities=[p.tolist() for p in probabilities],
            source_entropy_bits=entropies.tolist()))
        for k in args.ks:
            checkpoint = saved["k"].index(k)
            cache = CachedMoments(tables)
            records = []
            for row_index, (p, entropy) in enumerate(zip(probabilities, entropies)):
                row_started = time.monotonic()
                moments = count_multiplicity_moments(p, k)
                evaluation = fractional_cost(moments.mean, k, coding_depth, cache,
                    step=args.step, covariance=moments.covariance)
                evaluation.pop("projected_gradient_bits")
                records.append(dict(row_index=row_index,
                    predicted_codelength_bits=evaluation["corrected_bits"],
                    predicted_excess_bits=evaluation["corrected_bits"]-k*float(entropy),
                    upper_excess_bits=evaluation["jensen_bits"]-k*float(entropy),
                    evaluation=evaluation, covariance_diagnostics=moments.diagnostics,
                    elapsed_seconds=time.monotonic()-row_started))
                if row_index == 0 or (row_index+1) % 8 == 0:
                    print(f"alpha={alpha:g} k={k}: {row_index+1}/{rows} source rows; "
                          f"last row {records[-1]['elapsed_seconds']:.2f}s", flush=True)
            predicted = summary([row["predicted_excess_bits"] for row in records])
            upper = summary([row["upper_excess_bits"] for row in records])
            result["comparisons"].append(dict(d=d, alpha=alpha, source_depth=source_depth,
                k=k, rows=rows, coding_depth=coding_depth, prediction=predicted,
                uncorrected_upper=upper,
                saved_rho_mean=float(saved["rho_mean"][checkpoint]),
                saved_rho_SE=float(saved["rho_se"][checkpoint]), saved_rho_SE_ddof=0,
                saved_rho_SE_converted_to_ddof1=float(saved["rho_se"][checkpoint]*math.sqrt(rows/(rows-1))),
                predicted_minus_saved_mean_bits=predicted["mean"]-saved["rho_mean"][checkpoint],
                per_source=records))
            save()
            print(json.dumps({key:value for key,value in result['comparisons'][-1].items()
                              if key != 'per_source'}), flush=True)
        if time.monotonic()-started > args.stop_after_seconds:
            break
    result["complete"] = len(result["comparisons"]) == len(args.alphas)*len(args.ks)
    save()
    tables.close()


if __name__ == "__main__":
    main()

"""Grid, bounds, normalization and sequential checks on saved mixture tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from mixture_family_evidence import FamilyEvidence, sha256
from lsa_seq import SeqLSA
from shared_depth import SharedDepthLSA, ProductSharedDepthLSA


def validate(heldout):
    started = time.monotonic()
    refs = {name: [FamilyEvidence(name), FamilyEvidence(name, bounds=(-36, 36)),
                   FamilyEvidence(name, bounds=(-36, 36), step=1 / 128)]
            for name in ("product", "uniform_base")}
    worst, sequential_checks, normalization = {}, [], []
    for truth in ("product", "shared_base", "uniform_base"):
        with np.load(heldout / f"{truth}_prior.npz", allow_pickle=False) as saved:
            tokens, prefixes = saved["tokens"][:4], saved["prefixes"]
        for name, variants in refs.items():
            for i, x in enumerate(tokens):
                values = np.array([[q.log_joint(x[:t + 1]) for t in prefixes] for q in variants])
                errors = (np.abs(values[1:] - values[:-1]) / np.log(2)).max(axis=1)
                for key, error in zip(("bounds", "grid"), errors):
                    if error > worst.get(key, {}).get("bits", -1):
                        worst[key] = dict(bits=float(error), truth=truth, reference=name, sequence=i)
            x, reference = tokens[0, :16], variants[0]
            parent = reference.log_joint(x)
            probability = np.array([np.exp(reference.log_joint(np.r_[x, b]) - parent) for b in range(64)])
            normalization.append(dict(truth=truth, reference=name, length=16,
                                      sum_error=float(abs(probability.sum() - 1))))
            # The independent sequential method updates log integrands after
            # every observed transition, rather than rebuilding count profiles.
            x = tokens[0, :65]
            depths = list(range(1, 9)) if name == "product" else list(range(9))
            dimension = 8 if name == "product" else 64
            kernel = SeqLSA(dimension, depths=depths, rmax=65, u_lo=-30, u_hi=30)
            sequential = (ProductSharedDepthLSA(SharedDepthLSA(kernel), SharedDepthLSA(kernel))
                          if name == "product" else SharedDepthLSA(kernel))
            got = sequential.run(x)["log_evidence"][-1]
            expected = variants[0].log_joint(x)
            sequential_checks.append(dict(truth=truth, reference=name, length=65,
                                          abs_error_bits=float(abs(got - expected) / np.log(2))))
    passed = (max(v["bits"] for v in worst.values()) < 1e-6
              and max(v["sum_error"] for v in normalization) < 1e-8
              and max(v["abs_error_bits"] for v in sequential_checks) < 1e-6)
    return dict(schema="mixture-family-evidence-validation-v1", passed=passed,
                refinement=worst, normalization=normalization, batch_vs_sequential=sequential_checks,
                n_prior_sequences_per_cohort=4, prefixes=prefixes.tolist(),
                elapsed_seconds=time.monotonic() - started,
                code_sha256={name: sha256(ROOT / "code/transformer" / name) for name in
                             ("mixture_family_evidence.py", "validate_shared_depth.py", "phi_source.py",
                              "shared_depth.py", "lsa_seq.py")},
                source_sha256={name: sha256(heldout / f"{name}_prior.npz") for name in
                               ("product", "shared_base", "uniform_base")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new receipt path; existing validated results are immutable")
    receipt = validate(args.heldout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=receipt["passed"], refinement=receipt["refinement"],
                          elapsed_seconds=receipt["elapsed_seconds"])), flush=True)
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()

"""Bounded experimental timing/calibration of shared-base evidence SMC.

This is not a scientific-reference acceptance procedure. It reuses one
frozen held-out sequence; source parameters never initialize inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_evidence_smc import estimate_log_evidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_short_evidence import SharedBaseShortEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "output/multifamily_results_2026-09-16/heldout/shared_base_prior.npz")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[5, 17, 65, 1025])
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--temperatures", type=int, default=16)
    parser.add_argument("--moves", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--alpha-order", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=170)
    parser.add_argument("--store", type=Path, default=Path("/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod"))
    parser.add_argument("--src", type=Path, default=Path("/Users/ruediger/Projects/product_model_with_memory/src"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to replace an earlier experimental benchmark")
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("bounded benchmark time limit reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    with np.load(args.cohort, allow_pickle=False) as source:
        tokens = source["tokens"][args.sequence_index].copy()
        identity = source["sequence_seed"][args.sequence_index].tolist()
    nodes, weights = leggauss(args.alpha_order)
    kernel = RankedLSARowEvidence(64, 1+nodes/2, store=args.store, src=args.src,
                                  term_cache_size=1024, evidence_cache_size=100000)
    sampler = SharedBaseMCMC(64, kernel, weights/2)
    short = SharedBaseShortEvidence()
    report = dict(schema="shared-base-evidence-smc-experimental-benchmark-v1",
                  scientific_reference_validated=False, sequence_index=args.sequence_index,
                  sequence_seed=identity, prefixes=args.prefixes, particles=args.particles,
                  temperatures=args.temperatures, moves=args.moves, alpha_order=args.alpha_order,
                  replicates=args.replicates, records=[], summaries=[], stopped_by_time_limit=False,
                  cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  smc_sha256=hashlib.sha256((ROOT / "code/transformer/shared_base_evidence_smc.py").read_bytes()).hexdigest())
    current = None
    try:
        for length in args.prefixes:
            if length > len(tokens):
                raise ValueError("requested prefix exceeds saved sequence length")
            results = []
            for replica in range(args.replicates):
                current = dict(n_tokens=length, replica=replica)
                result = estimate_log_evidence(sampler, tokens[:length], particles=args.particles,
                           temperatures=args.temperatures, moves=args.moves,
                           seed=2026091620+1000*length+replica)
                result["replica"] = replica
                if length <= 5:
                    exact = short.log_joint(tokens[:length])
                    result["catalog_log_evidence"] = exact
                    result["estimate_minus_catalog_loss_bits"] = (exact-result["log_evidence"])/math.log(2)
                report["records"].append(result)
                results.append(result)
                print(json.dumps(dict(n_tokens=length, replica=replica,
                      seconds=result["elapsed_seconds"], log_evidence=result["log_evidence"],
                      min_ess=min(result["effective_sample_sizes"]),
                      catalog_loss_error_bits=result.get("estimate_minus_catalog_loss_bits"))), flush=True)
            losses = -np.array([result["log_evidence"] for result in results])/math.log(2)
            report["summaries"].append(dict(n_tokens=length,
                mean_loss_bits=float(losses.mean()), range_loss_bits=float(np.ptp(losses)),
                independent_replica_se_bits=float(losses.std(ddof=1)/np.sqrt(len(losses))) if len(losses)>1 else None))
    except TimeoutError:
        report["stopped_by_time_limit"] = True
        report["interrupted_calculation"] = current
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["note"] = "Timing and limited calibration only. No long-context accuracy claim follows from agreement between a few replicas."
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(out=str(args.out), elapsed_seconds=report["elapsed_seconds"],
                              stopped_by_time_limit=report["stopped_by_time_limit"])), flush=True)


if __name__ == "__main__":
    main()

"""Bounded complete-SMC GPU benchmark on declared short shared-base controls.

CUDA is mandatory when --device cuda is requested; no fallback is allowed.
Table preparation, device transfer, original CPU sampling, batched CPU
sampling and batched requested-device sampling are timed separately. This
does not implement longer sequences or certify a production reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from shared_base_evidence_gpu import (MomentTables, ShortTarget, checked_device,
                                      estimate_log_evidence, row_descriptors, synchronize)
from shared_base_evidence_smc import estimate_log_evidence as cpu_estimate
from shared_base_mcmc import SharedBaseMCMC, PrefixPosterior
from shared_base_short_evidence import SharedBaseShortEvidence
from ranked_lsa import RankedLSARowEvidence
from calibrate_shared_base_evidence import (atomic_json, file_hash, normalizer_summary,
                                           numerical_environment, utc_now)
from prepare_shared_base_cluster import verify_dependency


def load_cases(args, catalog):
    if args.cases is not None:
        data = json.loads(args.cases.read_text())
        if data.get("schema") not in ("shared-base-gpu-cases-v1", "shared-base-evidence-calibration-cases-v1"):
            raise ValueError("unrecognized cases file")
        cases = [dict(id=str(c["id"]), name=c.get("name", str(c["id"])),
                      tokens=list(c["tokens"] if data["schema"] == "shared-base-gpu-cases-v1" else c["tokens"][:5]),
                      declared_log_evidence=c.get("known_log_evidence")) for c in data["cases"]]
    else:
        cases = [dict(id=f"case_{i}", name=value, tokens=[int(x) for x in value.split(",")],
                      declared_log_evidence=None) for i, value in enumerate(args.case)]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("requires nonempty cases with unique ids")
    for c in cases:
        row_descriptors(c["tokens"], 64)
        exact = catalog.log_joint(c["tokens"])
        if c["declared_log_evidence"] is not None and abs(c["declared_log_evidence"]-exact) > 1e-10:
            raise ValueError("declared case likelihood disagrees with validated catalog")
        c["known_log_evidence"] = exact
    return cases


def table_hash(tables):
    digest = hashlib.sha256()
    for key, value in sorted(dict(base=tables.base_laws, alpha=tables.alpha_weights,
                                 depth=tables.depth_weights, **tables.moments).items()):
        a = np.ascontiguousarray(value, dtype="<f8")
        digest.update(key.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def validate_target(sampler, tables, tokens, device, count=24):
    """Same fixed states: original kernel, batched CPU, requested device."""
    rng = np.random.default_rng(2026091624)
    ranks = np.stack([rng.permutation(sampler.d) for _ in range(count)])
    ai = np.arange(count) % len(sampler.alpha_weights)
    original = PrefixPosterior(sampler, tokens)
    expected_prior, expected_likelihood = [], []
    for a, r in zip(ai, ranks):
        s = original.evaluate(int(a), r)
        first = sampler.kernel.log_base_laws[a, r[tokens[0]]]
        expected_prior.append(first+sampler.log_alpha_weights[a])
        expected_likelihood.append(s.log_likelihood-first)
    report, targets = {}, {}
    for name, where in (("cpu", "cpu"), ("requested_device", device)):
        started = time.monotonic()
        target = ShortTarget(tables, tokens, where)
        synchronize(target.device)
        transfer = time.monotonic()-started
        p, ll = target.terms(torch.tensor(ai, device=target.device),
                             torch.tensor(ranks, device=target.device))
        error = max(float(np.max(abs(p.cpu().numpy()-expected_prior))),
                    float(np.max(abs(ll.cpu().numpy()-expected_likelihood))))
        if not np.isfinite(error) or error > 1e-9:
            raise ValueError(f"batched {name} target differs from original: {error}")
        report[name] = dict(max_absolute_log_error=error, transfer_and_target_setup_seconds=transfer)
        targets[name] = target
    report.update(states=count, passed=True, comparison="original weighted-LSA kernel on identical states")
    return report, targets


def summaries(records, cases):
    result = []
    for case in cases:
        for implementation in sorted({r["implementation"] for r in records}):
            selected = [r for r in records if r["case_id"] == case["id"]
                        and r["implementation"] == implementation]
            if not selected:
                continue
            result.append(dict(case_id=case["id"], implementation=implementation,
                full_sampler_seconds=[r["result"]["elapsed_seconds"] for r in selected],
                statistics=normalizer_summary([r["result"]["log_evidence"] for r in selected],
                                             case["known_log_evidence"]),
                scientific_reference_validated=False))
    return result


def execute(args):
    device = checked_device(args.device)  # fail before expensive setup if CUDA is absent
    if args.out.exists():
        raise FileExistsError("benchmark output exists; choose a new output path")
    torch.set_num_threads(args.cpu_threads)
    setup_started = time.monotonic()
    environment = numerical_environment()
    dependency_root = args.src.resolve().parent
    if args.store.resolve() != dependency_root / "tables/anchors_prod":
        raise ValueError("store and src must come from the same verified dependency root")
    verify_dependency(dependency_root, args.dependency_manifest)
    catalog = SharedBaseShortEvidence()
    cases = load_cases(args, catalog)
    nodes, weights = leggauss(args.alpha_order)
    kernel = RankedLSARowEvidence(64, 1+nodes/2, store=args.store, src=args.src,
                                 u_lo=-32, u_hi=32, step=1/32,
                                 term_cache_size=1024, evidence_cache_size=100000)
    backend = kernel._get_backend()
    if not backend.read_only or not backend.sealed:
        raise ValueError("requires the original sealed store opened read-only")
    if not sys.modules["product_model_with_memory.kernel"].available:
        raise ValueError("compiled CPU interpolation kernel unavailable; build the verified dependency first")
    sampler = SharedBaseMCMC(64, kernel, weights/2)
    prepare_started = time.monotonic()
    tables = MomentTables.from_sampler(sampler)
    prep_seconds = time.monotonic()-prepare_started
    parity, targets = {}, {}
    for case in cases:
        parity[case["id"]], targets[case["id"]] = validate_target(sampler, tables, case["tokens"], device)
    code_files = [Path(__file__), ROOT / "code/transformer/shared_base_evidence_gpu.py",
                  ROOT / "code/scripts/shortprefix_shared_base_exact.py",
                  ROOT / "code/scripts/calibrate_shared_base_evidence.py",
                  ROOT / "code/scripts/prepare_shared_base_cluster.py"]
    code_files += [ROOT / "code/transformer" / name for name in
                  ("ranked_lsa.py", "weighted_lsa.py", "shared_base_mcmc.py", "shared_base_evidence_smc.py")]
    report = dict(schema="shared-base-gpu-benchmark-v1", scientific_reference_validated=False,
        scope="six declared short controls; no long-context or population reference", created_utc=utc_now(),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        torch_version=torch.__version__, dtype="float64", cpu_threads=args.cpu_threads,
        configuration=dict(particles=args.particles, temperatures=args.temperatures, moves=args.moves,
                           replicates=args.replicates, cpu_replicates=args.cpu_replicates,
                           torch_cpu_replicates=args.torch_cpu_replicates, alpha_order=args.alpha_order,
                           seed=args.seed, sampling_seconds_limit=args.seconds),
        cases=cases, cases_sha256=file_hash(args.cases) if args.cases else None,
        dependency_manifest_sha256=file_hash(args.dependency_manifest),
        code_sha256={str(p): file_hash(p) for p in code_files},
        sealed_store_sha256={p.name: file_hash(p) for p in sorted(args.store.iterdir())
                             if p.is_file() and (p.suffix == ".bin" or p.name.endswith(".json"))},
        reference_validation=catalog.validation, numerical_environment=environment,
        row_moment_validation=tables.validation, moment_tables_sha256=table_hash(tables),
        store_sealed=True, store_read_only=True, compiled_cpu_kernel_available=True,
        row_moment_preparation_seconds=prep_seconds, total_setup_seconds=time.monotonic()-setup_started,
        target_parity=parity, records=[], status="prepared",
        timing_scope="Full sampler includes initialization, weighting, resampling, MH, diagnostics and device synchronization. Moment preparation and transfers are separate. Compare torch CPU versus torch CUDA to isolate hardware; original CPU additionally differs in short-row precomputation and diagnostics. Original CPU uses a warm quadrature/row cache after parity checks; no long-context speedup is implied.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError("bounded benchmark sampling limit")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        max_reps = max(args.replicates, args.cpu_replicates, args.torch_cpu_replicates)
        for rep in range(max_reps):
            for ci, case in enumerate(cases):
                for implementation, count in (("torch_requested_device", args.replicates),
                                               ("torch_cpu", args.torch_cpu_replicates),
                                               ("original_cpu", args.cpu_replicates)):
                    if rep >= count:
                        continue
                    # Unique independent repetitions per case/implementation.
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}|{case['id']}|{rep}|{implementation}".encode()).digest()[:8], "little") % (2**63-1)
                    report["active"] = dict(case_id=case["id"], replica=rep, implementation=implementation)
                    report["status"] = "running"
                    atomic_json(args.out, report)
                    if any(file_hash(path) != expected for path, expected in report["code_sha256"].items()):
                        raise RuntimeError("benchmark or numerical code changed during the run")
                    options = dict(particles=args.particles, temperatures=args.temperatures, moves=args.moves, seed=seed)
                    if implementation == "original_cpu":
                        result = cpu_estimate(sampler, case["tokens"], **options)
                    else:
                        key = "cpu" if implementation == "torch_cpu" else "requested_device"
                        result = estimate_log_evidence(targets[case["id"]][key], **options)
                    report["records"].append(dict(case_id=case["id"], replica=rep,
                                                  implementation=implementation, result=result))
                    report["summary"] = summaries(report["records"], cases)
                    atomic_json(args.out, report)
                    print(json.dumps(dict(case=case["id"], replica=rep, implementation=implementation,
                                          seconds=result["elapsed_seconds"], log_evidence=result["log_evidence"])), flush=True)
        report["status"] = "completed experimental benchmark; accuracy requires independent review"
        report["active"] = None
    except TimeoutError:
        report["status"] = "partial: time limit; active repetition not included"
    except BaseException as error:
        report["status"] = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        report["sampling_elapsed_seconds"] = time.monotonic()-started
        report["summary"] = summaries(report["records"], cases)
        atomic_json(args.out, report)
    print(json.dumps(dict(out=str(args.out), status=report["status"])), flush=True)
    return 75 if report["status"].startswith("partial:") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--cases", type=Path)
    selected.add_argument("--case", action="append", help="comma-separated token string; repeat for more cases")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path,
                        default=ROOT / "code/transformer/configs/shared_base_dependency_manifest_2026-09-16.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--temperatures", type=int, default=64)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--cpu-replicates", type=int, default=1)
    parser.add_argument("--torch-cpu-replicates", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--alpha-order", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026091603)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if (args.particles < 2 or min(args.temperatures, args.moves, args.replicates,
            args.alpha_order, args.seconds, args.cpu_threads) < 1
            or min(args.cpu_replicates, args.torch_cpu_replicates, args.seed) < 0):
        parser.error("invalid numerical or timing setting")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()

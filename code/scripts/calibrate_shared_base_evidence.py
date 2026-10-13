"""Bounded, resumable calibration of shared-base evidence estimates.

Run one explicitly selected length and one sampler configuration per file.
This estimates the SAME full shared-base prior on a few fixed token strings,
not average performance over a source family. It never advances to a longer
length or declares the reference accurate. Independent repetitions are
averaged as likelihoods BEFORE taking logarithms. Likelihood-mean intervals
are approximate Monte Carlo diagnostics, not protection against missed modes
or inaccurate quadrature. Existing sealed-store row kernels are reused.

Examples (set --protocol to the predeclared calibration plan):
  python code/scripts/calibrate_shared_base_evidence.py --sampler original \
      --length 5 --particles 64 --temperatures 16 --moves 2 --replicates 16 \
      --families product shared_base uniform_base --out output/calibration/original.json
  python code/scripts/calibrate_shared_base_evidence.py --sampler row-block \
      --pattern 0 1 0 1 0 --out output/calibration/repeated_context.json

Reusing --out resumes missing repetitions only when all numerical settings,
code, source tokens, reference catalog and sealed-store identity are unchanged.
--seconds limits sampling after input/provenance setup and may increase on
resume. Setup time is reported separately. No within-SMC
checkpoint is claimed: an interrupted repetition restarts with the same seed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import t as student_t
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from ranked_lsa import RankedLSARowEvidence
from shared_base_evidence_smc import estimate_log_evidence
from shared_base_efficiency import SharedBaseEfficiencyRowMCMC
from shared_base_mcmc import SharedBaseMCMC
from shared_base_short_evidence import SharedBaseShortEvidence

FAMILIES = ("product", "shared_base", "uniform_base")
SCHEMA = "shared-base-evidence-calibration-v1"
SEED_NAMESPACE = "shared-base-evidence-calibration-replicates-v1"
CODE_FILES = ("shared_base_evidence_smc.py", "shared_base_mcmc.py", "shared_base_surrogate.py",
              "shared_base_efficiency.py", "ranked_lsa.py", "weighted_lsa.py", "phi_source.py",
              "lsa_seq.py", "shared_base_short_evidence.py")


def numerical_environment():
    for name in ("PMM_PHI_BIAS", "PMM_PHI_WAVE"):
        value = float(os.environ.get(name, "0") or 0.)
        if not np.isfinite(value) or value != 0:
            raise ValueError(f"calibration forbids numerical perturbation {name}")
    names = ("PMM_PHI_BIAS", "PMM_PHI_WAVE", "PMM_PHI_WAVE_SCALE", "PMM_PHI_LADDER",
             "PMM_PHI_LADDER_EVERY", "PMM_PHI_LADDER_DEGREE", "PMM_PHI_SADDLE_MIN_L",
             "PMM_SERIES_TAIL", "PMM_KERNEL", "PMM_INTERP_KERNEL", "PMM_BUILD_EXACT", "PMM_UNIVERSAL_TABLES")
    return {name: os.environ.get(name) for name in names}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def output_lock(path):
    with Path(str(path) + ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process is writing this calibration file") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def replicate_seed(sequence_id, replica, master_seed=2026091603):
    """The same fixed token string uses the same seed across sampler budgets."""
    value = canonical([SEED_NAMESPACE, int(master_seed), sequence_id, int(replica)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def normalizer_summary(log_estimates, exact_log_evidence=None):
    """Stable arithmetic mean of likelihoods, and approximate uncertainty.

    The t interval is formed for the mean of scaled positive likelihood
    estimates and then transformed monotonically into a loss interval.
    A nonpositive lower likelihood endpoint gives an unbounded upper loss.
    """
    logs = np.asarray(log_estimates, dtype=np.float64)
    if logs.ndim != 1 or len(logs) == 0 or not np.isfinite(logs).all():
        raise ValueError("need a nonempty finite vector of log likelihood estimates")
    count = len(logs)
    shift = float(logs.max())
    scaled = np.exp(logs - shift)
    mean = float(scaled.mean())
    log_mean = float(shift + np.log(mean))
    loss = -log_mean / np.log(2)
    result = dict(repetitions=count, log_mean_evidence_nats=log_mean,
                  loss_of_mean_evidence_bits=float(loss),
                  likelihood_scale_log_nats=shift, mean_scaled_likelihood=mean,
                  arithmetic_mean_replicate_loss_bits_diagnostic_only=float(-logs.mean() / np.log(2)),
                  replicate_loss_range_bits=float(np.ptp(logs) / np.log(2)),
                  effective_replica_count=float(scaled.sum() ** 2 / (scaled @ scaled)),
                  maximum_single_replica_likelihood_share=float(scaled.max() / scaled.sum()),
                  relative_mc_se=None, loss_mc_se_bits_delta_method=None,
                  approximate_loss_ci95_bits=None, unbounded_upper_loss_ci=False,
                  approximate_loss_ci95_max_halfwidth_bits=None,
                  interval_method="Student t interval for mean likelihood, then logarithm; approximate, does not bound missed-mode or quadrature error")
    if count > 1:
        se = float(scaled.std(ddof=1) / np.sqrt(count))
        relative_se = se / mean
        half = float(student_t.ppf(.975, count - 1)) * se
        lower, upper = mean - half, mean + half
        result.update(relative_mc_se=relative_se,
                      loss_mc_se_bits_delta_method=float(relative_se / np.log(2)),
                      approximate_loss_ci95_bits=[float(-(shift + np.log(upper)) / np.log(2)),
                                                  None if lower <= 0 else float(-(shift + np.log(lower)) / np.log(2))],
                      unbounded_upper_loss_ci=bool(lower <= 0))
        if lower > 0:
            result["approximate_loss_ci95_max_halfwidth_bits"] = max(abs(endpoint - loss) for endpoint in result["approximate_loss_ci95_bits"])
    if exact_log_evidence is not None:
        if not np.isfinite(exact_log_evidence):
            raise ValueError("exact reference must be finite")
        exact_loss = -float(exact_log_evidence) / np.log(2)
        result["exact_catalog_loss_bits"] = exact_loss
        result["estimated_minus_catalog_loss_bits"] = float(loss - exact_loss)
        halfwidth = result["approximate_loss_ci95_max_halfwidth_bits"]
        result["absolute_error_plus_ci95_max_halfwidth_bits"] = (None if halfwidth is None else
                abs(float(loss - exact_loss)) + halfwidth)
        interval = result["approximate_loss_ci95_bits"]
        result["approximate_loss_error_ci95_bits"] = (None if interval is None else
                [None if endpoint is None else float(endpoint - exact_loss) for endpoint in interval])
    return result


def summarize_records(records, cases, groups_of=4):
    summaries = {}
    for case in cases:
        selected = sorted((record for record in records if record["sequence_id"] == case["id"]),
                          key=lambda record: record["replica"])
        if not selected:
            continue
        exact = case.get("catalog_log_evidence")
        logs = [record["estimate"]["log_evidence"] for record in selected]
        item = normalizer_summary(logs, exact)
        item["case"] = case["name"]
        item["scope"] = ("calibration against independently computed short-prefix evidence"
                         if exact is not None else "repeatability only; absolute normalizer accuracy unknown")
        groups = []
        for begin in range(0, len(selected), groups_of):
            group = selected[begin:begin + groups_of]
            if len(group) != groups_of:
                continue
            groups.append(dict(replica_indices=[record["replica"] for record in group],
                               **normalizer_summary([record["estimate"]["log_evidence"] for record in group], exact)))
        item["complete_disjoint_repetition_groups"] = groups
        item["repetition_indices"] = [record["replica"] for record in selected]
        item["runtime_seconds"] = sum(record["estimate"]["elapsed_seconds"] for record in selected)
        item["minimum_particle_ess"] = min(min(record["estimate"]["effective_sample_sizes"]) for record in selected)
        ancestry = [record["estimate"].get("initial_particle_ancestry_counts", []) for record in selected]
        item["final_initial_ancestor_counts"] = [values[-1] if values else None for values in ancestry]
        summaries[case["id"]] = item
    values = list(summaries.values())
    aggregate = dict(scope="descriptive average over the listed fixed token strings only; not a source-family performance estimate",
                     completed_cases=len(values), expected_cases=len(cases),
                     uncertainty_scope="Monte Carlo repetitions only; between-sequence spread is descriptive, not a generalization interval")
    if values:
        means = np.array([item["loss_of_mean_evidence_bits"] for item in values])
        aggregate["mean_loss_of_mean_evidence_bits"] = float(means.mean())
        aggregate["between_fixed_sequence_sd_bits_descriptive"] = (float(means.std(ddof=1)) if len(values) > 1 else None)
        errors = [item["loss_mc_se_bits_delta_method"] for item in values]
        if all(error is not None for error in errors):
            se = float(np.sqrt(np.sum(np.square(errors))) / len(errors))
            aggregate["mc_se_bits_delta_method"] = se
            aggregate["approximate_mc_ci95_bits"] = [float(means.mean() - 1.96 * se), float(means.mean() + 1.96 * se)]
        else:
            aggregate.update(mc_se_bits_delta_method=None, approximate_mc_ci95_bits=None)
        if all("estimated_minus_catalog_loss_bits" in item for item in values):
            aggregate["mean_estimated_minus_catalog_loss_bits"] = float(np.mean([item["estimated_minus_catalog_loss_bits"] for item in values]))
    return dict(by_fixed_sequence=summaries, fixed_sequence_average=aggregate,
                acceptance="not assessed; apply the predeclared protocol before any larger run",
                diagnostics_caveat="Particle ESS, ancestry diversity, rank movement and replicate agreement do not certify normalizer accuracy")


def paired_budget_comparison(current, previous):
    """Diagnostic change in log(mean Z), paired using identical seeds."""
    out = dict(scope="paired Monte Carlo budget/proposal comparison; agreement is not an independent normalizer check", by_fixed_sequence={})
    for case in current["identity"]["cases"]:
        cid = case["id"]
        left = {record["replica"]: record for record in current["records"] if record["sequence_id"] == cid}
        right = {record["replica"]: record for record in previous["records"] if record["sequence_id"] == cid}
        common = sorted(set(left) & set(right))
        if len(common) < 2:
            continue
        if any(left[i]["seed"] != right[i]["seed"] for i in common):
            raise ValueError("comparison repetitions must use the same seed identities")
        x = np.array([left[i]["estimate"]["log_evidence"] for i in common])
        y = np.array([right[i]["estimate"]["log_evidence"] for i in common])
        sx, sy = np.exp(x - x.max()), np.exp(y - y.max())
        delta = float(-(logsumexp(x) - logsumexp(y)) / np.log(2))
        influence = -(sx / sx.mean() - sy / sy.mean()) / np.log(2)
        se = float(influence.std(ddof=1) / np.sqrt(len(common)))
        critical = float(student_t.ppf(.975, len(common) - 1))
        out["by_fixed_sequence"][cid] = dict(paired_repetitions=len(common),
            current_minus_previous_loss_of_mean_evidence_bits=delta,
            paired_mc_se_bits_delta_method=se,
            approximate_paired_ci95_bits=[delta - critical * se, delta + critical * se])
    keys = set(current["identity"]["algorithm"]) | set(previous["identity"]["algorithm"])
    out["changed_algorithm_fields"] = {key: [previous["identity"]["algorithm"].get(key), current["identity"]["algorithm"].get(key)]
        for key in sorted(keys) if previous["identity"]["algorithm"].get(key) != current["identity"]["algorithm"].get(key)}
    return out


def build_cases(args, catalog):
    cases, token_strings = [], {}
    if args.pattern is not None:
        x = np.asarray(args.pattern, dtype=np.int64)
        if len(x) != args.length or np.any(x < 0) or np.any(x >= 64):
            raise ValueError("--pattern must have --length tokens in 0..63")
        case = dict(kind="custom_fixed_control", name="pattern_" + "_".join(map(str, args.pattern)),
                    tokens=x.tolist(), token_sha256=hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest())
        case["id"] = fingerprint(dict(kind=case["kind"], token_sha256=case["token_sha256"]))
        cases.append(case)
        token_strings[case["id"]] = x
    else:
        for family in (args.families or FAMILIES):
            path = args.heldout / f"{family}_prior.npz"
            record = json.loads(path.with_suffix(".json").read_text())
            digest = file_hash(path)
            if digest != record["data_sha256"] or record["condition"]["kind"] != "prior":
                raise ValueError("frozen full-prior source record does not match data")
            with np.load(path, allow_pickle=False) as saved:
                if not 0 <= args.sequence_index < len(saved["tokens"]):
                    raise ValueError("sequence-index outside saved cohort")
                tokens = saved["tokens"][args.sequence_index]
                if args.length > len(tokens):
                    raise ValueError("requested length exceeds saved sequence")
                x = tokens[:args.length].astype(np.int64, copy=True)
                seed = saved["sequence_seed"][args.sequence_index].tolist()
            case = dict(kind="frozen_heldout_prefix", name=f"{family}_prior_{args.sequence_index}",
                        family_for_reporting_only=family, cohort_sha256=digest,
                        cohort_record_sha256=file_hash(path.with_suffix(".json")), source_plan_id=record["plan_id"],
                        sequence_index=args.sequence_index, sequence_seed=seed, tokens=x.tolist(),
                        token_sha256=hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest())
            case["id"] = fingerprint(dict(kind=case["kind"], sequence_seed=seed, token_sha256=case["token_sha256"]))
            cases.append(case)
            token_strings[case["id"]] = x
    for case in cases:
        if args.length == 5:
            case["catalog_log_evidence"] = catalog.log_joint(token_strings[case["id"]])
    return cases, token_strings


def make_identity(args, cases, catalog, kernel, sampler):
    store = args.store.expanduser().resolve()
    if not (store / "anchors.json").is_file():
        raise ValueError("a sealed store is required")
    store_names = ["manifest.json", "anchors.json"] + [f"level_{depth:02d}.{suffix}" for depth in range(2, 9) for suffix in ("bin", "index.json")]
    backend = args.src.expanduser().resolve() / "product_model_with_memory"
    code = {str(ROOT / "code/transformer" / name): file_hash(ROOT / "code/transformer" / name) for name in CODE_FILES}
    code[str(Path(__file__).resolve())] = file_hash(__file__)
    for name in ("universal_tables.py", "production_coding.py", "kernel.py", "_kernel.c", "mellin.py"):
        code[str(backend / name)] = file_hash(backend / name)
    algorithm = dict(sampler=args.sampler, particles=args.particles, temperatures=args.temperatures,
                     moves=args.moves, resample_fraction=args.resample_fraction,
                     alpha_order=args.alpha_order, alpha_nodes=kernel.alpha_nodes.tolist(),
                     alpha_weights=sampler.alpha_weights.tolist(), depth_prior=sampler.depth_prior.tolist(),
                     grid_bounds=list(args.bounds), grid_step=args.grid_step,
                     surrogate_steps=args.surrogate_steps if args.sampler == "row-block" else None,
                     concentration=args.concentration if args.sampler == "row-block" else None,
                     term_cache_size=args.term_cache_size, row_cache_size=args.row_cache_size,
                     proposal_spec=getattr(sampler, "proposal_spec", None), beta_schedule="(k/K)^3")
    protocol = (None if args.protocol is None else dict(path=str(args.protocol.resolve()),
                 sha256=file_hash(args.protocol), contents=args.protocol.read_text()))
    return dict(schema=SCHEMA, reference_family="shared_base", n_tokens=args.length,
                prior=dict(alpha_uniform=[.5, 1.5], depth_uniform_inclusive=[0, 8], ranks="uniform permutation"),
                cases=cases, algorithm=algorithm, replicates=args.replicates, groups_of=args.groups_of,
                seed_namespace=SEED_NAMESPACE, master_seed=args.seed, code_sha256=code,
                sealed_store_sha256={name: file_hash(store / name) for name in store_names},
                sealed_store=str(store), short_catalog_validation=catalog.validation,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                input_numerical_environment=numerical_environment(),
                thread_environment={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                predeclared_protocol=protocol,
                advancement="one explicitly selected length only; no automatic acceptance or larger run")


def check_resume(report, identity):
    if report.get("schema") != SCHEMA or report.get("identity_id") != fingerprint(identity) or report.get("identity") != identity:
        raise ValueError("existing calibration has different settings, code, inputs, or protocol; choose a new output")
    seen = set()
    cases = {case["id"] for case in identity["cases"]}
    for record in report["records"]:
        key = (record["sequence_id"], record["replica"])
        if (key in seen or key[0] not in cases or not 0 <= key[1] < identity["replicates"]
                or record["seed"] != replicate_seed(*key, identity["master_seed"])
                or not np.isfinite(record["estimate"]["log_evidence"])):
            raise ValueError("saved repetition identity or numerical value is inconsistent")
        seen.add(key)
    return seen


def execute(args):
    setup_started = time.monotonic()
    numerical_environment()  # refuse perturbed targets before constructing a numerical kernel
    catalog = SharedBaseShortEvidence()
    cases, token_strings = build_cases(args, catalog)
    nodes, weights = leggauss(args.alpha_order)
    kernel = RankedLSARowEvidence(64, 1 + nodes / 2, store=args.store, src=args.src,
                                 u_lo=args.bounds[0], u_hi=args.bounds[1], step=args.grid_step,
                                 term_cache_size=args.term_cache_size, evidence_cache_size=args.row_cache_size)
    sampler = (SharedBaseMCMC(64, kernel, weights / 2) if args.sampler == "original" else
               SharedBaseEfficiencyRowMCMC(64, kernel, weights / 2,
                                          surrogate_steps=args.surrogate_steps, concentration=args.concentration))
    identity = make_identity(args, cases, catalog, kernel, sampler)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(args.out):
        if args.out.exists():
            report = json.loads(args.out.read_text())
            completed = check_resume(report, identity)
        else:
            report = dict(schema=SCHEMA, identity_id=fingerprint(identity), identity=identity,
                          created_utc=utc_now(), records=[], invocations=[], status="not started",
                          scientific_scope="normalizer calibration or repeatability on fixed strings; no source-family result or accepted long-context reference")
            completed = set()
        if len(completed) == len(cases) * args.replicates:
            print(json.dumps(dict(out=str(args.out), status="already complete", repetitions=len(completed))), flush=True)
            return
        invocation = dict(started_utc=utc_now(), seconds_limit=args.seconds,
                          setup_seconds=time.monotonic() - setup_started,
                          time_limit_scope="sampling after input/provenance setup",
                          completed_before=len(completed), status="running", active_repetition=None)
        report["invocations"].append(invocation)
        previous_handler = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("bounded calibration time limit reached")
        signal.signal(signal.SIGALRM, expired)
        started = time.monotonic()
        signal.alarm(args.seconds)
        try:
            # Cycle through fixed strings so a timeout cannot silently leave
            # all repetitions concentrated on the first, easiest string.
            for replica in range(args.replicates):
                for case in cases:
                    key = (case["id"], replica)
                    if key in completed:
                        continue
                    for path, expected in identity["code_sha256"].items():
                        if file_hash(path) != expected:
                            raise RuntimeError("numerical code changed during calibration")
                    seed = replicate_seed(*key, args.seed)
                    invocation["active_repetition"] = dict(sequence_id=case["id"], replica=replica, seed=seed)
                    report["status"] = "running"
                    atomic_json(args.out, report)
                    result = estimate_log_evidence(sampler, token_strings[case["id"]], particles=args.particles,
                               temperatures=args.temperatures, moves=args.moves, seed=seed,
                               resample_fraction=args.resample_fraction)
                    if not np.isfinite(result["log_evidence"]):
                        raise FloatingPointError("nonfinite evidence estimate")
                    report["records"].append(dict(sequence_id=case["id"], case=case["name"], replica=replica,
                                                   seed=seed, estimate=result, row_cache=kernel.cache_info()))
                    completed.add(key)
                    invocation["active_repetition"] = None
                    report["summary"] = summarize_records(report["records"], cases, args.groups_of)
                    if args.compare_to is not None:
                        other = json.loads(args.compare_to.read_text())
                        report["budget_comparison"] = paired_budget_comparison(report, other)
                        report["budget_comparison"]["previous_file_sha256"] = file_hash(args.compare_to)
                    atomic_json(args.out, report)
                    print(json.dumps(dict(case=case["name"], replica=replica, seconds=result["elapsed_seconds"],
                          completed=len(completed), expected=len(cases) * args.replicates,
                          log_evidence=result["log_evidence"],
                          scope="short exact calibration" if args.length == 5 else "repeatability only")), flush=True)
            invocation["status"] = "completed"
            report["status"] = "completed; interpretation requires predeclared calibration gates"
        except TimeoutError:
            invocation["status"] = "time limit"
            report["status"] = "partial; resumable, interrupted repetition will restart"
        except BaseException as error:
            invocation["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            invocation["error"] = f"{type(error).__name__}: {error}"
            report["status"] = invocation["status"]
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            invocation.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                              completed_after=len(completed))
            report["summary"] = summarize_records(report["records"], cases, args.groups_of)
            atomic_json(args.out, report)
    print(json.dumps(dict(out=str(args.out), status=report["status"], repetitions=len(completed),
                          summary=report["summary"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampler", choices=("original", "row-block"), default="original")
    parser.add_argument("--length", type=int, choices=(5, 17, 65, 1025), default=5)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--families", nargs="+", choices=FAMILIES)
    source.add_argument("--pattern", nargs="+", type=int)
    parser.add_argument("--heldout", type=Path, default=ROOT / "output/multifamily_results_2026-09-16/heldout")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--temperatures", type=int, default=16)
    parser.add_argument("--moves", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026091603)
    parser.add_argument("--alpha-order", type=int, default=16)
    parser.add_argument("--surrogate-steps", type=int, default=8)
    parser.add_argument("--concentration", type=float, default=1.)
    parser.add_argument("--resample-fraction", type=float, default=.75)
    parser.add_argument("--bounds", nargs=2, type=float, default=(-32., 32.))
    parser.add_argument("--grid-step", type=float, default=1 / 32)
    parser.add_argument("--term-cache-size", type=int, default=1024)
    parser.add_argument("--row-cache-size", type=int, default=100000)
    parser.add_argument("--groups-of", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--store", type=Path, default=Path("/Users/ruediger/Projects/product_model_with_memory/tables/anchors_prod"))
    parser.add_argument("--src", type=Path, default=Path("/Users/ruediger/Projects/product_model_with_memory/src"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        report = json.loads(args.out.read_text())
        if report.get("schema") != SCHEMA:
            parser.error("output does not contain this calibration schema")
        print(json.dumps(summarize_records(report["records"], report["identity"]["cases"], report["identity"]["groups_of"]), indent=2), flush=True)
        return
    positive = (args.temperatures, args.moves, args.replicates, args.alpha_order, args.surrogate_steps,
                args.term_cache_size, args.row_cache_size, args.groups_of, args.seconds)
    if (min(positive) < 1 or args.particles < 2 or args.sequence_index < 0 or args.seed < 0
            or not 0 < args.resample_fraction <= 1 or args.concentration <= 0
            or not np.isfinite([*args.bounds, args.grid_step, args.concentration]).all()
            or args.grid_step <= 0 or args.bounds[0] >= args.bounds[1]
            or (args.families is not None and len(set(args.families)) != len(args.families))):
        parser.error("invalid numerical setting, family list, index or runtime limit")
    execute(args)


if __name__ == "__main__":
    main()

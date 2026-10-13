"""Portable independent-repetition shards for shared-base evidence calibration.

One SMC repetition stays intact in one process. Sharding changes neither its
global repetition index nor its seed. Each output has an exclusive writer;
merging requires every requested case/repetition exactly once, with matching
plan, numerical implementation, immutable store and source-token identities.
No completion or numerical gate approves production reference computation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from calibrate_shared_base_evidence import (CODE_FILES, atomic_json, canonical,
    file_hash, fingerprint, normalizer_summary, numerical_environment,
    output_lock, replicate_seed, utc_now)
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_evidence_smc import estimate_log_evidence

PLAN_SCHEMA = "shared-base-cluster-plan-v1"
SHARD_SCHEMA = "shared-base-cluster-shard-v1"
MERGE_SCHEMA = "shared-base-cluster-merge-v1"
PURPOSES = {"preflight", "exact_short", "confirmation", "quadrature_refinement", "repeatability"}
SHORT_GATE = dict(minimum_completed_replicates=16, tolerance_bits=.01,
                  minimum_effective_replica_count=8., maximum_single_replica_likelihood_share=.2)
SUPPORTED_PRIOR = dict(alphabet=64, alpha_uniform=[.5, 1.5], depth_uniform=list(range(9)), permutation="uniform")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, int) and value >= minimum,
            f"{name} must be an integer >= {minimum}")
    return value


def read_plan(path, stage):
    path = Path(path)
    raw = path.read_bytes()
    plan = json.loads(raw)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported plan schema")
    require(plan.get("prior") == SUPPORTED_PRIOR, "plan prior differs from the implemented shared-base prior")
    acceptance = plan.get("acceptance", {})
    require(acceptance.get("exact_short") == SHORT_GATE and acceptance.get("long_precision_bits") == .01
            and acceptance.get("production_reference_approved") is False, "unsupported acceptance criteria")
    manifest_path = ROOT / plan["dependency_manifest"]
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    require(hashlib.sha256(manifest_raw).hexdigest() == plan.get("dependency_manifest_sha256"),
            "dependency manifest hash differs from the frozen plan")
    require(manifest.get("schema") == "shared-base-dependency-manifest-v1" and bool(manifest.get("files")),
            "missing frozen dependency file manifest")
    require(isinstance(plan.get("cases"), list) and len(plan["cases"]) > 0, "plan needs fixed cases")
    cases = {}
    for case in plan["cases"]:
        require(isinstance(case.get("id"), str) and bool(case["id"]) and case["id"] not in cases,
                "case IDs must be distinct nonempty strings")
        require(isinstance(case.get("name"), str) and isinstance(case.get("kind"), str), "case name and kind required")
        tokens = np.asarray(case.get("tokens"))
        require(tokens.ndim == 1 and len(tokens) >= 2 and tokens.dtype.kind in "iu"
                and np.all((tokens >= 0) & (tokens < 64)), "fixed tokens must be integers in 0..63")
        if case.get("known_log_evidence") is not None:
            require(np.isfinite(case["known_log_evidence"]) and case["known_log_evidence"] <= 0,
                    "known log evidence must be finite and normalized")
        cases[case["id"]] = case
    require(stage in plan.get("stages", {}), f"stage {stage!r} is absent from plan")
    config = plan["stages"][stage]
    require(config.get("sampler") == "original", "cluster runner currently supports only the audited original sampler")
    require(config.get("purpose") in PURPOSES, "stage needs an explicit scientific purpose")
    require(isinstance(config.get("case_ids"), list) and len(config["case_ids"]) > 0
            and len(set(config["case_ids"])) == len(config["case_ids"])
            and all(cid in cases for cid in config["case_ids"]), "stage case IDs must be distinct known cases")
    for key, minimum in (("n_tokens", 2), ("particles", 2), ("temperatures", 1),
                         ("moves", 1), ("replicates", 1), ("alpha_order", 1), ("seed", 0)):
        integer(config.get(key), key, minimum)
    require(all(len(cases[cid]["tokens"]) >= config["n_tokens"] for cid in config["case_ids"]), "fixed case is too short for stage")
    bounds = config.get("grid_bounds")
    step = config.get("grid_step")
    require(isinstance(bounds, list) and len(bounds) == 2 and np.isfinite([*bounds, step]).all()
            and bounds[0] < bounds[1] and step > 0, "invalid row integration grid")
    require(np.isclose((bounds[1] - bounds[0]) / step, round((bounds[1] - bounds[0]) / step), atol=1e-8, rtol=0),
            "row integration interval must contain whole grid cells")
    if config["purpose"] in {"exact_short", "confirmation", "quadrature_refinement", "preflight"}:
        require(config["n_tokens"] == 5, "short calibration/preflight stages must use five tokens")
        require(all(len(cases[cid]["tokens"]) == 5 and cases[cid].get("known_log_evidence") is not None
                    for cid in config["case_ids"]), "short calibration needs an exact reference for every full case")
    requires = config.get("requires", [])
    require(isinstance(requires, list) and len(set(requires)) == len(requires)
            and all(name in plan["stages"] and name != stage for name in requires), "invalid stage prerequisites")
    return dict(plan=plan, plan_sha256=hashlib.sha256(raw).hexdigest(), stage=stage, config=config, cases=cases,
                dependency_manifest=manifest, dependency_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())


def verify_dependencies(context, store, src):
    """Check every portable dependency before importing or constructing its kernel."""
    verified = {}
    for name, expected in context["dependency_manifest"]["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe dependency manifest path")
        if name.startswith("src/"):
            actual = Path(src).expanduser().resolve() / name.removeprefix("src/")
        elif name.startswith("tables/anchors_prod/"):
            actual = Path(store).expanduser().resolve() / name.removeprefix("tables/anchors_prod/")
        else:
            raise ValueError(f"unsupported dependency manifest location: {name}")
        require(actual.is_file() and actual.stat().st_size == expected["size_bytes"], f"dependency missing or wrong size: {name}")
        digest = file_hash(actual)
        require(digest == expected["sha256"], f"dependency hash mismatch: {name}")
        verified[name] = digest
    return verified


def stage_id(context):
    return fingerprint(dict(plan_sha256=context["plan_sha256"], stage=context["stage"], config=context["config"]))


def case_identity(context, case_id):
    require(case_id in context["config"]["case_ids"], "case is not in this stage")
    case = context["cases"][case_id]
    tokens = case["tokens"][:context["config"]["n_tokens"]]
    return dict(id=case_id, name=case["name"], case_sha256=fingerprint(case),
                tokens_sha256=fingerprint(tokens), n_tokens=len(tokens))


def expected_seed(context, case_id, replica):
    # Stage name, worker number and shard boundaries deliberately do not enter.
    sequence_identity = fingerprint(dict(case_id=case_id,
        tokens=context["cases"][case_id]["tokens"][:context["config"]["n_tokens"]]))
    return replicate_seed(sequence_identity, replica, context["config"]["seed"])


def known_evidence(context, case_id):
    case = context["cases"][case_id]
    return case.get("known_log_evidence") if len(case["tokens"]) == context["config"]["n_tokens"] else None


def numerical_setup(context, store, src):
    """Reuse, never rebuild, the existing sealed numerical store."""
    started = time.monotonic()
    numerical_environment()  # rejects nonzero BIAS/WAVE before construction
    dependency_hashes = verify_dependencies(context, store, src)
    config = context["config"]
    nodes, weights = leggauss(config["alpha_order"])
    kernel = RankedLSARowEvidence(64, 1 + nodes / 2, store=store, src=src,
        u_lo=config["grid_bounds"][0], u_hi=config["grid_bounds"][1], step=config["grid_step"],
        term_cache_size=config.get("term_cache_size", 1024), evidence_cache_size=config.get("row_cache_size", 100000))
    backend = kernel._get_backend()
    require(backend.read_only and backend.sealed, "store must be sealed and opened read-only")
    sampler = SharedBaseMCMC(64, kernel, weights / 2)
    compiled = sys.modules["product_model_with_memory.kernel"]
    require(compiled.available, "compiled interpolation kernel unavailable; build it in the dependency checkout before calibration")
    package = Path(src).expanduser().resolve() / "product_model_with_memory"
    require(Path(compiled.__file__).resolve().parent == package, "compiled kernel imported from a different dependency checkout")
    code_hashes = {"code/transformer/" + name: file_hash(ROOT / "code/transformer" / name) for name in CODE_FILES}
    for name in ("code/transformer/shared_base_cluster.py", "code/scripts/shared_base_cluster.py", "code/scripts/calibrate_shared_base_evidence.py"):
        code_hashes[name] = file_hash(ROOT / name)
    backend_hashes = {name: file_hash(package / name) for name in
                     ("universal_tables.py", "production_coding.py", "kernel.py", "_kernel.c", "mellin.py")}
    store = Path(store).expanduser().resolve()
    files = ["manifest.json", "anchors.json"] + [f"level_{depth:02d}.{suffix}"
             for depth in range(2, 9) for suffix in ("bin", "index.json")]
    environment = numerical_environment()
    environment.pop("PMM_UNIVERSAL_TABLES")  # deployment path is not a numerical identity
    identity = dict(code_sha256=code_hashes, backend_sha256=backend_hashes,
        dependency_manifest_sha256=context["dependency_manifest_sha256"], dependency_sha256=dependency_hashes,
        store_file_sha256={name: file_hash(store / name) for name in files},
        effective_numerical_environment=environment,
        alpha_nodes=kernel.alpha_nodes.tolist(), alpha_weights=sampler.alpha_weights.tolist(),
        depth_prior=sampler.depth_prior.tolist(), numpy_version=np.__version__, scipy_version=scipy.__version__,
        python_version=platform.python_version(), compiled_kernel_available=True)
    resources = dict(hostname=socket.gethostname(), platform=platform.platform(),
        python_version=platform.python_version(), numpy_version=np.__version__, scipy_version=scipy.__version__,
        compiled_kernel_available=True, store_sealed=True, store_read_only=True,
        store=str(store), companion_src=str(package.parent), setup_seconds=time.monotonic() - started,
        thread_environment={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    return sampler, identity, resources


def shard_identity(context, case_id, numerical_identity):
    return dict(plan_sha256=context["plan_sha256"], stage=context["stage"], stage_id=stage_id(context),
                dependency_manifest_sha256=context["dependency_manifest_sha256"],
                prerequisite_receipts=context.get("prerequisite_receipts", {}),
                stage_config=context["config"], case=case_identity(context, case_id), numerical=numerical_identity)


def new_shard(context, case_id, numerical_identity, start, stop, resources=None):
    integer(start, "replica-start")
    integer(stop, "replica-stop", 1)
    require(start < stop <= context["config"]["replicates"], "replica range must be nonempty and within the plan")
    identity = shard_identity(context, case_id, numerical_identity)
    return dict(schema=SHARD_SCHEMA, identity=identity, identity_id=fingerprint(identity),
                shard_range=dict(start=start, stop=stop), records=[], records_sha256=fingerprint([]),
                status="pending", complete=False, invocations=[], runtime_resources=resources or {})


def append_record(shard, context, replica, estimate):
    case_id = shard["identity"]["case"]["id"]
    integer(replica, "replica")
    require(shard["shard_range"]["start"] <= replica < shard["shard_range"]["stop"], "replica outside shard range")
    require(all(record["replica"] != replica for record in shard["records"]), "duplicate repetition in shard")
    require(np.isfinite(estimate["log_evidence"]), "nonfinite log evidence")
    record = dict(case_id=case_id, replica=replica, seed=expected_seed(context, case_id, replica), estimate=estimate)
    record["record_sha256"] = fingerprint(record)
    shard["records"].append(record)
    shard["records"].sort(key=lambda value: value["replica"])
    shard["records_sha256"] = fingerprint(shard["records"])


def validate_shard(shard, context, numerical_identity=None, expected_range=None):
    require(shard.get("schema") == SHARD_SCHEMA, "not a cluster evidence shard")
    identity = shard["identity"]
    require(shard.get("identity_id") == fingerprint(identity), "shard identity hash mismatch")
    case_id = identity["case"]["id"]
    prerequisites = identity.get("prerequisite_receipts", {})
    require(set(prerequisites) == set(context["config"].get("requires", [])), "missing prerequisite receipt provenance")
    expected = shard_identity(dict(context, prerequisite_receipts=context.get("prerequisite_receipts", prerequisites)),
                              case_id, identity["numerical"] if numerical_identity is None else numerical_identity)
    require(identity == expected, "shard plan, config, source case or numerical identity mismatch")
    require(identity["numerical"].get("compiled_kernel_available") is True, "compiled kernel was not verified")
    require(all(shard.get("runtime_resources", {}).get(key) is True
                for key in ("compiled_kernel_available", "store_sealed", "store_read_only")), "shard runtime/store preflight failed")
    region = shard["shard_range"]
    start, stop = region["start"], region["stop"]
    integer(start, "shard start")
    integer(stop, "shard stop", 1)
    require(start < stop <= context["config"]["replicates"], "invalid shard repetition range")
    if expected_range is not None:
        require(region == dict(start=expected_range[0], stop=expected_range[1]), "resuming with different shard boundaries is forbidden")
    require(shard.get("records_sha256") == fingerprint(shard["records"]), "shard record-array hash mismatch")
    indices = set()
    for record in shard["records"]:
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        require(record.get("record_sha256") == fingerprint(payload), "raw repetition record hash mismatch")
        replica = record["replica"]
        integer(replica, "replica")
        require(record["case_id"] == case_id and start <= replica < stop
                and replica not in indices, "duplicate, wrong-case or out-of-range repetition")
        require(record["seed"] == expected_seed(context, case_id, replica), "repetition seed mismatch")
        require(np.isfinite(record["estimate"]["log_evidence"]), "nonfinite evidence estimate")
        for key in ("seed", "particles", "temperatures", "moves", "n_tokens"):
            expected_value = record["seed"] if key == "seed" else context["config"][key]
            require(record["estimate"].get(key) == expected_value, f"raw estimator {key} differs from its planned repetition")
        require(record["estimate"].get("sampler_class") == "SharedBaseMCMC", "raw estimator used an unplanned sampler")
        indices.add(replica)
    if shard.get("complete"):
        require(indices == set(range(start, stop)) and shard.get("status") == "completed", "falsely complete shard")
    return indices


def run_shard(plan, stage, case_id, start, stop, out, store, src, seconds=600, receipts=None):
    context = read_plan(plan, stage)
    require(case_id in context["config"]["case_ids"], "case is not in stage")
    integer(seconds, "seconds", 1)
    prerequisites = {}
    prerequisite_numerics = []
    for name in context["config"].get("requires", []):
        require(receipts is not None, "this stage requires --receipts pointing to verified merged prerequisite results")
        raw = (Path(receipts) / f"{name}.json").read_bytes()
        receipt = json.loads(raw)
        validate_prerequisite_receipt(context, name, receipt)
        prerequisite_numerics.append(receipt["numerical"])
        prerequisites[name] = hashlib.sha256(raw).hexdigest()
    context["prerequisite_receipts"] = prerequisites
    sampler, numerical, resources = numerical_setup(context, store, src)
    for previous in prerequisite_numerics:
        for key in ("code_sha256", "backend_sha256", "dependency_manifest_sha256", "dependency_sha256",
                    "store_file_sha256", "effective_numerical_environment", "numpy_version", "scipy_version", "python_version"):
            require(previous.get(key) == numerical.get(key), f"numerical implementation changed since prerequisite: {key}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        if out.exists():
            report = json.loads(out.read_text())
            completed = validate_shard(report, context, numerical, (start, stop))
            if report["complete"]:
                return report, 0
        else:
            report = new_shard(context, case_id, numerical, start, stop, resources)
            completed = set()
        require(report["identity"]["case"]["id"] == case_id, "resume case mismatch")
        invocation = dict(started_utc=utc_now(), sampling_seconds_limit=seconds,
                          runtime_resources=resources, active_replica=None, status="running")
        report["invocations"].append(invocation)
        started = time.monotonic()
        previous_handler = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise TimeoutError("sampling time limit reached")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(seconds)
        code = 75
        try:
            config = context["config"]
            tokens = np.asarray(context["cases"][case_id]["tokens"][:config["n_tokens"]], dtype=np.int64)
            for replica in range(start, stop):
                if replica in completed:
                    continue
                for path, expected in numerical["code_sha256"].items():
                    require(file_hash(ROOT / path) == expected, "code changed during sampling")
                report["status"] = "running"
                invocation["active_replica"] = replica
                atomic_json(out, report)
                estimate = estimate_log_evidence(sampler, tokens, particles=config["particles"],
                    temperatures=config["temperatures"], moves=config["moves"],
                    seed=expected_seed(context, case_id, replica), resample_fraction=config.get("resample_fraction", .75))
                append_record(report, context, replica, estimate)
                completed.add(replica)
                invocation["active_replica"] = None
                report["row_cache"] = sampler.kernel.cache_info()
                report["summary"] = normalizer_summary([r["estimate"]["log_evidence"] for r in report["records"]], known_evidence(context, case_id))
                atomic_json(out, report)
                print(canonical(dict(case_id=case_id, replica=replica, completed=len(completed),
                    requested=stop-start, elapsed_seconds=estimate["elapsed_seconds"])), flush=True)
            report.update(status="completed", complete=True)
            invocation["status"] = "completed"
            code = 0
        except TimeoutError:
            report.update(status="time_limit", complete=False)
            invocation["status"] = "time_limit"
        except BaseException as error:
            report.update(status="failed", complete=False)
            invocation.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            invocation.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
            if report["records"]:
                report["summary"] = normalizer_summary([r["estimate"]["log_evidence"] for r in report["records"]], known_evidence(context, case_id))
            atomic_json(out, report)
    return report, code


def summarize_stage(context, records):
    summaries = {}
    for case_id in context["config"]["case_ids"]:
        values = sorted((record for record in records if record["case_id"] == case_id), key=lambda value: value["replica"])
        summary = normalizer_summary([record["estimate"]["log_evidence"] for record in values], known_evidence(context, case_id))
        summary["case_name"] = context["cases"][case_id]["name"]
        summary["runtime_seconds"] = float(sum(record["estimate"].get("elapsed_seconds", 0.) for record in values))
        summary["replica_indices"] = [record["replica"] for record in values]
        summaries[case_id] = summary
    return summaries


def stage_gate(context, summaries):
    purpose = context["config"]["purpose"]
    gate = dict(purpose=purpose, production_approved=False, scope="only the fixed cases and numerical settings in this stage")
    if purpose == "preflight":
        return dict(gate, status="preflight_only", accuracy_approved=False, runtime_ready=True)
    if purpose in {"exact_short", "confirmation", "quadrature_refinement"}:
        criteria = {}
        for cid, summary in summaries.items():
            metric = summary.get("absolute_error_plus_ci95_max_halfwidth_bits")
            criteria[cid] = bool(summary["repetitions"] >= SHORT_GATE["minimum_completed_replicates"]
                and metric is not None and metric <= SHORT_GATE["tolerance_bits"]
                and summary["effective_replica_count"] >= SHORT_GATE["minimum_effective_replica_count"]
                and summary["maximum_single_replica_likelihood_share"] <= SHORT_GATE["maximum_single_replica_likelihood_share"])
        return dict(gate, status="passed_exact_short" if all(criteria.values()) else "failed_exact_short",
                    criteria=SHORT_GATE, passed_by_case=criteria, accuracy_approved=False,
                    exact_short_gate_passed=all(criteria.values()))
    criteria = {cid: bool(summary["repetitions"] >= 16
        and summary["approximate_loss_ci95_max_halfwidth_bits"] is not None
        and summary["approximate_loss_ci95_max_halfwidth_bits"] <= .01
        and summary["effective_replica_count"] >= 8
        and summary["maximum_single_replica_likelihood_share"] <= .2)
        for cid, summary in summaries.items()}
    return dict(gate, status="repeatability_only", precision_met=all(criteria.values()), precision_passed=all(criteria.values()),
                precision_met_by_case=criteria, accuracy_approved=False,
                caution="Small Monte Carlo intervals and repeatability do not establish normalizer accuracy; independent refinement remains required.")


def merge_reports(context, reports):
    require(bool(reports), "no shards supplied")
    numerical = reports[0]["identity"]["numerical"]
    prerequisites = reports[0]["identity"].get("prerequisite_receipts", {})
    records, seen, unfinished = [], set(), []
    for shard in reports:
        validate_shard(shard, context, numerical)
        require(shard["identity"].get("prerequisite_receipts", {}) == prerequisites, "shards use different prerequisite receipts")
        if not shard.get("complete") or shard.get("status") != "completed":
            unfinished.append(dict(case_id=shard["identity"]["case"]["id"], range=shard["shard_range"], status=shard.get("status")))
        for record in shard["records"]:
            key = (record["case_id"], record["replica"])
            require(key not in seen, "duplicate case/repetition across shards")
            seen.add(key)
            records.append(record)
    expected = {(cid, replica) for cid in context["config"]["case_ids"] for replica in range(context["config"]["replicates"])}
    missing = sorted(expected - seen)
    receipt = dict(schema=MERGE_SCHEMA, plan_sha256=context["plan_sha256"], stage=context["stage"],
        stage_id=stage_id(context), stage_config=context["config"], numerical=numerical,
        dependency_manifest_sha256=context["dependency_manifest_sha256"],
        prerequisite_receipts=prerequisites,
        expected_records=len(expected), observed_records=len(records), records_sha256=fingerprint(sorted(records, key=lambda r: (r["case_id"], r["replica"]))),
        complete=not missing and not unfinished, missing=[dict(case_id=cid, replica=replica) for cid, replica in missing],
        unfinished_shards=unfinished, coverage={cid: sorted(replica for case, replica in seen if case == cid) for cid in context["config"]["case_ids"]},
        no_auto_advance=True)
    receipt["cases"] = {cid: dict(records=sorted((r for r in records if r["case_id"] == cid), key=lambda r: r["replica"]))
                        for cid in context["config"]["case_ids"]}
    if not receipt["complete"]:
        receipt["gate"] = dict(status="incomplete", production_approved=False, accuracy_approved=False)
        return receipt, 75
    receipt["case_summaries"] = summarize_stage(context, records)
    for cid, summary in receipt["case_summaries"].items():
        receipt["cases"][cid]["statistics"] = summary
    receipt["gate"] = stage_gate(context, receipt["case_summaries"])
    receipt["uncertainty_scope"] = "independent full SMC repetitions on fixed token strings; likelihood estimates averaged before logarithms"
    return receipt, 0


def validate_prerequisite_receipt(context, prerequisite, receipt):
    """Launcher gate: recompute coverage, hashes and scientific criteria from raw repetitions."""
    require(prerequisite in context["config"].get("requires", []), "not a declared prerequisite")
    previous = dict(context, stage=prerequisite, config=context["plan"]["stages"][prerequisite])
    return validate_merge_receipt(previous, receipt, require_gate=True)


def validate_merge_receipt(context, receipt, *, require_gate=False):
    """Return a recomputed receipt after validating every retained raw record.

    Complete failed gates may be inspected when ``require_gate=False``. Neither
    this check nor a passed gate validates accuracy beyond the fixed short cases.
    """
    previous = dict(context, prerequisite_receipts=receipt.get("prerequisite_receipts", {}))
    require(receipt.get("schema") == MERGE_SCHEMA and receipt.get("complete") is True, "prerequisite is incomplete")
    require(receipt.get("plan_sha256") == context["plan_sha256"] and receipt.get("stage") == context["stage"]
            and receipt.get("stage_id") == stage_id(previous) and receipt.get("stage_config") == previous["config"]
            and receipt.get("dependency_manifest_sha256") == context["dependency_manifest_sha256"], "prerequisite identity mismatch")
    require(set(receipt.get("cases", {})) == set(previous["config"]["case_ids"]), "prerequisite case coverage mismatch")
    reports = []
    for cid in previous["config"]["case_ids"]:
        shard = new_shard(previous, cid, receipt["numerical"], 0, previous["config"]["replicates"],
                          dict(compiled_kernel_available=True, store_sealed=True, store_read_only=True))
        shard["records"] = receipt["cases"][cid]["records"]
        shard.update(records_sha256=fingerprint(shard["records"]), complete=True, status="completed")
        reports.append(shard)
    checked, code = merge_reports(previous, reports)
    for key in ("expected_records", "observed_records", "records_sha256", "coverage", "case_summaries", "gate"):
        require(receipt.get(key) == checked[key], f"prerequisite {key} differs from its raw repetitions")
    require(code == 0 and not receipt.get("missing") and not receipt.get("unfinished_shards"), "prerequisite has missing records")
    if not require_gate:
        return checked
    gate = checked["gate"]
    purpose = previous["config"]["purpose"]
    if purpose == "preflight":
        require(gate["status"] == "preflight_only" and gate["runtime_ready"], "preflight runtime check failed")
    elif purpose == "repeatability":
        require(gate["status"] == "repeatability_only" and gate["precision_passed"], "previous pilot fails Monte Carlo precision gate")
    else:
        require(gate["status"] == "passed_exact_short", "exact short calibration gate failed")
    return checked


def merge_shards(plan, stage, directory, out):
    context = read_plan(plan, stage)
    paths = sorted(path for path in Path(directory).rglob("*.json") if path.resolve() != Path(out).resolve())
    require(bool(paths), "no shard JSON files found")
    reports = []
    hashes = {}
    for path in paths:
        raw = path.read_bytes()
        reports.append(json.loads(raw))
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    result, code = merge_reports(context, reports)
    result["shard_sha256"] = hashes
    result["merge_code_sha256"] = file_hash(__file__)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        atomic_json(out, result)
    return result, code

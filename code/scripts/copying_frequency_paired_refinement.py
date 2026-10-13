"""Small paired stationary-frequency refinement for the copying-payoff gate.

The 16- and 32-node Gauss--Legendre rules reuse the same random transition
rule draws.  Each marginal calculation is valid for its quadrature rule, and
the paired delete-bank comparison removes irrelevant source-bank noise.  The
immutable allocation has sixteen independent banks, four workers, one attempt
per bank, and fixed wall-clock and resident-memory limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from stationary_frequency_moments import frequency_type, stationary_joint_type


BANKS = 16
REPETITIONS = 16
COARSE_ORDER = 16
FINE_ORDER = 32
DIMENSION = 64
MAX_WORKERS = 4
TASK_WALL_SECONDS = 300
MAX_RSS_BYTES = 1 * 2**30
MINIMUM_FREE_DISK_BYTES = 2 * 2**30
SEED = 20260921
DEFAULT_OUT = ROOT / "output/copying_frequency_paired_refinement_2026-09-21"
BASE = ROOT / "output/shared_base_shuffle_download_2026-09-18"
MANIFEST = ROOT / "code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text())


def output_path(path):
    path = Path(path).absolute()
    require(path.parent.resolve() == (ROOT / "output").resolve() and not path.is_symlink(),
            "Output must be a direct child of the project output directory")
    return path


def quadrature_rows(coarse_order=COARSE_ORDER, fine_order=FINE_ORDER):
    require(fine_order == 2 * coarse_order, "Paired rule requires twice as many fine nodes")
    cn, cw = np.polynomial.legendre.leggauss(coarse_order)
    fn, fw = np.polynomial.legendre.leggauss(fine_order)
    return [dict(fine_index=i, coarse_index=i // 2,
                 fine_alpha=float(1 + fn[i] / 2),
                 coarse_alpha=float(1 + cn[i // 2] / 2),
                 fine_alpha_weight=float(fw[i] / 2),
                 coarse_alpha_weight=float(cw[i // 2] / 4))
            for i in range(fine_order)]


def _q(alpha, repetitions, dimension):
    ranks = np.arange(1, dimension + 1, dtype=float)
    value = ranks[None, :] ** (-np.full(repetitions, alpha)[:, None])
    return value / value.sum(1, keepdims=True)


def _stationary(transition):
    transition = np.asarray(transition, dtype=float); d = transition.shape[-1]
    a = transition.transpose(0, 2, 1) - np.eye(d); a[:, -1] = 1
    rhs = np.zeros((len(transition), d, 1)); rhs[:, -1, 0] = 1
    value = np.linalg.solve(a, rhs)[..., 0]
    residual = np.abs(np.einsum("bi,bij->bj", value, transition) - value).sum(1)
    require(np.isfinite(value).all() and np.min(value) > 0
            and np.max(np.abs(value.sum(1) - 1)) < 1e-12
            and residual.max() < 1e-11, "Stationary solve failed")
    return value


def coupled_transition_laws(*, fine_alpha, coarse_alpha, depth, repetitions,
                            dimension, rng):
    require(depth >= 0 and repetitions >= 1 and dimension >= 2, "Invalid source batch")
    qf = _q(fine_alpha, repetitions, dimension)
    qc = _q(coarse_alpha, repetitions, dimension)
    if depth == 0:
        return qf, qf.copy(), qc, qc.copy()
    logw = np.log(rng.exponential(size=(depth, repetitions, dimension, dimension))).sum(0)
    weights = np.exp(logw - logw.max(2, keepdims=True))
    tf = qf[:, None, :] * weights; tf /= tf.sum(2, keepdims=True)
    tc = qc[:, None, :] * weights; tc /= tc.sum(2, keepdims=True)
    return qf, _stationary(tf), qc, _stationary(tc)


def paired_bank_metrics(fine, coarse):
    fine = np.asarray(fine, dtype=float); coarse = np.asarray(coarse, dtype=float)
    require(fine.shape == coarse.shape and fine.ndim == 2 and len(fine) >= 2,
            "Invalid paired bank estimates")
    difference = fine - coarse; center = difference.mean(0); banks = len(difference)
    mcse = np.sqrt((banks - 1) / banks * np.sum((difference - center) ** 2, axis=0))
    return dict(difference=center, comparison_mcse=mcse,
                comparison_bias=np.zeros_like(center))


def load_groups():
    manifest = read(MANIFEST); groups = []; inputs = {str(MANIFEST.relative_to(ROOT)): sha(MANIFEST)}
    for condition in manifest["conditions"]:
        path = BASE / "cohorts" / (condition["id"] + ".npz")
        with np.load(path, allow_pickle=False) as saved:
            tokens = saved["tokens"].copy(); rows = saved["transition"].copy()
        require(tokens.shape == (256, 1025) and rows.shape == (256, 64, 64),
                "Incomplete frozen cohort")
        groups.append(dict(condition=condition, tokens=tokens, rows=rows))
        inputs[str(path.relative_to(ROOT))] = sha(path)
    require([g["condition"]["id"] for g in groups] == ["a1_L0", "a1_L2", "a1_L3"],
            "Unexpected conditions")
    return groups, inputs


def all_types(groups):
    kinds = set()
    for group in groups:
        for row in group["tokens"]:
            for n in range(1, 9):
                prefix = row[:n].tolist(); kinds.add(frequency_type(prefix))
                labels = list(dict.fromkeys(prefix))
                labels.append(next(x for x in range(DIMENSION) if x not in prefix))
                for label in labels:
                    kinds.add(frequency_type(prefix + [label]))
    return sorted(kinds, key=lambda kind: (1 + kind[0] + sum(kind[1]), kind))


def kind_key(kind):
    return json.dumps(kind, separators=(",", ":"))


def build_plan(out):
    groups, inputs = load_groups(); kinds = all_types(groups)
    code = [Path(__file__), ROOT / "code/transformer/stationary_frequency_moments.py"]
    plan = dict(schema="copying-frequency-paired-refinement-plan-v1",
        output=str(output_path(out)), banks=BANKS, repetitions_per_fine_node_depth=REPETITIONS,
        coarse_order=COARSE_ORDER, fine_order=FINE_ORDER, dimension=DIMENSION,
        quadrature_coupling="Fine nodes are paired consecutively; each coarse node reuses both corresponding independent rule batches.",
        random_stream="SeedSequence[20260921,0x53544154,bank,fine_node,depth]",
        times=list(range(1, 9)), frequency_types=[kind_key(x) for x in kinds],
        resources=dict(max_workers=MAX_WORKERS, task_wall_seconds=TASK_WALL_SECONDS,
            max_rss_bytes_per_worker=MAX_RSS_BYTES, maximum_attempts_per_bank=1,
            minimum_free_disk_bytes=MINIMUM_FREE_DISK_BYTES),
        runtime=dict(python=list(sys.version_info[:3]), numpy=np.__version__, platform=sys.platform),
        python=sys.executable, inputs_sha256=inputs,
        code_sha256={str(path.relative_to(ROOT)): sha(path) for path in code},
        finite_shuffle_exact=False, payoff_forecast_approved=False)
    plan["plan_id"] = fingerprint(plan)
    return plan


def load_plan(out):
    out = output_path(out); plan = read(out / "plan.json")
    identity = {key: value for key, value in plan.items() if key != "plan_id"}
    require(plan["plan_id"] == fingerprint(identity), "Plan identity changed")
    require(plan["banks"] == BANKS and plan["resources"]["max_workers"] == MAX_WORKERS,
            "Production allocation changed")
    for name, digest in {**plan["inputs_sha256"], **plan["code_sha256"]}.items():
        require(sha(ROOT / name) == digest, "Frozen input changed: " + name)
    return plan


def generate_bank(index, kinds):
    fine_q = []; fine_nu = []; fine_weights = []
    coarse_q = []; coarse_nu = []; coarse_weights = []
    for row in quadrature_rows():
        for depth in range(9):
            repetitions = 1 if depth == 0 else REPETITIONS
            rng = np.random.default_rng(np.random.SeedSequence(
                [SEED, 0x53544154, index, row["fine_index"], depth]))
            qf, nuf, qc, nuc = coupled_transition_laws(fine_alpha=row["fine_alpha"],
                coarse_alpha=row["coarse_alpha"], depth=depth, repetitions=repetitions,
                dimension=DIMENSION, rng=rng)
            fine_q.append(qf); fine_nu.append(nuf)
            coarse_q.append(qc); coarse_nu.append(nuc)
            divisor = 9 * repetitions
            fine_weights.append(np.full(repetitions, row["fine_alpha_weight"] / divisor))
            coarse_weights.append(np.full(repetitions, row["coarse_alpha_weight"] / divisor))
    qf = np.concatenate(fine_q); nuf = np.concatenate(fine_nu); wf = np.concatenate(fine_weights)
    qc = np.concatenate(coarse_q); nuc = np.concatenate(coarse_nu); wc = np.concatenate(coarse_weights)
    require(abs(wf.sum() - 1) < 2e-13 and abs(wc.sum() - 1) < 2e-13,
            "Quadrature weights do not normalize")
    fine = np.asarray([stationary_joint_type(kind, qf, nuf, block_size=1024) @ wf for kind in kinds])
    coarse = np.asarray([stationary_joint_type(kind, qc, nuc, block_size=1024) @ wc for kind in kinds])
    require(np.isfinite(fine).all() and np.isfinite(coarse).all()
            and fine.min() > 0 and coarse.min() > 0, "Invalid bank moments")
    return fine, coarse, len(qf), len(qc)


def task_paths(out, index):
    stem = out / "tasks" / f"bank-{index:04d}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json"), stem.with_suffix(".log")


def worker(out, index):
    out = output_path(out); plan = load_plan(out)
    require(0 <= index < BANKS, "Invalid bank index")
    npz, report_path, _ = task_paths(out, index)
    require(not npz.exists() and not report_path.exists(), "Bank output already exists")
    began = time.monotonic(); kinds = [tuple(json.loads(value)[0:1]) for value in []]
    # JSON turns the nested count tuple into a list; reconstruct it explicitly.
    kinds = [(json.loads(value)[0], tuple(json.loads(value)[1])) for value in plan["frequency_types"]]
    fine, coarse, nf, nc = generate_bank(index, kinds)
    temporary = Path(str(npz) + ".tmp.npz"); npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, fine_joint=fine, coarse_joint=coarse)
    os.replace(temporary, npz)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin": peak *= 1024
    report = dict(schema="copying-frequency-paired-bank-v1", plan_id=plan["plan_id"],
        bank=index, fine_latent_rows=nf, coarse_latent_rows=nc,
        elapsed_seconds=time.monotonic() - began, peak_rss_bytes=int(peak),
        arrays_sha256=sha(npz), complete=True)
    write(report_path, report)
    print(json.dumps(report), flush=True)


def _delete_metrics(full, leave):
    full = np.asarray(full); leave = np.asarray(leave); estimates = leave.mean(1)
    center = estimates.mean(0); banks = len(estimates)
    covariance = (banks - 1) / banks * (estimates - center).T @ (estimates - center)
    bias = (banks - 1) * (center - full.mean(0))
    return dict(mcse=np.sqrt(np.maximum(0, np.diag(covariance))), bias=np.abs(bias))


def score(joints, groups, kinds):
    index = {kind_key(kind): i for i, kind in enumerate(kinds)}
    total = joints.sum(0); full = total / len(joints); leave = (total - joints) / (len(joints) - 1)
    result = {}
    for group in groups:
        loss = np.empty((256, 8)); loo = np.empty((len(joints), 256, 8))
        for i, row in enumerate(group["tokens"]):
            for n in range(1, 9):
                prefix = row[:n].tolist(); denominator = full[index[kind_key(frequency_type(prefix))]]
                denominators = leave[:, index[kind_key(frequency_type(prefix))]]
                prediction = np.empty(64); predictions = np.empty((len(joints), 64))
                for label in range(64):
                    column = index[kind_key(frequency_type(prefix + [label]))]
                    prediction[label] = full[column] / denominator
                    predictions[:, label] = leave[:, column] / denominators
                require(abs(prediction.sum() - 1) < 1e-10
                        and np.max(abs(predictions.sum(1) - 1)) < 1e-10,
                        "Predictive normalization failed")
                oracle = group["rows"][i, row[n - 1]]
                loss[i, n - 1] = -oracle @ np.log2(prediction)
                loo[:, i, n - 1] = -np.log2(predictions) @ oracle
        result[group["condition"]["id"]] = (loss, loo)
    return result


def merge(out):
    out = output_path(out); plan = load_plan(out); groups, _ = load_groups()
    kinds = [(json.loads(value)[0], tuple(json.loads(value)[1])) for value in plan["frequency_types"]]
    fine = []; coarse = []; reports = []
    for index in range(BANKS):
        npz, report_path, _ = task_paths(out, index); report = read(report_path)
        require(report["complete"] and report["plan_id"] == plan["plan_id"]
                and report["arrays_sha256"] == sha(npz), "Incomplete bank")
        with np.load(npz, allow_pickle=False) as saved:
            fine.append(saved["fine_joint"].copy()); coarse.append(saved["coarse_joint"].copy())
        reports.append(report)
    fine = score(np.asarray(fine), groups, kinds); coarse = score(np.asarray(coarse), groups, kinds)
    summaries = []
    for condition in ("a1_L0", "a1_L2", "a1_L3"):
        fl, fj = fine[condition]; cl, cj = coarse[condition]
        fm = _delete_metrics(fl, fj); dm = _delete_metrics(fl - cl, fj - cj)
        np.savez_compressed(out / (condition + ".npz"), fine_stationary_loss=fl,
            fine_stationary_leave_one_bank_out_loss=fj, coarse_stationary_loss=cl,
            coarse_stationary_leave_one_bank_out_loss=cj,
            paired_difference_leave_one_bank_out_loss=fj - cj)
        summaries.append(dict(condition=condition,
            mean_difference_bits=(fl - cl).mean(0).tolist(),
            fine_bank_mcse_bits=fm["mcse"].tolist(), fine_bank_bias_bits=fm["bias"].tolist(),
            paired_comparison_mcse_bits=dm["mcse"].tolist(),
            paired_comparison_bias_bits=dm["bias"].tolist()))
    summary = dict(schema="copying-frequency-paired-refinement-v1", complete=True,
        plan_id=plan["plan_id"], banks=BANKS, times=list(range(1, 9)), groups=summaries,
        maximum_worker_elapsed_seconds=max(x["elapsed_seconds"] for x in reports),
        maximum_worker_peak_rss_bytes=max(x["peak_rss_bytes"] for x in reports),
        finite_shuffle_exact=False, payoff_forecast_approved=False)
    write(out / "summary.json", summary)
    return summary


def rss_bytes(pid):
    if sys.platform == "darwin":
        import ctypes
        class TaskInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                ("virtual_size", "resident_size", "total_user", "total_system", "threads_user", "threads_system")] + [
                (name, ctypes.c_int32) for name in ("policy", "faults", "pageins", "cow_faults", "messages_sent",
                "messages_received", "syscalls_mach", "syscalls_unix", "csw", "threadnum", "numrunning", "priority")]
        library = ctypes.CDLL("/usr/lib/libproc.dylib"); info = TaskInfo()
        copied = library.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        return 0 if copied == 0 else int(info.resident_size)
    status = Path(f"/proc/{pid}/status")
    if not status.is_file(): return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"): return int(line.split()[1]) * 1024
    return 0


def stop(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def run(out):
    out = output_path(out)
    require(not out.exists(), "Immutable allocation already exists; choose a new output")
    require(shutil.disk_usage(out.parent).free >= MINIMUM_FREE_DISK_BYTES, "Insufficient free disk")
    plan = build_plan(out); out.mkdir(); (out / "tasks").mkdir(); write(out / "plan.json", plan)
    allocation = dict(schema="copying-frequency-paired-allocation-v1", plan_id=plan["plan_id"],
        status="running", dispatcher_pid=os.getpid(), started_unix=time.time(), attempts=[])
    write(out / "allocation.json", allocation)
    pending = list(range(BANKS)); active = {}; failed = False
    while pending or active:
        while pending and len(active) < MAX_WORKERS:
            index = pending.pop(0); _, _, log = task_paths(out, index); stream = log.open("x")
            process = subprocess.Popen([plan["python"], str(Path(__file__).resolve()), "worker",
                "--out", str(out), "--index", str(index)], cwd=ROOT,
                env={**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"},
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active[index] = dict(process=process, stream=stream, began=time.monotonic(), peak=0)
        time.sleep(.1)
        for index, item in list(active.items()):
            process = item["process"]; elapsed = time.monotonic() - item["began"]
            item["peak"] = max(item["peak"], rss_bytes(process.pid))
            forced = None
            if elapsed > TASK_WALL_SECONDS: forced = "time_limit"
            elif item["peak"] > MAX_RSS_BYTES: forced = "memory_limit"
            if forced: stop(process)
            code = process.poll()
            if code is None: continue
            item["stream"].close(); status = "ready" if code == 0 and forced is None else forced or "failed"
            attempt = dict(bank=index, status=status, exit_code=code, elapsed_seconds=elapsed,
                observed_peak_rss_bytes=item["peak"]); allocation["attempts"].append(attempt)
            del active[index]
            if status != "ready": failed = True; pending.clear()
        write(out / "allocation.json", allocation)
        if failed:
            for item in active.values(): stop(item["process"]); item["stream"].close()
            active.clear()
    if failed:
        allocation["status"] = "failed"; write(out / "allocation.json", allocation); return 3
    summary = merge(out); allocation.update(status="complete", finished_unix=time.time(), summary_sha256=sha(out / "summary.json"))
    write(out / "allocation.json", allocation)
    print(json.dumps(dict(complete=True, output=str(out), plan_id=plan["plan_id"],
        maximum_worker_seconds=summary["maximum_worker_elapsed_seconds"])), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "merge"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        require(args.index is not None, "Worker index required"); worker(args.out, args.index); return 0
    if args.action == "merge": merge(args.out); return 0
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

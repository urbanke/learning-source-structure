"""Exercise scaling orchestration with fake Slurm/CUDA; no training or inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path("code/transformer/kuma/submit_hidden_factor_scaling.sh")
BATCH = Path("code/transformer/kuma/hidden_factor_scaling_train.sbatch")
HELPER = Path("code/scripts/hidden_factor_scaling_benchmark.py")
PLAN = Path("code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json")
spec = importlib.util.spec_from_file_location("scaling_benchmark_test_target", ROOT / HELPER)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "isolated repository"
    for relative in (LAUNCHER, BATCH, HELPER, PLAN):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return root


def invoke(repo, *args, env=None):
    return subprocess.run(["bash", str(repo / LAUNCHER), *map(str, args)],
                          cwd=repo.parent, env=env, capture_output=True, text=True)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def prepare_benchmark(repo, out):
    run = out / "benchmark/large/seed0"
    run.mkdir(parents=True)
    plan = json.loads((repo / PLAN).read_text())
    cfg = dict(plan["arms"]["large"]["config"], seed=0, source_dimensions=plan["source_dimensions"],
               scaling_provenance=dict(plan_sha256=benchmark.sha(repo / PLAN), arm="large"))
    model = {k: cfg[k] for k in ("n_ctx", "n_layer", "n_head", "d_model")}
    model.update(vocab=16, mlp=True, dropout=0.0)
    receipt = dict(config=cfg, config_hash=digest(cfg), device="cuda", model=model,
                   n_params=plan["arms"]["large"]["n_params"], implementation_hashes=plan["implementation_sha256"])
    state = dict(step=1000, production_steps=100000, complete=False, reason="max_steps",
                 completed_sequences=64000, config_hash=digest(cfg),
                 cuda_peak_memory_allocated_bytes=123456789, cuda_peak_memory_reserved_bytes=234567890)
    write_json(run / "config.json", receipt)
    write_json(run / "status.json", state)
    write_json(run / "scaling_runtime.json", dict(plan_sha256=benchmark.sha(repo / PLAN), arm="large", seed=0,
        step=1000, device="cuda", return_code=75, cuda_device="FAKE H100 - test fixture",
        max_memory_allocated_bytes=123456789, max_memory_reserved_bytes=234567890, total_memory_bytes=80000000000))
    records = [dict(event="train", step=i, run_elapsed_seconds=10 + i*.04, loss_bits=3.2)
               for i in (100, 500, 1000)]
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    result = benchmark.collect(repo / PLAN, run)
    write_json(run / "benchmark.json", result)
    return run


def prepare_baseline(base):
    for relative in ("cohort.json", "cohort.npz", "references/manifest.json", "models/seed0/manifest.json", "models/seed1/manifest.json"):
        path = base / "eval" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test placeholder; no scientific data\n")
    for seed in (0, 1):
        for name in ("latest.pt", "model_100000.pt"):
            path = base / "train" / f"seed{seed}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test checkpoint placeholder\n")


def fake_runtime(tmp_path, repo, train_exit=0, array_id="0"):
    binary = tmp_path / "fakebin"
    binary.mkdir(exist_ok=True)
    calls = tmp_path / "fake_calls.jsonl"
    # Fake launch processes only record routing. They cannot contact Slurm or
    # execute the actual trainer/evaluator. The benchmark helper is stdlib-only.
    for command in ("sbatch", "srun", "python"):
        path = binary / command
        path.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['FAKE_CALLS'], 'a') as stream:
    stream.write(json.dumps(dict(program=name,args=args))+'\n')
if name == 'sbatch':
    print('FAKE_JOB_ID')
elif name == 'srun':
    if any(a.endswith('train_hidden_factor_scaling.py') for a in args):
        sys.exit(int(os.environ['FAKE_TRAIN_EXIT']))
elif args and args[0].endswith('hidden_factor_scaling_benchmark.py'):
    os.execv(sys.executable, [sys.executable,*args])
elif args[:2] == ['-m','pip']:
    print('fake-runtime==0')
elif args[:1] == ['-c']:
    print('FAKE CUDA; no real device inspected')
else:
    raise SystemExit('Unexpected fake Python invocation')
''')
        path.chmod(0o755)
    venv = tmp_path / "fakevenv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin/activate").write_text(f"export PATH={shlex.quote(str(binary))}:\"$PATH\"\n")
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               HIDDEN_FACTOR_VENV=str(venv), FAKE_CALLS=str(calls), FAKE_TRAIN_EXIT=str(train_exit),
               SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_ID="test-job", SLURM_ARRAY_TASK_ID=str(array_id))
    return env, calls


def read_calls(path):
    return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]


def test_frozen_plan_retains_source_training_budgets_and_development_test_set(repo):
    plan = json.loads((repo / PLAN).read_text())
    assert plan["source_dimensions"] == [2, 8] and plan["seeds"] == [0, 1]
    assert plan["baseline"]["completed_sequences"] == 6_400_000
    assert plan["arms"]["longsmall"]["additional_sequences"] == 19_200_000
    large = plan["arms"]["large"]["config"]
    assert large["steps"] * large["batch"] == 6_400_000
    assert large["experiment"] == "hidden-factor-dirichlet1-v1"
    assert (large["n_layer"], large["n_head"], large["d_model"]) == (4, 8, 256)
    assert plan["test"]["n_evaluate"] == 256 and plan["test"]["role"] == "development"
    assert plan["benchmark"]["steps"] == 1000 and plan["benchmark"]["separate_output"] is True


@pytest.mark.parametrize("flags", [(), ("--benchmark",), ("--summary",), ("--arm", "large"), ("--arm", "longsmall")])
def test_default_is_read_only_dry_run(repo, flags):
    before = tree(repo)
    result = invoke(repo, *flags)
    assert result.returncode == 0, result.stderr
    assert tree(repo) == before
    assert "hidden_factor_scaling" in result.stdout
    assert "hidden_factor_references" not in result.stdout
    assert "--plan" in result.stdout or str(PLAN) in result.stdout


@pytest.mark.parametrize("arm,array", [("all", "0-3%4"), ("longsmall", "0-1%2"), ("large", "2-3%2")])
def test_production_array_and_fixed_plan(repo, arm, array):
    result = invoke(repo, "--arm", arm)
    assert result.returncode == 0
    command = shlex.split(result.stdout)
    assert f"--array={array}" in command
    assert command[-1] == str(repo / PLAN)
    assert command[-2] == "train"
    assert f"--chdir={repo}" in command


def test_benchmark_and_summary_are_separate_routes(repo):
    benchmark_result = invoke(repo, "--benchmark")
    assert "--qos=debug" in benchmark_result.stdout
    assert "--time=00:30:00" in benchmark_result.stdout
    assert "--array" not in benchmark_result.stdout
    summary = shlex.split(invoke(repo, "--summary").stdout)
    assert "sbatch" not in summary and "summary" in summary
    assert "--baseline-eval" in summary and "--evaluations" in summary
    assert "hidden_factor_scaling_eval.py" in summary[1]


@pytest.mark.parametrize("flags", [("--unknown",), ("--arm", "tiny"), ("--arm",),
    ("--shape", "3x4"), ("--plan", "another.json"), ("--benchmark", "--arm", "large"),
    ("--summary", "--arm", "longsmall"), ("--benchmark", "--summary")])
def test_invalid_flags_fail_without_files(repo, flags):
    before = tree(repo)
    result = invoke(repo, *flags)
    assert result.returncode != 0
    assert tree(repo) == before


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant", "alias"])
def test_output_overlap_rejected_before_submission(repo, tmp_path, relation):
    base = tmp_path / "baseline"
    base.mkdir()
    if relation == "equal":
        out = base
    elif relation == "ancestor":
        out = tmp_path
    elif relation == "descendant":
        out = base / "new-results"
    else:
        out = tmp_path / "alias"
        out.symlink_to(base, target_is_directory=True)
    env, calls = fake_runtime(tmp_path, repo)
    result = invoke(repo, "--benchmark", "--submit", "--baseline-root", base, "--out", out, env=env)
    assert result.returncode != 0 and "separate" in result.stderr
    assert not any(r["program"] == "sbatch" for r in read_calls(calls))
    assert list(base.iterdir()) == []


def test_production_submission_requires_benchmark_then_existing_inputs(repo, tmp_path):
    out, base = tmp_path / "new-results", tmp_path / "baseline"
    env, calls = fake_runtime(tmp_path, repo)
    arguments = ("--submit", "--out", out, "--baseline-root", base)
    result = invoke(repo, *arguments, env=env)
    assert result.returncode != 0 and not out.exists()
    prepare_benchmark(repo, out)
    result = invoke(repo, *arguments, env=env)
    assert result.returncode != 0 and "Missing baseline evidence" in result.stderr
    prepare_baseline(base)
    baseline_before = tree(base)
    result = invoke(repo, *arguments, env=env)
    assert result.returncode == 0, result.stderr
    assert [r["program"] for r in read_calls(calls)].count("sbatch") == 1
    assert tree(base) == baseline_before


@pytest.mark.parametrize("index,arm,seed,step", [(0,"longsmall",0,400000), (1,"longsmall",1,400000),
                                                 (2,"large",0,100000), (3,"large",1,100000)])
def test_batch_maps_arm_seed_and_evaluates_only_final_checkpoint(repo, tmp_path, index, arm, seed, step):
    out, base = tmp_path / "results", tmp_path / "baseline"
    env, calls = fake_runtime(tmp_path, repo, array_id=index)
    result = subprocess.run(["bash", str(repo / BATCH), str(out), str(base), "train", str(repo / PLAN)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    runs = [r["args"] for r in read_calls(calls) if r["program"] == "srun"]
    assert len(runs) == 2
    training, evaluation = runs
    assert training[training.index("--arm")+1] == arm
    assert training[training.index("--seed")+1] == str(seed)
    expected_out = out / "train" / arm / f"seed{seed}"
    assert training[training.index("--out")+1] == str(expected_out)
    assert ("--parent-checkpoint" in training) == (arm == "longsmall")
    if arm == "longsmall":
        assert training[training.index("--parent-checkpoint")+1] == str(base / "train" / f"seed{seed}" / "latest.pt")
    assert "--max-steps" not in training
    assert evaluation[evaluation.index("--checkpoint")+1] == str(expected_out / f"model_{step}.pt")
    assert evaluation[evaluation.index("--baseline-eval")+1] == str(base / "eval")
    assert evaluation[evaluation.index("--out")+1] == str(out / "eval")
    assert "model" in evaluation and "summary" not in evaluation
    assert not any("reference" in arg and arg.endswith(".py") for run in runs for arg in run)


@pytest.mark.parametrize("exitcode", [1, 75])
def test_incomplete_training_does_not_evaluate(repo, tmp_path, exitcode):
    env, calls = fake_runtime(tmp_path, repo, train_exit=exitcode)
    result = subprocess.run(["bash", str(repo / BATCH), str(tmp_path / "out"), str(tmp_path / "base"), "train", str(repo / PLAN)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == exitcode
    assert len([r for r in read_calls(calls) if r["program"] == "srun"]) == 1


@pytest.mark.parametrize("exitcode,reason,expected_ok", [(75,"max_steps",True), (75,"signal",False),
                                                        (0,"max_steps",False), (1,"max_steps",False)])
def test_benchmark_accepts_only_intended_1000_step_cap(repo, tmp_path, exitcode, reason, expected_ok):
    out = tmp_path / "out"
    run = prepare_benchmark(repo, out)
    state = json.loads((run / "status.json").read_text())
    state["reason"] = reason
    write_json(run / "status.json", state)
    env, calls = fake_runtime(tmp_path, repo, train_exit=exitcode)
    result = subprocess.run(["bash", str(repo / BATCH), str(out), str(tmp_path / "base"), "benchmark", str(repo / PLAN)],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == expected_ok, result.stderr
    runs = [r["args"] for r in read_calls(calls) if r["program"] == "srun"]
    assert len(runs) == 1
    assert runs[0][runs[0].index("--max-steps")+1] == "1000"
    assert runs[0][runs[0].index("--out")+1] == str(run)


def test_helper_reports_measured_interval_not_full_walltime(repo, tmp_path):
    run = prepare_benchmark(repo, tmp_path / "out")
    result = benchmark.collect(repo / PLAN, run)
    assert result["seconds_per_step"] == pytest.approx(.04)
    assert result["projected_training_hours"] == pytest.approx(100000*.04/3600)
    assert result["completed_steps"] == 1000 and result["passed"] is True


@pytest.mark.parametrize("corruption", ["plan", "config", "model", "parameters", "counts", "production_steps",
    "identity", "false_identity", "device", "step", "complete", "nonfinite_loss", "negative_time", "missing_final_log",
    "implementation", "runtime_plan", "runtime_cap", "runtime_arm", "zero_memory", "excess_memory", "interrupted_clock"])
def test_helper_rejects_corrupt_benchmark_receipts(repo, tmp_path, corruption):
    run = prepare_benchmark(repo, tmp_path / "out")
    config = json.loads((run / "config.json").read_text())
    state = json.loads((run / "status.json").read_text())
    runtime = json.loads((run / "scaling_runtime.json").read_text())
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    if corruption == "plan":
        config["config"]["scaling_provenance"]["plan_sha256"] = "wrong"
    elif corruption == "config": config["config"]["n_layer"] = 9
    elif corruption == "model": config["model"]["vocab"] = 24
    elif corruption == "parameters": config["n_params"] -= 1
    elif corruption == "counts": state["completed_sequences"] -= 64
    elif corruption == "production_steps": state["production_steps"] = 1000
    elif corruption == "identity": state["config_hash"] = "wrong"
    elif corruption == "false_identity": config["config_hash"] = state["config_hash"] = "wrong"
    elif corruption == "device": config["device"] = "cpu"
    elif corruption == "step": state["step"] = 999
    elif corruption == "complete": state["complete"] = True
    elif corruption == "nonfinite_loss": records[0]["loss_bits"] = float("nan")
    elif corruption == "negative_time": records[-1]["run_elapsed_seconds"] = 0
    elif corruption == "missing_final_log": records.pop()
    elif corruption == "implementation": config["implementation_hashes"]["model.py"] = "wrong"
    elif corruption == "runtime_plan": runtime["plan_sha256"] = "wrong"
    elif corruption == "runtime_cap": runtime["step"] = 999
    elif corruption == "runtime_arm": runtime["arm"] = "longsmall"
    elif corruption == "zero_memory": runtime["max_memory_allocated_bytes"] = 0
    elif corruption == "excess_memory": runtime["max_memory_reserved_bytes"] = runtime["total_memory_bytes"] + 1
    elif corruption == "interrupted_clock": records[1]["run_elapsed_seconds"] = records[0]["run_elapsed_seconds"] / 2
    write_json(run / "config.json", config)
    write_json(run / "status.json", state)
    write_json(run / "scaling_runtime.json", runtime)
    (run / "log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    with pytest.raises((ValueError, KeyError)):
        benchmark.collect(repo / PLAN, run)


def test_batch_rejects_invalid_array_index_before_runtime(repo, tmp_path):
    env, calls = fake_runtime(tmp_path, repo, array_id=4)
    result = subprocess.run(["bash", str(repo / BATCH), str(tmp_path / "out"), str(tmp_path / "base"), "train", str(repo / PLAN)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and not calls.exists()

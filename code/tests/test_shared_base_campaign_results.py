"""Independent arithmetic and rejection checks for imported campaign scores.

Fixtures use normalized artificial predictions, not model inference.  They
exercise pairing, source support, uncertainty and the fixed training contract.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import shared_base_campaign_results as results


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@pytest.fixture
def score_fixture():
    """Vary both time and draw so a population-SD SE gives the wrong answer."""
    n, nt, d = 256, 6, 64
    source_ids = np.asarray([f"{index:064x}" for index in range(n)])
    times = np.asarray([1, 4, 16, 64, 256, 1024])
    i = np.arange(n)[:, None, None]
    t = np.arange(nt)[None, :, None]
    b = np.arange(d)[None, None, :]
    q = 1 + ((3 * i + 5 * t + 7 * b) % 19) / 13
    q = q / q.sum(-1, keepdims=True)
    oracle = np.broadcast_to(1 + ((2 * i + 3 * t + b) % 11), q.shape).astype(float).copy()
    oracle[..., ::2] = 0  # Zero true probabilities must contribute zero to KL.
    oracle /= oracle.sum(-1, keepdims=True)
    logp = np.log(q)
    # Scalar calculation is deliberately separate from the vectorized importer.
    expected = np.empty((n, nt))
    regret = np.empty((n, nt))
    for row in range(n):
        for position in range(nt):
            expected[row, position] = -sum(float(p) * np.log2(float(pred))
                                          for p, pred in zip(oracle[row, position], q[row, position]))
            regret[row, position] = sum(float(p) * np.log2(float(p) / float(pred))
                                        for p, pred in zip(oracle[row, position], q[row, position]) if p > 0)
    frozen = dict(oracle_probs=oracle, source_ids=source_ids,
                  source_indices=np.arange(n), times=times,
                  tokens=np.tile(np.arange(1025) % d, (n, 1)))
    arrays = dict(source_ids=source_ids.copy(), source_indices=np.arange(n),
                  times=times.copy(), log_predictive_probs=logp,
                  expected_loss_bits=expected, regret_bits=regret)
    return arrays, frozen


def test_score_reconstruction_with_zero_oracle_support(score_fixture):
    arrays, frozen = score_fixture
    regret, expected, checks = results.score_arrays(arrays, frozen)
    np.testing.assert_allclose(regret, arrays["regret_bits"], rtol=0, atol=2e-13)
    np.testing.assert_allclose(expected, arrays["expected_loss_bits"], rtol=0, atol=2e-13)
    assert isinstance(checks, dict)


@pytest.mark.parametrize("field", ["source_ids", "source_indices", "times"])
def test_rejects_unpaired_coordinates(score_fixture, field):
    arrays, frozen = score_fixture
    arrays[field] = arrays[field][::-1].copy()
    with pytest.raises(ValueError):
        results.score_arrays(arrays, frozen)


@pytest.mark.parametrize("field", ["expected_loss_bits", "regret_bits"])
def test_rejects_saved_score_tampering(score_fixture, field):
    arrays, frozen = score_fixture
    arrays[field][7, 3] += .02
    with pytest.raises(ValueError):
        results.score_arrays(arrays, frozen)


def test_rejects_unnormalized_predictions(score_fixture):
    arrays, frozen = score_fixture
    arrays["log_predictive_probs"][0, 0] += np.log(1.001)
    with pytest.raises(ValueError):
        results.score_arrays(arrays, frozen)


def test_rejects_nonfinite_prediction(score_fixture):
    arrays, frozen = score_fixture
    arrays["log_predictive_probs"][0, 0, 1] = float("nan")
    with pytest.raises(ValueError):
        results.score_arrays(arrays, frozen)


def test_rejects_truncated_cohort(score_fixture):
    arrays, frozen = score_fixture
    arrays["log_predictive_probs"] = arrays["log_predictive_probs"][:-1]
    with pytest.raises(ValueError):
        results.score_arrays(arrays, frozen)


def test_mean_and_paired_standard_error(score_fixture):
    arrays, _ = score_fixture
    regret = arrays["regret_bits"]
    means = [sum(map(float, x)) / len(x) for x in regret.T]
    errors = [(sum((float(x) - mean)**2 for x in col) / (len(col) * (len(col) - 1)))**.5
              for col, mean in zip(regret.T, means)]
    group = dict(mean_regret_bits=means, se_regret_bits=errors)
    assert results.compare_statistics(regret, group) < 1e-13
    wrong = deepcopy(group)
    wrong["se_regret_bits"] = (regret.std(0, ddof=0) / 16).tolist()
    with pytest.raises(ValueError):
        results.compare_statistics(regret, wrong)
    wrong = deepcopy(group)
    wrong["mean_regret_bits"][2] += .001
    with pytest.raises(ValueError):
        results.compare_statistics(regret, wrong)


@pytest.fixture
def config_fixture():
    plan_path = ROOT / "code/transformer/configs/shared_base_campaign_plan_2026-09-18.json"
    parent_path = ROOT / "output/shared_base_scaling_download_2026-09-17/seed0/config.json"
    if not parent_path.is_file():
        pytest.skip("Pinned parent receipt is not available locally")
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    parent = json.loads(parent_path.read_text())
    arm = "continue_large"
    cfg = dict(schema="shared-base-campaign-training-v1", **plan["arms"][arm]["config"],
               **plan["source"], seed=0, arm=arm, continuation_lr=3e-5,
               parent=dict(directory="/home/urbanke/Projects/example/seed0",
                           latest_sha256="a" * 64,
                           weights_sha256=plan["parent"]["seeds"]["0"]["model_100000.pt"],
                           config_hash=parent["config_hash"], step=100000))
    receipt = deepcopy(parent)
    receipt.update(config=cfg, config_hash=digest(cfg), plan_sha256=plan_hash,
                   implementation_hashes=plan["implementation_sha256"])
    status = dict(schema="shared-base-campaign-status-v1", step=400000,
                  production_steps=400000, complete=True, reason="completed",
                  completed_sequences=25600000, config_hash=digest(cfg),
                  plan_sha256=plan_hash, training_elapsed_seconds=20000.)
    return plan, plan_hash, receipt, status, arm, 0, parent


def test_valid_completed_continuation_contract(config_fixture):
    metadata = results.validate_config(*config_fixture)
    assert isinstance(metadata, dict)


def test_fresh_larger_arm_has_no_parent(config_fixture):
    plan, plan_hash, receipt, status, _, seed, _ = config_fixture
    arm = "xlarge"
    cfg = dict(schema="shared-base-campaign-training-v1", **plan["arms"][arm]["config"],
               **plan["source"], seed=seed, arm=arm, parent=None, continuation_lr=None)
    receipt.update(config=cfg, config_hash=digest(cfg), n_params=25809920,
                   model=dict(vocab=64, n_ctx=1024, n_layer=8, n_head=8,
                              d_model=512, mlp=True, dropout=0.))
    status["config_hash"] = receipt["config_hash"]
    assert isinstance(results.validate_config(plan, plan_hash, receipt, status, arm, seed, None), dict)
    cfg["parent"] = dict(step=100000)
    receipt["config_hash"] = digest(cfg)
    status["config_hash"] = receipt["config_hash"]
    with pytest.raises(ValueError):
        results.validate_config(plan, plan_hash, receipt, status, arm, seed, None)


@pytest.mark.parametrize("mutation", ["parent_weights", "parent_step", "parent_config", "config_digest",
                                      "architecture", "source_prior", "exposure", "precision", "tf32"])
def test_rejects_false_training_contract(config_fixture, mutation):
    plan, plan_hash, receipt, status, arm, seed, parent = config_fixture
    if mutation == "parent_weights":
        receipt["config"]["parent"]["weights_sha256"] = "f" * 64
    elif mutation == "parent_step":
        receipt["config"]["parent"]["step"] = 99999
    elif mutation == "parent_config":
        receipt["config"]["parent"]["config_hash"] = "f" * 64
    elif mutation == "config_digest":
        receipt["config_hash"] = "f" * 64
    elif mutation == "architecture":
        receipt["model"]["n_layer"] = 8
    elif mutation == "source_prior":
        receipt["config"]["alpha"] = [.5, 2.]
    elif mutation == "exposure":
        status["completed_sequences"] = 19200000  # Additional is not total exposure.
    elif mutation == "precision":
        receipt["precision"] = "bfloat16"
    elif mutation == "tf32":
        receipt["runtime"]["cuda_matmul_allow_tf32"] = True
    # Recompute identity after edits so content validation, not only stale hashes,
    # must detect an unauthorized ancestry/source modification.
    if mutation in {"parent_weights", "parent_step", "parent_config", "source_prior"}:
        receipt["config_hash"] = digest(receipt["config"])
        status["config_hash"] = receipt["config_hash"]
    with pytest.raises(ValueError):
        results.validate_config(plan, plan_hash, receipt, status, arm, seed, parent)


@pytest.fixture
def runtime_fixture(config_fixture):
    _, plan_hash, _, status, arm, seed, _ = config_fixture
    runtime = dict(schema="shared-base-campaign-runtime-v1", plan_sha256=plan_hash,
                   arm=arm, seed=seed, step=status["step"], device="cuda", return_code=0,
                   cuda_device="NVIDIA H100", max_memory_allocated_bytes=1_000_000,
                   max_memory_reserved_bytes=2_000_000, total_memory_bytes=80_000_000)
    return runtime, status, plan_hash, arm, seed


def test_completed_and_interrupted_runtime(runtime_fixture):
    runtime, status, plan_hash, arm, seed = runtime_fixture
    results.validate_runtime(runtime, status, plan_hash, arm, seed)
    # A signal stops at any update, not necessarily a saved evaluation milestone.
    runtime.update(arm="xlarge", step=291592, return_code=75)
    status.update(step=291592, completed_sequences=291592 * 64, complete=False, reason="SIGUSR1")
    results.validate_runtime(runtime, status, plan_hash, "xlarge", seed)


@pytest.mark.parametrize("field,value", [("step", 399999), ("device", "cpu"), ("return_code", 75),
                                         ("seed", 1), ("plan_sha256", "f" * 64)])
def test_rejects_conflicting_runtime(runtime_fixture, field, value):
    runtime, status, plan_hash, arm, seed = runtime_fixture
    runtime[field] = value
    with pytest.raises(ValueError):
        results.validate_runtime(runtime, status, plan_hash, arm, seed)

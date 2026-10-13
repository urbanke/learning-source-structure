import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))

import copying_frequency_paired_refinement as paired


def test_coupled_quadrature_has_correct_marginal_weights():
    rows = paired.quadrature_rows(16, 32)
    assert len(rows) == 32
    assert [row["coarse_index"] for row in rows] == [i // 2 for i in range(32)]
    np.testing.assert_allclose(
        sum(row["fine_alpha_weight"] for row in rows), 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        [sum(row["coarse_alpha_weight"] for row in rows if row["coarse_index"] == i)
         for i in range(16)],
        np.polynomial.legendre.leggauss(16)[1] / 2,
        rtol=0, atol=2e-15,
    )


def test_shared_rule_draws_give_valid_stationary_marginals():
    rng = np.random.default_rng(123)
    qf, nuf, qc, nuc = paired.coupled_transition_laws(
        fine_alpha=.73, coarse_alpha=.71, depth=2, repetitions=3,
        dimension=4, rng=rng)
    assert qf.shape == nuf.shape == qc.shape == nuc.shape == (3, 4)
    np.testing.assert_allclose(qf.sum(1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(qc.sum(1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(nuf.sum(1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(nuc.sum(1), 1, rtol=0, atol=1e-14)
    assert np.all(qf > 0) and np.all(qc > 0)
    assert np.all(nuf > 0) and np.all(nuc > 0)


def test_paired_delete_bank_error_cancels_common_noise():
    rng = np.random.default_rng(4)
    common = rng.normal(size=(12, 3))
    fine = common + np.arange(3) / 100
    coarse = common.copy()
    metrics = paired.paired_bank_metrics(fine, coarse)
    np.testing.assert_allclose(metrics["difference"], np.arange(3) / 100)
    np.testing.assert_allclose(metrics["comparison_mcse"], 0, atol=2e-16)
    np.testing.assert_allclose(metrics["comparison_bias"], 0, atol=2e-16)


def test_production_allocation_is_small_and_bounded():
    assert paired.BANKS == 16
    assert paired.REPETITIONS == 16
    assert paired.COARSE_ORDER == 16
    assert paired.FINE_ORDER == 32
    assert paired.MAX_WORKERS == 4
    assert paired.MAX_RSS_BYTES <= 2 * 2**30
    assert paired.TASK_WALL_SECONDS <= 300

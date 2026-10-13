"""Independent exact checks of the conditional B/U crossing calculation.

The finite examples integrate binary source depths zero and one exactly.
No production numerical tables or simulation estimates are used here.
"""
from fractions import Fraction
from itertools import product
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import conditional_crossing_summary as summary
from eval_conditional_crossing import (array_hash, conditional_metrics, draw_source, evaluate_sequence,
                                      fingerprint, load_record, save_record, validate_record)
from shared_depth import SharedDepthLSA
from source import Source


LN2 = math.log(2)


def row_evidence(counts, depth):
    k = sum(counts)
    if depth == 0:
        return Fraction(1, 2 ** k)
    assert depth == 1
    return Fraction(math.factorial(counts[0]) * math.factorial(counts[1]),
                    math.factorial(k + 1))


def source_counts(prefix):
    counts = [[0, 0], [0, 0]]
    for a, b in zip(prefix[:-1], prefix[1:]):
        counts[a][b] += 1
    return counts


def source_joint(prefix, depth):
    if not prefix:
        return Fraction(1)
    counts = source_counts(prefix)
    return Fraction(1, 2) * row_evidence(counts[0], depth) * row_evidence(counts[1], depth)


def bayes_joint(prefix):
    return (source_joint(prefix, 0) + source_joint(prefix, 1)) / 2


def bayes_predict(prefix):
    return np.array([float(bayes_joint(prefix + (b,)) / bayes_joint(prefix)) for b in (0, 1)])


def unigram_predict(prefix):
    return np.array([(prefix.count(b) + 1) / (len(prefix) + 2) for b in (0, 1)])


class ExactKernel:
    d = 2
    rmax = 32

    def __init__(self, depths=(0, 1)):
        self.depths = list(depths)

    def new_table(self):
        return ExactTable(self)


class ExactTable:
    def __init__(self, kernel):
        self.kernel = kernel
        self.k = 0
        self.m = np.zeros(2, dtype=int)
        self.logQ = None

    def _logQ_after(self, count):
        counts = self.m.copy()
        counts[np.flatnonzero(counts == count)[0]] += 1
        return np.array([math.log(float(row_evidence(counts.tolist(), depth)))
                         for depth in self.kernel.depths])

    def predictive(self):
        before = sum(row_evidence(self.m.tolist(), depth) for depth in self.kernel.depths)
        p = []
        for b in (0, 1):
            after = self.m.copy()
            after[b] += 1
            p.append(float(sum(row_evidence(after.tolist(), depth) for depth in self.kernel.depths) / before))
        return np.array(p)

    def observe(self, symbol):
        loss = -math.log2(self.predictive()[symbol])
        self.logQ = self._logQ_after(self.m[symbol])
        self.m[symbol] += 1
        self.k += 1
        return loss


def test_metrics_use_oracle_rows_and_score_only_supplied_next_targets():
    b = np.array([[[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]])
    u = np.array([[[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]])
    r = np.array([[[0.9, 0.1], [0.2, 0.8], [0.65, 0.35]]])
    targets = np.array([[1, 0, 1]])
    result = conditional_metrics(b, u, r, targets)
    expected_b = np.sum(r * np.log2(r / b), axis=-1)
    expected_u = np.sum(r * np.log2(r / u), axis=-1)
    np.testing.assert_allclose(result["bayes_regret_bits"], expected_b, atol=1e-14)
    np.testing.assert_allclose(result["unigram_regret_bits"], expected_u, atol=1e-14)
    np.testing.assert_allclose(result["instantaneous_gap_bits"], np.sum(r * np.log2(b / u), axis=-1), atol=1e-14)
    np.testing.assert_allclose(result["bayes_loss_bits"], -np.log2([[0.7, 0.8, 0.6]]), atol=1e-14)
    np.testing.assert_allclose(result["unigram_loss_bits"], -np.log2([[0.3, 0.6, 0.5]]), atol=1e-14)
    np.testing.assert_allclose(result["oracle_loss_bits"], -np.log2([[0.1, 0.2, 0.35]]), atol=1e-14)
    np.testing.assert_array_equal(result["cumulative_gap_bits"][:, 0], [0.0])
    np.testing.assert_allclose(np.diff(result["cumulative_gap_bits"], axis=-1),
                               result["instantaneous_gap_bits"], atol=1e-14)


@pytest.mark.parametrize("which", ["bayes", "unigram", "oracle"])
def test_material_raw_probability_error_is_rejected_not_renormalized(which):
    values = {name: np.array([[0.4, 0.6]]) for name in ("bayes", "unigram", "oracle")}
    values[which] = values[which] * 1.01
    with pytest.raises((ValueError, FloatingPointError)):
        conditional_metrics(values["bayes"], values["unigram"], values["oracle"], np.array([0]))


@pytest.mark.parametrize("source_depth", [0, 1])
def test_exact_binary_source_expectations_match_observed_loss_gaps(source_depth):
    length = 7
    paths = list(product((0, 1), repeat=length))
    weights = np.array([float(source_joint(path, source_depth)) for path in paths])
    assert sum(source_joint(path, source_depth) for path in paths) == 1
    bayes = np.array([[bayes_predict(path[:n]) for n in range(1, length)] for path in paths])
    unigram = np.array([[unigram_predict(path[:n]) for n in range(1, length)] for path in paths])
    source_mean, mean_row_entropy = [], []
    for path in paths:
        means, entropies = [], []
        for n in range(1, length):
            counts = np.array(source_counts(path[:n])[path[n - 1]], dtype=float)
            mu = np.full(2, 0.5) if source_depth == 0 else (counts + 1) / (counts.sum() + 2)
            # E[H(T_row)|prefix] for a Beta(count+1) row posterior.
            h = 1.0 if source_depth == 0 else float(
                (digamma(counts.sum() + 3) - mu @ digamma(counts + 2)) / LN2)
            means.append(mu)
            entropies.append(h)
        source_mean.append(means)
        mean_row_entropy.append(entropies)
    source_mean, mean_row_entropy = np.array(source_mean), np.array(mean_row_entropy)
    result = conditional_metrics(bayes, unigram, source_mean, np.array(paths)[:, 1:])
    # The gap is linear in the true row, so its posterior mean gives the
    # exact integrated gap. Individual regrets additionally need the row
    # entropy correction; the test does not identify that mean with T.
    correction = -np.sum(source_mean * np.log2(source_mean), axis=-1) - mean_row_entropy
    integrated_b = weights @ (result["bayes_regret_bits"] + correction)
    integrated_u = weights @ (result["unigram_regret_bits"] + correction)
    realized_b = weights @ (result["bayes_loss_bits"] - mean_row_entropy)
    realized_u = weights @ (result["unigram_loss_bits"] - mean_row_entropy)
    np.testing.assert_allclose(integrated_b, realized_b, atol=2e-14)
    np.testing.assert_allclose(integrated_u, realized_u, atol=2e-14)
    exact_gap = weights @ result["instantaneous_gap_bits"]
    np.testing.assert_allclose(exact_gap, weights @ (result["unigram_loss_bits"] - result["bayes_loss_bits"]), atol=2e-14)
    expected_first_gap = 0.5 * math.log2(9 / 8)
    assert exact_gap[0] == pytest.approx(expected_first_gap, abs=1e-14)
    assert integrated_b[0] == pytest.approx(0 if source_depth == 0 else 1 - 1 / (2 * LN2), abs=1e-14)
    cumulative = weights @ result["cumulative_gap_bits"]
    assert cumulative[0] == 0
    np.testing.assert_allclose(np.diff(cumulative), exact_gap, atol=2e-14)


def test_sequence_evaluator_matches_exact_joint_ratios_and_free_first_token():
    bayes = SharedDepthLSA(ExactKernel())
    unigram = ExactKernel(depths=(1,))
    matrix = np.array([[0.25, 0.75], [0.625, 0.375]])
    for path in product((0, 1), repeat=6):
        arrays, checks = evaluate_sequence(np.array(path), matrix, bayes, unigram)
        np.testing.assert_array_equal(arrays["prefix_lengths"], np.arange(1, 6))
        np.testing.assert_array_equal(arrays["cumulative_prefix_lengths"], np.arange(1, 7))
        np.testing.assert_array_equal(arrays["targets"], path[1:])
        np.testing.assert_array_equal(arrays["oracle_probs"], matrix[np.array(path[:-1])])
        for n in range(1, 6):
            np.testing.assert_allclose(arrays["bayes_probs"][n - 1], bayes_predict(path[:n]), atol=1e-14)
            np.testing.assert_allclose(arrays["unigram_probs"][n - 1], unigram_predict(path[:n]), atol=1e-14)
        expected_b_loss = -math.log2(float(2 * bayes_joint(path)))
        counts = [path.count(b) for b in (0, 1)]
        expected_u_loss = -math.log2(float(2 * row_evidence(counts, 1)))
        assert arrays["bayes_loss_bits"].sum() == pytest.approx(expected_b_loss, abs=2e-14)
        assert arrays["unigram_loss_bits"].sum() == pytest.approx(expected_u_loss, abs=2e-14)
        assert checks["source_averaged_first_gap_bits"] == pytest.approx(0.5 * math.log2(9 / 8), abs=1e-14)
        assert arrays["observed_cumulative_gap_bits"][0] == 0
        assert arrays["observed_cumulative_gap_bits"][-1] == pytest.approx(expected_u_loss - expected_b_loss, abs=2e-14)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_conditional_source_matches_existing_generator_and_relabels_both_axes(depth):
    seed, d, n_ctx = 20260925, 4, 12
    actual = draw_source(d, n_ctx, depth, seed)
    source = Source(d, n_ctx + 1, alpha=0, ls=depth, permute=True)
    expected_tokens, parameters = source.batch(1, np.random.default_rng(seed))
    np.testing.assert_array_equal(actual["tokens"], expected_tokens[0])
    np.testing.assert_array_equal(actual["permutation"], parameters["perm"][0])
    rng = np.random.default_rng(seed)
    alpha, depths = source.draw_params(1, rng)
    initial, raw_matrix = source.matrices(alpha, depths, rng)
    inverse = np.argsort(parameters["perm"][0])
    np.testing.assert_array_equal(actual["initial_probs"], initial[0, inverse])
    np.testing.assert_array_equal(actual["transition_probs"], raw_matrix[0][inverse][:, inverse])
    raw_tokens = inverse[actual["tokens"]]
    observed_path_likelihood = np.prod(actual["transition_probs"][actual["tokens"][:-1], actual["tokens"][1:]])
    raw_path_likelihood = np.prod(raw_matrix[0, raw_tokens[:-1], raw_tokens[1:]])
    assert observed_path_likelihood == raw_path_likelihood
    np.testing.assert_allclose(actual["initial_probs"], 1 / d, atol=1e-15)


def exact_record(seed=20260925):
    source = draw_source(2, 5, 1, seed)
    arrays, checks = evaluate_sequence(source["tokens"], source["transition_probs"],
                                      SharedDepthLSA(ExactKernel()), ExactKernel(depths=(1,)),
                                      conditioned_bayes=SharedDepthLSA(ExactKernel(depths=(1,))))
    arrays = dict(**source, **arrays)
    identity = dict(source=dict(d=2, n_tokens=6),
                    prediction=dict(bayes_depths=[0, 1], bayes_depth_prior=[0.5, 0.5],
                                    diagnostic_conditioned_depth=1, prob_tol=1e-7, loss_tol=2e-6))
    metadata = dict(identity=identity, cache_id=fingerprint(identity), checks=checks,
                    array_sha256={key: array_hash(value) for key, value in arrays.items()})
    return arrays, metadata


def test_cache_preserves_all_arrays_and_refuses_overwriting(tmp_path):
    arrays, metadata = exact_record()
    path = tmp_path / "sequence.npz"
    save_record(path, arrays, metadata)
    restored, saved_metadata = load_record(path, metadata["identity"])
    assert metadata == saved_metadata
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], restored[key])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_record(path, arrays, metadata)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="inputs changed"):
        load_record(path, dict(changed=True))


def test_cache_rejects_inconsistent_target_and_oracle_alignment_even_with_fresh_hashes():
    arrays, metadata = exact_record()
    arrays["targets"] = 1 - arrays["targets"]
    metadata["array_sha256"]["targets"] = array_hash(arrays["targets"])
    with pytest.raises(ValueError, match="indexing"):
        validate_record(arrays, metadata)
    arrays, metadata = exact_record()
    arrays["oracle_probs"] = arrays["oracle_probs"][:, ::-1]
    metadata["array_sha256"]["oracle_probs"] = array_hash(arrays["oracle_probs"])
    with pytest.raises(ValueError, match="current-row"):
        validate_record(arrays, metadata)


def test_conditioned_diagnostic_is_source_marginal_prediction_not_matrix_oracle():
    tokens = np.array([0, 0, 1, 1, 0, 1])
    matrix = np.array([[0.25, 0.75], [0.625, 0.375]])
    arrays, _ = evaluate_sequence(tokens, matrix, SharedDepthLSA(ExactKernel()), ExactKernel(depths=(1,)),
                                  conditioned_bayes=SharedDepthLSA(ExactKernel(depths=(1,))))
    expected = []
    for n in range(1, len(tokens)):
        prefix = tuple(tokens[:n])
        expected.append([float(source_joint(prefix + (b,), 1) / source_joint(prefix, 1)) for b in (0, 1)])
    np.testing.assert_allclose(arrays["conditioned_bayes_probs"], expected, atol=1e-14)
    assert not np.allclose(arrays["conditioned_bayes_probs"], arrays["oracle_probs"])
    # At the first prediction the depth-known and family mixtures are both
    # uniform, even though the actual transition row is not.
    assert arrays["conditioned_to_bayes_kl_bits"][0] == pytest.approx(0, abs=1e-14)
    assert arrays["bayes_regret_bits"][0] > 0
    exact_conditional_code = float(2 * source_joint(tuple(tokens), 1))
    assert arrays["conditioned_loss_bits"].sum() == pytest.approx(-math.log2(exact_conditional_code), abs=1e-14)
    np.testing.assert_allclose(arrays["conditioned_gap_bits"],
                               arrays["conditioned_to_unigram_kl_bits"] - arrays["conditioned_to_bayes_kl_bits"], atol=1e-14)
    assert arrays["conditioned_cumulative_gap_bits"][0] == 0


def test_cumulative_se_pairs_paths_before_averaging_correlated_positions():
    # Each path's two errors cancel. Adding positionwise standard errors
    # (or treating positions as independent) would give a false error bar.
    gaps = np.array([[1, -1], [2, -2], [3, -3], [4, -4.0]])
    result = summary.statistics(summary.cumul(gaps))
    np.testing.assert_allclose(result["mean"], [0, 2.5, 0], atol=1e-15)
    assert result["se"][0] == 0 and result["se"][-1] == 0
    assert result["se"][1] == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)
    naive_independent = np.linalg.norm(summary.statistics(gaps)["se"])
    assert naive_independent > 0


def test_finite_window_signs_distinguish_recrossings_ties_and_uncertainty():
    result = summary.finite_window_signs([1, -2, 0, 3, -1, 2, 1], np.zeros(7), np.arange(1, 8))
    assert result["first_positive_mean_suffix"] == 6
    assert result["negative_mean_positions"] == [2, 5]
    assert result["zero_mean_positions"] == [3]
    assert summary.finite_window_signs([1, 0], [0, 0], [1, 2])["first_positive_mean_suffix"] is None
    cumulative = summary.finite_window_signs([0, 0.4, 0.3], [0, 1, 1], [1, 2, 3])
    assert cumulative["first_positive_mean_suffix"] == 2  # G(1)=0 is not a crossing
    assert cumulative["pointwise_95_normal_lower_positive_count"] == 0
    assert "Evaluated positions only" in cumulative["scope"]


@pytest.mark.parametrize("mean,se,coordinates", [
    ([1, np.nan], [0, 0], [1, 2]),
    ([1, 2], [0, np.inf], [1, 2]),
    ([1, 2], [0, -1], [1, 2]),
    ([1, 2], [0, 0], [2, 1]),
    ([1, 2], [0, 0], [1, 1]),
    ([1, 2], [0, 0], [1.0, 2.0]),
    ([[1, 2]], [[0, 0]], [[1, 2]]),
])
def test_invalid_sign_curves_are_rejected(mean, se, coordinates):
    with pytest.raises(ValueError):
        summary.finite_window_signs(mean, se, coordinates)


def aggregate_fixture(directory, monkeypatch, stale=None):
    expected, records = {}, []
    for index in range(2):
        arrays, metadata = exact_record(seed=20260925 + index)
        identity = metadata["identity"]
        identity.update(sequence_index=index, sequence_seed=20260925 + index, master_seed=20260925,
                        backend=dict(store="test-only-store", src="test-only-source"),
                        code_sha256=dict(test_fixture="current"))
        identity["source"]["source_depth"] = 1
        identity["prediction"].update(u_lo=-30.0, u_hi=30.0)
        expected[index] = deepcopy(identity)
        if index == 1 and stale is not None:
            if stale == "code":
                identity["code_sha256"]["test_fixture"] = "stale"
            elif stale == "seed":
                identity["sequence_seed"] = identity["sequence_seed"] - 1
        metadata["cache_id"] = fingerprint(identity)
        save_record(directory / f"seq_{index:06d}.npz", arrays, metadata)
        records.append(arrays)

    # Keep the ordinary record/hash/metric validation real; substitute only
    # the production-store identity constructor for this exact tiny fixture.
    def expected_identity(d, n_ctx, source_depth, master_seed, index, store, src, *settings):
        assert (d, n_ctx, source_depth, master_seed, store, src) == (
            2, 5, 1, 20260925, "test-only-store", "test-only-source")
        return expected[index]

    monkeypatch.setattr(summary, "make_identity", expected_identity)
    return records


def test_aggregate_preserves_paired_uncertainty_conditioned_diagnostic_and_inputs(tmp_path, monkeypatch):
    records = aggregate_fixture(tmp_path, monkeypatch)
    report = summary.aggregate(tmp_path, 2)
    gaps = np.stack([np.sum(r["oracle_probs"] * np.log2(r["bayes_probs"] / r["unigram_probs"]), axis=1)
                     for r in records])
    totals = np.cumsum(gaps, axis=1)
    np.testing.assert_allclose(report["cumulative"]["gap"]["mean"][1:], totals.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(report["cumulative"]["gap"]["se"][1:], totals.std(axis=0, ddof=1) / math.sqrt(2), atol=1e-14)
    assert "conditional_gap" in report["instantaneous"]
    assert report["prefix_lengths"] == [1, 2, 3, 4, 5]
    assert report["cumulative_prefix_lengths"] == [1, 2, 3, 4, 5, 6]
    assert len(report["inputs"]) == 2
    assert all(item["sha256"] == summary.sha(Path(item["path"])) for item in report["inputs"])
    with pytest.raises(FileNotFoundError):
        summary.aggregate(tmp_path, 3)


@pytest.mark.parametrize("stale", ["code", "seed"])
def test_aggregate_rejects_stale_code_or_reused_source_seed(tmp_path, monkeypatch, stale):
    aggregate_fixture(tmp_path, monkeypatch, stale=stale)
    with pytest.raises(ValueError, match="inputs changed"):
        summary.aggregate(tmp_path, 2)

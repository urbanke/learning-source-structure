"""Validate saved short-prefix evidence without rerunning LSA quadrature."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from shared_base_short_evidence import (DEFAULT_CATALOG, DEFAULT_FIRST_STEP,
                                       SharedBaseShortEvidence)


@pytest.fixture(scope="module")
def reference():
    return SharedBaseShortEvidence()


def canonical(xs):
    order = list(dict.fromkeys(xs))
    return tuple(order.index(x) for x in xs)


def test_short_joint_matches_direct_sum_over_actual_label_continuations(reference):
    # Deliberately enumerate all 64^2 labelled suffixes here instead of
    # using the adapter's equality-pattern multiplicities.
    catalog = {tuple(r["pattern"]): r["prefix_probability"]
               for r in json.loads(DEFAULT_CATALOG.read_text())["records"]}
    for prefix in [(7, 7), (7, 53)]:
        expected = math.fsum(catalog[canonical(prefix + suffix)]
                             for suffix in itertools.product(range(64), repeat=2))
        assert math.exp(reference.log_joint(prefix)) == pytest.approx(expected, rel=1e-14)


def test_first_prediction_matches_independent_rank_integration(reference):
    first = json.loads(DEFAULT_FIRST_STEP.read_text())["reports"][-1]
    p = reference.next_probs([47])
    assert p[47] == pytest.approx(first["seen_probability"], abs=1e-11)
    np.testing.assert_allclose(np.delete(p, 47), first["unseen_probability"], atol=1e-11, rtol=0)
    assert reference.log_joint([]) == 0
    assert reference.log_joint([47]) == pytest.approx(-math.log(64), abs=1e-10)


def test_catalog_joints_and_all_next_label_ratios_are_preserved(reference):
    rng = np.random.default_rng(7411)
    for record in json.loads(DEFAULT_CATALOG.read_text())["records"]:
        relabelling = rng.permutation(64)
        prefix = tuple(int(relabelling[x]) for x in record["pattern"])
        assert math.exp(reference.log_joint(prefix)) == pytest.approx(record["prefix_probability"], rel=1e-14)
        expected = np.full(64, record["unseen_probability"])
        expected[list(dict.fromkeys(prefix))] = record["seen_probs"]
        np.testing.assert_array_equal(reference.next_probs(prefix), expected)
        children = np.array([math.exp(reference.log_joint(prefix + (a,))) for a in range(64)])
        np.testing.assert_allclose(children / record["prefix_probability"], expected, rtol=1e-14, atol=0)
        # An independent sequential factorization also checks time alignment.
        for next_token in [prefix[0], prefix[-1], int(relabelling[record["n_seen"]])]:
            sequence = prefix + (next_token,)
            sequential = sum(math.log(reference.next_probs(sequence[:t])[token])
                             for t, token in enumerate(sequence))
            assert sequential == pytest.approx(reference.log_joint(sequence), abs=1e-13)


def test_validation_records_the_raw_numerical_residual_without_rescaling(reference):
    report = reference.validation
    assert report["passed"] and not report["renormalized"]
    assert report["pattern_counts"] == [1, 1, 2, 5, 15, 52]
    assert 1e-12 < report["normalization_errors"]["4"] < 1e-9
    assert report["maximum_child_parent_relative_error"] < 1e-9
    assert report["maximum_independent_first_step_error"] < 1e-11
    assert str(DEFAULT_CATALOG) in report["verified_file_sha256"]


@pytest.mark.parametrize("tokens", [[-1], [64], [1.0], [True], [[1]], list(range(6))])
def test_unavailable_or_invalid_sequences_are_rejected(reference, tokens):
    with pytest.raises(ValueError):
        reference.log_joint(tokens)


def test_predictions_stop_at_four_observed_tokens(reference):
    with pytest.raises(ValueError, match="lengths 0 through 4"):
        reference.next_probs([0, 1, 2, 3, 4])


def test_catalog_tampering_is_rejected_against_committed_receipt(tmp_path):
    catalog = json.loads(DEFAULT_CATALOG.read_text())
    catalog["records"][0]["prefix_probability"] *= 1.01
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="catalog hash"):
        SharedBaseShortEvidence(path, receipt_path=DEFAULT_CATALOG.parent / "time4_summary.json")


def test_scaled_evidence_is_rejected_even_with_a_matching_receipt(tmp_path):
    catalog = json.loads(DEFAULT_CATALOG.read_text())
    for record in catalog["records"]:
        record["prefix_probability"] *= 1.01
    # Simulate relocation without optional raw integration reports. The
    # independent normalization check must still catch a missing constant.
    for i, record in enumerate(catalog["provenance"]["numerical_reports"]):
        record["path"] = str(tmp_path / f"unavailable_{i}.json")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    receipt = json.loads((DEFAULT_CATALOG.parent / "time4_summary.json").read_text())
    receipt["provenance"]["reference_catalog_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="do not normalize"):
        SharedBaseShortEvidence(path, receipt_path=receipt_path)

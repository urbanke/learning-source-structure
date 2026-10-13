"""Absolute shared-base evidence for at most five tokens from saved quadrature.

The validated length-four catalog already contains ordered-sequence
probabilities, not probabilities of equality patterns. Shorter marginals
are obtained by summing continuations, with the multiplicity of unseen
labels included. Length-five evidence is the saved joint times its saved
next-token probability. No quadrature or posterior simulation is run here.

All numerical values are retained without renormalization. Only the empty
sequence is assigned its exact probability one; the catalog's small
normalization residual is checked and recorded, including at this boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "results/transformer/shared_base_conditional_curves/shortprefix_reference.json"
DEFAULT_FIRST_STEP = ROOT / "results/transformer/conditional_shared_base/first_step.json"
EXPECTED_PRIOR = {"alpha": [.5, 1.5], "depths": list(range(9)), "ranks": "uniform permutation"}
TOLERANCE = 1e-9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _patterns(length):
    """Restricted-growth strings: exactly one representative per relabelling."""
    result = [()]
    for _ in range(length):
        result = [prefix + (a,) for prefix in result
                  for a in range(max(prefix, default=-1) + 2)]
    return result


def _canonical(tokens):
    labels = {}
    return tuple(labels.setdefault(int(token), len(labels)) for token in tokens)


def _tokens(values, maximum):
    array = np.asarray(values)
    _require(array.ndim == 1 and (array.size == 0 or array.dtype.kind in "iu"),
             "tokens must be a one-dimensional integer sequence")
    _require(len(array) <= maximum, f"only token lengths 0 through {maximum} are available")
    _require(np.all((array >= 0) & (array < 64)), "tokens must lie in 0..63")
    return tuple(map(int, array))


class SharedBaseShortEvidence:
    """The original 64-symbol, full-prior shared-base source.

    ``log_joint(tokens)`` returns a natural logarithm for lengths 0..5.
    ``next_probs(prefix)`` returns 64 probabilities for lengths 0..4.
    ``validation`` records both checks and all files actually verified.
    Raw numerical reports are optional after relocation; committed catalog,
    receipt, first-step reference and their generating scripts are required.
    """

    d = 64

    def __init__(self, catalog_path=DEFAULT_CATALOG, first_step_path=DEFAULT_FIRST_STEP,
                 receipt_path=None):
        catalog_path, first_step_path = Path(catalog_path), Path(first_step_path)
        receipt_path = (catalog_path.parent / "time4_summary.json" if receipt_path is None
                        else Path(receipt_path))
        catalog = json.loads(catalog_path.read_text())
        first = json.loads(first_step_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        _require(catalog.get("schema") == "shared-base-shortprefix-reference-catalog-v1"
                 and catalog.get("time") == 4 and catalog.get("passed") is True,
                 "a passed length-four shared-base reference catalog is required")
        _require(catalog.get("prior") == EXPECTED_PRIOR, "catalog uses a different source prior")
        _require(first.get("schema") == "shared-base-first-step-exact-ranks-v1"
                 and first.get("time") == 1 and first.get("passed") is True
                 and first.get("prior") == EXPECTED_PRIOR,
                 "a passed independent first-step reference for the same prior is required")
        _require(receipt.get("schema") == "shared-base-shortprefix-paired-summary-v1"
                 and receipt.get("reference_validation_passed") is True,
                 "a passed catalog validation receipt is required")
        _require(receipt["provenance"]["reference_catalog_sha256"] == _sha(catalog_path),
                 "catalog hash does not match its validation receipt")

        verified = {str(path): _sha(path) for path in (catalog_path, first_step_path, receipt_path)}
        scripts = {
            ROOT / "code/scripts/shortprefix_shared_base_exact.py":
                catalog["provenance"]["exact_script_sha256"],
            ROOT / "code/scripts/shortprefix_shared_base_summary.py":
                catalog["provenance"]["summary_script_sha256"],
            ROOT / "code/scripts/conditional_shared_base_first_step.py": first["script_sha256"],
        }
        for path, expected in scripts.items():
            _require(path.is_file() and _sha(path) == expected,
                     f"reference-generating script hash mismatch: {path}")
            verified[str(path)] = expected

        numerical = catalog["validation"]
        refinements = [float(value) for record in numerical["refinements"].values()
                       for value in record.values()]
        errors = refinements + numerical["all_pattern_probability_sum_errors"] + numerical["row_marginal_max_errors"]
        _require(bool(errors) and np.isfinite(errors).all() and min(errors) >= 0
                 and max(errors) <= 1e-7, "catalog numerical validation failed")
        _require(bool(numerical["small_d_exhaustive"])
                 and all(record.get("passed") is True
                         and 0 <= record["max_absolute_error"] <= 1e-7
                         for record in numerical["small_d_exhaustive"]),
                 "catalog lacks passed independent exhaustive validation")

        missing_reports = []
        report_files = catalog["provenance"]["numerical_reports"]
        _require(len(report_files) == 3, "catalog requires its three refinement reports")
        for i, record in enumerate(report_files):
            path = Path(record["path"])
            if not path.is_file() and "output" in path.parts:
                path = ROOT.joinpath(*path.parts[path.parts.index("output"):])
            if not path.is_file():
                missing_reports.append(record)
                continue
            _require(_sha(path) == record["sha256"], f"numerical report hash mismatch: {path}")
            verified[str(path)] = record["sha256"]
            report = json.loads(path.read_text())
            _require(report.get("prior") == EXPECTED_PRIOR
                     and report.get("script_sha256") == catalog["provenance"]["exact_script_sha256"],
                     "numerical report prior or generating script mismatch")
            if i == len(report_files) - 1:
                _require(report["records"] == catalog["records"],
                         "catalog differs from its finest numerical report")

        self._joint = {(): 1.0}
        self._prediction = {}
        entries = catalog["records"]
        _require([tuple(entry["pattern"]) for entry in entries] == _patterns(4),
                 "catalog must contain all 15 length-four equality patterns, once each")
        for entry in entries:
            pattern = tuple(entry["pattern"])
            k = max(pattern) + 1
            joint = float(entry["prefix_probability"])
            seen = np.asarray(entry["seen_probs"], dtype=float)
            unseen = float(entry["unseen_probability"])
            prediction = np.asarray(entry["predictive_probs"], dtype=float)
            _require(entry["n_seen"] == k and seen.shape == (k,) and prediction.shape == (64,),
                     "catalog probability dimensions are inconsistent")
            expected_prediction = np.r_[seen, np.repeat(unseen, 64-k)]
            _require(np.array_equal(prediction, expected_prediction),
                     "catalog seen/unseen probabilities disagree with its predictive row")
            _require(np.isfinite(joint) and 0 < joint <= 1
                     and np.isfinite(prediction).all() and np.all(prediction > 0)
                     and np.all(prediction <= 1)
                     and abs(float(prediction.sum()) - 1) <= TOLERANCE,
                     "invalid catalog joint or predictive probability")
            self._joint[pattern] = joint
            self._prediction[pattern] = prediction.copy()
            for token in range(k+1):
                self._joint[pattern + (token,)] = joint * prediction[token]

        # Summation is over all labels, with all unseen labels exchangeable.
        for length in (3, 2, 1):
            for pattern in _patterns(length):
                k = max(pattern) + 1
                self._joint[pattern] = (math.fsum(self._joint[pattern + (a,)] for a in range(k))
                                        + (64-k) * self._joint[pattern + (k,)])

        total_errors, child_errors = {}, []
        for length in range(6):
            total = math.fsum(self._joint[pattern] * math.prod(range(64-max(pattern, default=-1), 65))
                              for pattern in _patterns(length))
            total_errors[str(length)] = abs(total - 1)
            _require(abs(total - 1) <= TOLERANCE,
                     f"absolute ordered-sequence probabilities at length {length} do not normalize")
            if length < 5:
                for pattern in _patterns(length):
                    k = max(pattern, default=-1) + 1
                    children = (math.fsum(self._joint[pattern + (a,)] for a in range(k))
                                + (64-k) * self._joint[pattern + (k,)])
                    child_errors.append(abs(children / self._joint[pattern] - 1))
        _require(max(child_errors) <= TOLERANCE, "child evidence does not sum to parent evidence")
        first_errors = []
        for record in first["reports"]:
            _require(record["d"] == 64 and record["depths"] == list(range(9)),
                     "independent first-step reference has a different alphabet or depths")
            probabilities = np.asarray(record["probs"], dtype=float)
            _require(probabilities.shape == (64,) and np.isfinite(probabilities).all()
                     and np.all(probabilities > 0) and abs(probabilities.sum()-1) <= TOLERANCE,
                     "invalid independent first-step probabilities")
            first_errors.append(float(np.max(np.abs(self.next_probs([0]) - probabilities))))
            first_errors.append(abs(self._joint[(0, 0)] - probabilities[0]/64))
            first_errors.append(abs(self._joint[(0, 1)] - probabilities[1]/64))
        _require(bool(first_errors) and max(first_errors) <= TOLERANCE,
                 "backward marginalization disagrees with the independent first-step reference")
        self.validation = dict(
            schema="shared-base-short-evidence-validation-v1", passed=True,
            available_token_lengths=list(range(6)), available_prediction_times=list(range(5)),
            prior=EXPECTED_PRIOR, pattern_counts=[len(_patterns(n)) for n in range(6)],
            normalization_errors=total_errors, maximum_child_parent_relative_error=max(child_errors),
            maximum_independent_first_step_error=max(first_errors), tolerance=TOLERANCE,
            renormalized=False, verified_file_sha256=verified,
            unavailable_optional_numerical_reports=missing_reports,
        )

    def log_joint(self, tokens):
        """Natural log of the marginal probability of an ordered token sequence."""
        return math.log(self._joint[_canonical(_tokens(tokens, 5))])

    def next_probs(self, prefix):
        """Predict every actual token label from a prefix of length at most four."""
        tokens = _tokens(prefix, 4)
        pattern = _canonical(tokens)
        seen = list(dict.fromkeys(tokens))
        k = len(seen)
        if len(tokens) == 4:
            values = self._prediction[pattern]
            result = np.full(64, values[k])
            result[seen] = values[:k]
        else:
            parent = self._joint[pattern]
            result = np.full(64, self._joint[pattern + (k,)] / parent)
            result[seen] = [self._joint[pattern + (a,)] / parent for a in range(k)]
        return result

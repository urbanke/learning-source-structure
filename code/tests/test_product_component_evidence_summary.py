"""Cohort/provenance refusal and lossless curation checks, without model runs."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import product_component_evidence_summary as evidence


@pytest.fixture
def declared_spec(tmp_path, monkeypatch):
    spec = copy.deepcopy(evidence.EXPECTED_PROTOCOL)
    spec["runtime"] = dict(device="cpu", torch_threads=1)
    spec["provenance"] = {}
    for name in evidence.REQUIRED_PROVENANCE:
        path = tmp_path / "provenance" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "config":
            path.write_text(json.dumps(dict(family="product:8x8", d=64, ls=[1, 8], seed=0, n_ctx=1024)))
        else:
            path.write_text("independent-test-fixture:" + name)
        spec["provenance"][name] = dict(path=str(path), sha256=evidence.frozen.sha(path))
    # Fixtures have their own explicit immutable model identity. The test
    # still requires the identity named by the validator to match the file.
    monkeypatch.setattr(evidence, "CHECKPOINT_SHA256", spec["provenance"]["checkpoint"]["sha256"])
    monkeypatch.setattr(evidence, "CONFIG_SHA256", spec["provenance"]["config"]["sha256"])
    return spec


@pytest.mark.parametrize("key,wrong", [
    ("n_seq", 32), ("n_ctx", 128), ("master_seed", 20260928),
    ("source_depths", list(range(9))), ("objective", "squared error"),
])
def test_protocol_refuses_wrong_cohort_or_scientific_spec(declared_spec, key, wrong):
    evidence.verify_spec(declared_spec)
    changed = copy.deepcopy(declared_spec)
    changed[key] = wrong
    with pytest.raises(ValueError, match="Evidence protocol mismatch"):
        evidence.verify_spec(changed)
    with pytest.raises(ValueError, match="does not permit"):
        evidence.verify_spec(declared_spec, allow_smoke=True)


@pytest.mark.parametrize("tamper", ["file", "missing_identity", "different_checkpoint"])
def test_provenance_cannot_be_changed_or_silently_omitted(declared_spec, tamper):
    if tamper == "file":
        Path(declared_spec["provenance"]["source.py"]["path"]).write_text("changed sampler")
    elif tamper == "missing_identity":
        del declared_spec["provenance"]["families.py"]
    else:
        declared_spec["provenance"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        evidence.verify_spec(declared_spec)


def test_missing_draw_refuses_before_reading_data_or_replacing_results(declared_spec, tmp_path, capsys):
    directory = tmp_path / "input"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps(declared_spec))
    # A deliberately unreadable present record ensures cohort completeness
    # is checked before any per-record numerical aggregation begins.
    (directory / "seq_000000.npz").write_bytes(b"not a numerical archive")
    output = tmp_path / "curated"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("existing validated result")
    full_output = tmp_path / "full"
    code = evidence.main(["--input", str(directory), "--out", str(output), "--full-out", str(full_output)])
    assert code == 1
    error = json.loads(capsys.readouterr().out)
    assert "Missing 1023" in error["error"]
    assert "seq_000001.npz" in error["error"]
    assert sentinel.read_text() == "existing validated result"
    assert not full_output.exists()


def test_curated_values_are_exact_and_omissions_are_bound_to_full_report(tmp_path):
    full = {key: {"preserved_field": key} for key in evidence.CURATED_FIELDS}
    full.update(schema="full", sensitivity={"trajectory": [0.1, 0.2]},
                crossings=[{"index": 0, "events": [3, 9]}],
                diagnostics={"fit_kl": {"mean": [0.00123], "se": [0.00001], "count": [1024]}})
    path = tmp_path / "full.json"
    path.write_text(json.dumps(full))
    curated = evidence.curate_report(full, path)
    for key in evidence.CURATED_FIELDS:
        assert curated[key] == full[key]
    assert "crossings" not in curated and "sensitivity" not in curated
    assert curated["retention"]["omitted_fields"] == ["crossings", "sensitivity"]
    assert curated["retention"]["full_summary"]["sha256"] == evidence.frozen.sha(path)
    assert curated["retention"]["full_summary"]["size_bytes"] == path.stat().st_size

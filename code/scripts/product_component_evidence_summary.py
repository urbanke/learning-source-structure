"""Validate the complete 1,024-source product experiment and curate its results.

The frozen pilot consumer supplies the unchanged aggregate function and all
numerical checks. Its function is bound to a private copy of its namespace,
replacing only the pilot-specific protocol validator with the strict cohort
validator below. No frozen module, data, prior, or numerical rule is changed.

All 1,024 immutable records must exist before aggregation starts. Full
per-sequence event histories and sensitivity traces stay in ignored output;
the curated report retains the numerical evidence needed by the paper plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from types import FunctionType

import product_component_pilot_summary as frozen

ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = Path(frozen.__file__).resolve()
FROZEN_SHA256 = "21ca97f59bc074dcc01eda9ebc8a8b4485d2689d6ca69a4b7871bc1b6b8996a2"
CHECKPOINT_SHA256 = "29cac698253475a539a0b1bb76a47436a6eee0b0ab3d8c651a61bfe239b9b271"
CONFIG_SHA256 = "7369b744e526a007d45e8b3e9e5333a1021ca9f7582da971dd4bb6430a3f2a06"
SCHEMA = "product-component-evidence-summary-v1"

REQUIRED_PROVENANCE = frozenset((
    "eval_product_components.py", "component_fit.py", "product_components.py", "shared_depth.py",
    "lsa_seq.py", "phi_source.py", "source.py", "families.py", "model.py", "eval_conditional_crossing.py",
    "checkpoint", "config", "store/anchors.json", "store/manifest.json",
    "pmm/production_coding.py", "pmm/universal_tables.py", "pmm/mellin.py", "pmm/kernel.py",
))
EXPECTED_PROTOCOL = dict(
    schema=frozen.SCHEMA, n_seq=1024, n_ctx=1024, master_seed=20260929,
    source_family="product:8x8", source_depths=list(range(1, 9)), alphabet_sizes=[8, 8],
    depth_scope="independent coordinate depths; each shared across all coordinate rows",
    first_token="supplied free", seed_rule="seed_of(schema,master_seed,index)",
    objective="KL(transformer||four-component mixture),bits; free simplex weights",
    optimality_gap_tolerance_nats=1e-7, identification_sigma_threshold=0.02,
    sensitivity_epsilon_l2=0.001, prior_unchanged=True,
    transformer_probability_calculation="float32 model logits; float64 normalized log-softmax",
)

CURATED_FIELDS = (
    "passed", "units", "spec", "spec_id", "protocol", "inputs", "n_sequences", "prefix_lengths",
    "paired_weights", "diagnostics", "eligible_fraction", "early_middle_late_windows",
    "maximum_check_errors", "per_sequence_time_average_fit_kl_bits", "uncertainty",
    "weight_interpretation", "robustness_interpretation", "crossing_interpretation", "full_copy_event",
    "script", "script_sha256", "reused_consumer",
)


def verify_spec(spec, *, allow_smoke=False):
    """The evidence cohort has no partial-cohort or smoke-validation mode."""
    if allow_smoke:
        raise ValueError("The 1,024-source evidence report does not permit a smoke or subset mode")
    for key, expected in EXPECTED_PROTOCOL.items():
        if spec.get(key) != expected:
            raise ValueError(f"Evidence protocol mismatch: {key}")
    if (spec.get("runtime", {}).get("device") != "cpu"
            or spec.get("runtime", {}).get("torch_threads") != 1):
        raise ValueError("The declared evidence evaluator uses CPU with one torch thread")
    provenance = spec.get("provenance", {})
    if set(provenance) != REQUIRED_PROVENANCE:
        raise ValueError("Missing or unexpected source/model/store provenance entries")
    if (provenance["checkpoint"]["sha256"] != CHECKPOINT_SHA256
            or provenance["config"]["sha256"] != CONFIG_SHA256):
        raise ValueError("The checkpoint/configuration differs from the fixed product model")
    for name, value in provenance.items():
        if frozen.sha(value["path"]) != value["sha256"]:
            raise ValueError(f"Current provenance hash changed: {name}")
    configuration = json.loads(Path(provenance["config"]["path"]).read_text())
    if (configuration["family"] != "product:8x8" or configuration["d"] != 64
            or configuration["ls"] != [1, 8] or configuration["seed"] != 0
            or configuration["n_ctx"] != 1024):
        raise ValueError("Checkpoint configuration does not match the declared product evidence")


def require_complete_cohort(directory, n_sequences=1024):
    expected = {f"seq_{index:06d}.npz" for index in range(n_sequences)}
    present = {path.name for path in Path(directory).glob("seq_*.npz") if path.is_file()}
    unexpected = sorted(present - expected)
    if unexpected:
        raise ValueError(f"Unexpected source record: {unexpected[0]}")
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Missing {len(missing)} of {n_sequences} declared source draws; "
                         f"first is {missing[0]}. No partial aggregation is permitted")


def aggregate(directory):
    """Run the frozen numerical aggregation with a replacement cohort gate."""
    directory = Path(directory)
    if frozen.sha(FROZEN_PATH) != FROZEN_SHA256:
        raise ValueError("The frozen numerical consumer changed")
    spec = json.loads((directory / "protocol.json").read_text())
    verify_spec(spec)
    require_complete_cohort(directory)
    # FunctionType keeps the exact function code, defaults and numerical
    # dependencies. Only the protocol lookup differs; the imported module's
    # own namespace and validator remain untouched, including on exceptions.
    namespace = dict(frozen.aggregate.__globals__)
    namespace["_verify_spec"] = verify_spec
    reused_aggregate = FunctionType(frozen.aggregate.__code__, namespace,
                                    frozen.aggregate.__name__, frozen.aggregate.__defaults__,
                                    frozen.aggregate.__closure__)
    reused_aggregate.__kwdefaults__ = dict(frozen.aggregate.__kwdefaults__ or {})
    report = reused_aggregate(directory, allow_smoke=False)
    report.update(schema="product-component-evidence-full-summary-v1",
                  script=frozen.relative(__file__), script_sha256=frozen.sha(__file__),
                  reused_consumer=dict(path=frozen.relative(FROZEN_PATH), sha256=FROZEN_SHA256,
                      adaptation="Only the cohort-specific protocol validator is replaced in a private namespace; "
                                 "the aggregate function code and all numerical functions are unchanged."))
    return report


def curate_report(full_report, full_summary_path):
    """Retain exact numerical arrays; only omit separately preserved histories."""
    result = {key: full_report[key] for key in CURATED_FIELDS}
    omitted = sorted(set(full_report) - set(CURATED_FIELDS) - {"schema", "tables"})
    path = Path(full_summary_path)
    result.update(schema=SCHEMA,
                  retention=dict(full_summary=dict(path=frozen.relative(path), sha256=frozen.sha(path),
                                                    size_bytes=path.stat().st_size),
                                 omitted_fields=omitted,
                                 explanation="Per-sequence crossing histories, timing diagnostics and sensitivity "
                                             "traces remain in the full ignored summary. The curated numerical "
                                             "values are copied exactly; no new timing claim is made."))
    return result


def save_reports(full_report, output_directory, full_directory):
    """Publish only after every record passes validation; JSON is published last."""
    output_directory, full_directory = Path(output_directory), Path(full_directory)
    full_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".summary-", dir=full_directory) as temporary:
        staging = Path(temporary)
        frozen.save_tables(full_report, staging)
        for name in ("summary.tsv", "crossings.tsv", "windows.tsv"):
            (staging / name).replace(full_directory / name)
        full_report["tables"] = {name: dict(path=frozen.relative(full_directory / name),
                                            sha256=frozen.sha(full_directory / name))
                                 for name in ("summary.tsv", "crossings.tsv", "windows.tsv")}
        full_path = full_directory / "summary.json"
        (staging / "summary.json").write_text(json.dumps(full_report, indent=2, allow_nan=False) + "\n")
        (staging / "summary.json").replace(full_path)
    curated = curate_report(full_report, full_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".summary-", dir=output_directory) as temporary:
        staging = Path(temporary)
        for name in ("summary.tsv", "windows.tsv"):
            shutil.copyfile(full_directory / name, staging / name)
            (staging / name).replace(output_directory / name)
        curated["tables"] = {name: dict(path=frozen.relative(output_directory / name),
                                        sha256=frozen.sha(output_directory / name))
                             for name in ("summary.tsv", "windows.tsv")}
        (staging / "summary.json").write_text(json.dumps(curated, indent=2, allow_nan=False) + "\n")
        (staging / "summary.json").replace(output_directory / "summary.json")
    return curated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/product_component_evidence/run_2026-09-15")
    parser.add_argument("--out", type=Path, default=ROOT / "results/transformer/product_component_evidence")
    parser.add_argument("--full-out", type=Path, default=ROOT / "output/product_component_evidence/summary_full")
    args = parser.parse_args(argv)
    try:
        full_report = aggregate(args.input)
        report = save_reports(full_report, args.out, args.full_out)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps(dict(schema=SCHEMA, passed=False, error=str(error), input=frozen.relative(args.input))))
        return 1
    early = report["early_middle_late_windows"][0]
    print(json.dumps(dict(passed=True, n_sequences=report["n_sequences"],
                          time_average_fit_kl_bits=report["per_sequence_time_average_fit_kl_bits"],
                          early_eligible_fraction=early["eligible_fraction"],
                          output=frozen.relative(args.out),
                          curated_bytes=(args.out / "summary.json").stat().st_size,
                          full_summary=report["retention"]["full_summary"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

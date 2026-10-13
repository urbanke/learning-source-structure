"""Audit every retained original-source prediction and summarize context regret."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from eval_shared_base_models import atomic_json, canonical, fingerprint, sha_file
from eval_shared_base_reference import load_inputs


RUNS = ("d64", "d64_seed1")
TIMES = (16, 64, 256, 1024)


def summarize(folder):
    cohort_path = folder / "cohort.json"
    cohort = json.loads(cohort_path.read_text())
    count = cohort["n_sequences"]
    if count != 1024 or cohort["prefixes"] != list(TIMES):
        raise ValueError("the declared complete cohort must contain 1024 sequences at the four specified times")
    values = {run: [] for run in RUNS}
    provenance = []
    normalization_error = regret_error = 0.0
    for index, entry in enumerate(cohort["records"]):
        if index != entry["index"]:
            raise ValueError("source order differs from the fixed cohort")
        stem = f"seq_{index:06d}"
        source_path = folder / "data" / f"{stem}.json"
        observations_path = folder / "observations" / f"{stem}.json"
        source = json.loads(source_path.read_text())
        if (sha_file(source_path) != entry["file_sha256"]
                or sha_file(observations_path) != entry["observations_sha256"]
                or json.loads(observations_path.read_text()) != {"tokens": source["tokens"]}):
            raise ValueError("source/observation identity failed")
        tokens = np.asarray(source["tokens"])
        transition = np.asarray(source["oracle"]["transition_probs"])
        record = dict(index=index, source_id=source["source_id"], source_sha256=entry["file_sha256"],
                      observations_sha256=entry["observations_sha256"], models={})
        for run in RUNS:
            path = folder / "models" / run / f"{stem}.json"
            model = json.loads(path.read_text())
            arrays_path = path.with_suffix(".npz")
            if (model["cache_id"] != fingerprint(model["identity"])
                    or model["arrays_sha256"] != sha_file(arrays_path)
                    or model["source_id"] != source["source_id"]
                    or model["tokens_sha256"] != source["tokens_sha256"]
                    or model["identity"]["source_file_sha256"] != entry["file_sha256"]):
                raise ValueError("model identity failed")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["prefixes"], TIMES)
                np.testing.assert_array_equal(arrays["targets"], tokens[list(TIMES)])
                q = transition[tokens[np.asarray(TIMES) - 1]]
                p = arrays["predictive_probs"]
                np.testing.assert_array_equal(arrays["oracle_probs"], q)
                np.testing.assert_array_equal(model["predictive_probs"], p)
                if not np.isfinite(p).all() or (p <= 0).any():
                    raise ValueError("invalid predictive probabilities")
                normalization_error = max(normalization_error, float(abs(p.sum(axis=1) - 1).max()))
                # Independent scalar sum, with the source's current row, rather
                # than the vector log-probability expression used by inference.
                regret = np.asarray([sum(float(q[j, k]) * np.log2(float(q[j, k]) / float(p[j, k]))
                                         for k in range(64)) for j in range(len(TIMES))])
                regret_error = max(regret_error, float(abs(regret - arrays["expected_regret_bits"]).max()))
                values[run].append(regret)
            record["models"][run] = dict(report_sha256=sha_file(path), arrays_sha256=sha_file(arrays_path))
        provenance.append(record)
    if len(provenance) != count or normalization_error > 1e-12 or regret_error > 1e-12:
        raise ValueError("incomplete cohort or numerical audit failed")
    compatibility = []
    for index, t in ((0, 16), (0, 1024), (1023, 64)):
        source, observed, oracle, models, _ = load_inputs(folder, index, t)
        assert observed.tolist() == source["tokens"][:t]
        compatibility.append(dict(index=index, observed_length=t, passed=True))
    summaries = []
    for run in RUNS:
        array = np.asarray(values[run])
        for j, t in enumerate(TIMES):
            summaries.append(dict(run=run, observed_length=t, n_sequences=count,
                                  mean_expected_regret_bits=float(array[:, j].mean()),
                                  se_expected_regret_bits=float(array[:, j].std(ddof=1) / np.sqrt(count))))
    summary = dict(schema="shared-base-context-model-summary-v1", n_sequences=count,
                   n_tokens_per_sequence=1025, observed_lengths=list(TIMES), master_seed=cohort["master_seed"],
                   quantity="Expected next-token regret in bits: KL(true current transition row || transformer prediction).",
                   uncertainty="One standard error across independent source sequences; each trained checkpoint is reported separately.",
                   selection=cohort["selection"], depth_counts=cohort["depth_counts"], mean_alpha=cohort["mean_alpha"],
                   results=summaries)
    receipt = dict(schema="shared-base-context-model-audit-v1", input_root=str(folder.resolve()),
                   cohort_sha256=sha_file(cohort_path), n_model_records=2 * count,
                   max_probability_sum_error=normalization_error,
                   max_independent_scalar_KL_error_bits=regret_error,
                   bayes_input="Each observations JSON contains only tokens.",
                   reference_loader_compatibility=compatibility,
                   code_sha256={name: sha_file(Path(__file__).resolve().parents[1] / name)
                                for name in ("scripts/shared_base_context_model_summary.py", "transformer/extend_shared_base_models.py", "transformer/eval_shared_base_models.py", "transformer/source.py", "transformer/model.py")},
                   checkpoints={run: sha_file(Path(__file__).resolve().parents[2] / "results/transformer/runs" / run / "model_100000.pt") for run in RUNS},
                   records=provenance)
    return summary, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/shared_base_context_accuracy_2026-09-15"))
    parser.add_argument("--out", type=Path, default=Path("results/transformer/shared_base_context_accuracy"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    summary, receipt = summarize(args.input)
    output = io.StringIO()
    output.write("run\tobserved_length\tn_sequences\tmean_expected_regret_bits\tse_expected_regret_bits\n")
    for row in summary["results"]:
        output.write("\t".join(str(row[key]) for key in ("run", "observed_length", "n_sequences", "mean_expected_regret_bits", "se_expected_regret_bits")) + "\n")
    for name, value in (("summary.json", summary), ("receipt.json", receipt)):
        path = args.out / name
        if args.check_only:
            if canonical(json.loads(path.read_text())) != canonical(value):
                raise ValueError(f"saved output differs: {path}")
        else:
            atomic_json(path, value)
    table = args.out / "summary.tsv"
    if args.check_only:
        if table.read_text() != output.getvalue():
            raise ValueError("saved summary table differs")
    else:
        table.write_text(output.getvalue())
    print(json.dumps(dict(passed=True, sequences=summary["n_sequences"], model_records=receipt["n_model_records"],
                          compatibility=receipt["reference_loader_compatibility"])), flush=True)


if __name__ == "__main__":
    main()

"""Recheck the two retained coin conditions from existing saved evaluations.

No training or new evaluation cohort is used. Binary prediction vectors are
recovered from the saved selected-token log probability and normalization.
A fixed four-sequence sample per existing evaluation is replayed against
the checkpoint; all metric uncertainty is across independent source draws.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, xlogy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "code" / "transformer"
sys.path.insert(0, str(HERE))
from source import seed_of

LN2 = math.log(2)
PRIORS = ("twopoint", "uniform")
RUNS = {prior: "bin_" + prior for prior in PRIORS}
REPLAY_INDICES = [0, 7, 127, 4095]
REPLAY_LOGP_TOL = 2e-5


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entropy(p):
    p = np.asarray(p, dtype=float)
    return -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / LN2


def posterior_prediction(ones, n, prior):
    if prior == "uniform":
        return (np.asarray(ones) + 1) / (np.asarray(n) + 2)
    if prior != "twopoint":
        raise ValueError("only the two selected priors are supported")
    return 0.2 + 0.6 * expit((2 * np.asarray(ones) - np.asarray(n)) * math.log(4))


def binary_kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("binary transformer probabilities must be strictly interior")
    return (xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))) / LN2


def source_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("summary requires one finite value per independent sequence")
    return dict(mean=float(values.mean()), source_se=float(values.std(ddof=1) / math.sqrt(len(values))),
                n_sources=len(values))


def reconstruct_binary(logp_true, targets):
    logp = np.asarray(logp_true, dtype=float)
    targets = np.asarray(targets)
    if logp.shape != targets.shape or not np.isfinite(logp).all() or np.any(logp >= 0):
        raise ValueError("invalid selected-token log probabilities")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")
    p_true = np.exp(logp)
    # expm1 avoids cancellation when the observed zero is very probable.
    p_one = np.where(targets == 1, p_true, -np.expm1(logp))
    if np.any(p_one <= 0) or np.any(p_one >= 1):
        raise ValueError("saved precision cannot reconstruct an interior binary law")
    return p_one


def exact_prior_regret(prior, predictions=256):
    """Count-enumerated prior expectation, with one observed unscored token."""
    base_entropy = float(entropy(0.2)) if prior == "twopoint" else 1 / (2 * LN2)
    increments = []
    for n in range(1, predictions + 1):
        counts = np.arange(n + 1)
        weights = (binom.pmf(counts, n, 0.2) + binom.pmf(counts, n, 0.8)) / 2 if prior == "twopoint" else np.full(n + 1, 1 / (n + 1))
        increments.append(float(weights @ entropy(posterior_prediction(counts, n, prior)) - base_entropy))
    return dict(cumulative_regret_bits=np.cumsum(increments).tolist(),
                final_regret_bits=float(np.sum(increments)),
                observed_first_token=True, scored_predictions=predictions,
                finite_prior_limit_bits=float(entropy(0.2)) if prior == "twopoint" else None)


class Audit:
    def __init__(self):
        self.checks, self.inputs = [], {}

    def require(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))
        if not passed:
            raise ValueError(name)

    def track(self, name, path):
        value = dict(path=str(Path(path).resolve()), sha256=sha_file(path))
        self.inputs[name] = value
        return value


def replay_checkpoint(audit, run, arrays, config, binding):
    import torch
    from model import GPT
    torch.set_num_threads(1)
    checkpoint = torch.load(ROOT / "results" / "transformer" / "runs" / run / "model_20000.pt",
                            map_location="cpu", weights_only=False)
    audit.require(run + ":checkpoint_config", checkpoint["step"] == 20000
                  and json.loads(json.dumps(checkpoint["cfg"])) == config)
    model_config = checkpoint["cfg"]["model"]
    model = GPT(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    x = arrays["tokens"][REPLAY_INDICES].astype(np.int64)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.from_numpy(x[:, :-1])).float(), dim=-1).numpy()
    target_lp = np.take_along_axis(lp, x[:, 1:, None], axis=2)[:, :, 0]
    error = float(np.max(np.abs(target_lp - arrays["logp_true"][REPLAY_INDICES])))
    reconstructed = reconstruct_binary(arrays["logp_true"][REPLAY_INDICES], x[:, 1:])
    p_error = float(np.max(np.abs(np.exp(lp[:, :, 1].astype(float)) - reconstructed)))
    audit.require(run + ":checkpoint_replay", error <= REPLAY_LOGP_TOL,
                  max_log_probability_error=error)
    return dict(passed=True, binding=binding, sequence_indices=REPLAY_INDICES,
                prediction_positions="all 256 positions", device="cpu",
                max_log_probability_error=error, max_probability_error=p_error,
                log_probability_tolerance=REPLAY_LOGP_TOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "transformer" / "coins_rechecked")
    parser.add_argument("--smoke", action="store_true", help="recompute saved-array metrics using a bound prior checkpoint replay")
    parser.add_argument("--replay-validation", type=Path, help="prior full summary, required for --smoke")
    args = parser.parse_args()
    if args.smoke and args.replay_validation is None:
        parser.error("--smoke requires --replay-validation")
    if args.smoke and (args.out / "summary.json").resolve() == args.replay_validation.resolve():
        parser.error("smoke output must differ from the checkpoint replay validation file")
    audit, rows, details, replay = Audit(), [], {}, {}
    prior_replay = {}
    script_hash = sha_file(Path(__file__))
    try:
        if args.smoke:
            prior = json.loads(args.replay_validation.read_text())
            audit.track("prior_replay_validation", args.replay_validation)
            audit.require("prior_replay_script", prior["provenance"]["script_sha256"] == script_hash
                          and prior["passed"] is True)
            prior_replay = prior["checkpoint_replay"]
        for filename in ("source.py", "families.py", "refs_family.py", "eval_model.py", "model.py", "figures_fam.py", "analyze_fam.py"):
            audit.track("current_code:" + filename, HERE / filename)
        figure_path = ROOT / "results" / "transformer" / "fig_coins.json"
        audit.track("old_figure_summary", figure_path)
        figure = json.loads(figure_path.read_text())
        for test_prior in PRIORS:
            tag, spec = "binary_" + test_prior, "binary:" + test_prior
            reference_path = ROOT / "results" / "transformer" / "refs_fam" / (tag + ".npz")
            audit.track("reference:" + tag, reference_path)
            with np.load(reference_path, allow_pickle=False) as stored:
                ref = {key: stored[key].copy() for key in stored.files if key != "eightpoint"}
            x = ref["tokens"]
            audit.require(tag + ":source_shape", x.shape == (4096, 257) and np.all((x == 0) | (x == 1))
                          and str(ref["spec"].item()) == spec)
            md5 = hashlib.md5(x.tobytes()).hexdigest()
            audit.require(tag + ":token_checksum", md5 == str(ref["md5"].item()))
            rng = np.random.default_rng(seed_of("eval", spec, 256, 4096))
            theta = rng.choice(np.array([0.2, 0.8]), size=4096) if test_prior == "twopoint" else rng.uniform(0, 1, size=4096)
            regenerated = (rng.random((4096, 257)) < theta[:, None]).astype(np.int16)
            audit.require(tag + ":source_regeneration", np.array_equal(x, regenerated)
                          and np.array_equal(ref["h_true"], np.repeat(entropy(theta)[:, None], 256, axis=1).astype(np.float32)))
            ones = np.cumsum(x, axis=1)[:, :-1]
            lengths, targets = np.arange(1, 257)[None, :], x[:, 1:]
            bayes = {prior: posterior_prediction(ones, lengths, prior) for prior in PRIORS}
            bayes_loss = {prior: -np.log2(np.where(targets == 1, p, 1 - p)) for prior, p in bayes.items()}
            oracle_loss = -np.log2(np.where(targets == 1, theta[:, None], 1 - theta[:, None]))
            bayes_records = {}
            for prior in PRIORS:
                key = "twopoint" if prior == "twopoint" else "laplace"
                error = float(np.max(np.abs(bayes_loss[prior] - ref[key])))
                audit.require(tag + ":reference_formula:" + prior, error <= 1e-6, max_loss_error_bits=error)
                label = "two-point Bayes" if prior == "twopoint" else "add-one (uniform Bayes)"
                legacy = float((ref[key].astype(float) - ref["h_true"].astype(float)).sum(axis=1).mean())
                audit.require(tag + ":old_bayes_summary:" + prior, abs(legacy - figure[tag][label]) <= 1e-9)
                bayes_records[prior] = dict(
                    legacy_entropy_subtracted_regret=source_summary((ref[key].astype(float) - ref["h_true"].astype(float)).sum(axis=1)),
                    realized_oracle_log_ratio=source_summary((bayes_loss[prior] - oracle_loss).sum(axis=1)),
                    conditional_kl_regret=source_summary(binary_kl(theta[:, None], bayes[prior]).sum(axis=1)))
            details[tag] = dict(n_sources=4096, n_tokens=257, scored_predictions=256,
                                source_seed=seed_of("eval", spec, 256, 4096), bayes=bayes_records,
                                exact_matched_prior_bayes=exact_prior_regret(test_prior))
            transformer_totals = {}
            for trained_prior in PRIORS:
                run = RUNS[trained_prior]
                path = ROOT / "results" / "transformer" / "eval" / run / (tag + "_step20000.npz")
                array_binding = audit.track("evaluation:" + run + ":" + tag, path)
                with np.load(path, allow_pickle=False) as stored:
                    arrays = {key: stored[key].copy() for key in stored.files}
                audit.require(run + ":" + tag + ":tokens", np.array_equal(arrays["tokens"], x)
                              and str(arrays["md5"].item()) == md5 and str(arrays["spec"].item()) == spec
                              and int(arrays["step"]) == 20000 and int(arrays["d"]) == 2)
                p_one = reconstruct_binary(arrays["logp_true"], targets)
                full = arrays["logp_full"]
                positions = arrays["positions"]
                audit.require(run + ":" + tag + ":full_vector_shape", full.shape == (16, 64, 2)
                              and np.array_equal(positions, np.arange(0, 256, 4)) and full.dtype == np.float16)
                selected = np.take_along_axis(full, targets[:16, positions, None], axis=2)[:, :, 0]
                audit.require(run + ":" + tag + ":full_target_alignment",
                              np.array_equal(selected, arrays["logp_true"][:16, positions].astype(np.float16)))
                reconstructed_log = np.stack([np.log1p(-p_one[:16, positions]), np.log(p_one[:16, positions])], axis=2)
                upper = np.nextafter(full, np.float16(np.inf)).astype(float) - full.astype(float)
                lower = full.astype(float) - np.nextafter(full, np.float16(-np.inf)).astype(float)
                tolerance = np.maximum(upper, lower) / 2 + 2e-6
                full_error = np.abs(reconstructed_log - full.astype(float))
                audit.require(run + ":" + tag + ":full_vector_precision", np.all(full_error <= tolerance),
                              max_log_error=float(full_error.max()), max_error_to_tolerance=float(np.max(full_error / tolerance)))
                run_folder = ROOT / "results" / "transformer" / "runs" / run
                checkpoint_binding = audit.track("checkpoint:" + run, run_folder / "model_20000.pt")
                config_binding = audit.track("config:" + run, run_folder / "config.json")
                config = json.loads((run_folder / "config.json").read_text())
                audit.require(run + ":configuration", config["family"] == "binary:" + trained_prior
                              and config["n_ctx"] == 256 and config["d"] == 2 and config["steps"] == 20000
                              and config["seed"] == 0)
                binding = dict(evaluation_sha256=array_binding["sha256"], checkpoint_sha256=checkpoint_binding["sha256"],
                               config_sha256=config_binding["sha256"], model_code_sha256=audit.inputs["current_code:model.py"]["sha256"])
                replay_key = run + ":" + tag
                if args.smoke:
                    receipt = prior_replay[replay_key]
                    audit.require(replay_key + ":replay_receipt", receipt["passed"] is True and receipt["binding"] == binding
                                  and receipt["sequence_indices"] == REPLAY_INDICES
                                  and receipt["max_log_probability_error"] <= REPLAY_LOGP_TOL)
                else:
                    receipt = replay_checkpoint(audit, run, arrays, config, binding)
                replay[replay_key] = receipt
                transformer_loss = -arrays["logp_true"].astype(float) / LN2
                transformer_totals[trained_prior] = transformer_loss.sum(axis=1)
                label = "transformer trained on the " + ("two-point" if trained_prior == "twopoint" else "uniform") + " prior"
                legacy = float((transformer_loss - ref["h_true"].astype(float)).sum(axis=1).mean())
                audit.require(replay_key + ":old_transformer_summary", abs(legacy - figure[tag][label]) <= 1e-9)
                for reference_prior in PRIORS:
                    delta = (transformer_loss - bayes_loss[reference_prior]).sum(axis=1)
                    kl = binary_kl(bayes[reference_prior], p_one)
                    tv = np.abs(bayes[reference_prior] - p_one)
                    rows.append(dict(test_prior=test_prior, trained_prior=trained_prior, bayes_prior=reference_prior,
                        matched=test_prior == trained_prior == reference_prior,
                        paired_cumulative_loss_gap_bits=source_summary(delta),
                        cumulative_transformer_loss_bits=source_summary(transformer_loss.sum(axis=1)),
                        cumulative_bayes_loss_bits=source_summary(bayes_loss[reference_prior].sum(axis=1)),
                        cumulative_predictive_kl_bits=source_summary(kl.sum(axis=1)),
                        mean_predictive_kl_bits=source_summary(kl.mean(axis=1)),
                        final_predictive_kl_bits=source_summary(kl[:, -1]),
                        mean_total_variation=source_summary(tv.mean(axis=1)),
                        final_total_variation=source_summary(tv[:, -1]),
                        max_total_variation=float(tv.max()),
                        final_observed_prefix_length=256, scored_predictions=256,
                        source_se_unit="one full sequence, including its freshly drawn bias"))
                    if test_prior == trained_prior == reference_prior:
                        reference_label = "two-point Bayes" if reference_prior == "twopoint" else "add-one (uniform Bayes)"
                        old_gap = figure[tag][label] - figure[tag][reference_label]
                        details[tag]["matched_loss_gap_recomputation"] = dict(
                            historical_gap_bits=old_gap, recomputed_gap_bits=float(delta.mean()),
                            difference_bits=float(delta.mean()) - old_gap,
                            note="Exact float64 posterior formulas replace the legacy float32 reference-loss arrays. Saved transformer log probabilities are unchanged; this explains the small change in the paired loss gap.")
            details[tag]["paired_training_priors"] = dict(
                direction="uniform minus two-point, on exactly the same observed sequences",
                transformer_cumulative_loss_difference_bits=source_summary(
                    transformer_totals["uniform"] - transformer_totals["twopoint"]),
                bayes_cumulative_loss_difference_bits=source_summary(
                    (bayes_loss["uniform"] - bayes_loss["twopoint"]).sum(axis=1)))
    except (OSError, KeyError, TypeError, ValueError, AssertionError) as error:
        audit.checks.append(dict(name="audit_exception", passed=False, error=str(error)))
    report = dict(schema="two-prior-coin-audit-v1", passed=bool(audit.checks) and all(c["passed"] for c in audit.checks),
        rows=rows, conditions=details, checks=audit.checks, checkpoint_replay=replay,
        provenance=dict(script_sha256=script_hash, inputs=audit.inputs, smoke=args.smoke,
                        runtime={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")}),
        limitations=[
            "One trained checkpoint per prior, both training seed zero; source SE excludes training variation.",
            "Historical files lack checkpoint hashes. Fixed-sample replay checks consistency with current checkpoints but does not authenticate every historical output.",
            "Binary probabilities are reconstructed from float32 selected-token log probabilities; full-vector float16 checks and checkpoint replay validate the reconstruction.",
            "Predictive KL is KL(Bayes predictive || transformer predictive), averaged over the saved prefixes; it is not KL of averaged probability vectors.",
            "Paired cumulative loss gaps use the same observed tokens. Entropy-subtracted regret is retained only to audit the old figure, not to establish equality.",
            "Each evaluation condition contains 4096 fresh independent bias draws and 257 tokens per draw, with 256 scored predictions after the free first token.",
            "Neither small loss gaps nor this single-checkpoint audit establishes exact Bayesian prediction at every prefix."])
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = ["test_prior\ttrained_prior\tbayes_prior\tmatched\tloss_gap_bits\tloss_gap_source_se\tmean_kl_bits\tmean_kl_source_se\tfinal_kl_bits\tfinal_kl_source_se\tmean_tv\tmean_tv_source_se"]
    for row in rows:
        values = [row[key] for key in ("test_prior", "trained_prior", "bayes_prior", "matched")]
        for key in ("paired_cumulative_loss_gap_bits", "mean_predictive_kl_bits",
                    "final_predictive_kl_bits", "mean_total_variation"):
            values += [row[key]["mean"], row[key]["source_se"]]
        table.append("\t".join(map(str, values)))
    (args.out / "summary.tsv").write_text("\n".join(table) + "\n")
    print(json.dumps(dict(out=str(output), passed=report["passed"], checks=len(audit.checks),
                          failed=[c for c in audit.checks if not c["passed"]],
                          matched=[row for row in rows if row["matched"]])))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

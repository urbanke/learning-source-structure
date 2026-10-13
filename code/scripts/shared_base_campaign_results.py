"""Audit transferred shared-base campaign scores and plot both training runs.

Only saved JSON/NPZ and pinned local source inputs are read. Regrets, expected
losses and paired uncertainty are independently reconstructed; model inference
and Bayesian sampling are not repeated. The manuscript release uses the final
400k checkpoints and records the available informed Bayesian contexts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/layered-campaign-matplotlib")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "code/transformer/configs/shared_base_campaign_plan_2026-09-18.json"
INPUT = ROOT / "output/shared_base_campaign_download_2026-09-19"
COHORTS = ROOT / "output/shared_base_campaign_inputs_2026-09-18"
PARENTS = ROOT / "output/shared_base_scaling_download_2026-09-17"
BASELINE = ROOT / "results/transformer/shared_base_scaling_2026-09-17"
OUT = ROOT / "results/transformer/shared_base_campaign_2026-09-19"
DEPTHS, TIMES, SEEDS = [0, 1, 4, 8], [1, 4, 16, 64, 256, 1024], [0, 1]
REQUIRED = {"continue_large": [200000, 400000], "xlarge": [100000, 200000]}
SOURCE_CONTRACT = dict(version="shared-base-scaling-source-v1", alphabet=64,
    alpha_uniform=[.5, 1.5], depth_uniform_inclusive=[0, 8], alphabet_permutation=True,
    parameters="Fresh exponent, common depth, rank permutation and transition matrix per sequence.",
    initial_token="Drawn from the same base law q.", model_input="Tokens only.",
    sampling="indexed-batch-seedsequence-v1")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def digest_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_config(plan, plan_hash, receipt, status, arm, seed, parent_receipt):
    """Validate remote training metadata without pretending to rehash weights."""
    require(arm in plan["arms"] and seed in SEEDS, "Unplanned arm or seed")
    cfg = receipt["config"]
    intended = plan["arms"][arm]
    expected = dict(schema="shared-base-campaign-training-v1", **intended["config"],
                    **plan["source"], arm=arm, seed=seed, parent=cfg["parent"],
                    continuation_lr=intended.get("continuation_lr"))
    require(cfg == expected and receipt["config_hash"] == fingerprint(cfg),
            "Campaign configuration or config hash differs")
    require(receipt["plan_sha256"] == plan_hash and
            receipt["implementation_hashes"] == plan["implementation_sha256"],
            "Training plan or implementation hashes differ")
    model = dict(vocab=64, n_ctx=1024, n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                 d_model=cfg["d_model"], mlp=True, dropout=0.)
    d = cfg["d_model"]
    count = cfg["n_layer"] * (12*d*d + 13*d) + d*(1024 + 2*64 + 2)
    require(receipt["model"] == model and receipt["n_params"] == intended["n_params"] == count,
            "Campaign architecture or parameter count differs")
    require(all(receipt[k] == v for k, v in dict(family="lsa", d=64, n_ctx=1024,
            alpha=[.5, 1.5], ls=[0, 8], no_permute=False, seed=seed).items()),
            "Enclosing source or seed metadata differs")
    contract = SOURCE_CONTRACT
    require(receipt["source_contract"] == contract and receipt["source_contract_hash"] == fingerprint(contract),
            "Indexed source contract differs")
    if parent_receipt is not None:
        require(receipt["validation_hashes"] == parent_receipt["validation_hashes"],
                "Training monitoring cohort differs")
    runtime = receipt["runtime"]
    require(receipt["precision"] == "float32" and receipt["device"] == "cuda" and
            runtime["device"] == "cuda" and runtime["float32_matmul_precision"] == "highest" and
            runtime["cuda_matmul_allow_tf32"] is False and runtime["cudnn_allow_tf32"] is False,
            "Training precision or GPU arithmetic contract differs")
    if arm == "continue_large":
        require(parent_receipt is not None and receipt["runtime"] == parent_receipt["runtime"],
                "Continuation runtime differs from its parent")
        parent = cfg["parent"]
        require(isinstance(parent, dict) and parent["step"] == 100000 and
                parent["weights_sha256"] == plan["parent"]["seeds"][str(seed)]["model_100000.pt"] and
                parent["config_hash"] == parent_receipt["config_hash"] and digest_value(parent["latest_sha256"]),
                "Continuation parent identity differs")
    else:
        require(cfg["parent"] is None, "Fresh eight-layer model has a parent")
    step = status["step"]
    require(status["schema"] == "shared-base-campaign-status-v1" and
            type(step) is int and 0 <= step <= cfg["steps"] and
            status["production_steps"] == cfg["steps"] and status["complete"] is (step == cfg["steps"]) and
            status["completed_sequences"] == step*cfg["batch"] and
            status["config_hash"] == receipt["config_hash"] and status["plan_sha256"] == plan_hash,
            "Training progress, exposure or identity differs")
    require(status["reason"] == "completed" if status["complete"] else
            status["reason"] in ("SIGINT", "SIGTERM", "SIGUSR1", "max_steps"),
            "Transferred training receipt does not describe a completed or saved stopped invocation")
    require(np.isfinite(status["training_elapsed_seconds"]) and status["training_elapsed_seconds"] >= 0,
            "Invalid elapsed training time")
    return dict(arm=arm, seed=seed, model=model, n_params=count, training_step_at_transfer=step,
                production_steps=cfg["steps"], training_complete=status["complete"],
                completed_sequences_at_transfer=step*cfg["batch"], batch=cfg["batch"],
                config_hash=receipt["config_hash"], parent=cfg["parent"],
                continuation_lr=cfg["continuation_lr"], precision=receipt["precision"])


def validate_runtime(report, status, plan_hash, arm, seed):
    """Check the saved invocation receipt; this is not an independent training replay."""
    require(report["schema"] == "shared-base-campaign-runtime-v1" and
            report["plan_sha256"] == plan_hash and report["arm"] == arm and report["seed"] == seed and
            report["step"] == status["step"] and report["device"] == "cuda" and
            report["return_code"] == (0 if status["complete"] else 75) and
            isinstance(report["cuda_device"], str) and bool(report["cuda_device"]),
            "Runtime invocation identity or saved termination differs")
    memory = [report[k] for k in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "total_memory_bytes")]
    require(all(type(v) is int for v in memory) and 0 <= memory[0] <= memory[1] <= memory[2] and memory[2] > 0,
            "Invalid CUDA memory receipt")
    return dict(invocation_receipt_verified=True, cuda_device=report["cuda_device"],
                invocation_return_code=report["return_code"], training_log_verified=False)


def score_arrays(arrays, frozen):
    """Reconstruct losses using the true current row; zero oracle entries are valid."""
    for name in ("source_ids", "source_indices", "times"):
        require(np.array_equal(arrays[name], frozen[name]), "Prediction/source alignment differs: " + name)
    oracle = np.asarray(frozen["oracle_probs"], dtype=float)
    logp = np.asarray(arrays["log_predictive_probs"], dtype=float)
    require(oracle.shape == logp.shape == (256, 6, 64) and np.isfinite(logp).all()
            and np.isfinite(oracle).all() and np.all(oracle >= 0)
            and np.max(np.abs(oracle.sum(-1)-1)) < 1e-12, "Invalid prediction/oracle dimensions or values")
    normalization = float(np.max(np.abs(np.exp(logp).sum(-1)-1)))
    require(normalization < 1e-12, "Raw predictive probabilities do not normalize")
    positive = oracle > 0
    terms = np.zeros_like(oracle)
    terms[positive] = oracle[positive] * (np.log2(oracle[positive]) - logp[positive]/np.log(2))
    regret = terms.sum(-1)
    expected = -np.einsum("ijk,ijk->ij", oracle, logp)/np.log(2)
    require(np.min(regret) > -1e-10, "Negative reconstructed KL")
    errors = {}
    for name, actual in (("regret_bits", regret), ("expected_loss_bits", expected)):
        saved = np.asarray(arrays[name])
        require(saved.shape == (256, 6) and np.isfinite(saved).all(), "Invalid stored score array")
        error = float(np.max(np.abs(saved-actual)))
        require(error < 2e-12, "Independent score reconstruction differs: " + name)
        errors[name] = error
    return regret, expected, dict(max_normalization_error=normalization, **errors)


def stats(values, times=TIMES):
    values = np.asarray(values, dtype=float)
    require(values.shape == (256, len(times)) and np.isfinite(values).all(), "Incomplete finite-score cohort")
    # Direct centered sums independently reproduce the remote np.std aggregation.
    mean = np.sum(values, axis=0)/256
    se = np.sqrt(np.sum((values-mean)**2, axis=0)/(256*255))
    return dict(n=256, times=list(times), mean=mean.tolist(), se=se.tolist())


def compare_statistics(regret, group):
    actual = stats(regret)
    error = max(float(np.max(np.abs(np.asarray(group["mean_regret_bits"])-actual["mean"]))),
                float(np.max(np.abs(np.asarray(group["se_regret_bits"])-actual["se"]))))
    require(error < 2e-12, "Evaluation mean or standard error differs")
    return error


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, digest=None):
        path = Path(path)
        actual = sha(path)
        require(digest is None or digest == actual, "Input checksum differs: " + str(path))
        key = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
        require(key not in self.hashes or self.hashes[key] == actual, "Input changed during import: " + key)
        self.hashes[key] = actual
        return actual

    def read(self, path, digest=None):
        self.bind(path, digest)
        return json.loads(Path(path).read_text())

    def arrays(self, path, digest=None):
        self.bind(path, digest)
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name].copy() for name in saved.files}


def load_frozen(plan, folder, inputs):
    manifest = inputs.read(folder/"manifest.json", plan["evaluation"]["inputs_manifest_sha256"])
    require(manifest["schema"] == "shared-base-campaign-inputs-v1" and
            manifest["times"] == TIMES and manifest["n_sequences"] == 256 and manifest["n_tokens"] == 1025,
            "Frozen input declaration differs")
    frozen = {}
    for depth in DEPTHS:
        name = f"a1_L{depth}.npz"
        a = inputs.arrays(folder/name, manifest["files"][name])
        records = manifest["source_records"][str(depth)]
        require(a["tokens"].shape == (256, 1025) and np.array_equal(a["times"], TIMES) and
                np.array_equal(a["source_indices"], np.arange(256)) and
                a["source_ids"].tolist() == records["source_ids"] and len(set(records["source_ids"])) == 256 and
                np.all((a["tokens"] >= 0) & (a["tokens"] < 64)), "Frozen source identities/indices/times differ")
        for basename in ("cohort.json", "protocol.json"):
            relative = f"output/shared_base_conditional_curves_2026-09-15/a1_L{depth}/{basename}"
            digest = plan["evaluation"]["frozen_cohort_sha256"][relative]
            require(records[basename.split(".")[0]+"_sha256"] == digest, "Manifest cohort binding differs")
            value = inputs.read(ROOT/relative, digest)
            if basename == "cohort.json":
                require([r["index"] for r in value["records"]] == list(range(256)) and
                        [r["source_id"] for r in value["records"]] == records["source_ids"] and
                        [r["file_sha256"] for r in value["records"]] == records["record_sha256"],
                        "Compact inputs do not match the original paired cohort")
        frozen[depth] = a
    return frozen


def load_baseline(folder, frozen, inputs):
    from shared_base_scaling_figures import load_scaling
    paths, previous, panels = load_scaling(folder)
    for path in paths:
        inputs.bind(path)
    audit = inputs.read(folder/"validation.json")
    arrays = {}
    for depth in DEPTHS:
        row = next(r for r in audit["paired_scores"] if r["depth"] == depth)
        a = inputs.arrays(folder/row["filename"], row["sha256"])
        require(np.array_equal(a["times"], TIMES) and np.array_equal(a["source_indices"], frozen[depth]["source_indices"])
                and np.array_equal(a["bayes_times"], [1, 4, 16, 64]), "Baseline score coordinates differ")
        require(a["baseline_regret_bits"].shape == a["larger_regret_bits"].shape == (256, 6, 2)
                and a["bayes_regret_bits"].shape == (256, 4), "Baseline/Bayes score shape differs")
        for seed in SEEDS:
            panel = next(p for p in panels if p["depth"] == depth and p["training_seed"] == seed)
            for key, saved_key, times in (("baseline", "baseline_regret_bits", TIMES),
                                        ("larger", "larger_regret_bits", TIMES),
                                        ("bayes", "bayes_regret_bits", [1, 4, 16, 64])):
                values = a[saved_key] if key == "bayes" else a[saved_key][:, :, seed]
                actual = stats(values, times)
                require(all(np.allclose(actual[k], panel[key][k], rtol=0, atol=2e-12) for k in ("mean", "se")),
                        "Curated baseline/Bayes arrays differ from their validated release")
        arrays[depth] = a
    return arrays


def import_results(folder=INPUT, cohorts=COHORTS, plan_path=PLAN, parent_root=PARENTS, baseline=BASELINE):
    inputs = Inputs()
    plan = inputs.read(plan_path)
    plan_hash = sha(plan_path)
    require(plan["schema"] == "shared-base-campaign-plan-v1" and plan["seeds"] == SEEDS and
            plan["source"] == dict(d=64, n_ctx=1024, alpha=[.5, 1.5], ls=[0, 8], permute=True) and
            plan["evaluation"]["conditions"] == DEPTHS and plan["evaluation"]["times"] == TIMES and
            plan["evaluation"]["n_sequences"] == 256 and plan["evaluation"]["n_tokens"] == 1025,
            "Campaign scope differs")
    for name, digest in plan["implementation_sha256"].items():
        inputs.bind(ROOT/"code/transformer"/name, digest)
    inputs.bind(ROOT/plan["parent"]["plan_path"], plan["parent"]["plan_sha256"])
    evaluator_hash = inputs.bind(ROOT/"code/transformer/shared_base_campaign_eval.py")
    parents = {seed: inputs.read(parent_root/f"seed{seed}/config.json",
                plan["parent"]["seeds"][str(seed)]["config.json"]) for seed in SEEDS}
    frozen = load_frozen(plan, Path(cohorts), inputs)
    old = load_baseline(Path(baseline), frozen, inputs)
    milestones = {arm: list(steps) for arm, steps in REQUIRED.items()}
    optional = [(Path(folder)/f"eval/xlarge/seed{s}/step400000/summary.json").is_file() for s in SEEDS]
    require(not any(optional) or all(optional), "Eight-layer 400k evaluation exists for only one seed; wait for both")
    if all(optional):
        milestones["xlarge"].append(400000)
    scores, training, evaluations = {}, [], []
    errors = dict(max_normalization_error=0., regret_bits=0., expected_loss_bits=0., summary_statistics=0.)
    for arm, steps in milestones.items():
        for seed in SEEDS:
            run = Path(folder)/arm/f"seed{seed}"
            config = inputs.read(run/"config.json")
            status = inputs.read(run/"status.json")
            metadata = validate_config(plan, plan_hash, config, status, arm, seed, parents[seed])
            metadata.update(validate_runtime(inputs.read(run/"campaign_runtime.json"), status, plan_hash, arm, seed))
            require(status["step"] >= max(steps), "Training status has not reached the requested milestone")
            training.append(metadata)
            eval_root = Path(folder)/"eval"/arm/f"seed{seed}"
            eval_status = inputs.read(eval_root/"status.json")
            require(eval_status["arm"] == arm and eval_status["seed"] == seed and
                    set(steps) <= set(eval_status["evaluated_steps"]) and
                    set(eval_status["evaluated_steps"]).isdisjoint(eval_status["missing_steps"]) and
                    sorted(eval_status["evaluated_steps"]+eval_status["missing_steps"]) == plan["arms"][arm]["config"]["weight_steps"] and
                    eval_status["complete"] is (not eval_status["missing_steps"]), "Evaluation milestone coverage differs")
            for step in steps:
                out = eval_root/f"step{step}"
                summary = inputs.read(out/"summary.json")
                require(summary["schema"] == "shared-base-campaign-evaluation-v1" and
                        summary["plan_sha256"] == plan_hash and summary["arm"] == arm and summary["seed"] == seed and
                        summary["step"] == step and summary["inputs_manifest_sha256"] == plan["evaluation"]["inputs_manifest_sha256"] and
                        summary["evaluation_code_sha256"] == evaluator_hash and summary["times"] == TIMES and
                        summary["n_sequences"] == 256 and summary["role"] == plan["evaluation"]["role"] and
                        summary["bayesian_reference_computed"] is False and digest_value(summary["weights_sha256"]),
                        "Milestone evaluation identity or scope differs")
                require(len(summary["groups"]) == 4 and {g["depth"] for g in summary["groups"]} == set(DEPTHS),
                        "Milestone evaluation omits or repeats a source depth")
                weights = run/f"model_{step}.pt"
                if weights.exists():
                    inputs.bind(weights, summary["weights_sha256"])
                evaluations.append(dict(arm=arm, seed=seed, step=step, n_params=metadata["n_params"],
                    completed_sequences=step*64, additional_sequences=(step-100000)*64 if arm=="continue_large" else step*64,
                    interim=step<400000, weights_sha256=summary["weights_sha256"], weights_locally_rehashed=weights.exists(),
                    evaluator_sha256=evaluator_hash, config_hash=config["config_hash"], summary_sha256=sha(out/"summary.json")))
                for group in summary["groups"]:
                    depth = group["depth"]
                    require(group["file"] == f"a1_L{depth}.npz", "Unexpected milestone filename")
                    a = inputs.arrays(out/group["file"], group["sha256"])
                    regret, _, checks = score_arrays(a, frozen[depth])
                    for key, value in checks.items():
                        errors[key] = max(errors[key], value)
                    errors["summary_statistics"] = max(errors["summary_statistics"], compare_statistics(regret, group))
                    require(abs(checks["max_normalization_error"]-group["max_normalization_error"]) < 2e-15,
                            "Saved normalization check differs")
                    scores[arm, seed, step, depth] = regret
    groups = []
    for depth in DEPTHS:
        b = old[depth]["bayes_regret_bits"]
        for seed in SEEDS:
            original = old[depth]["baseline_regret_bits"][:, :, seed]
            parent = old[depth]["larger_regret_bits"][:, :, seed]
            arrays = {"original100k": original, "large100k": parent}
            model_info = {"original100k": dict(layers=2, n_params=544256, step=100000, completed_sequences=6400000),
                          "large100k": dict(layers=4, n_params=3454464, step=100000, completed_sequences=6400000)}
            for arm, steps in milestones.items():
                for step in steps:
                    key = f"{arm}_{step}"
                    arrays[key] = scores[arm, seed, step, depth]
                    model_info[key] = dict(layers=plan["arms"][arm]["config"]["n_layer"],
                        n_params=plan["arms"][arm]["n_params"], step=step, completed_sequences=step*64,
                        interim=step<400000, arm=arm)
            groups.append(dict(depth=depth, alpha=1., seed=seed, n_sequences=256,
                source_ids=frozen[depth]["source_ids"].tolist(), bayes=stats(b, [1,4,16,64]),
                predictors={key: dict(**model_info[key], **stats(a)) for key, a in arrays.items()},
                paired={key: dict(minus_original100k=stats(a-original), minus_large100k=stats(a-parent),
                                  minus_bayes=stats(a[:, :4]-b, [1,4,16,64])) for key, a in arrays.items()}))
    report = dict(schema="shared-base-campaign-results-v1", passed=True, plan_sha256=plan_hash,
        n_sequences_per_depth=256, depths=DEPTHS, times=TIMES, seeds=SEEDS, milestones=milestones,
        missing_planned_milestones=[] if all(optional) else [dict(arm="xlarge", seed=s, step=400000) for s in SEEDS],
        campaign_complete=all(optional) and all(r["training_complete"] for r in training),
        training=training, evaluations=evaluations, groups=groups,
        bayes_times=[1,4,16,64], full_cohort_long_context_bayes_available=False,
        cohort_role=plan["evaluation"]["role"], uncertainty="One sequence SE; paired differences use the same 256 histories, each training seed separate.",
        checkpoint_scope="Weights hashes are retained from the frozen evaluator; locally rehashed only when the weight file is present. No raw logits, inference replay or training-log verification. Local checks reconstruct scores from saved log probabilities; checkpoint-to-output provenance relies on the frozen remote evaluator's checks.",
        scientific_scope="Finite training/capacity comparison on reused diagnostic cohorts; no asymptotic convergence or isolated depth claim.",
        sampling_performed=False, inference_performed=False)
    return report, dict(passed=True, checks=errors, inputs=inputs.hashes,
                       source_histories_checked=1024, model_source_arrays_checked=len(scores)*256)


def render(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.titlesize":9,
        "axes.labelsize":8.5, "xtick.labelsize":8, "ytick.labelsize":8,
        "axes.spines.top":False, "axes.spines.right":False, "pdf.fonttype":42})
    display_specs={
        "original100k": dict(color="#777777", style="-", label="2 layers, 6.4M", width=1.25),
        "large100k": dict(color="#1565a8", style="--", label="4 layers, 6.4M", width=1.15),
        "continue_large_400000": dict(color="#1565a8", style=":", label="4 layers, 25.6M", width=1.35),
        f"xlarge_{max(report['milestones']['xlarge'])}": dict(
            color="#d77719", style="-.", label="8 layers, 25.6M", width=1.35),
    }
    fig, axes=plt.subplots(4,2,figsize=(5.5,7.15),sharex=True,sharey="row")
    fig.subplots_adjust(left=.105,right=.95,bottom=.10,top=.855,wspace=.18,hspace=.38)
    for row, depth in enumerate(DEPTHS):
        pair=[g for g in report["groups"] if g["depth"]==depth]
        extrema=[0.]
        for seed, ax in enumerate(axes[row]):
            group=next(g for g in pair if g["seed"]==seed)
            # Retain all audited milestones in the data release, but present the
            # completed continuation and the latest available eight-layer model.
            shown = list(display_specs)
            for name in shown:
                value = group["predictors"][name]
                mean,se=np.asarray(value["mean"]),np.asarray(value["se"])
                spec=display_specs[name]
                ax.fill_between(TIMES,mean-se,mean+se,color=spec["color"],alpha=.18,
                                edgecolor=spec["color"],linewidth=.35)
                ax.plot(TIMES,mean,color=spec["color"],ls=spec["style"],lw=spec["width"])
                extrema.extend((mean-se).tolist());extrema.extend((mean+se).tolist())
            b=group["bayes"]
            ax.errorbar(b["times"],b["mean"],yerr=b["se"],color="black",marker="o",mfc="white",
                        ls="none",ms=3.5,lw=.7,capsize=1.5,zorder=5)
            ax.set_xscale("log",base=2);ax.set_xlim(.9,1200)
            ax.set_xticks(TIMES,[str(t) for t in TIMES]);ax.minorticks_off()
            ax.set_title(f"$L={depth}$; run {seed+1}",loc="left",pad=4)
            ax.grid(axis="y",color=".88",lw=.45);ax.tick_params(length=2.5,pad=2)
        span=max(extrema)-min(extrema)
        axes[row,0].set_ylim(min(extrema)-.02*span,max(extrema)+.06*span)
    handles=[Line2D([],[],color=spec["color"],ls=spec["style"],lw=spec["width"],label=spec["label"])
             for spec in display_specs.values()]
    handles.append(Line2D([],[],color="black",marker="o",mfc="white",ls="none",ms=3.5,
                          label="Informed Bayes"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.997),ncol=3,
               frameon=False,fontsize=7.5,columnspacing=.8,handlelength=1.8,labelspacing=.35)
    fig.supylabel("Expected next-token regret (bits)",x=.008,fontsize=9)
    fig.supxlabel("Observed tokens",y=.057,fontsize=9)
    fig.text(.52,.018,"Same 256 histories per depth; ±1 sequence SE; separate training runs.",ha="center",fontsize=8)
    for extension in ("pdf","png"):
        fig.savefig(out/f"scaling_comparison.{extension}",dpi=240,
                    metadata={"Creator":"shared_base_campaign_results.py"})
    plt.close(fig)


def load_campaign(source=OUT):
    """Read the compact audited release for the manuscript's main-panel builder.

    Returns (paths, report, panels), with baseline/larger/bayes panel keys matching
    shared_base_scaling_figures.load_scaling. Here 'larger' is the latest common
    eight-layer checkpoint, final at 400k in the manuscript release. No raw data
    or inference is needed to read the hash-bound curated release.
    """
    source = Path(source)
    paths = [source/"comparison.json", source/"validation.json", PLAN]
    report, audit, plan = [json.loads(p.read_text()) for p in paths]
    require(report["schema"] == "shared-base-campaign-results-v1" and report["passed"] is True and
            audit["schema"] == "shared-base-campaign-import-validation-v1" and audit["passed"] is True and
            audit["summary_sha256"] == sha(paths[0]) and report["plan_sha256"] == sha(PLAN),
            "Campaign release differs from its passing independent audit")
    require(report["depths"] == DEPTHS and report["times"] == TIMES and report["seeds"] == SEEDS and
            report["bayes_times"] == [1,4,16,64] and report["n_sequences_per_depth"] == 256 and
            report["full_cohort_long_context_bayes_available"] is False and
            len(report["groups"]) == 8 and
            {(g["depth"], g["seed"]) for g in report["groups"]} == {(d,s) for d in DEPTHS for s in SEEDS},
            "Campaign release coverage or Bayesian scope differs")
    latest = max(report["milestones"]["xlarge"])
    require(latest in (200000,400000), "Unexpected latest common eight-layer milestone")
    panels = []
    for group in sorted(report["groups"], key=lambda g: (g["depth"],g["seed"])):
        curves = dict(baseline=group["predictors"]["original100k"],
                      larger=group["predictors"][f"xlarge_{latest}"], bayes=group["bayes"])
        for key, value in curves.items():
            times = [1,4,16,64] if key == "bayes" else TIMES
            require(value["n"] == 256 and value["times"] == times and
                    len(value["mean"]) == len(value["se"]) == len(times) and
                    np.isfinite(value["mean"]).all() and np.isfinite(value["se"]).all() and
                    np.all(np.asarray(value["se"]) >= 0), "Invalid displayed campaign statistics")
        panels.append(dict(depth=group["depth"], training_seed=group["seed"], n_sequences=256,
            n_tokens=1025, baseline={k:curves["baseline"][k] for k in ("times","mean","se")},
            larger={k:curves["larger"][k] for k in ("times","mean","se")},
            bayes={k:curves["bayes"][k] for k in ("times","mean","se")},
            larger_layers=8, larger_parameters=plan["arms"]["xlarge"]["n_params"],
            larger_updates=latest, larger_interim=latest < 400000,
            comparison_scope="Both model capacity and training exposure change."))
    return paths, report, panels


def save(report, validation, out, *, plot=True):
    out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    rows=[]
    for g in report["groups"]:
        for name, value in g["predictors"].items():
            gap=g["paired"][name]
            rows.append(dict(depth=g["depth"],seed=g["seed"],model=name,layers=value["layers"],
                parameters=value["n_params"],updates=value["step"],training_sequences=value["completed_sequences"],
                regret_t1024=value["mean"][-1],regret_t1024_se=value["se"][-1],
                minus_large100k_t1024=gap["minus_large100k"]["mean"][-1],
                minus_large100k_t1024_se=gap["minus_large100k"]["se"][-1],
                minus_bayes_t64=gap["minus_bayes"]["mean"][-1],minus_bayes_t64_se=gap["minus_bayes"]["se"][-1]))
    with (out/"endpoints.tsv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    if plot:
        render(report,out)
    names=["comparison.json","endpoints.tsv"]+(["scaling_comparison.pdf","scaling_comparison.png"] if plot else [])
    validation.update(schema="shared-base-campaign-import-validation-v1",script_sha256=sha(__file__),
                      summary_sha256=sha(out/"comparison.json"),outputs={name:sha(out/name) for name in names},
                      plotting=plot,inference_replayed=False,bayesian_references_recomputed=False)
    (out/"validation.json").write_text(json.dumps(validation,indent=2,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=INPUT)
    p.add_argument("--cohorts",type=Path,default=COHORTS)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--parent-root",type=Path,default=PARENTS)
    p.add_argument("--baseline",type=Path,default=BASELINE)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--no-plot",action="store_true",help="Audit/import only; defer figure generation")
    a=p.parse_args()
    report,validation=import_results(a.input,a.cohorts,a.plan,a.parent_root,a.baseline)
    save(report,validation,a.out,plot=not a.no_plot)
    print(json.dumps(dict(passed=True,milestones=report["milestones"],checks=validation["checks"],
                         missing=report["missing_planned_milestones"],out=str(a.out)),indent=2))


if __name__=="__main__":
    main()

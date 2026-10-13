"""Review the fixed independent ABACA follow-up without changing old gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
from shared_base_cluster import (atomic_json, file_hash, fingerprint, output_lock,
                                 read_plan, require, validate_merge_receipt)

DEFAULT_ORIGINAL = ROOT / "code/transformer/configs/shared_base_cluster_plan_2026-09-16.json"
DEFAULT_FOLLOWUP = ROOT / "code/transformer/configs/shared_base_abaca_plan_2026-09-16.json"
CONTROLS = ("cal_product_0", "cal_shared_base_0", "cal_uniform_base_0",
            "cal_aaaaa", "cal_ababa", "cal_abaca")
ABACA = "cal_abaca"


def checked_receipt(context, path):
    raw = Path(path).read_bytes()
    saved = json.loads(raw)
    checked = validate_merge_receipt(context, saved, require_gate=False)
    for cid, case in checked["cases"].items():
        require(saved["cases"][cid].get("statistics") == case["statistics"],
                "nested statistics differ from raw repetitions")
    return checked, hashlib.sha256(raw).hexdigest()


def review(original_plan, followup_plan, original_receipt, followup_receipt):
    old = read_plan(original_plan, "calibrate")
    new = read_plan(followup_plan, "abaca_check")
    a, b = old["config"], new["config"]
    require(set(a["case_ids"]) == set(CONTROLS) and a["purpose"] == "exact_short"
            and a["particles"] == 8192 and a["replicates"] == 64,
            "original stage must be the six-control 8192-particle, 64-repetition calibration")
    require(b["case_ids"] == [ABACA] and set(new["cases"]) == {ABACA}
            and set(new["plan"]["stages"]) == {"abaca_check"}
            and b["purpose"] == "confirmation" and b.get("requires") == []
            and b["particles"] == 16384 and b["replicates"] == 128,
            "follow-up must be the independent ABACA-only 16384-particle, 128-repetition check")
    varied = {"purpose", "requires", "case_ids", "particles", "replicates", "seed"}
    require({k: v for k, v in a.items() if k not in varied}
            == {k: v for k, v in b.items() if k not in varied}, "numerical stage settings changed")
    require(b["seed"] not in {stage["seed"] for stage in old["plan"]["stages"].values()},
            "follow-up must use a master seed absent from every original stage")
    require(old["cases"][ABACA] == new["cases"][ABACA]
            and new["cases"][ABACA]["tokens"] == [0, 1, 0, 2, 0],
            "ABACA source, tokens or exact reference changed")
    for key in ("prior", "acceptance", "dependency_manifest_sha256"):
        require(old["plan"][key] == new["plan"][key], f"follow-up changed {key}")
    binding = new["plan"].get("followup_of", {})
    require(binding.get("original_plan_sha256") == old["plan_sha256"]
            and binding.get("original_stage") == "calibrate"
            and (ROOT / binding.get("original_plan", "")).resolve() == Path(original_plan).resolve(),
            "follow-up does not bind this original calibration plan")
    previous, old_hash = checked_receipt(old, original_receipt)
    following, new_hash = checked_receipt(new, followup_receipt)
    require(previous["numerical"] == following["numerical"], "numerical implementation or store identity changed")
    old_passes = previous["gate"]["passed_by_case"]
    require(not old_passes[ABACA] and all(old_passes[cid] for cid in CONTROLS if cid != ABACA),
            "this follow-up requires original ABACA failure and five other passing controls")
    old_seeds = {r["seed"] for c in previous["cases"].values() for r in c["records"]}
    new_seeds = [r["seed"] for r in following["cases"][ABACA]["records"]]
    require(len(set(new_seeds)) == 128 and old_seeds.isdisjoint(new_seeds),
            "follow-up reused a repetition seed")
    sources = dict(original=dict(plan_sha256=old["plan_sha256"], receipt_sha256=old_hash,
                                 stage="calibrate", gate=previous["gate"]),
                   followup=dict(plan_sha256=new["plan_sha256"], receipt_sha256=new_hash,
                                 stage="abaca_check", gate=following["gate"]))
    evidence = {}
    for cid in CONTROLS:
        label, receipt = ("followup", following) if cid == ABACA else ("original", previous)
        case = receipt["cases"][cid]
        evidence[cid] = dict(source=label, plan_sha256=sources[label]["plan_sha256"],
            receipt_sha256=sources[label]["receipt_sha256"], stage=receipt["stage"],
            records_sha256=fingerprint(case["records"]), statistics=case["statistics"],
            meets_threshold=receipt["gate"]["passed_by_case"][cid])
    return dict(schema="shared-base-abaca-review-v1", sources=sources, evidence_by_control=evidence,
        original_abaca=previous["cases"][ABACA]["statistics"],
        followup_abaca=following["cases"][ABACA]["statistics"],
        all_six_controls_meet_threshold_with_followup=all(c["meets_threshold"] for c in evidence.values()),
        original_calibration_gate_preserved=previous["gate"]["status"],
        production_approved=False, no_auto_promotion=True, pooled_estimates=False,
        criteria=previous["gate"]["criteria"], review_code_sha256=file_hash(__file__),
        scope="A follow-up selected after observing an exact-control failure; all six fixed controls are retained. "
              "Original and follow-up ABACA estimates use different particle counts and remain separate. "
              "This does not satisfy original-stage prerequisites, validate long sequences or establish quadrature accuracy.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-plan", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--followup-plan", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--original-receipt", type=Path, required=True)
    parser.add_argument("--followup-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(args.out.resolve() not in {getattr(args, k).resolve() for k in
                ("original_plan", "followup_plan", "original_receipt", "followup_receipt")},
                "review output must not overwrite an input")
        result = review(args.original_plan, args.followup_plan, args.original_receipt, args.followup_receipt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with output_lock(args.out):
            atomic_json(args.out, result)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    compact = {k: result[k] for k in ("original_calibration_gate_preserved",
        "all_six_controls_meet_threshold_with_followup", "production_approved", "no_auto_promotion")}
    for key in ("original_abaca", "followup_abaca"):
        compact[key] = {k: result[key][k] for k in ("repetitions", "estimated_minus_catalog_loss_bits",
            "approximate_loss_error_ci95_bits", "absolute_error_plus_ci95_max_halfwidth_bits")}
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

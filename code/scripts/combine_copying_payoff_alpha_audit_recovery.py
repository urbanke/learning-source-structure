"""Combine the six validated producers for the eight alpha refinements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import copying_payoff_alpha_refinement as alpha
import copying_payoff_alpha_memory_recovery as first_recovery
import copying_payoff_alpha_audit_only_salvage as t5_audit_only
import copying_payoff_alpha_sampling_memory_recovery as sampling_recovery
import copying_payoff_alpha_t6_audit_only_salvage as t6_audit_only
import copying_payoff_alpha_t7_t8_single_worker_recovery as t7_t8_recovery
import copying_payoff_alpha_t8_audit_only_salvage as t8_audit_only
import copying_payoff_alpha_memory_tail as unused_tail
import copying_payoff_alpha_t7_t8_recovery as unused_t7_t8


ROOT = alpha.ROOT
SCHEMA = "copying-payoff-alpha-final-combined-overlay-v1"
OVERLAY_SCHEMA = SCHEMA
BASELINE_DRAWS = alpha.BASELINE_DRAWS
TARGET_DRAWS = alpha.TARGET_DRAWS
DEFAULT_FIRST = ROOT / "output/copying_payoff_alpha_memory_recovery_2026-09-21"
DEFAULT_T5_AUDIT = ROOT / "output/copying_payoff_alpha_audit_only_salvage_2026-09-21"
DEFAULT_SAMPLING = ROOT / "output/copying_payoff_alpha_sampling_memory_recovery_2026-09-21"
DEFAULT_T6_AUDIT = ROOT / "output/copying_payoff_alpha_t6_audit_only_salvage_2026-09-21"
DEFAULT_T7_T8 = ROOT / "output/copying_payoff_alpha_t7_t8_single_worker_recovery_2026-09-21"
DEFAULT_T8_AUDIT = ROOT / "output/copying_payoff_alpha_t8_audit_only_salvage_2026-09-21"
DEFAULT_UNUSED_TAIL = ROOT / "output/copying_payoff_alpha_memory_tail_2026-09-21"
DEFAULT_UNUSED_T7_T8 = ROOT / "output/copying_payoff_alpha_t7_t8_recovery_2026-09-21"
DEFAULT_OUT = ROOT / "output/copying_payoff_alpha_final_combined_overlay_2026-09-21"
FIRST_IDS = {alpha.TARGETS[0][2]}
T5_AUDIT_IDS = {alpha.TARGETS[1][2]}
SAMPLING_IDS = {alpha.TARGETS[2][2]}
T6_AUDIT_IDS = {alpha.TARGETS[3][2]}
T7_T8_IDS = {item[2] for item in alpha.TARGETS[4:7]}
T8_AUDIT_IDS = {alpha.TARGETS[7][2]}
COMPONENT_IDS = (FIRST_IDS, T5_AUDIT_IDS, SAMPLING_IDS, T6_AUDIT_IDS,
                 T7_T8_IDS, T8_AUDIT_IDS)
EXPECTED_IDS = set().union(*COMPONENT_IDS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bound(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Combined-overlay input missing or linked")
    return dict(path=str(path.resolve()), sha256=sha(path), size_bytes=path.stat().st_size)


def replacement_item(producer, plan, spec):
    checked = producer.validate_replacement(plan, spec, require_pass=True)
    p, task = checked["paths"], spec["task"]
    item = dict(time=task["time"], order=task["order"], task_id=task["id"], task=task,
        producer_schema=producer.SCHEMA, producer_plan=bound(Path(plan["output"]) / "plan.json"),
        parent_plan=spec["parent_plan"], baseline_draws=producer.BASELINE_DRAWS,
        target_draws=producer.TARGET_DRAWS, parent_files=spec["parent_files"],
        replacement_files={name: bound(p[name]) for name in (*producer.PARENT_FILES, "receipt")},
        sample_prefix=checked["prefix"], rng_continuity=checked["rng"], audit_passed=True,
        sampling_performed=checked["report"].get("sampling_performed", True),
        audit_only_recovery=checked["report"].get("audit_only_recovery", False))
    return item, dict(producer=producer, plan=plan, spec=spec, checked=checked, item=item)


def component(producer, root, expected_ids):
    root = Path(root); plan = producer.load_plan(root)
    specs = {spec["task"]["id"]: spec for spec in plan["tasks"]}
    require(expected_ids <= set(specs), "Combined component plan lost required tasks")
    items, records = [], {}
    for task_id in sorted(expected_ids):
        item, record = replacement_item(producer, plan, specs[task_id])
        items.append(item); records[task_id] = record
    return dict(producer_schema=producer.SCHEMA, plan=bound(root / "plan.json"),
                allocation=bound(root / "allocation.json"), task_ids=sorted(expected_ids)), items, records


def collect(first=DEFAULT_FIRST, t5_audit=DEFAULT_T5_AUDIT, sampling=DEFAULT_SAMPLING,
            t6_audit=DEFAULT_T6_AUDIT, t7_t8=DEFAULT_T7_T8,
            t8_audit=DEFAULT_T8_AUDIT,
            unused_tail_root=DEFAULT_UNUSED_TAIL,
            unused_t7_t8_root=DEFAULT_UNUSED_T7_T8):
    first, t5_audit, sampling = Path(first), Path(t5_audit), Path(sampling)
    t6_audit, t7_t8, t8_audit = Path(t6_audit), Path(t7_t8), Path(t8_audit)
    unused_tail_root, unused_t7_t8_root = Path(unused_tail_root), Path(unused_t7_t8_root)
    require(sum(map(len, COMPONENT_IDS)) == len(EXPECTED_IDS) == 8,
            "Final alpha component partition overlaps or is incomplete")
    first_allocation = read(first / "allocation.json")
    first_ready = {item["task"]["id"] for item in first_allocation.get("attempts", [])
                   if item.get("status") == "ready"}
    statuses = {item["task"]["id"]: item.get("status")
                for item in first_allocation.get("attempts", [])}
    require(first_allocation.get("status") == "failed" and first_ready == FIRST_IDS
            and statuses.get(alpha.TARGETS[1][2]) == "memory_limit"
            and "memory_limit" in first_allocation.get("reason", ""),
            "First recovery did not preserve exactly its one passed record")
    t5_audit_allocation = read(t5_audit / "allocation.json")
    t5_audit_ready = {t5_audit_allocation.get("attempt", {}).get("task", {}).get("id")}
    require(t5_audit_allocation.get("status") == "ready" and t5_audit_ready == T5_AUDIT_IDS,
            "Audit-only component is not the ready t5/order-32 record")
    sampling_allocation = read(sampling / "allocation.json")
    sampling_ready = {item["task"]["id"] for item in sampling_allocation.get("attempts", [])
                      if item.get("status") == "ready"}
    sampling_statuses = {item["task"]["id"]: item.get("status")
                         for item in sampling_allocation.get("attempts", [])}
    require(sampling_allocation.get("status") == "failed"
            and sampling_ready == SAMPLING_IDS
            and sampling_statuses.get(alpha.TARGETS[3][2]) == "memory_limit"
            and sampling_allocation.get("pending_task_indices") == [2, 3, 4, 5]
            and "memory_limit" in sampling_allocation.get("reason", ""),
            "Sampling-memory recovery did not preserve exactly its ready t6/order-16 record")
    t6_audit_allocation = read(t6_audit / "allocation.json")
    t6_audit_ready = {t6_audit_allocation.get("attempt", {}).get("task", {}).get("id")}
    require(t6_audit_allocation.get("status") == "ready" and t6_audit_ready == T6_AUDIT_IDS,
            "Audit-only component is not the ready t6/order-32 record")
    t7_t8_allocation = read(t7_t8 / "allocation.json")
    t7_t8_ready = {item["task"]["id"] for item in t7_t8_allocation.get("attempts", [])
                   if item.get("status") == "ready"}
    t7_t8_statuses = {item["task"]["id"]: item.get("status")
                      for item in t7_t8_allocation.get("attempts", [])}
    require(t7_t8_allocation.get("status") == "failed" and t7_t8_ready == T7_T8_IDS
            and t7_t8_statuses.get(alpha.TARGETS[7][2]) == "memory_limit"
            and not t7_t8_allocation.get("pending_task_indices")
            and "memory_limit" in t7_t8_allocation.get("reason", ""),
            "Serial time-7--8 recovery did not preserve its three ready records")
    t8_audit_allocation = read(t8_audit / "allocation.json")
    t8_audit_ready = {t8_audit_allocation.get("attempt", {}).get("task", {}).get("id")}
    require(t8_audit_allocation.get("status") == "ready" and t8_audit_ready == T8_AUDIT_IDS,
            "Audit-only component is not the ready t8/order-32 record")
    tail_plan = unused_tail.load_plan(unused_tail_root)
    require({spec["task"]["id"] for spec in tail_plan["tasks"]}
            == {item[2] for item in alpha.TARGETS[6:]}
            and not (unused_tail_root / "allocation.json").exists(),
            "Prepared tail must remain unused")
    t7_t8_plan = unused_t7_t8.load_plan(unused_t7_t8_root)
    require({spec["task"]["id"] for spec in t7_t8_plan["tasks"]}
            == {item[2] for item in alpha.TARGETS[4:]}
            and not (unused_t7_t8_root / "allocation.json").exists(),
            "Prepared two-worker time-7--8 plan must remain unused")
    first_component, first_items, first_records = component(first_recovery, first, FIRST_IDS)
    t5_audit_component, t5_audit_items, t5_audit_records = component(
        t5_audit_only, t5_audit, T5_AUDIT_IDS)
    sampling_component, sampling_items, sampling_records = component(
        sampling_recovery, sampling, SAMPLING_IDS)
    t6_audit_component, t6_audit_items, t6_audit_records = component(
        t6_audit_only, t6_audit, T6_AUDIT_IDS)
    t7_t8_component, t7_t8_items, t7_t8_records = component(
        t7_t8_recovery, t7_t8, T7_T8_IDS)
    t8_audit_component, t8_audit_items, t8_audit_records = component(
        t8_audit_only, t8_audit, T8_AUDIT_IDS)
    records = {**first_records, **t5_audit_records, **sampling_records,
               **t6_audit_records, **t7_t8_records, **t8_audit_records}
    require(set(records) == EXPECTED_IDS and len(records) == 8,
            "Final combined alpha replacement family is incomplete")
    result = dict(schema=SCHEMA, complete=True, baseline_draws=alpha.BASELINE_DRAWS,
        target_draws=alpha.TARGET_DRAWS, original_acceptance_criteria_unchanged=True,
        exact_rng_and_retained_prefix_required=True, automatic_retries=False,
        components=[first_component, t5_audit_component, sampling_component,
                    t6_audit_component, t7_t8_component, t8_audit_component],
        unused_plans=[
            dict(plan=bound(unused_tail_root / "plan.json"), sampling_started=False,
                 excluded_from_components=True,
                 reason="Superseded before launch after the post-sampling memory peak was localized."),
            dict(plan=bound(unused_t7_t8_root / "plan.json"), sampling_started=False,
                 excluded_from_components=True,
                 reason="Superseded before launch by the serial allocation after measured peak memory."),
        ],
        replacements=sorted(first_items + t5_audit_items + sampling_items
                            + t6_audit_items + t7_t8_items + t8_audit_items,
                            key=lambda item: (item["time"], item["order"])),
        combiner=bound(__file__))
    return result, records


def overlay(out, *, write_manifest, first=DEFAULT_FIRST, t5_audit=DEFAULT_T5_AUDIT,
            sampling=DEFAULT_SAMPLING, t6_audit=DEFAULT_T6_AUDIT,
            t7_t8=DEFAULT_T7_T8, t8_audit=DEFAULT_T8_AUDIT,
            unused_tail_root=DEFAULT_UNUSED_TAIL,
            unused_t7_t8_root=DEFAULT_UNUSED_T7_T8):
    out = Path(out)
    result, records = collect(first, t5_audit, sampling, t6_audit, t7_t8, t8_audit,
                              unused_tail_root, unused_t7_t8_root)
    if write_manifest:
        require(not out.exists(), "Use a new final combined-overlay output directory")
        out.mkdir()
        alpha.write(out / "overlay.json", result, immutable=True)
    else:
        require(read(out / "overlay.json") == result, "Final combined alpha overlay changed")
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--first", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--t5-audit", type=Path, default=DEFAULT_T5_AUDIT)
    parser.add_argument("--sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--t6-audit", type=Path, default=DEFAULT_T6_AUDIT)
    parser.add_argument("--t7-t8", type=Path, default=DEFAULT_T7_T8)
    parser.add_argument("--t8-audit", type=Path, default=DEFAULT_T8_AUDIT)
    parser.add_argument("--unused-tail", type=Path, default=DEFAULT_UNUSED_TAIL)
    parser.add_argument("--unused-t7-t8", type=Path, default=DEFAULT_UNUSED_T7_T8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result, _ = overlay(args.out, write_manifest=args.action == "write",
        first=args.first, t5_audit=args.t5_audit, sampling=args.sampling,
        t6_audit=args.t6_audit, t7_t8=args.t7_t8, t8_audit=args.t8_audit,
        unused_tail_root=args.unused_tail, unused_t7_t8_root=args.unused_t7_t8)
    print(json.dumps(dict(schema=result["schema"], complete=result["complete"],
        replacements=len(result["replacements"]), components=len(result["components"]))))


if __name__ == "__main__":
    main()

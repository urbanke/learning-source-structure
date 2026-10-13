"""Run or strictly merge portable CPU shards of independent evidence repetitions."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/transformer"))
# The implementation deliberately has the same descriptive filename in its
# package directory; it is imported by path order, not this __main__ module.
from shared_base_cluster import merge_shards, run_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--case-id", required=True)
    run.add_argument("--replica-start", type=int, required=True)
    run.add_argument("--replica-stop", type=int, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--src", type=Path, required=True)
    run.add_argument("--seconds", type=int, default=600)
    run.add_argument("--receipts", type=Path, help="Directory containing merged prerequisite stage receipts")
    merge = commands.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--stage", required=True)
    merge.add_argument("--shards", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        report, code = run_shard(args.plan, args.stage, args.case_id, args.replica_start,
                                args.replica_stop, args.out, args.store, args.src, args.seconds, args.receipts)
        print(json.dumps(dict(out=str(args.out), complete=report["complete"], status=report["status"],
                              records=len(report["records"]))), flush=True)
    else:
        report, code = merge_shards(args.plan, args.stage, args.shards, args.out)
        print(json.dumps(dict(out=str(args.out), complete=report["complete"], gate=report["gate"])), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

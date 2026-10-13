"""One-use bounded Kuma submission for an isolated larger pooled campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("benchmark", "train"))
    p.add_argument("--out", type=Path, default=ROOT/"output/pooled_transformer_large_2026-09-22")
    p.add_argument("--plan", type=Path, default=ROOT/"code/transformer/configs/pooled_transformer_large_plan_2026-09-22.json")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--hours-per-seed", type=int, default=12)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    if not 1 <= a.hours_per_seed <= 24:
        p.error("Each seed must have a finite allocation of 1..24 hours")
    out, plan, sources = a.out.resolve(), a.plan.resolve(), a.sources.resolve()
    if not out.is_relative_to(ROOT/"output") or out == ROOT/"output" or out == sources or out in sources.parents or sources in out.parents:
        p.error("Use a fresh separate directory beneath output")
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    source_manifest_hash = hashlib.sha256((sources/"manifest.json").read_bytes()).hexdigest()
    code_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        "code/scripts/pooled_transformer_large.py", "code/scripts/submit_pooled_transformer_large.py",
        "code/transformer/kuma/pooled_transformer_large.sbatch")}
    if a.mode == "train":
        b = json.loads((out/"benchmark/benchmark.json").read_text())
        if not b["passed"] or b["plan_sha256"] != plan_hash or not math.isfinite(b["projected_training_hours_including_monitoring"]):
            p.error("Matching successful benchmark required")
        if b["projected_training_hours_including_monitoring"] >= a.hours_per_seed-0.5:
            p.error("Allocation needs at least 30 minutes beyond measured training projection")
    name = "pooled-large-bench" if a.mode == "benchmark" else "pooled-large-train"
    command = ["sbatch", "--parsable", "--account=lthc", f"--chdir={ROOT}", f"--job-name={name}"]
    if a.mode == "benchmark":
        command += ["--qos=debug", "--time=00:30:00"]
    else:
        command += ["--array=0-1%2", f"--time={a.hours_per_seed:02d}:00:00"]
    command += [f"--output={out}/logs/{a.mode}-%A_%a.out", str(ROOT/"code/transformer/kuma/pooled_transformer_large.sbatch"), a.mode, str(out), str(plan), str(sources)]
    print(json.dumps(dict(mode=a.mode, maximum_gpu_hours=.5 if a.mode=="benchmark" else 2*a.hours_per_seed,
                         concurrent_gpus=1 if a.mode=="benchmark" else 2, automatic_retries=False)))
    print(shlex.join(command), flush=True)
    if not a.submit:
        return
    active = subprocess.run(["squeue", "-u", "urbanke", "-h", "-o", "%j"], check=True, text=True, capture_output=True).stdout.splitlines()
    if name in active:
        p.error("An allocation with this campaign name is active; inspect before another submission")
    (out/"logs").mkdir(parents=True, exist_ok=True)
    claim = out/f"{a.mode}_submission.json"
    with claim.open("x") as handle:
        receipt = dict(status="submitting", command=command, plan_sha256=plan_hash,
                       source_manifest_sha256=source_manifest_hash, implementation_sha256=code_hashes)
        json.dump(receipt, handle)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    receipt.update(status="submitted", job_id=result.stdout.strip(), scheduler_message=result.stderr)
    claim.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

"""Release four bounded specialist runs only after both timing gates pass."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import specialist_transformer_large as campaign
ROOT=campaign.ROOT
NAMES=("code/scripts/specialist_transformer_large.py",
       "code/transformer/kuma/specialist_transformer_large.sbatch",
       "code/scripts/specialist_transformer_release.py",
       "code/transformer/kuma/specialist_transformer_release.sbatch")


def gate(out,plan):
    launch=campaign.read(out/"train_submission.json")
    benchmark_launch=campaign.read(out/"benchmark_submission.json")
    if launch["parent_benchmark_job"]!=benchmark_launch["job_id"]:
        raise ValueError("Benchmark job identity changed")
    for key,value in (("plan_sha256",campaign.sha(plan)),
                      ("cohort_manifest_sha256",campaign.sha(out/"cohorts/manifest.json"))):
        if launch[key]!=value or benchmark_launch[key]!=value:
            raise ValueError(f"Launch input changed:{key}")
    for name,digest in launch["implementation_sha256"].items():
        if campaign.sha(ROOT/name)!=digest:raise ValueError(f"Launch code changed:{name}")
    for name,digest in campaign.read(plan)["implementation_sha256"].items():
        if campaign.sha(ROOT/name)!=digest:raise ValueError(f"Numerical code changed:{name}")
    for name,digest in benchmark_launch["implementation_sha256"].items():
        if launch["implementation_sha256"].get(name)!=digest:
            raise ValueError("Training and timing implementations differ")
    measurements={}
    for family in campaign.FAMILIES:
        b=campaign.read(out/f"benchmark/{family}/benchmark.json")
        hours=b["projected_training_hours"]
        if (b["passed"] is not True or b["plan_sha256"]!=launch["plan_sha256"] or
            b["family"]!=family or b["seed"]!=0 or b["completed_steps"]!=1000 or
            b["launch_binding"]!=benchmark_launch or not math.isfinite(hours) or not 0<hours<=3.5):
            raise ValueError(f"Timing or provenance gate failed:{family}")
        measurements[family]=hours
    return dict(passed=True,maximum_hours_per_run=4,measurements=measurements,
                plan_sha256=launch["plan_sha256"],cohort_manifest_sha256=launch["cohort_manifest_sha256"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("submit","run"));p.add_argument("--out",type=Path,default=campaign.OUT)
    p.add_argument("--plan",type=Path,default=campaign.PLAN);p.add_argument("--after-job")
    p.add_argument("--task",type=int,choices=range(4));p.add_argument("--submit",action="store_true")
    a=p.parse_args();out=a.out.resolve();plan=a.plan.resolve()
    if a.mode=="run":
        if a.task is None:p.error("Array task required")
        print(json.dumps(dict(event="specialist_release_check",task=a.task)),flush=True)
        try:report=gate(out,plan)
        except Exception as e:
            campaign.write(out/f"release_checks/task{a.task}.json",dict(passed=False,error=str(e)))
            raise
        campaign.write(out/f"release_checks/task{a.task}.json",report)
        return subprocess.run([sys.executable,str(ROOT/"code/scripts/specialist_transformer_large.py"),
            "train","--out",str(out),"--plan",str(plan),"--task",str(a.task)],check=False).returncode
    if not out.is_relative_to(ROOT/"output") or out==ROOT/"output":p.error("Separate campaign output required")
    prior=campaign.read(out/"benchmark_submission.json")
    if not a.after_job or not a.after_job.isdecimal() or prior["job_id"]!=a.after_job or prior["status"]!="submitted":
        p.error("Exact submitted benchmark array required")
    if prior["plan_sha256"]!=campaign.sha(plan) or prior["cohort_manifest_sha256"]!=campaign.sha(out/"cohorts/manifest.json"):
        p.error("Original benchmark input binding changed")
    binding=dict(plan_sha256=campaign.sha(plan),cohort_manifest_sha256=prior["cohort_manifest_sha256"],
                 implementation_sha256={name:campaign.sha(ROOT/name) for name in NAMES},parent_benchmark_job=a.after_job)
    if any(binding["implementation_sha256"].get(k)!=v for k,v in prior["implementation_sha256"].items()):
        p.error("Benchmark implementation changed")
    command=["sbatch","--parsable","--account=lthc",f"--chdir={ROOT}","--job-name=specialist-large-train",
        "--array=0-3%4","--time=04:00:00",f"--dependency=afterok:{a.after_job}","--kill-on-invalid-dep=yes",
        f"--output={out}/logs/train-%A_%a.out",str(ROOT/"code/transformer/kuma/specialist_transformer_release.sbatch"),str(out),str(plan)]
    print(json.dumps(dict(maximum_gpu_hours=16,automatic_retries=False,command=command)),flush=True)
    if not a.submit:return 0
    active=subprocess.run(["squeue","-u","urbanke","-h","-o","%j"],text=True,capture_output=True,check=True).stdout.splitlines()
    if "specialist-large-train" in active:p.error("Inspect active training before submitting again")
    claim=out/"train_submission.json"
    with claim.open("x") as f:json.dump(dict(binding,status="submitting",command=command),f)
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    receipt=dict(binding,status="submitted",command=command,job_id=result.stdout.strip(),scheduler_message=result.stderr)
    campaign.write(claim,receipt);print(json.dumps(receipt),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())

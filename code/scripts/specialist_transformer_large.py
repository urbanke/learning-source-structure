"""Larger product and uniform-base models on the existing priors and test cohorts.

Reuse train.py without numerical edits:100000 updates,batch64,seeds0/1.
Match priors and sample budgets; do not claim exact historical training batches.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"code/transformer/configs/specialist_transformer_large_plan_2026-09-22.json"
OUT = ROOT/"output/specialist_transformer_large_2026-09-22"
sys.path[:0] = [str(ROOT/"code/transformer"),str(ROOT/"code/scripts")]
FAMILIES = ("product", "uniform_base")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def prepare(out):
    import numpy as np
    from context_accuracy_audit import record
    times = 2**np.arange(11)
    manifest = dict(schema="specialist-large-cohorts-v1",times=times.tolist(),families={})
    for family in FAMILIES:
        rows = dict(tokens=[],true_probs=[],small_regret=[],bayes_regret=[],groups=[])
        summary_path=ROOT/f"results/transformer/conditional_accuracy/{family}/summary.json"
        summary=read(summary_path)
        if family=="product":
            for group,case in enumerate(summary["cases"]):
                a,b=case["depth_pair"]; folder=ROOT/f"output/conditional_accuracy/product/L{a}_{b}"
                digest=hashlib.sha256()
                for index in range(256):
                    paths={p:folder/p/f"seq_{index:06d}.npz" for p in ("data","references","model")}
                    for part,path in paths.items():
                        digest.update(f"{part}/{path.name}:{sha(path)}".encode())
                    with np.load(paths["data"],allow_pickle=False) as z:
                        tokens=z["tokens"].copy(); true=z["true_probs"].copy()
                    with np.load(paths["references"],allow_pickle=False) as z:
                        bayes=z["bayes_probs"].copy()
                    with np.load(paths["model"],allow_pickle=False) as z:
                        logp=z["logp"].astype(float)
                    entropy=np.sum(np.where(true>0,true*np.log2(np.maximum(true,1e-300)),0),axis=-1)
                    for key,value in dict(tokens=tokens,true_probs=true,small_regret=entropy-np.sum(true*logp,axis=-1)/np.log(2),
                                          bayes_regret=entropy-np.sum(true*np.log2(bayes),axis=-1),groups=group).items():
                        rows[key].append(value)
                if digest.hexdigest()!=case["raw_records_sha256"]:
                    raise ValueError("Product raw cohort changed")
            cases=summary["cases"]
        else:
            folder=ROOT/"output/copier_time/run_2026-09-15"; protocol=read(folder/"protocol.json")
            digest=hashlib.sha256()
            keys=("tokens","prefix_lengths","source_depth","transition_probs","intact_probs","bayes_probs",
                  "metric_oracle_regret_intact","metric_oracle_regret_bayes")
            for index in range(1024):
                z,_=record(folder/f"seq_{index:06d}.npz",protocol,keys,digest)
                for key,value in dict(tokens=z["tokens"],true_probs=z["transition_probs"][z["tokens"][times-1]],
                                      small_regret=z["metric_oracle_regret_intact"][times-1],
                                      bayes_regret=z["metric_oracle_regret_bayes"][times-1],groups=int(z["source_depth"])).items():
                    rows[key].append(value)
                if index%256==255:print(json.dumps(dict(event="cohort_verified",family=family,count=index+1)),flush=True)
            if digest.hexdigest()!=summary["raw_arrays_sha256"]:
                raise ValueError("Uniform raw cohort changed")
            cases=summary["panels"]
        arrays={k:np.asarray(v) for k,v in rows.items()}
        for group,case in enumerate(cases):
            mask=arrays["groups"]==group
            if int(mask.sum())!=case["n_sequences"]:raise ValueError("Incomplete frozen group")
            for key,old in (("small_regret","transformer"),("bayes_regret","informed_bayes")):
                values=arrays[key][mask]
                if not np.allclose(values.mean(0),case[old]["mean"],rtol=0,atol=2e-12):
                    raise ValueError("Frozen means changed")
                if not np.allclose(values.std(0,ddof=1)/np.sqrt(len(values)),case[old]["se"],rtol=0,atol=2e-12):
                    raise ValueError("Frozen uncertainties changed")
        out.mkdir(parents=True,exist_ok=True)
        path=out/f"{family}.npz"
        if path.exists():raise ValueError("Fresh cohort output required")
        np.savez_compressed(path,**arrays,times=times)
        manifest["families"][family]=dict(file=path.name,sha256=sha(path),n_sequences=len(arrays["tokens"]),
            original_summary_sha256=sha(summary_path),groups=[dict(id=i,condition=c.get("depth_pair",c.get("details")),n_sequences=c["n_sequences"]) for i,c in enumerate(cases)])
    write(out/"manifest.json",manifest)
    print(json.dumps(dict(event="cohorts_prepared",path=str(out/"manifest.json"))),flush=True)


def submit(args):
    out=args.out.resolve();stage=args.stage
    if not out.is_relative_to(ROOT/"output") or out==ROOT/"output":raise ValueError("Separate output required")
    if not 1<=args.hours<=8:raise ValueError("Finite1..8-hour reservations required")
    if stage=="train":
        for family in FAMILIES:
            b=read(out/f"benchmark/{family}/benchmark.json")
            if not b["passed"] or b["plan_sha256"]!=sha(args.plan) or b["projected_training_hours"]>=args.hours-.5:
                raise ValueError("Both matching benchmarks and a30-minute evaluation margin are required")
    code=["code/scripts/specialist_transformer_large.py","code/transformer/kuma/specialist_transformer_large.sbatch"]
    binding=dict(plan_sha256=sha(args.plan),implementation_sha256={p:sha(ROOT/p) for p in code},
                 cohort_manifest_sha256=sha(out/"cohorts/manifest.json"))
    name="specialist-large-"+stage
    command=["sbatch","--parsable","--account=lthc",f"--chdir={ROOT}",f"--job-name={name}",
        "--array=0-1%2" if stage=="benchmark" else "--array=0-3%4",
        "--time=00:30:00" if stage=="benchmark" else f"--time={args.hours:02d}:00:00",
        f"--output={out}/logs/{stage}-%A_%a.out"]
    if stage=="benchmark":command+=["--qos=debug"]
    command += [str(ROOT/"code/transformer/kuma/specialist_transformer_large.sbatch"),stage,str(out),str(args.plan.resolve())]
    print(json.dumps(dict(stage=stage,maximum_gpu_hours=1 if stage=="benchmark" else 4*args.hours,automatic_retries=False)))
    print(shlex.join(command),flush=True)
    if not args.submit:return
    active=subprocess.run(["squeue","-u","urbanke","-h","-o","%j"],text=True,capture_output=True,check=True).stdout.splitlines()
    if name in active:raise ValueError("Existing active allocations must be checked first")
    (out/"logs").mkdir(parents=True,exist_ok=True)
    claim=out/f"{stage}_submission.json"
    with claim.open("x") as f:json.dump(dict(binding,status="submitting",command=command),f)
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    receipt=dict(binding,status="submitted",command=command,job_id=result.stdout.strip(),scheduler_message=result.stderr)
    write(claim,receipt);print(json.dumps(receipt),flush=True)


def evaluate(out,family,seed,plan):
    import numpy as np
    import torch
    from model import GPT
    from eval_mixture import check_causality
    manifest=read(out/"cohorts/manifest.json");entry=manifest["families"][family]
    path=out/"cohorts"/entry["file"]
    if sha(path)!=entry["sha256"]:raise ValueError("Frozen evaluation changed")
    checkpoint=out/f"train/{family}/seed{seed}/model_100000.pt"
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["step"]!=100000:raise ValueError("Final planned checkpoint required")
    model=GPT(**saved["cfg"]["model"]).cuda().eval();model.load_state_dict(saved["model"])
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    tokens,true,times=arrays["tokens"],arrays["true_probs"],arrays["times"]
    regrets=[]
    for left in range(0,len(tokens),16):
        x=torch.as_tensor(tokens[left:left+16].astype(np.int64),device="cuda")
        if left==0:causality=check_causality(model,x,times)
        with torch.inference_mode(): lp=model(x[:,:-1])[:,times-1].double().log_softmax(-1).cpu().numpy()
        p=true[left:left+16]
        regrets.append(np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-300)),0)-p*lp/np.log(2),axis=-1))
    regret=np.concatenate(regrets)
    if not np.isfinite(regret).all() or regret.min() < -1e-8:raise ValueError("Invalid regret")
    dest=out/f"eval/{family}/seed{seed}";dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/"scores.npz",regret=regret,times=times,groups=arrays["groups"])
    result=dict(complete=True,family=family,seed=seed,checkpoint_sha256=sha(checkpoint),
                cohort_sha256=entry["sha256"],plan_sha256=sha(plan),causality=causality,groups=[])
    for group in entry["groups"]:
        mask=arrays["groups"]==group["id"]
        values=dict(transformer=regret[mask],informed_bayes=arrays["bayes_regret"][mask],
                    transformer_minus_bayes=regret[mask]-arrays["bayes_regret"][mask],
                    improvement_over_small=arrays["small_regret"][mask]-regret[mask])
        result["groups"].append(dict(group,**{k:dict(mean=v.mean(0).tolist(),se=(v.std(0,ddof=1)/np.sqrt(len(v))).tolist()) for k,v in values.items()}))
    write(dest/"summary.json",result)
    print(json.dumps(dict(event="specialist_evaluation_complete",family=family,seed=seed)),flush=True)


def run(args):
    family=FAMILIES[args.task if args.mode=="benchmark" else args.task//2]
    seed=0 if args.mode=="benchmark" else args.task%2
    print(json.dumps(dict(event="specialist_start",family=family,seed=seed,mode=args.mode)),flush=True)
    plan=read(args.plan); binding=read(args.out/f"{args.mode}_submission.json")
    if binding["plan_sha256"]!=sha(args.plan) or binding["cohort_manifest_sha256"]!=sha(args.out/"cohorts/manifest.json"):
        raise ValueError("Frozen launch plan/cohort changed")
    for name,digest in dict(plan["implementation_sha256"],**binding["implementation_sha256"]).items():
        if sha(ROOT/name)!=digest:raise ValueError(f"Frozen code changed:{name}")
    baseline=ROOT/plan["families"][family]["baseline_config"]
    if sha(baseline)!=plan["families"][family]["baseline_config_sha256"]:raise ValueError("Baseline configuration changed")
    cfg=read(baseline)
    if (plan["schema"]!="specialist-transformer-large-plan-v1" or
        plan["architecture"]!={"n_layer":4,"n_head":8,"d_model":256} or
        (cfg["steps"],plan["steps"])!=(100000,100000) or (cfg["batch"],plan["batch"])!=(64,64)):
        raise ValueError("Unexpected architecture or training budget")
    import torch
    import train as trainer
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError("CUDA allocation required")
    run_dir=args.out/f"{args.mode}/{family}/seed{seed}"
    run_dir.mkdir(parents=True,exist_ok=False)
    cfg.update(**plan["architecture"],seed=seed,out=str(run_dir),resume=False)
    parameters=("d","n_ctx","family","n_layer","n_head","d_model","batch","steps","lr","warmup","wd","clip",
                "seed","workers","log_every","val_every","ckpt_every","val_settings","val_n","out")
    argv=[str(ROOT/"code/transformer/train.py"),"--device","cuda"]
    for name in parameters:argv += ["--"+name,str(cfg[name])]
    argv += ["--alpha",*map(str,cfg["alpha"]),"--ls",*map(str,cfg["ls"])]
    if cfg["no_permute"]:argv += ["--no_permute"]
    if cfg["no_mlp"]:argv += ["--no_mlp"]
    captured={};original_model=trainer.GPT;original_adam=torch.optim.AdamW
    def model(*a,**kw):
        captured["model"]=original_model(*a,**kw)
        if captured["model"].n_params()!=3454464:raise ValueError("Unexpected larger architecture")
        return captured["model"]
    def adam(*a,**kw):captured["optimizer"]=original_adam(*a,**kw);return captured["optimizer"]
    trainer.GPT=model;torch.optim.AdamW=adam
    validation_seconds=[];original_eval=trainer.evaluate
    def timed_eval(*a,**kw):
        t=time.monotonic();v=original_eval(*a,**kw);validation_seconds.append(time.monotonic()-t);return v
    trainer.evaluate=timed_eval
    if args.mode=="benchmark":
        original_loader=torch.utils.data.DataLoader
        torch.utils.data.DataLoader=lambda *a,**kw:itertools.islice(original_loader(*a,**kw),plan["benchmark_steps"])
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();sys.argv=argv;trainer.main()
    recorded=read(run_dir/"config.json")
    expected_config=dict(plan["architecture"],steps=plan["steps"],batch=plan["batch"],seed=seed)
    if any(recorded[k]!=v for k,v in expected_config.items()) or recorded["n_params"]!=3454464:
        raise ValueError("Completed model differs from the frozen architecture/budget")
    records=[json.loads(x) for x in (run_dir/"log.jsonl").read_text().splitlines()]
    logs=[r for r in records if "elapsed" in r]
    expected=plan["benchmark_steps"] if args.mode=="benchmark" else 100000
    if logs[-1]["step"]!=expected or not all(math.isfinite(r["loss_bits"]) for r in logs):raise ValueError("Incomplete/nonfinite training")
    result=dict(passed=True,plan_sha256=sha(args.plan),family=family,seed=seed,completed_steps=expected,
                wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                source_relationship=plan["source_relationship"],launch_binding=binding)
    if args.mode=="benchmark":
        start=time.monotonic()
        torch.save(dict(model=captured["model"].state_dict(),opt=captured["optimizer"].state_dict(),step=expected),run_dir/"benchmark_checkpoint.pt")
        ckpt=time.monotonic()-start
        rate=(logs[-1]["elapsed"]-logs[0]["elapsed"])/(logs[-1]["step"]-logs[0]["step"])
        # Training logs exclude the final validation; account for100 monitoring passes and3 writes/checkpoint.
        hours=(rate*100000+sum(validation_seconds)*100+ckpt*60)/3600
        result.update(seconds_per_step=rate,validation_seconds=sum(validation_seconds),checkpoint_seconds=ckpt,projected_training_hours=hours)
        write(args.out/f"benchmark/{family}/benchmark.json",result)
    else:
        write(run_dir/"status.json",dict(result,complete=True))
        del captured["model"],captured["optimizer"]
        evaluate(args.out,family,seed,args.plan)
    print(json.dumps(dict(event="specialist_stage_complete",**result)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","submit","benchmark","train"))
    p.add_argument("--plan",type=Path,default=PLAN);p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--stage",choices=("benchmark","train"),default="benchmark")
    p.add_argument("--hours",type=int,default=4);p.add_argument("--submit",action="store_true")
    p.add_argument("--task",type=int,default=0);args=p.parse_args()
    if args.mode=="prepare":prepare(args.out/"cohorts")
    elif args.mode=="submit":submit(args)
    else:run(args)


if __name__=="__main__":main()

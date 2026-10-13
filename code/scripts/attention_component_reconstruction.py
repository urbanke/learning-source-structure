"""Check complete probability reconstruction with fixed copying candidates.

All coefficients are descriptive reconstruction fits, not identified
internal gates. This consumer never selects a head or modifies a network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code/transformer"))
from component_mixture_fit import fit_component_mixture, fit_support_mass
from eval_conditional_crossing import array_hash, fingerprint

CONFIGS = {"attention_deleted": ("attention", "deleted"),
           "attention_uniform": ("attention", "uniform"),
           "empirical_uniform": ("empirical", "uniform"),
           "support_uniform": ("support_uniform", "uniform")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def stat(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0,mean=None,se=None,p95=None)
    return dict(n=len(x),mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
                p95=float(np.quantile(x,.95)))


def ratio_stat(residual, baseline):
    use = np.isfinite(residual) & np.isfinite(baseline)
    r,b = residual[use],baseline[use]
    if not len(r) or b.mean() <= 1e-20:
        return dict(n=int(use.sum()),mean=None,se=None)
    ratio = 1-r.mean()/b.mean()
    influence = ((b-r)-ratio*b)/b.mean()
    return dict(n=len(r),mean=float(ratio),se=float(influence.std(ddof=1)/np.sqrt(len(r))) if len(r)>1 else None)


def point_summary(metrics):
    report = {key:stat(value) for key,value in metrics.items()}
    report["explained_squared_error"] = ratio_stat(metrics["sse"],metrics["baseline_sse"])
    report["heldout_explained_squared_error"] = ratio_stat(metrics["heldout_sse"],metrics["baseline_sse"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"output/component_feasibility/attention_2026-09-15")
    parser.add_argument("--out",type=Path,default=ROOT/"results/transformer/component_feasibility")
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    if args.check_only:
        report = json.loads((args.out/"reconstruction.json").read_text())
        for path,digest in report["hashes"].items():
            if sha(ROOT/path) != digest:
                raise ValueError(f"Changed input or artifact: {path}")
        print("Verified complete reconstruction report and input/artifact hashes")
        return
    protocol = json.loads((args.input/"protocol.json").read_text())
    raw = Path(protocol["input"])
    if protocol["n_sequences"] != 1024 or protocol["sequence_length"] != 1025:
        raise ValueError("Expected full fixed cohort")
    for path,digest in protocol["producer_hashes"].items():
        if sha(ROOT/path) != digest:
            raise ValueError("Candidate producer changed after run")
    if sha(raw/"protocol.json") != protocol["source_protocol_sha256"]:
        raise ValueError("Source protocol changed")
    expected = {f"seq_{i:06d}.npz" for i in range(1024)}
    if {p.name for p in args.input.glob("seq_*.npz")} != expected:
        raise ValueError("Candidate cohort incomplete")
    metrics = {config:{} for config in CONFIGS}
    defined = np.empty((1024,1024),dtype=bool)
    no_opportunity_tv = {base:np.full((1024,1024),np.nan) for base in ("deleted","uniform")}
    depths = np.empty(1024,dtype=np.int64)
    inputs = []
    checks = dict(intact_replay_max_abs=0.,truncated_replay_max_abs=0.)
    for index in range(1024):
        path = args.input/f"seq_{index:06d}.npz"
        source = raw/path.name
        with np.load(path,allow_pickle=False) as f:
            receipt = json.loads(str(f["metadata_json"]))
            arrays = {key:f[key] for key in f.files if key != "metadata_json"}
        if receipt["index"] != index or receipt["protocol_id"] != fingerprint(protocol):
            raise ValueError("Candidate identity changed")
        source_hash = sha(source)
        if source_hash != receipt["source_sha256"]:
            raise ValueError("Source record changed")
        for key,value in arrays.items():
            if array_hash(value) != receipt["array_hashes"][key]:
                raise ValueError("Candidate array changed")
        for key in checks:
            if not np.isfinite(receipt[key]) or not 0 <= receipt[key] <= 1e-12:
                raise ValueError("Nonfinite or excessive intact/truncated replay error")
            checks[key] = max(checks[key],receipt[key])
        with np.load(source,allow_pickle=False) as f:
            target,deleted,counts = f["intact_probs"],f["copier_deleted_probs"],f["metric_row_counts"]
            depth = int(f["source_depth"])
        depths[index] = depth
        if depth != int(arrays["source_depth"]):
            raise ValueError("Source depth changed")
        totals = counts.sum(-1)
        available = totals > 0
        defined[index] = available
        if not np.array_equal(available, arrays["defined"]):
            raise ValueError("Candidate availability disagrees with row counts")
        support = counts > 0
        candidates = {"attention":arrays["attention_candidate"]}
        for name,values in (("empirical",counts),("support_uniform",support)):
            candidates[name] = np.full_like(target,1/64)
            candidates[name][available] = values[available]/values[available].sum(-1,keepdims=True)
        for config,(copy_name,base_name) in CONFIGS.items():
            base = deleted if base_name == "deleted" else np.full_like(target,1/64)
            no_opportunity_tv[base_name][index] = np.where(~available,.5*np.abs(target-base).sum(-1),np.nan)
            candidate = candidates[copy_name]
            result = fit_component_mixture(target,base,candidate)
            mass = fit_support_mass(target,base,support)
            # Coefficients from outside-support mass are evaluated only
            # when they define an actual convex combination, without repair.
            valid_mass = available & mass["identified"] & mass["strictly_convex"]
            mass_fit = base + np.where(valid_mass,mass["weights_raw"],0.)[:,None]*(candidate-base)
            values = dict(weight=result["weights"],sse=result["squared_error"],
                          baseline_sse=result["baseline_squared_error"],tv=result["tv_error"],
                          baseline_tv=.5*np.abs(target-base).sum(-1),
                          heldout_sse=result["cross_validation"]["heldout_squared_error"],
                          mass_tv=np.where(valid_mass,.5*np.abs(target-mass_fit).sum(-1),np.nan),
                          mass_sse=np.where(valid_mass,((target-mass_fit)**2).sum(-1),np.nan),
                          mass_weight=np.where(mass["identified"],mass["weights_raw"],np.nan),
                          mass_outside_interval=np.where(mass["identified"],~mass["strictly_convex"],np.nan))
            for key,value in values.items():
                if key not in metrics[config]:
                    metrics[config][key] = np.full((1024,1024),np.nan)
                metrics[config][key][index] = np.where(available,value,np.nan)
        inputs.append(dict(index=index,source_depth=depth,source=str(source.relative_to(ROOT)),source_sha256=source_hash,
                           candidate=str(path.relative_to(ROOT)),candidate_sha256=sha(path)))
        if (index+1)%256 == 0:
            print(f"Analyzed {index+1}/1024",flush=True)
    groups = {}
    table = []
    for name,selected in [("all",np.ones(1024,dtype=bool)),*((f"depth_{i}",depths==i) for i in range(9))]:
        group = dict(n_sequences=int(selected.sum()),
                     available_fraction={str(t):float(defined[selected,t-1].mean()) for t in (16,64,256,1024)},
                     no_copy_opportunity={str(t):{base:stat(value[selected,t-1]) for base,value in no_opportunity_tv.items()}
                                          for t in (16,64,256,1024)},fits={})
        for config,all_metrics in metrics.items():
            selected_metrics = {k:v[selected] for k,v in all_metrics.items()}
            points = {str(t):point_summary({k:v[:,t-1] for k,v in selected_metrics.items()}) for t in (16,64,256,1024)}
            sequence_means = {key:np.nanmean(value,axis=1) for key,value in selected_metrics.items()}
            group["fits"][config] = dict(points=points,whole_sequence_available_times=point_summary(sequence_means))
            for t,statistics in points.items():
                for key,value in statistics.items():
                    table.append(dict(group=name,config=config,time=int(t),quantity=key,**value))
        groups[name] = group
    args.out.mkdir(parents=True,exist_ok=True)
    table_path = args.out/"reconstruction.tsv"
    with table_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["group","config","time","quantity","n","mean","se","p95"],
                              delimiter="\t",lineterminator="\n",restval="NA")
        writer.writeheader()
        writer.writerows({k:"NA" if v is None else v for k,v in row.items()} for row in table)
    report = dict(schema="attention-component-reconstruction-v1",n_sequences=1024,sequence_length=1025,
                  scope="Predictive candidate fit, not identified internal mixture weights",groups=groups,checks=checks,
                  truncated_replay_checked_indices=[0],
                  source_depth_counts=np.bincount(depths,minlength=9).tolist(),inputs=inputs,
                  definitions=dict(tv="half the sum of absolute errors over all64token probabilities",
                      explained_squared_error="one minus summed fitted squared error divided by summed baseline squared error, over available independent sequences",
                      heldout="fit one scalar on even labels, score odd labels and reverse; no half renormalization",
                      mass_weight="one minus target probability outside observed successor support divided by baseline probability there; never clipped",
                      candidates=protocol["pooling"],availability="Only times with at least one completed matching transition have a copying candidate"))
    plot(report,args.out)
    paths = [Path(__file__), ROOT/"code/transformer/component_mixture_fit.py",args.input/"protocol.json",table_path,args.out/"reconstruction.png"]
    paths += [ROOT/p for p in protocol["producer_hashes"]]
    report["hashes"] = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    for item in inputs:
        report["hashes"][item["source"]] = item["source_sha256"]
        report["hashes"][item["candidate"]] = item["candidate_sha256"]
    (args.out/"reconstruction.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({config:groups["all"]["fits"][config]["points"]["1024"] for config in CONFIGS},indent=2))


def plot(report,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    selected=(0,1,4,8)
    times=np.array([16,64,256,1024])
    for config,color,label in (("attention_uniform","#167aaa","Attention copier"),
                               ("empirical_uniform","#d79340","Frequency copier")):
        points=report["groups"]["all"]["fits"][config]["points"]
        for key,style in (("explained_squared_error","-"),("heldout_explained_squared_error","--")):
            values=np.array([points[str(t)][key]["mean"] for t in times])
            errors=np.array([points[str(t)][key]["se"] for t in times])
            axes[0].errorbar(times,100*values,yerr=196*errors,fmt=style+"o",color=color,lw=1.7,capsize=3,
                             label=label+(": fitted" if style=="-" else ": held out"))
    axes[0].set(xscale="log",xlabel="Time t (number of observed tokens)",ylabel="Squared probability error explained (%)",
                title="Reconstruction across the full source prior",xticks=times,
                xticklabels=[f"{t}\nN={n:,}" for t,n in zip(times,(264,638,1003,1024))])
    axes[0].legend(fontsize=8,loc="lower left")
    x=np.arange(len(selected))
    for j,(config,key,label,color) in enumerate((("attention_uniform","baseline_tv","Uniform alone","#999999"),
                                               ("attention_uniform","tv","Attention copier mixture","#167aaa"),
                                               ("empirical_uniform","tv","Frequency copier mixture","#d79340"))):
        stats=[report["groups"][f"depth_{d}"]["fits"][config]["points"]["1024"][key] for d in selected]
        axes[1].bar(x+(j-1)*.25,[100*s["mean"] for s in stats],width=.25,label=label,color=color,
                    yerr=[196*s["se"] for s in stats],capsize=2)
    axes[1].set(xticks=x,xticklabels=[f"L={d}" for d in selected],ylabel="Total variation error (%)",
                title="Full prediction error at t=1,024")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    fig.savefig(out/"reconstruction.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

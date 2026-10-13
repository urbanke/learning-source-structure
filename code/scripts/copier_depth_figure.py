"""Plot source-depth comparisons from the frozen, independently audited summary.

This consumer performs no inference or experiments. It checks the complete
cohort, source identities, summary dimensions, every curated TSV row, and
the existing audit/artifact hashes before drawing any curves. --check-only
(also --smoke) verifies the saved receipt and new figure hashes without
regenerating them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (112, 101, 127, 111, 115, 96, 118, 125, 119)
SELECTED_DEPTHS = (0, 1, 4, 8)
MAIN_PREDICTORS = ("intact", "bayes")
GRID_PREDICTORS = ("intact", "bayes", "copier_deleted")
SCHEMA = "natural-copier-depth-figures-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def verify_file(record):
    if sha(resolve(record["path"])) != record["sha256"]:
        raise ValueError(f"Input or audited artifact changed: {record['path']}")


def check_statistic(value, n, shape):
    if value["n"] != n:
        raise ValueError("A statistic has the wrong independent-sequence count")
    for key in ("mean", "se"):
        array = np.asarray(value[key], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid statistic shape or nonfinite value")
        if key == "se" and np.any(array < 0):
            raise ValueError("Negative standard error")
    if not math.isclose(value["ci95_multiplier"], student_t.ppf(.975, n-1), abs_tol=1e-12):
        raise ValueError("Unexpected confidence-interval multiplier")


def table_entries(report):
    """Reconstruct the complete curated table from independent JSON fields."""
    entries = {}
    points = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    for group, values in report["groups"].items():
        for key, stat in values["curves"].items():
            for point in points:
                entries[(group,key,"prefix",point,point)] = (
                    stat["mean"][point-1],stat["se"][point-1],stat["n"],stat["ci95_multiplier"])
        for scope, field, first, last in (("whole_sequence_mean","time_averages",1,1024),
                                          ("final_prefix","final_prefix",1024,1024)):
            for key, stat in values[field].items():
                entries[(group,key,scope,first,last)] = (stat["mean"],stat["se"],stat["n"],stat["ci95_multiplier"])
        for window in values["windows"]:
            for key, stat in window["statistics"].items():
                entries[(group,key,"sequence_window_mean",window["first_prefix"],window["last_prefix"])] = (
                    stat["mean"],stat["se"],stat["n"],stat["ci95_multiplier"])
    return entries


def verify_table(report, path):
    expected = table_entries(report)
    seen = set()
    maximum_error = 0.0
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = (row["group"], row["quantity"], row["scope"], int(row["first_prefix"]), int(row["last_prefix"]))
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicated TSV row")
            seen.add(key)
            mean, se, n, multiplier = expected[key]
            wanted = np.array([mean, se, mean-multiplier*se, mean+multiplier*se])
            actual = np.array([float(row[field]) for field in ("mean","sequence_standard_error","ci95_lower","ci95_upper")])
            if not np.isfinite(actual).all() or int(row["n_sequences"]) != n:
                raise ValueError("Invalid TSV statistic")
            error = float(np.max(np.abs(wanted-actual)))
            maximum_error = max(maximum_error,error)
            if error > 1e-12:
                raise ValueError("TSV disagrees with the audited JSON summary")
    if seen != set(expected):
        raise ValueError("The curated table is incomplete")
    return dict(rows=len(seen),maximum_absolute_error=maximum_error)


def inspect(summary_path, table_path):
    report = read_json(summary_path)
    if any(report.get(key) != value for key,value in dict(schema="natural-copier-analysis-v1",passed=True,
           smoke=False,publication_cohort=True,n_sequences=1024).items()):
        raise ValueError("Require the complete, audited production cohort")
    spec = report["spec"]
    expected = dict(schema="natural-copier-sequence-v1",master_seed=20261004,d=64,n_seq=1024,n_ctx=1024,
                    alpha=[0.,0.],source_depths=list(range(9)),source_relabelled=True,
                    conditions=["intact","copier_deleted","control_deleted","bayes"],
                    candidate_head=[0,1],control_head=[0,0],
                    bayes_prior="uniform shared depth 0..8, uniform base; source parameters withheld")
    if any(spec.get(key) != value for key,value in expected.items()) or report["spec_id"] != fingerprint(spec):
        raise ValueError("Wrong source, trained model intervention, or Bayesian prior")
    if report["prefix_lengths"] != list(range(1,1025)):
        raise ValueError("Incomplete or reordered prediction times")
    if report["source_depth_counts"] != {str(i):n for i,n in enumerate(COUNTS)}:
        raise ValueError("Unexpected source-depth counts")
    groups = report["groups"]
    if set(groups) != {"all","depth_positive",*(f"depth_{i}" for i in range(9))}:
        raise ValueError("Missing or unexpected source group")
    for group,values in groups.items():
        n = 1024 if group == "all" else 912 if group == "depth_positive" else COUNTS[int(group[6:])]
        if values["n_sequences"] != n:
            raise ValueError("Wrong source-group denominator")
        for stat in values["curves"].values():
            check_statistic(stat,n,(1024,))
        for field in ("time_averages","final_prefix"):
            for stat in values[field].values():
                check_statistic(stat,n,())
        for window in values["windows"]:
            for stat in window["statistics"].values():
                check_statistic(stat,n,())
        for name in GRID_PREDICTORS:
            if abs(values["curves"]["excess_"+name]["mean"][0]) > 1e-12:
                raise ValueError("Successor excess must be zero before any completed transition")
    for name in GRID_PREDICTORS:
        rebuilt = sum(COUNTS[i] * np.asarray(groups[f"depth_{i}"]["curves"]["excess_"+name]["mean"])
                      for i in range(9))/1024
        if not np.allclose(rebuilt,groups["all"]["curves"]["excess_"+name]["mean"],rtol=0,atol=1e-12):
            raise ValueError("Depth means do not reproduce the source-weighted overall mean")
    inputs = report["inputs"]
    if len(inputs) != 1024 or sorted(record["index"] for record in inputs) != list(range(1024)):
        raise ValueError("Incomplete source identities")
    if tuple(sum(record["source_depth"] == i for record in inputs) for i in range(9)) != COUNTS:
        raise ValueError("Source identities disagree with group counts")
    for record in (report["script"],report["protocol"],report["numerical_validation"],*report["artifacts"].values()):
        verify_file(record)
    if Path(resolve(report["artifacts"]["summary.tsv"]["path"])).resolve() != Path(table_path).resolve():
        raise ValueError("The table is not the audited summary's table")
    for record in spec["provenance"].values():
        verify_file(record)
    checked = verify_table(report,table_path)
    return report,dict(passed=True,summary=dict(path=relative(summary_path),sha256=sha(summary_path)),
                      table=dict(path=relative(table_path),sha256=sha(table_path)),table_check=checked,
                      audited_producer=report["script"],spec_id=report["spec_id"],n_sequences=1024,
                      source_depth_counts=report["source_depth_counts"],prefix_count=1024,
                      numerical_validation=report["numerical_validation"])


def draw_curve(ax,positions,stat,color,style,linewidth=1.1,alpha=.065):
    means = 100*np.asarray(stat["mean"])
    radius = 100*stat["ci95_multiplier"]*np.asarray(stat["se"])
    ax.fill_between(positions,means-radius,means+radius,color=color,alpha=alpha,linewidth=0,zorder=1)
    ax.plot(positions,means,color=color,ls=style,lw=linewidth,zorder=3)


def style_axis(ax, *, grid=False):
    from matplotlib.ticker import FixedLocator,ScalarFormatter
    ax.set_xscale("log",base=2)
    ax.set_xlim(1,1024)
    ax.xaxis.set_major_locator(FixedLocator((1,16,1024) if grid else (1,4,16,64,256,1024)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="both",labelsize=6.5 if grid else 8)
    ax.grid(axis="y",color="#E6E9ED",lw=.5)
    ax.axhline(0,color="#959CA3",lw=.55)
    ax.spines[["top","right"]].set_visible(False)


def figures(report,out):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"output/copier_time/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"axes.linewidth":.65})
    positions=np.asarray(report["prefix_lengths"])
    colors={0:"#7B3294",1:"#0072B2",4:"#D55E00",8:"#009E73"}
    dashed=(0,(4.2,2.7))
    outputs={}
    def save(fig,name):
        for extension in ("pdf","png"):
            path=out/f"{name}.{extension}"
            fig.savefig(path,dpi=220,metadata={"CreationDate":None,"ModDate":None} if extension=="pdf" else None)
            outputs[path.name]=dict(path=relative(path),sha256=sha(path))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.65))
    for depth in SELECTED_DEPTHS:
        curves=report["groups"][f"depth_{depth}"]["curves"]
        for name in MAIN_PREDICTORS:
            draw_curve(ax,positions,curves["excess_"+name],colors[depth],"-" if name=="intact" else dashed)
    style_axis(ax)
    ax.set_ylim(-1,80)
    ax.set_yticks((0,20,40,60,80))
    ax.set_xlabel("Tokens observed in the sequence",fontsize=8.5)
    ax.set_ylabel("Successor probability above uniform\n(percentage points)",fontsize=8.5)
    fig.legend(handles=[Line2D([],[],color=colors[d],lw=2,label=f"$L_{{\\mathrm{{s}}}}={d}$ ($N={COUNTS[d]}$)") for d in SELECTED_DEPTHS],
               loc="upper center",bbox_to_anchor=(.54,.995),ncol=4,frameon=False,fontsize=7.6,
               handlelength=1.35,columnspacing=1.0,handletextpad=.45)
    fig.legend(handles=[Line2D([],[],color="#343A40",lw=1.25,ls=style,label=label)
                        for style,label in (("-","Transformer"),(dashed,"Informed Bayes"))],
               loc="upper center",bbox_to_anchor=(.54,.921),ncol=2,frameon=False,fontsize=8,
               handlelength=2.8,columnspacing=2)
    fig.subplots_adjust(left=.14,right=.96,bottom=.155,top=.795)
    save(fig,"by_depth")
    fig,axes=plt.subplots(3,3,figsize=(5.5,6.7))
    predictor_colors={"intact":"#0072B2","bayes":"#D55E00","copier_deleted":"#44856C"}
    for depth,ax in enumerate(axes.flat):
        curves=report["groups"][f"depth_{depth}"]["curves"]
        maximum=0.
        for name in GRID_PREDICTORS:
            stat=curves["excess_"+name]
            draw_curve(ax,positions,stat,predictor_colors[name],dashed if name=="bayes" else "-",linewidth=.8,alpha=.09)
            maximum=max(maximum,float(np.max(100*(np.asarray(stat["mean"])+stat["ci95_multiplier"]*np.asarray(stat["se"])))))
        style_axis(ax,grid=True)
        ax.set_title(f"$L_{{\\mathrm{{s}}}}={depth}$, $N={COUNTS[depth]}$",fontsize=8,pad=5)
        upper=max(15,10*math.ceil(maximum/10))
        ax.set_ylim(-.5,upper)
        ax.set_yticks((0,upper/2,upper))
    fig.legend(handles=[Line2D([],[],color=predictor_colors[name],lw=1.3,
                      ls=dashed if name=="bayes" else "-",label=label) for name,label in
                      (("intact","Transformer"),("bayes","Informed Bayes"),("copier_deleted","Copying head removed"))],
               loc="upper center",bbox_to_anchor=(.54,.995),ncol=3,frameon=False,fontsize=7.5,
               handlelength=2.1,columnspacing=1)
    fig.supxlabel("Tokens observed in the sequence",fontsize=8.5,y=.035)
    fig.supylabel("Successor probability above uniform (percentage points)",fontsize=8.5,x=.02)
    fig.subplots_adjust(left=.125,right=.96,bottom=.09,top=.916,wspace=.33,hspace=.42)
    save(fig,"all_depths")
    return outputs


def receipt(checked,artifacts):
    return dict(schema=SCHEMA,passed=True,inputs=checked,
                script=dict(path=relative(__file__),sha256=sha(__file__)),artifacts=artifacts,
                main_depths=list(SELECTED_DEPTHS),main_predictors=list(MAIN_PREDICTORS),
                appendix_depths=list(range(9)),appendix_predictors=list(GRID_PREDICTORS),
                observable="Probability of previously observed successors of the current token minus the uniform probability of that same set; values are multiplied by 100 to give percentage points.",
                uncertainty="Pointwise 95% Student-t confidence intervals, using source-level standard errors from the independently audited complete cohort.",
                conditioning="Realized source depth only selects reported sequences; Bayes retains its uniform shared-depth prior 0..8 in every panel.",
                averaging="All histories in each depth group, including empty successor sets. No probability repair, smoothing, selected threshold, or new prediction.",
                main_encoding="Color gives source depth; solid gives transformer and dashed gives informed Bayes. Exactly eight mean curves on one axis.",
                appendix_encoding="All nine depth groups; blue transformer, dashed orange Bayes, green copier-deleted. Vertical scales vary by panel.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"results/transformer/copier_time/summary.json")
    parser.add_argument("--table",type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"results/transformer/copier_time")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="Alias for --check-only; no partial cohorts are accepted")
    args=parser.parse_args(argv)
    report,checked=inspect(args.input,args.table or args.input.with_suffix(".tsv"))
    target=args.out/"by_depth.json"
    if args.check_only or args.smoke:
        saved=read_json(target)
        if set(saved.get("artifacts",{})) != {f"{name}.{extension}" for name in ("by_depth","all_depths") for extension in ("pdf","png")}:
            raise ValueError("Incomplete saved figures")
        if saved != receipt(checked,saved["artifacts"]):
            raise ValueError("Saved figure receipt is stale or differs from the checked cohort")
        for artifact in saved["artifacts"].values():
            verify_file(artifact)
    else:
        args.out.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(receipt(checked,figures(report,args.out)),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True,check_only=args.check_only or args.smoke,
                          n_sequences=1024,table_rows=checked["table_check"]["rows"],
                          table_maximum_error=checked["table_check"]["maximum_absolute_error"],
                          receipt=relative(target))))


if __name__ == "__main__":
    main()

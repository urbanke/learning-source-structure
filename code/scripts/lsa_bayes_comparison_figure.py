"""Compare three actual codes and show paired LSA-minus-Bayes gaps.

MPLCONFIGDIR=/tmp/la-mpl python code/scripts/lsa_bayes_comparison_figure.py
Smoke: python code/scripts/lsa_bayes_comparison_figure.py --out /tmp/lsa_bayes_comparison.pdf --summary /tmp/lsa_bayes_comparison.json

Reads existing audited measurements only. No formula curves, fitted values,
control-variate total costs or new simulations enter the figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"results/mechanism/optimal_context_figure_check.json"
AUDIT=ROOT/"results/mechanism/optimal_context_figure_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"figures/lsa_bayes_comparison.pdf")
    parser.add_argument("--summary",type=Path,default=ROOT/"results/mechanism/lsa_bayes_comparison_figure.json")
    args=parser.parse_args()
    source=json.loads(SOURCE.read_text())
    audit=json.loads(AUDIT.read_text())
    assert source["complete"] and audit["status"]=="passed"
    assert audit["source_sha256"]==sha(SOURCE)
    assert audit["raw_sha256"]==source["raw_sha256"]
    assert (source["d"],source["alpha"],source["L"],source["repetitions"])==(256,1.,2,16)
    assert source["se_ddof"]==1
    n=np.asarray(source["tokens"],dtype=float)
    costs=source["costs"]
    paired=source["paired_differences"]
    series=[("lsa_frequency_full_cost","lsa_frequency_minus_bayes","#2369a0","o","Symmetric LSA $F$"),
            ("lsa_context_full_cost","lsa_context_minus_bayes","#bd6023","s","Symmetric LSA $C$"),
            ("bayes_full_cost",None,"#323232","D","Bayes (knows $q,L$)")]
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"legend.fontsize":7,
                         "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.8))
    displayed={}
    for name,difference,color,marker,label in series:
        mean=np.asarray(costs[name]["mean"])/n
        se=np.asarray(costs[name]["SE"])/n
        assert np.all(np.isfinite(mean)) and np.all(se>=0)
        axes[0].plot(n,mean,color=color,marker=marker,ms=3,mfc="white",lw=1.3)
        displayed[name]=dict(mean_bits_per_token=mean.tolist(),SE_bits_per_token=se.tolist())
        if difference is not None:
            gap=np.asarray(paired[difference]["mean"])/n
            gap_se=np.asarray(paired[difference]["SE"])/n
            assert np.allclose(gap,mean-np.asarray(costs["bayes_full_cost"]["mean"])/n,
                               rtol=1e-11,atol=1e-12)
            axes[1].errorbar(n,gap,yerr=2*gap_se,color=color,marker=marker,ms=3,
                             mfc="white",lw=1.3,elinewidth=.75,capsize=2)
            displayed[difference]=dict(mean_bits_per_token=gap.tolist(),
                                       paired_SE_bits_per_token=gap_se.tolist())
    axes[0].set_title("Three actual codes")
    axes[0].set_ylabel("Cumulative cost / tokens\n(bits per token)")
    axes[1].set_title("Excess cost over Bayes")
    axes[1].set_ylabel("LSA cost minus Bayes cost\n(bits per token)")
    axes[1].set_ylim(0,.95)
    axes[1].axhline(0,color=".45",lw=.6)
    for ax in axes:
        ax.set_xscale("log",base=2)
        ax.set_xlim(1250,25500)
        ax.set_xticks([2000,4000,8000,16000],["2000","4000","8000","16000"])
        ax.set_xlabel("Sequence length $n$ (tokens)")
        ax.grid(alpha=.16,lw=.5)
    handles=[Line2D([],[],color=color,marker=marker,mfc="white",ms=3,lw=1.3,label=label)
             for _,_,color,marker,label in series]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1.015),
               ncol=3,frameon=False,columnspacing=1.5,handlelength=2)
    fig.text(.52,.877,r"$d=256,\ \alpha=1,\ L=2$; 16 shared paths; right bars: $\pm2$ paired SE",
             ha="center",va="center",fontsize=7)
    fig.subplots_adjust(left=.09,right=.985,bottom=.19,top=.75,wspace=.43)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    comparisons=[]
    for selected in [5663,22652]:
        j=source["tokens"].index(selected)
        record=dict(tokens=selected)
        for name,difference,_,_,_ in series:
            record[name]=dict(mean_bits_per_token=costs[name]["mean"][j]/selected,
                             SE_bits_per_token=costs[name]["SE"][j]/selected)
            if difference is not None:
                record[difference]=dict(mean_bits_per_token=paired[difference]["mean"][j]/selected,
                                        paired_SE_bits_per_token=paired[difference]["SE"][j]/selected)
        comparisons.append(record)
    summary=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=sha(SOURCE),
        audit=str(AUDIT.relative_to(ROOT)),audit_sha256=sha(AUDIT),
        d=256,alpha=1.,L=2,repetitions=16,source_coding_cutoff=source["source_coding_cutoff"],
        tokens=source["tokens"],displayed=displayed,comparisons_at_tokens=comparisons,
        qualifications=["Left panel uses means of the raw full costs on the same sixteen paths; no formula or control-variate total-cost curve is shown.",
            "Right panel uses within-path LSA-minus-Bayes differences; bars are two sample standard errors with ddof=1, including covariance.",
            "The symmetric LSA codes use the default depth mixture. The matched Bayes code knows q and the source depth L; its advantage includes that information.",
            "Lines join measured checkpoint means. No crossing fit, new simulation or claim of equal code costs is made."],
        figure_sha256=sha(args.out),script_sha256=sha(Path(__file__)))
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(figure=str(args.out),comparisons_at_tokens=comparisons),indent=2))


if __name__=="__main__":
    main()

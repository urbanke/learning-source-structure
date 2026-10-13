"""Figures for the Bayesian-only unigram/copier feasibility study."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/transformer/section4_copying_pilot_2026-09-17'
BLUE, ORANGE = '#1565a8', '#d77719'
TIMES = np.array([2**j for j in range(11)])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    return rows


def matrix(rows, depth, branch, key):
    selected = [r for r in rows if int(r['depth_max']) == depth and r['branch'] == branch and float(r['prior_copy']) == .25]
    indices = sorted({int(r['index']) for r in selected})
    require(len(indices) == 256, 'Expected 256 sequences per branch')
    lookup = {(int(r['index']),int(r['time'])):r for r in selected}
    require(len(lookup) == len(selected) == 256 * len(TIMES), 'Incomplete or duplicate records')
    return np.array([[float(lookup[i,int(t)][key]) for t in TIMES] for i in indices])


def statistics(rows, depth, prior, branch):
    gap = matrix(rows, depth, branch, 'evidence_gap_bits')
    weights = expit(np.log(prior/(1-prior)) + np.log(2)*gap)
    delta = matrix(rows, depth, branch, 'component_tv')
    seen = matrix(rows, depth, branch, 'current_row_count') > 0
    distinguishable = delta >= .05
    dominant = weights > .5
    eligible_dominant = dominant & distinguishable
    seen_dominant = eligible_dominant & seen
    def binomial(a):
        mean = a.mean(axis=0)
        return dict(mean=mean.tolist(), standard_error=np.sqrt(mean*(1-mean)/(len(a)-1)).tolist())
    def crossings(a):
        any_cross = a.any(axis=1)
        first = np.where(any_cross, TIMES[a.argmax(axis=1)], 0)
        return dict(first_tested_histogram={str(t):int((first==t).sum()) for t in TIMES},
            unreached=int((~any_cross).sum()), ever=int(any_cross.sum()),
            recrossed=int((np.diff(a.astype(int),axis=1)<0).any(axis=1).sum()))
    raw_first = dominant.argmax(axis=1)
    any_first = dominant.any(axis=1)
    seen_first = seen[np.arange(len(weights)), raw_first]
    return dict(depth_max=depth,prior_copy=prior,branch=branch,n=256,
        mean_weight=weights.mean(axis=0).tolist(),
        weight_quantiles={str(q):np.quantile(weights,q,axis=0).tolist() for q in (.1,.5,.9)},
        dominant=binomial(dominant), distinguishable_dominant=binomial(eligible_dominant),
        seen_distinguishable_dominant=binomial(seen_dominant),
        indistinguishable=binomial(~distinguishable), seen=binomial(seen),
        posterior_crossing=crossings(dominant),
        distinguishable_crossing=crossings(eligible_dominant),
        seen_distinguishable_crossing=crossings(seen_dominant),
        first_posterior_crossing_current_row_unseen=int((any_first & ~seen_first).sum()))


def plot(rows, out, selected_depth=4, selected_prior=.25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    settings = {'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
                'axes.labelsize':8.5,'axes.linewidth':.6,'pdf.fonttype':42}
    groups = [statistics(rows,d,p,b) for d in (4,8) for p in (.25,.125) for b in ('iid','copy')]
    def find(d,p,b):
        return next(g for g in groups if (g['depth_max'],g['prior_copy'],g['branch'])==(d,p,b))
    def axes_style(ax):
        ax.set_xscale('log',base=2)
        ax.set_xlim(1,1024)
        ax.set_xticks([1,4,16,64,256,1024],[1,4,16,64,256,1024])
        ax.set_ylim(-.025,1.025)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.13)
        ax.set_xlabel('Observed tokens')
        ax.minorticks_off()
    def draw(axes,d,p,intervals=True):
        for b,color in [('copy',BLUE),('iid',ORANGE)]:
            g=find(d,p,b)
            axes[0].plot(TIMES,g['mean_weight'],color=color,lw=1.5)
            if intervals:
                axes[0].fill_between(TIMES,g['weight_quantiles']['0.1'],g['weight_quantiles']['0.9'],color=color,alpha=.13,lw=0)
            for metric,ls in [('distinguishable_dominant','-'),('seen_distinguishable_dominant','--')]:
                vals=g[metric]
                axes[1].plot(TIMES,vals['mean'],color=color,ls=ls,lw=1.4)
        axes[0].axhline(.5,color='.5',lw=.65,ls=':')
        for ax in axes: axes_style(ax)
        axes[0].set_ylabel('Copier posterior weight')
        axes[1].set_ylabel('Fraction of sequences')
    artifacts=[]
    def save(fig,name):
        for ext in ('pdf','png'):
            filename=name+'.'+ext
            fig.savefig(out/filename,dpi=240)
            artifacts.append(filename)
        plt.close(fig)
    with plt.rc_context(settings):
        fig,axes=plt.subplots(1,2,figsize=(6.4,3.4))
        draw(axes,selected_depth,selected_prior)
        axes[0].set_title('(a) Weight assigned to the copier')
        axes[1].set_title('(b) Copier favored in prediction')
        handles=[Line2D([],[],color=BLUE,lw=1.5,label='Copier-generated sequences'),
                 Line2D([],[],color=ORANGE,lw=1.5,label='Independent sequences')]
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.07))
        fig.text(.5,.025,'Panel (b): solid = distinct predictions; dashed = also an observed continuation',ha='center',fontsize=7.5)
        fig.subplots_adjust(left=.09,right=.965,top=.89,bottom=.32,wspace=.30)
        save(fig,'transition')
        fig,axes=plt.subplots(4,2,figsize=(6.4,9.1))
        for pair,(d,p) in zip(axes,[(4,.25),(4,.125),(8,.25),(8,.125)]):
            draw(pair,d,p)
            pair[0].set_title(f'Depths 1-{d}; copier prior {p:g}')
            pair[1].set_title('Distinct predictions / also seen context')
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.006))
        fig.subplots_adjust(left=.09,right=.965,top=.965,bottom=.075,hspace=.66,wspace=.30)
        save(fig,'all_candidates')
    output=dict(schema='section4-copying-presentation-v1',times=TIMES.tolist(),
        selected_for_display=dict(depth_max=selected_depth,prior_copy=selected_prior),
        groups=groups,records_sha256=sha(out/'records.csv'))
    (out/'plot_data.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    receipt=dict(passed=True,inputs={str(out/'records.csv'):sha(out/'records.csv'),
        str(Path(__file__).resolve()):sha(__file__)},
        outputs={name:sha(out/name) for name in artifacts+['plot_data.json']})
    (out/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--depth-max',type=int,default=4)
    parser.add_argument('--prior-copy',type=float,default=.25)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        r=json.loads((args.out/'figure_receipt.json').read_text())
        for name,digest in r['inputs'].items(): require(sha(name)==digest,'Changed input '+name)
        for name,digest in r['outputs'].items(): require(sha(args.out/name)==digest,'Changed output '+name)
        old=json.loads((args.out/'plot_data.json').read_text())
        rows=load_records(args.out/'records.csv')
        groups=[statistics(rows,d,p,b) for d in (4,8) for p in (.25,.125) for b in ('iid','copy')]
        require(groups==old['groups'],'Statistics replay differs')
        print(json.dumps(dict(passed=True,groups=len(groups))))
    else:
        plot(load_records(args.out/'records.csv'),args.out,args.depth_max,args.prior_copy)
        print(json.dumps(dict(passed=True,figures=2)))

if __name__=='__main__': main()

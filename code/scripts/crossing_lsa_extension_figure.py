"""Plot the concentrated/deep LSA checks without repeating simulations.

MPLCONFIGDIR=/tmp/la-mpl python code/scripts/crossing_lsa_extension_figure.py
Smoke: same command with --out /tmp/lsa_extension.pdf --table /tmp/lsa_extension.tex --summary /tmp/lsa_extension_figure.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'figures/crossing_lsa_extension.pdf')
    ap.add_argument('--table',type=Path,default=ROOT/'results/mechanism/crossing_lsa_extension_table.tex')
    ap.add_argument('--summary',type=Path,default=ROOT/'results/mechanism/crossing_lsa_extension_figure.json')
    args=ap.parse_args()
    source=ROOT/'results/mechanism/crossing_lsa_extension.json'
    data=json.loads(source.read_text()); assert data['complete']
    alpha2_source=ROOT/'results/mechanism/crossing_lsa_alpha2.json'
    alpha2=json.loads(alpha2_source.read_text()); assert alpha2['complete']
    order=((6,1.),(6,1.5),(10,1.5))
    cells=[next(c for c in data['cells'] if (c['L'],c['alpha'],c['d'])==(L,a,512)) for L,a in order]
    assert all(c['repetitions']==256 for c in cells)
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(6.5,2.5))
    displayed=[]
    for i,(ax,c) in enumerate(zip(axs,cells)):
        n=np.array(c['tokens']); methods=c['methods']
        pred=np.array(methods['count']['gap']['mean'])/n
        pse=np.array(methods['count']['gap']['SE'])/n
        actual=np.array(methods['chain']['gap']['mean'])/n
        ase=np.array(methods['chain']['gap']['SE'])/n
        ax.plot(n,pred,color='#704a9e',lw=1.3)
        ax.fill_between(n,pred-2*pse,pred+2*pse,color='#704a9e',alpha=.14,lw=0)
        ax.errorbar(n,actual,yerr=2*ase,fmt='o',mfc='white',color='.15',ms=2.7,elinewidth=.65,capsize=1)
        ax.axhline(0,color='.3',lw=.7)
        ax.set_xscale('log'); ax.set_xlim(8,n[-1])
        ax.set_xlabel('Sequence length $n$')
        ax.text(.04,.94,f'({chr(97+i)})',transform=ax.transAxes,va='top')
        ax.grid(alpha=.15,lw=.5)
        displayed.append(dict(d=512,L=c['L'],alpha=c['alpha'],repetitions=256,
            tokens=n.tolist(),predicted_mean_gap_per_token=pred.tolist(),predicted_SE_per_token=pse.tolist(),
            measured_mean_gap_per_token=actual.tolist(),measured_SE_per_token=ase.tolist()))
    axs[0].set_ylabel('Frequency cost minus copying cost\n(bits per token)')
    fig.legend(handles=[Line2D([],[],color='#704a9e',label='Predicted difference'),
        Line2D([],[],color='.15',marker='o',mfc='white',ls='none',label='Measured mean difference')],
        loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.54,1.01))
    fig.subplots_adjust(left=.095,right=.992,bottom=.20,top=.82,wspace=.36)
    fig.savefig(args.out); plt.close(fig)
    lines=[r'\begin{tabular}{rrrrrrl}',r'\toprule',
        r'$d$ & $L_{\mathrm{s}}$ & $\alpha$ & Draws & Predicted & Measured & Relative difference\\',r'\midrule']
    for c in sorted(data['cells']+alpha2['cells'],key=lambda c:(c['d'],c['L'],c['alpha'])):
        p=c['methods']['count']['mean_curve_crossing']; a=c['methods']['chain']['mean_curve_crossing']
        if p['value'] is None or a['value'] is None:
            vals=r'--- & --- & Positive from first check'
        else:
            ci=c['bootstrap']['count_over_chain_minus_one']['conditional_on_located_quantiles']
            diff=100*(p['value']/a['value']-1)
            vals=f"{p['value']:.1f} & {a['value']:.1f} & ${diff:+.1f}\\%\\;[{100*ci[0]:+.1f},{100*ci[2]:+.1f}]$"
        lines.append(f"{c['d']} & {c['L']} & {c['alpha']:g} & {c['repetitions']} & "+vals+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    args.table.write_text('\n'.join(lines)+'\n')
    args.summary.write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),displayed=displayed,
        additional_table_source=str(alpha2_source.relative_to(ROOT)),
        additional_table_source_sha256=hashlib.sha256(alpha2_source.read_bytes()).hexdigest(),
        qualification='Pointwise two-SE bars; relative crossing intervals are paired bootstrap intervals conditional on located roots.'),indent=2)+'\n')
    print('Wrote extension figure, table, and source summary.')


if __name__=='__main__': main()

"""Restyle appendix Figures 5 and 8 from the unchanged Figure 1 data receipt."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from appendix_figure_style import apply_style,style_axis,BLUE,DARK,LINEWIDTH,MARKERSIZE
import section3_compact_figure as source

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/transformer/appendix_style_2026-09-24'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def axis(ax,top):
    style_axis(ax)
    ax.set_xscale('log',base=2);ax.set_xlim(.87,1382)
    ticks=[1,4,16,64,256,1024]
    ax.set_xticks(ticks,[str(t) for t in ticks]);ax.set_ylim(0,top)
    ax.set_yticks([0,1,2,3])

def forecast(ax,value,color,style=None):
    t,y,se=[np.asarray(value[k]) for k in ('times','mean','se')]
    if style:
        ax.fill_between(t,np.maximum(0,y-se),y+se,color=color,alpha=.08,lw=0)
        ax.plot(t,y,color=color,ls=style,lw=LINEWIDTH)
    else:
        ax.errorbar(t,y,yerr=se,color=color,ls='none',marker='o',ms=MARKERSIZE,
                    mfc='white',mew=.75,elinewidth=.6,capsize=1.2,zorder=5)

def legend(fig):
    handles=[Line2D([],[],color=DARK,lw=LINEWIDTH,label='Two-layer transformer'),
        Line2D([],[],color=DARK,lw=LINEWIDTH,ls='--',label='Four-layer transformer'),
        Line2D([],[],color=DARK,ls='none',marker='o',mfc='white',ms=MARKERSIZE,label='Informed Bayes')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.995),
               ncol=3,frameon=False,columnspacing=1,handlelength=1.8,handletextpad=.5)

def save(fig,number):
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'figure{number}.{ext}',dpi=260,
                    metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)

def main():
    receipt_path=ROOT/'results/transformer/section3_compact/receipt.json'
    saved=json.loads(receipt_path.read_text());old_outputs=saved.pop('outputs')
    assert saved==source.collect(),'Underlying Figure 1 receipt changed'
    for name,digest in old_outputs.items():assert sha(receipt_path.parent/name)==digest,name
    OUT.mkdir(parents=True,exist_ok=True)
    with apply_style():
        fig,axes=plt.subplots(2,2,figsize=(5.5,4.5),sharex=True)
        fig.subplots_adjust(left=.12,right=.975,bottom=.13,top=.78,wspace=.26,hspace=.78)
        legend(fig)
        for row,ident in enumerate(('b','d')):
            panel=next(p for p in saved['panels'] if p['id']==ident)
            for run,ax in enumerate(axes[row]):
                axis(ax,3.9)
                for c in panel['cases']:
                    forecast(ax,c['original_transformer'],c['color'],'-')
                    forecast(ax,c['larger_runs'][run],c['color'],'--')
                    forecast(ax,c['bayes'],c['color'])
                title='Known product' if row==0 else 'Uniform base'
                ax.set_title(f'({chr(97+2*row+run)}) {title}, run {run+1}',loc='left',pad=29)
                hs=[Line2D([],[],color=c['color'],lw=LINEWIDTH) for c in panel['cases']]
                ax.legend(hs,[c['label'] for c in panel['cases']],loc='lower left',
                    bbox_to_anchor=(-.04,1.015),frameon=False,ncol=4,borderaxespad=0,
                    columnspacing=.55,handlelength=.7,handletextpad=.3)
        fig.supylabel('Expected next-token regret (bits)',x=.012,fontsize=8)
        fig.supxlabel('Observed tokens, $t$',y=.025,fontsize=8)
        save(fig,5)
        fig,axes=plt.subplots(1,2,figsize=(5.5,2.5),sharey=True)
        fig.subplots_adjust(left=.12,right=.975,bottom=.21,top=.76,wspace=.22)
        legend(fig);panel=saved['panels'][-1];assert panel['id']=='f'
        for run,ax in enumerate(axes):
            axis(ax,3)
            c=next(c for c in panel['cases'] if c['condition']['training_seed']==run)
            forecast(ax,c['original_transformer'],BLUE,'-')
            forecast(ax,c['transformer'],BLUE,'--')
            forecast(ax,panel['bayes'],BLUE)
            ax.set_title(f'({chr(97+run)}) Training run {run+1}',loc='left',pad=6)
        fig.supylabel('Expected next-token regret (bits)',x=.012,fontsize=8)
        fig.supxlabel('Observed tokens, $t$',y=.035,fontsize=8)
        save(fig,8)
    paths=[Path(__file__),ROOT/'code/scripts/appendix_figure_style.py',receipt_path]
    (OUT/'specialist_pooled_receipt.json').write_text(json.dumps(dict(
        schema='appendix-figure-presentation-v1',figures=[5,8],data_recomputed=False,
        estimates_and_uncertainties_unchanged=True,
        all_pooled_bayes_points_numerically_validated=saved['all_bayesian_points_numerically_validated'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        outputs={f'figure{n}.{ext}':sha(OUT/f'figure{n}.{ext}') for n in (5,8) for ext in ('pdf','png')}),indent=2)+'\n')
if __name__=='__main__':main()

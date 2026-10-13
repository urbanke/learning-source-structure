"""Plot saved hidden-factor evaluation summaries; never recompute predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STYLES={
    'seed0':('#2468A0','-','Transformer, seed 0'),
    'seed1':('#288167','-','Transformer, seed 1'),
    'bayes_hidden':('#BF611C','--','Bayes, encoding hidden'),
    'bayes_encoding_given':('#8A4E98',':','Bayes, encoding given'),
    'flat_dirichlet':('#666666','-.','Unstructured Bayes'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_scope(summary):
    """Distinguish a completed prefix from completion of the frozen test cohort."""
    full=next(g for g in summary['groups'] if g['id']=='full_prior')
    planned=int(summary.get('planned_n_sequences',summary['cohort']['identity']['n_seq']))
    evaluated=int(summary.get('evaluated_n_sequences',full['n_seq']))
    partial=summary.get('partial_evaluation',evaluated<planned)
    if (not 2<=evaluated<=planned or evaluated!=full['n_seq']
            or planned!=summary['cohort']['identity']['n_seq']
            or partial!=(evaluated<planned)
            or summary.get('evaluation_complete',not partial)==partial
            or (partial and summary['two_repeat_comparison_complete'])):
        raise ValueError('inconsistent evaluation completion or sequence counts')
    label=(f'Preliminary: first {evaluated:,} of {planned:,} test sequences'
           if partial else f'{evaluated:,} independent test sequences')
    if not summary['production_size']:
        label+=' · Pipeline smoke test — not scientific evidence'
    return dict(partial_evaluation=partial,evaluated_n_sequences=evaluated,
        planned_n_sequences=planned,evaluation_complete=not partial,label=label)


def scope_label(fig,scope,x=.52):
    fig.text(x,.008,scope['label'],ha='center',
        color='#8D3434' if scope['partial_evaluation'] or 'smoke test' in scope['label'] else '#444444',
        fontsize=7.5)


def draw(ax,times,values,name):
    color,style,label=STYLES[name]
    mean,se=np.asarray(values['mean']),np.asarray(values['se'])
    if (mean.shape!=times.shape or se.shape!=times.shape or not np.isfinite(mean).all()
            or not np.isfinite(se).all() or np.any(se<0)):
        raise ValueError('invalid saved figure statistics')
    ax.plot(times,mean,color=color,linestyle=style,lw=1.3,label=label)
    ax.fill_between(times,mean-se,mean+se,color=color,alpha=.12,linewidth=0)


def decorate(ax,times):
    ax.set_xscale('log',base=2)
    ticks=[int(t) for t in times if t in [1,16,1024]]
    if int(times[-1]) not in ticks:ticks.append(int(times[-1]))
    ax.set_xticks(ticks,[str(t) for t in ticks]);ax.minorticks_off()
    ax.set_xlim(.85,times[-1]*1.2)
    ax.grid(axis='y',color='#D8DEE5',lw=.5)
    ax.tick_params(labelsize=8,length=3)


def render(summary,out):
    times=np.asarray(summary['times']);full=next(g for g in summary['groups'] if g['id']=='full_prior')
    scope=evaluation_scope(summary)
    available=[name for name in STYLES if name in full['predictors']]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(8.0,2.7))
        fig.subplots_adjust(left=.07,right=.985,bottom=.20,top=.70,wspace=.43)
        for name in available:
            draw(axes[0],times,full['predictors'][name]['next_kl_bits'],name)
        draw(axes[1],times,full['paired']['bayes_hidden_minus_bayes_encoding_given']['cumulative_loss_bits'],
             'bayes_hidden')
        axes[0].set_title('(a) Prediction regret',loc='left')
        axes[1].set_title('(b) Cost of hidden encoding',loc='left')
        axes[2].set_title('(c) Excess over hidden Bayes',loc='left')
        axes[0].set_ylabel('Bits per prediction');axes[1].set_ylabel('Cumulative bits')
        axes[2].set_ylabel('Cumulative bits')
        for name in ['seed0','seed1']:
            pair=f'{name}_minus_bayes_hidden'
            if pair in full['paired']:
                draw(axes[2],times,full['paired'][pair]['cumulative_loss_bits'],name)
        if not axes[2].lines:
            axes[2].text(.5,.5,'Awaiting trained models',ha='center',va='center',
                transform=axes[2].transAxes,color='#666666')
            axes[2].set_yticks([])
        for ax in axes:decorate(ax,times);ax.set_xlabel('Observed tokens, $t$')
        axes[1].axhline(0,color='#777777',lw=.6)
        axes[2].axhline(0,color='#777777',lw=.6)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=7.5)
        scope_label(fig,scope)
        for ext in ['pdf','png']:fig.savefig(out/f'uncertainty_cost.{ext}',dpi=240,metadata={'Creator':'hidden_factor_figures.py'})
        plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(5.5,4.0))
        fig.subplots_adjust(left=.105,right=.972,bottom=.14,top=.76,wspace=.28,hspace=.48)
        for ax,group in zip(axes.flat,[g for g in summary['groups'] if g['id']!='full_prior']):
            if group['status']!='complete':
                ax.set_axis_off();ax.text(.1,.5,'Insufficient sequences');continue
            for name in ['seed0','seed1','bayes_hidden','bayes_encoding_given']:
                if name in group['predictors']:draw(ax,times,group['predictors'][name]['next_kl_bits'],name)
            ax.set_title(f"{group['entropy_label']} entropy ($N={group['n_seq']}$)",loc='left')
            decorate(ax,times)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=7.5)
        fig.text(.54,.845,'Coordinate 1 / coordinate 2: low or high mean row entropy',ha='center',fontsize=8)
        fig.supylabel('Expected next-token regret (bits)',x=.008,fontsize=8.5)
        fig.supxlabel('Observed tokens, $t$',y=.052,fontsize=8.5)
        scope_label(fig,scope,x=.54)
        for ext in ['pdf','png']:fig.savefig(out/f'conditional_curves.{ext}',dpi=240,metadata={'Creator':'hidden_factor_figures.py'})
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();summary=json.loads(args.summary.read_text())
    if summary['schema']!='hidden-factor-evaluation-v1':raise ValueError('wrong evaluation summary')
    if not summary['production_size'] and not args.allow_smoke:
        raise ValueError('smoke results require explicit --allow-smoke and are marked as nonscientific')
    scope=evaluation_scope(summary)
    receipt=dict(schema='hidden-factor-figures-v1',summary_sha256=sha(args.summary),script_sha256=sha(__file__),
        production_size=summary['production_size'],two_repeat_comparison_complete=summary['two_repeat_comparison_complete'],
        partial_evaluation=scope['partial_evaluation'],evaluated_n_sequences=scope['evaluated_n_sequences'],
        planned_n_sequences=scope['planned_n_sequences'],evaluation_complete=scope['evaluation_complete'],
        bayes_only=summary['bayes_only'],new_computation=False,
        conditional_rule=summary['cohort']['identity']['stratum_rule'],uncertainty=summary['uncertainty'])
    outputs=[f'{name}.{ext}' for name in ['uncertainty_cost','conditional_curves'] for ext in ['pdf','png']]
    if args.check_only:
        saved=json.loads((args.out/'receipt.json').read_text());hashes=saved.pop('outputs')
        if saved!=receipt or hashes!={name:sha(args.out/name) for name in outputs}:raise ValueError('figure receipt differs')
        print('Saved summary and all figure artifacts verified.');return
    args.out.mkdir(parents=True,exist_ok=True);render(summary,args.out)
    receipt['outputs']={name:sha(args.out/name) for name in outputs}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(args.out)


if __name__=='__main__':main()

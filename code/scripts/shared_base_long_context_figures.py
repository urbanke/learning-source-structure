"""Plot the eight fixed shared-base pilot sequences, without population pooling.

Short-context points reuse the same individual source indices in the validated
capacity release. The new Bayesian points are taken from the independently
checked long-context release. Every panel is one original sequence, not a mean.
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
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/transformer/shared_base_long_context_2026-09-17'
CAPACITY=ROOT/'results/transformer/shared_base_scaling_2026-09-17'
PILOT=ROOT/'code/transformer/configs/shared_base_posterior_plan_2026-09-16.json'
DEPTHS=[0,1,4,8]
TIMES=[1,4,16,64,256,1024]
COLORS={'original':'#2468A0','larger':'#288167','bayes':'#222222'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(source):
    summary_path=Path(source)/'summary.json'
    data=read(summary_path)
    records=data['records']
    require(len(records)==16 and {(r['depth'],r['index'],r['time']) for r in records}==
            {(d,i,t) for d in DEPTHS for i in (0,1) for t in (256,1024)},'Pilot records are missing, repeated or selected')
    require(data.get('passed',data.get('all_numerical_checks_passed',False)) is True,
            'Long-context release has no passing numerical audit')
    capacity=read(CAPACITY/'comparison.json');audit=read(CAPACITY/'validation.json');pilot=read(PILOT)
    require(audit['passed'] is True and audit['summary_sha256']==sha(CAPACITY/'comparison.json'),
            'Capacity scores no longer match their independent audit')
    require(capacity['times']==TIMES and capacity['seed_order']==[0,1],'Changed capacity time/seed convention')
    identities={(int(s['setting'].split('_L')[1]),s['index']):s for s in pilot['source_sequences']}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [summary_path,CAPACITY/'comparison.json',CAPACITY/'validation.json',PILOT]}
    panels=[]
    for depth in DEPTHS:
        path=CAPACITY/f'paired_scores_L{depth}.npz'
        receipt=next(p for p in capacity['provenance'] if p['depth']==depth)
        require(sha(path)==receipt['scores_sha256'],'Per-source capacity scores changed')
        inputs[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path,allow_pickle=False) as a:
            require(np.array_equal(a['source_indices'],np.arange(256)) and np.array_equal(a['times'],TIMES)
                    and np.array_equal(a['bayes_times'],TIMES[:4]),'Changed source/time indices')
            for index in (0,1):
                matching=sorted([r for r in records if r['depth']==depth and r['index']==index],key=lambda r:r['time'])
                require(all(r['source_id']==identities[depth,index]['source_id'] for r in matching),
                        'Bayesian long-context source differs from the frozen pilot')
                bayes=list(map(float,a['bayes_regret_bits'][index]))+[float(r['bayes']['regret_bits']) for r in matching]
                errors=[None]*4+[float(r['bayes']['mcse_bits']) for r in matching]
                require(np.isfinite(bayes).all() and min(bayes)>=0 and all(e>=0 and np.isfinite(e) for e in errors[4:]),
                        'Invalid Bayesian regrets or numerical errors')
                for seed in (0,1):
                    values={}
                    for role,array_name in (('original','baseline_regret_bits'),('larger','larger_regret_bits')):
                        values[role]=a[array_name][index,:,seed].tolist()
                        require(np.isfinite(values[role]).all() and min(values[role])>=0,'Invalid transformer regret')
                        for row,j in zip(matching,(4,5)):
                            require(abs(values[role][j]-row['predictors'][f'{role}_seed{seed}']['regret_bits'])<1e-12,
                                    'Long-context transformer record is not paired to the same source score')
                    panels.append(dict(depth=depth,index=index,source_id=identities[depth,index]['source_id'],
                                       training_seed=seed,times=TIMES,bayes=bayes,bayes_mcse=errors,**values))
    return inputs,panels


def render(panels,out):
    # A fixed scale per depth, shared by both source draws and both training runs.
    maxima={d:max(max(p[k]) for p in panels if p['depth']==d for k in ('original','larger','bayes'))
            for d in DEPTHS}
    limits={d:float(MaxNLocator(nbins=4).tick_values(0,1.08*v)[-1]) for d,v in maxima.items()}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8.,'axes.titlesize':8.,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
                         'pdf.fonttype':42,'ps.fonttype':42}):
        for seed in (0,1):
            fig,axes=plt.subplots(4,2,figsize=(5.5,7.),sharex=True,sharey='row')
            fig.subplots_adjust(left=.105,right=.985,bottom=.09,top=.905,wspace=.20,hspace=.36)
            for row,depth in enumerate(DEPTHS):
                for index in (0,1):
                    ax=axes[row,index]
                    p=next(p for p in panels if p['depth']==depth and p['index']==index and p['training_seed']==seed)
                    for role in ('original','larger'):
                        ax.plot(TIMES,p[role],color=COLORS[role],lw=1.05,marker='o',ms=2.2,
                                markeredgewidth=0,zorder=3)
                    ax.plot(TIMES,p['bayes'],color=COLORS['bayes'],lw=.85,ls='--',marker='o',ms=3.7,
                            mfc='white',mew=.75,zorder=5)
                    ax.errorbar(TIMES[-2:],p['bayes'][-2:],yerr=p['bayes_mcse'][-2:],fmt='none',
                                ecolor=COLORS['bayes'],elinewidth=.7,capsize=1.6,zorder=6)
                    ax.set_xscale('log',base=2);ax.set_xlim(.72,1430);ax.set_ylim(0,limits[depth])
                    ax.set_xticks(TIMES,[str(t) for t in TIMES]);ax.minorticks_off()
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(labelsize=7.,length=2.5,pad=2)
                    ax.grid(axis='y',color='#D8DEE5',lw=.45);ax.set_axisbelow(True)
                    ax.set_title(f'$L={depth}$, source draw {index+1}',loc='left',pad=5)
            handles=[Line2D([],[],color=COLORS[r],lw=1.2,marker='o',ms=2.5) for r in ('original','larger')]
            handles.append(Line2D([],[],color=COLORS['bayes'],lw=.9,ls='--',marker='o',mfc='white',ms=3.7))
            fig.legend(handles,['Original model','Larger model','Informed Bayes'],ncol=3,frameon=False,
                       loc='upper center',bbox_to_anchor=(.535,.97),fontsize=8.,handlelength=2.,columnspacing=1.1)
            fig.text(.535,.99,f'Shared-base LSA: eight individual test sequences, training run {seed+1}',
                     ha='center',va='top',fontsize=8.5)
            fig.supylabel('Expected next-token regret (bits)',x=.008,fontsize=8.)
            fig.supxlabel('Observed tokens, $t$',y=.040,fontsize=8.)
            fig.text(.535,.009,'Individual sequences; Bayes bars at 256/1024 show ±1 numerical MCSE.',
                     ha='center',va='bottom',fontsize=7.)
            from matplotlib.text import Text
            require(all(t.get_fontsize()>=7 for t in fig.findobj(Text) if t.get_text()),'Text smaller than seven points')
            fig.savefig(out/f'individual_sequences_seed{seed}.pdf',metadata={'CreationDate':None,'ModDate':None})
            fig.savefig(out/f'individual_sequences_seed{seed}.png',dpi=260)
            plt.close(fig)
    return limits


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=DEFAULT);ap.add_argument('--out',type=Path)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();out=args.out or args.source/'figures'
    inputs,panels=prepare(args.source)
    receipt=dict(schema='shared-base-long-context-figures-v1',passed=True,script_sha256=sha(__file__),
                 inputs=inputs,panels=panels,n_distinct_sequences=8,n_sequences_per_depth=2,
                 tested_times=TIMES,primary_seed=0,training_seeds_displayed_separately=[0,1],
                 display=dict(size_inches=[5.5,7.],minimum_font_points=7,rows_depths=DEPTHS,
                              columns_source_indices=[0,1],same_depth_scale_across_draws_and_training_runs=True),
                 averaging=False,smoothing=False,interpolation='Thin lines only connect the six evaluated points.',
                 uncertainty='At256/1024, Bayes bars are one numerical MCSE; no population standard errors. Early Bayesian points reuse individually validated references.',
                 colors=COLORS,new_inference=False,new_bayesian_computation=False)
    names=[f'individual_sequences_seed{seed}.{ext}' for seed in (0,1) for ext in ('pdf','png')]
    if args.check_only:
        saved=read(out/'receipt.json');saved.pop('y_max_by_depth');outputs=saved.pop('outputs')
        require(saved==receipt and outputs=={name:sha(out/name) for name in names},'Figure provenance or files changed')
        print('Verified all16 panels, unchanged source identities, individual regret values and figure hashes.')
        return
    out.mkdir(parents=True,exist_ok=True)
    limits=render(panels,out);receipt['y_max_by_depth']={str(k):v for k,v in limits.items()}
    receipt['outputs']={name:sha(out/name) for name in names}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(figures=names,directory=str(out),distinct_sequences=8)))


if __name__=='__main__':main()

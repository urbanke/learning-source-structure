"""Audit fresh product context forecasts and draw the Section 4 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/transformer/product_context_confirmation_2026-09-17'
COLORS = ['#2468A0','#BF611C','#288167','#8A4E98']
MARKERS = ['o','s','^','D']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def audit(out):
    s=read(out/'summary.json'); plan=s['plan']; base=Path(plan['out'])
    require(s['complete'] and s['confirmation'] and s['plan_id']==plan['plan_id'],
            'Incomplete confirmation')
    require(plan['n_sequences_per_pair']==1024 and plan['targets_bits']==[.1,.2,.5]
            and plan['depth_pairs']==[[1,1],[1,8],[4,4],[8,8]],'Changed scientific scope')
    for path,digest in {**plan['inputs'],**s['inputs']}.items():
        require(sha(path)==digest,'Changed bound input: '+path)
    forecast=read(base/'forecast.json'); seal=read(base/'forecast_seal.json')
    started=read(base/'model_started.json')
    require(sha(base/'forecast.json')==seal['sha256']==s['forecast_sha256']==started['forecast_sha256'],
            'Forecast seal differs')
    require(forecast['sealed_utc'] <= seal['created_utc'] <= started['started_utc'],
            'Forecast was not sealed before model evaluation')
    require(forecast['numerical_validation']['passed'] and forecast['complete'],
            'Numerical validation failed')
    for path,digest in forecast['inputs'].items():
        require(sha(path)==digest,'Changed forecast source/reference: '+path)
    max_error=0.; total=0
    for row in s['curves']:
        folder=Path(row['raw_directory']); digest=hashlib.sha256(); risks=[]
        require(row['n_sequences']==1024,'Incomplete condition')
        for part in ('data','references','model'):
            require(sorted(p.name for p in (folder/part).glob('seq_*.npz'))==
                    [f'seq_{i:06d}.npz' for i in range(1024)],'Missing or extra sequence')
        for i in range(1024):
            records={}
            for part in ('data','references','model'):
                path=folder/part/f'seq_{i:06d}.npz'
                digest.update(f'{part}/{path.name}:{sha(path)}'.encode())
                with np.load(path,allow_pickle=False) as a:records[part]={k:a[k] for k in a.files}
            data,ref,net=(records[k] for k in ('data','references','model'))
            for item in records.values():require(np.array_equal(item['prefixes'],plan['times']),'Times differ')
            for item in (ref,net):
                require(read_metadata(item)['tokens_sha256']==str(data['tokens_sha256']),
                        'Sequence identities differ')
            p=data['true_probs']; predictions=[ref['bayes_probs'],np.exp(net['logp'].astype(float))]
            require(p.shape==(len(plan['times']),64),'Source dimensions differ')
            for q in [p,*predictions]:
                require(np.isfinite(q).all() and q.min()>=0 and np.max(np.abs(q.sum(-1)-1))<2e-6,
                        'Invalid probability vector')
            risks.append(np.stack([rel_entr(p,q).sum(-1)/np.log(2) for q in predictions],axis=-1))
        require(digest.hexdigest()==row['raw_records_sha256'],'Raw cohort digest differs')
        risks=np.asarray(risks); means=risks.mean(0); ses=risks.std(0,ddof=1)/32
        t=s['statistics']['conditions'][row['id']]
        for actual,saved in ((means,t['mean_risk_bits']),(ses,t['standard_error_bits'])):
            error=float(np.max(np.abs(actual-np.asarray(saved))))
            require(error<2e-12,'Independent probability-to-risk aggregation differs')
            max_error=max(max_error,error)
        for target in t['thresholds']:
            for k,name in enumerate(('bayes','transformer')):
                indices=np.flatnonzero(means[:,k]<=target['target_bits'])
                point=int(plan['times'][indices[0]]) if len(indices) else None
                require(point==target[name]['point_context'],'Independent threshold differs')
        total+=len(risks)
    return s,dict(passed=True,sequences=total,raw_records=3*total,
        independent_aggregation_max_error=max_error,forecast_precedes_model=True,
        numerical_validation=forecast['numerical_validation'])


def read_metadata(record):return json.loads(str(record['metadata_json']))


def draw_forecast(ax,s):
    """Categorical final tick retains nonattainment without assigning infinity."""
    from matplotlib.lines import Line2D
    sentinel=11.
    needs_nr=any(t[p]['compatible']['unreached_possible']
        for c in s['statistics']['conditions'].values() for t in c['thresholds']
        for p in ('bayes','transformer'))
    transform=lambda n:sentinel if n is None else np.log2(n)
    if needs_nr:
        ax.axvspan(10.65,11.4,color='.965',zorder=-3)
        ax.axhspan(10.65,11.4,color='.965',zorder=-3)
    ax.plot([5,10],[5,10],color='.65',ls=':',lw=1,zorder=-2)
    handles=[]
    for row,color,marker in zip(s['curves'],COLORS,MARKERS):
        condition=s['statistics']['conditions'][row['id']]
        for item in condition['thresholds']:
            b,t=item['bayes'],item['transformer']
            x,y=transform(b['point_context']),transform(t['point_context'])
            def bounds(prediction):
                c=prediction['compatible']; v=list(c['tested_contexts'])
                values=[np.log2(n) for n in v]
                if c['unreached_possible']:values.append(sentinel)
                return min(values),max(values)
            bx,by=bounds(b),bounds(t)
            ax.errorbar(x,y,xerr=[[max(0,x-bx[0])],[max(0,bx[1]-x)]],
                yerr=[[max(0,y-by[0])],[max(0,by[1]-y)]],color=color,
                fmt=marker,mfc='none',mew=1.,ms=4.8,capsize=1.5,lw=.75,alpha=.8)
        handles.append(Line2D([],[],color=color,marker=marker,mfc='none',ls='none',
                            ms=4.8,label='(%d,%d)'%tuple(row['depth_pair'])))
    ax.set_xlim(5.5,11.4 if needs_nr else 10.35);ax.set_ylim(5.5,11.4 if needs_nr else 10.35)
    ticks=[6,7,8,9,10]+([11] if needs_nr else [])
    labels=['64','128','256','512','1024']+(['NR'] if needs_nr else [])
    ax.set_xticks(ticks,labels);ax.set_yticks(ticks,labels)
    ax.set_xlabel('Bayesian context requirement');ax.set_ylabel('Transformer context requirement')
    ax.legend(handles=handles,title='Coordinate depths',fontsize=6.4,title_fontsize=6.4,
              frameon=False,loc='upper left',ncol=2,handletextpad=.2,columnspacing=.5)
    ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)


def outputs(out,s,checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from section4_unseen_row_transfer import draw_panel
    transfer=ROOT/'results/transformer/section4_unseen_row_transfer'
    tr=read(transfer/'summary.json'); receipt=read(transfer/'validation.json')
    require(receipt['passed'] and receipt['files']['summary.json']==sha(transfer/'summary.json'),
            'Transfer source receipt differs')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
                         'axes.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(5.5,2.85))
        draw_forecast(axes[0],s)
        axes[0].set_title('(a) Product: required context',loc='left')
        draw_panel(axes[1],tr,16,legend=False)
        axes[1].set_title('');axes[1].set_title('(b) Shared base: new context',loc='left')
        axes[1].set_ylim(-.12,.86)
        handles,_=axes[1].get_legend_handles_labels()
        axes[1].legend(handles,['Bayes','TF run 1','TF run 2','Token counts'],
            fontsize=7,frameon=False,loc='upper right',ncol=2,columnspacing=.5,handletextpad=.25)
        axes[1].set_xlabel('Source depth; 16 observed tokens')
        fig.tight_layout(pad=.65,w_pad=1.4)
        for ext in ('pdf','png'):fig.savefig(out/f'overview.{ext}',dpi=240,metadata={'Creator':'product_context_figures'})
        plt.close(fig)
        # Main-text figure at its native manuscript width, with readable labels.
        plt.rcParams.update({'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8})
        fig,axes=plt.subplots(2,2,figsize=(5.5,4.4),sharex=True,sharey=True)
        for ax,row in zip(axes.flat,s['curves']):
            c=s['statistics']['conditions'][row['id']]; times=np.asarray(c['times'])
            keep=times>=32
            for k,color,style in ((0,'#d77719','--'),(1,'#1565a8','-')):
                mean=np.asarray(c['mean_risk_bits'])[:,k]
                lower=np.asarray(c['lower_band_bits'])[:,k]
                upper=np.asarray(c['upper_band_bits'])[:,k]
                ax.plot(times[keep],mean[keep],color=color,ls=style,lw=1.2,
                        marker='.',markersize=2.5)
                ax.fill_between(times[keep],np.maximum(lower[keep],1e-6),upper[keep],color=color,alpha=.12,lw=0)
                require(np.max(upper[keep])<1.2 and np.min(lower[keep])>.04,
                        'Context plot clips a simultaneous uncertainty band')
            for target in s['plan']['targets_bits']:ax.axhline(target,color='.65',lw=.65,ls=':')
            ax.set_title('Coordinate depths (%d,%d)'%tuple(row['depth_pair']))
            ax.set_xscale('log',base=2);ax.set_yscale('log')
            ax.set_xticks([32,64,128,256,512,1024],['32','64','128','256','512','1024'])
            ax.set_yticks([.05,.1,.2,.5,1],['.05','.1','.2','.5','1'])
            ax.set_ylim(.04,1.2);ax.minorticks_off()
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=.10);ax.spines[['top','right']].set_visible(False)
        fig.supxlabel('Observed tokens, $t$',fontsize=8.5)
        fig.supylabel('Expected next-token regret (bits)',fontsize=8.5)
        fig.legend(handles=[Line2D([],[],color='#1565a8',label='Transformer'),
            Line2D([],[],color='#d77719',ls='--',label='Informed Bayes')],loc='upper center',
            ncol=2,frameon=False,fontsize=8.5)
        fig.tight_layout(rect=(.02,.025,1,.95))
        for ext in ('pdf','png'):fig.savefig(out/f'risk_curves.{ext}',dpi=240,metadata={'Creator':'product_context_figures'})
        plt.close(fig)
    lines=['depths\ttarget_bits\tbayes_context\ttransformer_context\tbayes_compatible\ttransformer_compatible']
    for row in s['curves']:
        for t in s['statistics']['conditions'][row['id']]['thresholds']:
            values=[str(row['depth_pair']),str(t['target_bits'])]
            values.extend(str(t[k]['point_context']) if t[k]['point_context'] is not None else 'not_reached'
                          for k in ('bayes','transformer'))
            for k in ('bayes','transformer'):
                c=t[k]['compatible']; values.append(','.join(map(str,c['tested_contexts']))+
                    (',not_reached' if c['unreached_possible'] else ''))
            lines.append('\t'.join(values))
    (out/'requirements.tsv').write_text('\n'.join(lines)+'\n')
    files=['overview.pdf','overview.png','risk_curves.pdf','risk_curves.png','requirements.tsv']
    validation=dict(schema='product-context-figure-validation-v1',**checks,
        inputs={str(p):sha(p) for p in [out/'summary.json',Path(__file__),
            Path(__file__).with_name('section4_unseen_row_transfer.py'),transfer/'summary.json',transfer/'validation.json']},
        outputs={name:sha(out/name) for name in files})
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();s,checks=audit(args.out)
    if args.check_only:
        v=read(args.out/'validation.json')
        require(all(v[k]==value for k,value in checks.items()),'Audit changed')
        for p,h in v['inputs'].items():require(sha(p)==h,'Figure input changed')
        for p,h in v['outputs'].items():require(sha(args.out/p)==h,'Figure output changed')
    else:outputs(args.out,s,checks)
    print(json.dumps(checks))


if __name__=='__main__':main()

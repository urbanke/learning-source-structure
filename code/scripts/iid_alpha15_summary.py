"""Collect completed follow-up draws and draw the alphabet-cutoff comparison.

python code/scripts/iid_alpha15_summary.py --run --collect --plot --report
Uses completed raw summaries, independent finite predictions, and no fitted
parameters. --plot alone needs only the committed aggregate and predictions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAMES=('a15_main','a15_large','a1_control','a2_control','a14_neighbor','a16_neighbor')
OUT=ROOT/'results/mechanism/iid_alpha15_followup.json'
PRED=ROOT/'results/mechanism/iid_tilt_finite_prediction.json'


def run_campaign():
    cases=[('a15_main',1.5,[4096,16384,65536],512,1/64),
           ('a15_large',1.5,[262144],512,1/64),
           ('a1_control',1.,[4096,65536],128,1/128),
           ('a2_control',2.,[4096,65536],256,1/64),
           ('a14_neighbor',1.4,[4096,65536],256,1/64),
           ('a16_neighbor',1.6,[4096,65536],256,1/64)]
    commands=[[sys.executable,str(ROOT/'code/scripts/iid_alpha15_followup.py'),
               '--alpha',str(a),'--ds',*[str(d) for d in ds],'--reps',str(n),
               '--step',str(step),'--out',str(ROOT/f'output/iid_alpha15_followup/{name}.json')]
              for name,a,ds,n,step in cases]
    def run(command):subprocess.run(command,check=True,cwd=ROOT)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,commands))


def collect():
    pred=json.loads(PRED.read_text())
    lookup={(r['alpha'],r['d'],r['k']):r for r in pred['rows']}
    cells=[];sources=[];comparisons=[]
    for name in NAMES:
        path=ROOT/f'output/iid_alpha15_followup/{name}.json'
        z=json.loads(path.read_text());assert z['complete']
        sources.append(dict(path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            command_argv=z['command_argv'],seed=z['seed'],
            repetition_start=z['repetition_start']))
        for c in z['cells']:
            cells.append(c)
            for j,k in enumerate(c['k']):
                p=lookup.get((c['alpha'],c['d'],k))
                if p is None:continue
                f=p['finite_random_scale'];m=c['redundancy_mean'][j];s=c['redundancy_se'][j]
                comparisons.append(dict(alpha=c['alpha'],d=c['d'],k=k,
                    measured_mean_bits=m,measured_SE_bits=s,
                    finite_prediction_bits=f['mean_bits'],finite_prediction_SE_bits=f['SE_bits'],
                    finite_prediction_relative_to_mean=f['mean_bits']/m-1,
                    difference_combined_SE=(f['mean_bits']-m)/math.hypot(s,f['SE_bits']),
                    infinite_leading_bits=p.get('infinite_normalizer_leading_bits')))
    # These cells use the same source/sample random prefixes, including the
    # separately executed largest alphabet; compute paired differences.
    a=np.load(ROOT/'output/iid_alpha15_followup/a15_main.npz')
    b=np.load(ROOT/'output/iid_alpha15_followup/a15_large.npz')
    allr=np.concatenate((a['redundancy'],b['redundancy']),axis=1)
    ds=[4096,16384,65536,262144];paired=[]
    for j in range(1,len(ds)):
        dif=allr[:,j]-allr[:,j-1]
        paired.append(dict(d_small=ds[j-1],d_large=ds[j],k=a['ks'].tolist(),
            mean_difference_bits=dif.mean(0).tolist(),
            difference_SE_bits=(dif.std(0,ddof=1)/math.sqrt(len(dif))).tolist()))
    result=dict(schema_version=1,source_depth=3,complete=True,sources=sources,
        prediction_sha256=hashlib.sha256(PRED.read_bytes()).hexdigest(),
        seed_protocol='Paired across d and alpha using independent weight and uniform streams; see source command records.',
        target='Bayes learning cost of one iid normalized tilted distribution, known q and L',
        qualifications=['Finite random-scale prediction is an approximation, not an exact Bayes identity.',
            'Its independent normalizer draws use a separate seed from the cost experiments.',
            'The infinite leading term is compared only for alpha>1.',
            'Means across alphabets/alpha and sample checkpoints are correlated; paired differences retain this dependence.'],
        cells=cells,comparisons=comparisons,paired_alpha15_adjacent_alphabet_differences=paired)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',len(cells),'configurations and',len(comparisons),'prediction comparisons')


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    z=json.loads(OUT.read_text());p=json.loads(PRED.read_text())
    lookup={(r['alpha'],r['d'],r['k']):r for r in p['rows']}
    cells=sorted((c for c in z['cells'] if c['alpha']==1.5),key=lambda c:c['d'])
    ds=[c['d'] for c in cells]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(11.7,8.3))
    fig.text(.065,.951,'Why the α = 1.5 prediction missed the measured cost',fontsize=18,weight='bold')
    fig.text(.065,.913,'One distribution, source depth L = 3. Increasing the alphabet reveals a substantial finite-alphabet effect.',fontsize=11)
    axes=[fig.add_axes([.085,.48,.385,.335]),fig.add_axes([.575,.48,.385,.335])]
    for ax,k in zip(axes,[4096,16384]):
        ms=[];ss=[];fs=[];es=[]
        for c in cells:
            j=c['k'].index(k);q=lookup[1.5,c['d'],k]
            ms.append(c['redundancy_mean'][j]);ss.append(c['redundancy_se'][j])
            fs.append(q['finite_random_scale']['mean_bits']);es.append(q['finite_random_scale']['SE_bits'])
        lead=q['infinite_normalizer_leading_bits']
        ax.axhline(lead,color='#B34735',ls='--',lw=1.7)
        ax.plot(ds,fs,color='#0072B2',lw=1.8)
        ax.fill_between(ds,np.array(fs)-2*np.array(es),np.array(fs)+2*np.array(es),color='#0072B2',alpha=.16)
        ax.errorbar(ds,ms,yerr=2*np.array(ss),fmt='o',color='#222222',ms=5,capsize=4,lw=1.2)
        ax.set_xscale('log',base=4);ax.set_xticks(ds,[f'{d:,}' for d in ds])
        ax.set_xlim(ds[0]/1.2,ds[-1]*1.2)
        ax.set_ylabel('Learning cost (bits)');ax.set_xlabel('Alphabet size, d')
        ax.set_title(f'{k:,} samples from the distribution',fontsize=12)
        ax.grid(axis='y',color='.91',lw=.7)
    handles=[Line2D([0],[0],marker='o',color='#222222',ls='',label='Measured Bayes cost (±2 SE)'),
             Line2D([0],[0],color='#0072B2',label='Finite-alphabet approximation'),
             Line2D([0],[0],color='#B34735',ls='--',label='Infinite-alphabet leading term')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.062,.885),ncol=3,frameon=False)
    fig.text(.085,.383,'The same finite-alphabet approximation also fits the neighboring exponents and controls.',fontsize=11)
    fig.text(.085,.35,'At d = 65,536 and k = 4,096:',fontsize=10)
    table_ax=fig.add_axes([.085,.132,.87,.197]);table_ax.axis('off')
    rows=[]
    for alpha in [1.,1.4,1.5,1.6,2.]:
        r=next(r for r in z['comparisons'] if r['alpha']==alpha and r['d']==65536 and r['k']==4096)
        rows.append([f'{alpha:g}',f'{r["measured_mean_bits"]:,.1f} ± {r["measured_SE_bits"]:.1f}',
                     f'{r["finite_prediction_bits"]:,.1f}',f'{100*r["finite_prediction_relative_to_mean"]:+.2f}%'])
    tbl=table_ax.table(cellText=rows,colLabels=['α','Measured cost ± SE (bits)','Finite approximation (bits)','Prediction / measured - 1'],
                       colWidths=[.08,.34,.29,.29],bbox=[0,0,1,1],cellLoc='center')
    tbl.auto_set_font_size(False);tbl.set_fontsize(10)
    for (i,j),cell in tbl.get_celld().items():
        cell.set_edgecolor('#dddddd');cell.set_linewidth(.5)
        if i==0:cell.set_facecolor('#edf2f5');cell.set_text_props(weight='bold')
    fig.text(.085,.08,'The finite curve retains the alphabet cutoff and averages over the true random normalizer. No parameters are fitted.',fontsize=9.5)
    fig.text(.085,.053,'Cost draws per setting: 512 at α = 1.5; 256 at 1.4, 1.6 and 2; 128 at 1. Independent normalizer draws: 8,192.',fontsize=9.5)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path);plt.close(fig)
    print(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--report',action='store_true')
    ap.add_argument('--collect',action='store_true');ap.add_argument('--plot',action='store_true')
    ap.add_argument('--pdf-out',type=Path,default=ROOT/'figures/iid_alpha15_followup.pdf')
    args=ap.parse_args()
    if args.run:run_campaign()
    if args.collect:collect()
    if args.plot:plot(args.pdf_out)
    if args.report:
        from iid_tilt_check_outputs import rich_text
        rich_text(ROOT/'notes/reviews/mechanisms_alpha15_followup_2026-09-06.md')


if __name__=='__main__':main()

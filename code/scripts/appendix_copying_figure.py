"""Figure12 presentation wrapper; existing validated releases remain immutable."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/layered-appendix-copying-mpl')
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code/scripts'),str(ROOT/'code/transformer')]
from appendix_figure_style import apply_style,style_axis,BLUE,ORANGE,GRAY,LINEWIDTH
from section4_compact_figure import load_extended_copying
from copying_precision_figure_input import load as load_bayes
from source import seed_of
OUT=ROOT/'results/transformer/appendix_style_2026-09-24'
RUN2=OUT/'copying_run2_early16'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def export_run2(source):
 source=Path(source);s=read(source/'summary.json');p=read(source/'plan.json');r=read(source/'source_receipt.json')
 assert s['complete'] and s['schema']=='copying-run2-early16-full16384-v1' and s['plan_id']==p['plan_id'] and s['plan_sha256']==sha(source/'plan.json')
 assert s['n_sequences']==16384 and s['seed']==1 and s['retained_sequences']==1024 and s['times']==list(range(1,17))
 assert s['paired_scores_sha256']==sha(source/'paired_scores.npz') and s['source_receipt_sha256']==sha(source/'source_receipt.json')
 assert r['all_rows_retained'] and r['source_generation'] is False and r['training'] is False and r['input_sha256']==p['input_sha256']
 assert max(s['checks'].values())<2e-12
 saved={"times":np.arange(1,17)};checks={}
 with np.load(source/'paired_scores.npz',allow_pickle=False) as raw,np.load(ROOT/'results/transformer/copying_bayes_precision_2026-09-24/paired_scores.npz',allow_pickle=False) as original:
  assert np.array_equal(raw['times'],np.arange(1,17))
  for depth in (2,3):
   condition=f'a1_L{depth}';g=next(x for x in s['groups'] if x['depth']==depth);assert g['n_sequences']==16384 and g['seed']==1
   roles={}
   for role in ('ordered','shuffled'):
    v=raw[f'{condition}_{role}'];assert v.shape==(16384,16) and np.isfinite(v).all()
    e=float(np.max(abs(v[:256]-original[f'{condition}_{role}_seed1'])));assert e<2e-12;checks[f'L{depth}_{role}_original256_score_error']=e;roles[role]=v
   gap=roles['shuffled']-roles['ordered'];rng=np.random.default_rng(seed_of('shuffle-extension-bootstrap-v1',1,depth,'all16384'));counts=rng.multinomial(16384,np.full(16384,1/16384),size=1000).astype(float)
   for metric,values in [('instantaneous',gap),('cumulative',gap.cumsum(1))]:
    mean=values.mean(0);se=values.std(0,ddof=1)/128;bounds=np.quantile(counts@values/16384,[.025,.975],axis=0)
    current=dict(mean=mean,se=se,lower95=bounds[0],upper95=bounds[1]);error=max(float(np.max(abs(v-g[metric][k]))) for k,v in current.items());assert error<2e-12
    checks[f'L{depth}_{metric}_replay_error']=error
    for k,v in current.items():saved[f'L{depth}_{metric}_{k}']=v
 RUN2.mkdir(parents=True,exist_ok=True);np.savez_compressed(RUN2/'curves.npz',**saved)
 receipt=dict(schema='copying-run2-early16-figure-release-v1',passed=True,n_sequences=16384,retained_sequences=1024,seed=1,times=list(range(1,17)),depths=[2,3],bootstrap_replicates=1000,checks=checks,curves_sha256=sha(RUN2/'curves.npz'),source_artifact_sha256={n:sha(source/n) for n in ('summary.json','plan.json','source_receipt.json','projection.json','paired_scores.npz')},source_summary=s,source_plan=p,source_receipt=r,uncertainty='Pointwise95% paired whole-source bootstrap; no training-seed uncertainty; no smoothing.')
 with gzip.GzipFile(filename=str(RUN2/'provenance.json.gz'),mode='wb',mtime=0) as f:f.write((json.dumps(receipt,indent=2)+'\n').encode())
 return checks

def load_run2():
 paths=[RUN2/'curves.npz',RUN2/'provenance.json.gz']
 with gzip.open(paths[1],'rt') as f:r=json.load(f)
 assert r['passed'] and r['curves_sha256']==sha(paths[0]) and r['n_sequences']==16384 and r['seed']==1 and r['times']==list(range(1,17)) and r['depths']==[2,3] and r['bootstrap_replicates']==1000 and max(r['checks'].values())<2e-12
 with np.load(paths[0],allow_pickle=False) as z:
  assert np.array_equal(z['times'],np.arange(1,17));groups={d:{m:{k:z[f'L{d}_{m}_{k}'].copy() for k in ('mean','se','lower95','upper95')} for m in ('instantaneous','cumulative')} for d in (2,3)}
 return groups,paths,r

def draw():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 run1,n,checks,paths=load_extended_copying();assert n==16384
 run2,run2_paths,r2=load_run2();bayes,bchecks,bpaths=load_bayes(ROOT/'results/transformer/copying_bayes_precision_2026-09-24');paths+=run2_paths+bpaths
 data={0:run1,1:run2};times=np.arange(1,17);bt=np.array([1,2,4,8,16]);colors={2:BLUE,3:ORANGE};summary={}
 with apply_style():
  fig,axes=plt.subplots(2,2,figsize=(5.5,3.65),sharey='row');fig.subplots_adjust(left=.12,right=.985,bottom=.12,top=.86,hspace=.43,wspace=.34)
  for row,metric in enumerate(('instantaneous','cumulative')):
   for col,seed in enumerate((0,1)):
    ax=axes[row,col];summary[f'run{seed+1}_{metric}']={}
    for depth in (2,3):
     st={k:np.asarray(v)[:16] for k,v in data[seed][depth][metric].items()};assert all(v.shape==(16,) and np.isfinite(v).all() for v in st.values())
     assert np.all(st['lower95']<=st['upper95'])
     ax.fill_between(times,st['lower95'],st['upper95'],color=colors[depth],alpha=.14,linewidth=0,zorder=1)
     ax.plot(times,st['mean'],color=colors[depth],ls='--',lw=LINEWIDTH,zorder=2)
     b=bayes[depth][metric];ax.errorbar(bt,b['bayes_mean'][bt-1],yerr=b['bayes_source_se'][bt-1],fmt='o' if depth==2 else 's',linestyle='none',color=colors[depth],mfc='white',mec=colors[depth],mew=.85,ms=3.5,elinewidth=.7,capsize=1.8,zorder=4)
     summary[f'run{seed+1}_{metric}'][f'L{depth}']=dict(mean=st['mean'].tolist(),lower95=st['lower95'].tolist(),upper95=st['upper95'].tolist(),positive_mean_times=times[st['mean']>0].tolist(),positive_interval_times=times[st['lower95']>0].tolist())
    ax.axhline(0,color=GRAY,lw=.6,zorder=0);style_axis(ax);ax.set_xscale('log',base=2);ax.set_xlim(.93,17.1);ax.set_xticks(bt,[str(i) for i in bt]);ax.set_xlabel('Observed tokens, $t$' if row else '')
    if col==0:ax.set_ylabel('Next-token gain (bits)' if row==0 else 'Cumulative gain (bits)')
    label=chr(97+row*2+col);desc='Next-token gain' if row==0 else 'Cumulative gain';ax.set_title(f'({label}) Run {seed+1}: {desc}',loc='left',pad=5)
  legend=[Line2D([],[],color=BLUE,ls='--',lw=LINEWIDTH,marker='o',mfc='white',ms=3.5,label='$L=2$'),Line2D([],[],color=ORANGE,ls='--',lw=LINEWIDTH,marker='s',mfc='white',ms=3.5,label='$L=3$'),Line2D([],[],color=GRAY,ls='--',lw=LINEWIDTH,label='Four-layer transformer'),Line2D([],[],color=GRAY,marker='o',mfc='white',mew=.85,ls='none',ms=3.5,label='Informed Bayes')]
  fig.legend(handles=legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.53,.985),handlelength=2.3,columnspacing=1.2)
  OUT.mkdir(parents=True,exist_ok=True)
  for ext in ('pdf','png'):fig.savefig(OUT/f'figure12.{ext}',dpi=220,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
  plt.close(fig)
 receipt=dict(schema='appendix-copying-figure12-v1',passed=True,model_layers=4,transformer_style='dashed',source_colors={'L2':BLUE,'L3':ORANGE},transformer_source_count_per_depth_per_run=16384,bayes_source_count_per_depth=256,bayes_times=bt.tolist(),bootstrap_replicates=1000,transformer_uncertainty='Pointwise95% paired whole-source bootstrap intervals, no smoothing or training-seed uncertainty',bayes_uncertainty='One paired source standard error, original validated complete256-source cohort',bayes_comparator='Source-induced stationary-frequency surrogate minus informed Bayes; not exact finite-shuffle Bayes.',inputs={str(x.relative_to(ROOT)):sha(x) for x in paths},plotter_sha256=sha(__file__),common_style_sha256=sha(ROOT/'code/scripts/appendix_figure_style.py'),checks={**checks,**bchecks,**r2['checks']},plots={ext:sha(OUT/f'figure12.{ext}') for ext in ('pdf','png')},narrative_values=summary)
 (OUT/'copying_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(figure=str(OUT/'figure12.pdf'),transformer_n=16384,bayes_n=256)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run2-source',type=Path);v=a.parse_args()
 if v.run2_source:export_run2(v.run2_source)
 draw()

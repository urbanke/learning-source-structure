"""Fixed-k normalized-row information and random-normalizer checks above alpha=1."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--in',dest='inp',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    data=json.loads(args.inp.read_text());plt.rcParams.update({'font.size':9,'mathtext.fontset':'cm'})
    fig,axs=plt.subplots(2,2,figsize=(6.9,5.3),gridspec_kw={'hspace':.38,'wspace':.30})
    colors=['#b0413e','#7a4f9e','#2b6a99','#2a8a84']
    for col,alpha in enumerate([1.5,2.]):
        rows=sorted([r for r in data['rows'] if r['alpha']==alpha],key=lambda r:r['d'])
        for i,row in enumerate(rows):
            k=np.asarray(row['k']);scale=k**(1/alpha);m=np.array(row['redundancy_mean'])/scale;se=np.array(row['redundancy_se'])/scale
            axs[0,col].plot(k,m,color=colors[i],marker='o',ms=2.5,label=f"$d={row['d']}$")
            axs[0,col].fill_between(k,m-2*se,m+2*se,color=colors[i],alpha=.10,lw=0)
        row=rows[-1];k=np.asarray(row['k']);scale=k**(1/alpha)
        axs[0,col].plot(k,np.array(row['poisson_comparison_bits'])/scale,':',color='black',lw=1.,label='Poisson comparison, largest $d$')
        axs[0,col].set_xscale('log');axs[0,col].set_title(f'$\\alpha={alpha}$')
        axs[0,col].set_xlabel('$k$ (iid samples)');axs[0,col].set_ylabel('$R_q(k)/k^{1/\\alpha}$ [bits]')
        axs[0,col].legend(frameon=False,fontsize=6)
        for L,marker,color in [(1,'o','#2b6a99'),(3,'s','#b0413e')]:
            nr=sorted([r for r in data['normalizers'] if r['alpha']==alpha and r['L']==L],key=lambda r:r['d'])
            d=np.array([r['d'] for r in nr]);v=2**L-1
            y=np.array([r['normalizer_variance'] for r in nr])/v;se=np.array([r['normalizer_variance_se'] for r in nr])/v
            axs[1,col].errorbar(d,y,yerr=2*se,color=color,marker=marker,ms=3,lw=.8,capsize=2,label=f'measured $L={L}$')
        exact=np.array([r['exact_finite_variance'] for r in nr])/v
        limit=nr[0]['infinite_alphabet_variance']/v
        axs[1,col].plot(d,exact,'--',color='black',lw=1,label='finite-$d$ formula')
        axs[1,col].axhline(limit,color='#777777',ls=':',lw=1,label='infinite-alphabet limit')
        axs[1,col].set_xscale('log');axs[1,col].set_xlabel('$d$ (symbols)')
        axs[1,col].set_ylabel('$\\operatorname{Var}(S_q)/(2^L-1)$')
        axs[1,col].legend(frameon=False,fontsize=6)
    args.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,bbox_inches='tight');fig.savefig(args.out.with_suffix('.png'),dpi=160,bbox_inches='tight')

if __name__=='__main__':main()

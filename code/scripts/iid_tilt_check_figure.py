"""Plot the new iid Bayes-cost checks, with an explicit prediction-error panel.

python code/scripts/iid_tilt_check_figure.py
Uses the saved aggregate only; no simulations or fitted parameters.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from iid_layered_tilt_check import scalar_sum
from tilted_row_reference import zipf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path('results/mechanism/iid_layered_tilt_check.json'))
    ap.add_argument('--out', type=Path, default=Path('figures/iid_tilt_checks.pdf'))
    args = ap.parse_args()
    cells = json.loads(args.input.read_text())['cells']
    data = {(x['d'], x['alpha']): x for x in cells}
    master = json.loads(Path('results/mechanism/master_curves.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {256: '#0072B2', 1024: '#D55E00', 4096: '#009E73'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        for page, alphas in enumerate(((.75, .9, 1.), (1.25, 1.5, 2.)), 1):
            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3),
                                     gridspec_kw={'height_ratios': [1.5, 1.]})
            fig.subplots_adjust(left=.075, right=.975, top=.77, bottom=.19,
                                wspace=.28, hspace=.26)
            title = ('Single-distribution check where the crossing is not quadratic'
                     if page == 1 else 'Above one: the measured cost exceeds the uncorrected prediction')
            fig.suptitle(title, x=.055, y=.96, ha='left', fontsize=17, weight='bold')
            fig.text(.055, .915, 'Source depth L = 3. Points: measured Bayes learning cost. Lines: original single-distribution approximation.', fontsize=11)
            fig.text(.055, .885, 'Learning cost = expected code length minus the cost when the sampled distribution is known.', fontsize=10)
            legend = [Line2D([0], [0], color=c, marker='o', markersize=4,
                             label=f'd = {d:,}') for d, c in colors.items()]
            fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.05, .867),
                       frameon=False, ncol=3)
            for j, a in enumerate(alphas):
                top, bottom = axes[:, j]
                top.set_title(rf'$\alpha = {a:g}$', fontsize=13, pad=10)
                for d, color in colors.items():
                    z = data[d, a]
                    k = np.array(z['k'])
                    mean, se = np.array(z['redundancy_mean']), np.array(z['redundancy_se'])
                    pred = np.array(z['poisson_comparison_bits'])
                    dense = np.geomspace(k[0], k[-1], 160)
                    curve = [scalar_sum(zipf(d, a), kk, 3, master)[0] for kk in dense]
                    top.plot(dense, curve, color=color, lw=1.6)
                    top.errorbar(k, mean, yerr=2*se, color=color, fmt='o', ms=4,
                                 capsize=2, lw=.9)
                    bottom.errorbar(k, 100*(mean/pred-1), yerr=200*se/pred,
                                    color=color, fmt='o-', ms=4, capsize=2, lw=.9)
                top.set_xscale('log'); top.set_yscale('log')
                bottom.set_xscale('log')
                top.tick_params(labelbottom=False)
                bottom.axhline(0, color='.3', lw=.8, ls='--')
                for ax in (top, bottom):
                    ax.grid(True, which='major', color='.91', lw=.6)
                    ax.set_xlim(12, 22000)
                bottom.set_xlabel('Samples from one distribution, k')
                if j == 0:
                    top.set_ylabel('Learning cost (bits)')
                    bottom.set_ylabel('Measured / prediction - 1 (%)')
                bottom.set_ylim((-5, 12) if page == 1 else (-2, 55))
            fig.text(.075, .11, 'Upper panels compare costs. Lower panels show the discrepancy, so close curves can still be judged.', fontsize=10)
            fig.text(.075, .083, 'Bars show ±2 sample standard errors: 512 independent draws at d = 256 and 1,024; 256 at d = 4,096.', fontsize=10)
            fig.text(.075, .056, 'The Bayes code knows q and L. These are iid distributions; no stationary-frequency approximation is used.', fontsize=10)
            fig.text(.975, .035, f'{page} / 2', ha='right', color='.4')
            pdf.savefig(fig)
            plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()

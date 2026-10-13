"""Plot the validated finite-sum family forecast and saved paired estimates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/transformer/section4_family_forecast_2026-09-18'
FAMILIES = ['product', 'shared_base', 'uniform_base']
BLUE, ORANGE, PURPLE = '#1565a8', '#d77719', '#8056a5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inputs(out):
    summary = json.loads((out / 'summary.json').read_text())
    receipt = json.loads((out / 'validation.json').read_text())
    require(receipt['passed'] is True and receipt['summary_sha256'] == sha(out / 'summary.json'),
            'Forecast does not have a matching validation receipt')
    require(receipt['script_sha256'] == sha(ROOT / 'code/scripts/section4_family_forecast.py'),
            'Numerical consumer changed since validation')
    require(summary['schema'] == 'section4-family-forecast-v1' and summary['families'] == FAMILIES
            and summary['empirical_times'] == [1, 4] and summary['numerical_masses_rescaled'] is False,
            'Unexpected forecast scope')
    times = [r['scored_predictions'] for r in summary['records']]
    require(times == list(range(5)), 'Forecast times changed')
    plotted = dict(prediction_times=times,
                   bayes_mean=[r['equal_family_cost_bits'] for r in summary['records']],
                   transformer_times=summary['empirical_times'], groups={})
    for group in ['equal_family_average'] + FAMILIES:
        data = summary['empirical'][group]
        plotted['groups'][group] = dict(bayes=data['bayes_forecast_bits'], models={})
        for model in ['pooled_seed0', 'pooled_seed1']:
            values = data['models'][model]['control_variate_transformer_cost']
            paired = data['models'][model]['paired_transformer_minus_mixture']
            require(values['coefficient'] == 1 and values['n_seq'] == (3072 if group == 'equal_family_average' else 1024),
                    'Unexpected estimator or sequence count')
            require(np.allclose(values['mean'], np.array(data['bayes_forecast_bits']) + paired['mean'], rtol=0, atol=1e-14)
                    and values['se'] == paired['se'], 'Paired estimator inconsistent')
            plotted['groups'][group]['models'][model] = values
    return plotted


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.5, zorder=0)
    ax.tick_params(length=3, width=.6)


def series(ax, x, values, index, label=None):
    ax.errorbar(x, values['mean'], yerr=values['se'], color=[BLUE, PURPLE][index],
                marker=['s', 'D'][index], markersize=4.3, linestyle='none',
                capsize=2.5, elinewidth=.9, label=label, zorder=4)


def forecast(ax, data, both=False):
    ax.plot(data['prediction_times'], data['bayes_mean'], color=ORANGE, marker='o',
            markerfacecolor='white', markersize=4.5, linewidth=1.25,
            label='Bayesian forecast', zorder=3)
    for k in range(2 if both else 1):
        # Offsets only separate the two training runs in the appendix.
        offset = (-.045 if k == 0 else .045) if both else 0
        series(ax, np.array(data['transformer_times']) + offset,
               data['groups']['equal_family_average']['models'][f'pooled_seed{k}'],
               k, f'Transformer, run {k + 1}')
    ax.set(xlim=(-.13, 4.23), ylim=(-.006, .235), xticks=range(5),
           xlabel='Number of predictions', ylabel='Extra cumulative loss (bits)')
    style(ax)


def save(fig, out, stem):
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{stem}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def draw(data, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.titlesize': 9, 'axes.labelsize': 8.5,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(4.8, 2.9), layout='constrained')
    forecast(ax, data)
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    save(fig, out, 'forecast')

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), layout='constrained')
    forecast(axes[0], data, both=True)
    axes[0].set_title('(a) Equal average over the three families', loc='left')
    axes[0].legend(frameon=False, loc='upper left', fontsize=7.5)
    ax = axes[1]
    x = np.arange(3)
    ax.plot(x, [data['groups'][f]['bayes'][1] for f in FAMILIES], color=ORANGE,
            marker='o', markerfacecolor='white', markersize=4.5, linestyle='none')
    for k in range(2):
        values = {stat: [data['groups'][f]['models'][f'pooled_seed{k}'][stat][1]
                        for f in FAMILIES] for stat in ('mean', 'se')}
        series(ax, x + [-.10, .10][k], values, k)
    ax.set(xticks=x, xticklabels=['Product', 'Shared-base\nLSA', 'Uniform-base\ncontrol'],
           xlim=(-.4, 2.4), ylim=(0, .33), ylabel='Extra cumulative loss (bits)')
    ax.set_title('(b) Four predictions, by generating family', loc='left')
    style(ax)
    save(fig, out, 'forecast_runs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    data = inputs(args.out)
    binding = dict(summary_sha256=sha(args.out / 'summary.json'),
                   validation_sha256=sha(args.out / 'validation.json'),
                   script_sha256=sha(__file__), plotted=data)
    names = [stem + suffix for stem in ['forecast', 'forecast_runs'] for suffix in ['.pdf', '.png']]
    if args.check_only:
        receipt = json.loads((args.out / 'figures_validation.json').read_text())
        require(all(receipt[k] == v for k, v in binding.items()), 'Figure inputs changed')
        require(receipt['passed'] is True and receipt['artifacts'] == {name: sha(args.out / name) for name in names},
                'Figure artifacts changed')
        print('Both figures match the validated forecast and paired estimates.')
        return
    draw(data, args.out)
    receipt = dict(schema='section4-family-forecast-figures-v1', passed=True, **binding,
                   artifacts={name: sha(args.out / name) for name in names},
                   horizontal_offsets='Appendix only: training runs displaced slightly for visibility; prediction times are 1 and 4.')
    (args.out / 'figures_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Created main-text forecast and appendix comparison.')


if __name__ == '__main__':
    main()

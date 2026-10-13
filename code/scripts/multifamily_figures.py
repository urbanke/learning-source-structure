"""Plot the saved pooled-family evaluation; no inference or Bayes computation.

The cumulative comparison is pooled transformer minus specialist transformer,
not a measurement of the Bayesian family-identification cost. Both training
seeds remain separate. --check-only binds all plotted statistics to the saved
summary and verifies the source, script, and output hashes.
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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'results/transformer/multifamily_2026-09-16'
SOURCE = DEFAULT_OUT / 'comparison.json'
VALIDATION = DEFAULT_OUT / 'validation.json'
FAMILIES = [('product', 'Product', '#2468A0'),
            ('shared_base', 'Shared base', '#BF611C'),
            ('uniform_base', 'Uniform base', '#288167')]
COLORS = ['#2468A0', '#BF611C', '#288167', '#8A4E98']
TIMES = [1, 4, 16, 64, 256, 1024]
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8., 'axes.titlesize': 8.,
         'axes.labelsize': 8., 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specialist(family, seed):
    return f'shared_specialist_seed{seed}' if family == 'shared_base' else {
        'product': 'product_specialist', 'uniform_base': 'uniform_specialist'}[family]


def statistic(data, n_seq, signed=False):
    mean, se = [np.asarray(data[k], dtype=float) for k in ('mean', 'se')]
    require(data['n_seq'] == n_seq, 'Unexpected sample count')
    require(mean.shape == se.shape == (6,), 'Unexpected time grid size')
    require(np.isfinite(mean).all() and np.isfinite(se).all() and (se >= 0).all(),
            'Invalid mean or standard error')
    require(signed or (mean >= 0).all(), 'Negative expected next-token regret')
    return dict(times=TIMES, mean=mean.tolist(), se=se.tolist(), n_sequences=n_seq)


def read_source(path=SOURCE):
    data = json.loads(Path(path).read_text())
    validation = json.loads(VALIDATION.read_text())
    require(validation['schema'] == 'multifamily-independent-validation-v1'
            and validation['status'] == 'passed', 'Independent numerical validation failed')
    require(validation['summary_sha256'] == sha(path)
            and validation['source_plan_id'] == data['source_plan_id'],
            'Source differs from independently validated summary')
    require(data['schema'] == 'pooled-three-family-evaluation-v1', 'Unexpected summary schema')
    require(data['prefixes'] == TIMES, 'Unexpected time grid')
    require(len(data['models']) == 6 and len(data['paired_differences']) == 6,
            'Missing models or paired comparisons')
    require(data['bayesian_references']['mixture'] == 'pending',
            'Bayesian status changed; review the transformer-only presentation')
    return data


def gap_cases(data, condition_id):
    condition = data['models']['pooled_seed0']['cohorts'][condition_id]['condition']
    family, n_seq = condition['family'], condition['n_seq']
    curves = []
    for seed in (0, 1):
        reference = specialist(family, seed)
        pair = f'pooled_seed{seed}_minus_{reference}'
        measured = data['paired_differences'][pair][condition_id]['cumulative_loss_bits']
        curves.append(dict(seed=seed, pooled_model=f'pooled_seed{seed}',
                           specialist_model=reference, **statistic(measured, n_seq, signed=True)))
    return dict(condition=condition, n_tokens=1025, curves=curves)


def pooled_panel(path=SOURCE):
    data = read_source(path)
    cases = []
    for family, label, color in FAMILIES:
        case = gap_cases(data, f'{family}_prior')
        require(case['condition']['n_seq'] == 1024 and case['condition']['kind'] == 'prior',
                'Main panel must use the full-prior cohort')
        cases.append(dict(label=label, color=color, **case))
    return dict(id='f', title='Pooled training', status='measured', cases=cases,
                quantity='Cumulative log loss of pooled transformer minus specialist transformer (bits).',
                averaging='1024 independent sequences from each full family prior; training seeds shown separately.',
                target_tokens='2 through t+1, conditioned on the first observed token',
                reference_pairing='Matching seeds for shared-base specialists; the one existing specialist for each other family.',
                bayesian_mixture_comparison='pending; not estimated by this figure')


def collect():
    data = read_source()
    full_prior = []
    for family, label, color in FAMILIES:
        curves = []
        names = [f'pooled_seed{seed}' for seed in (0, 1)]
        names += list(dict.fromkeys(specialist(family, seed) for seed in (0, 1)))
        for name in names:
            record = data['models'][name]['cohorts'][f'{family}_prior']
            curves.append(dict(model=name, **statistic(record['metrics']['next_kl_bits'], 1024)))
        full_prior.append(dict(family=family, label=label, color=color, n_tokens=1025, curves=curves))
    conditional = []
    groups = [('Product', [f'product_L{a}_{b}' for a, b in [(1, 1), (1, 8), (4, 4), (8, 8)]]),
              ('Uniform base', [f'uniform_base_L{l}' for l in [0, 1, 4, 8]])]
    groups += [(f'Shared ($\\alpha={alpha}$)',
                [f'shared_base_a{alpha}_L{l}' for l in [0, 1, 4, 8]])
               for alpha in ['0.5', '1', '1.5']]
    for title, ids in groups:
        cases = []
        for condition_id, color in zip(ids, COLORS):
            case = gap_cases(data, condition_id)
            condition = case['condition']
            require(condition['kind'] == 'conditional' and condition['n_seq'] == 256,
                    'Unexpected conditional cohort')
            label = ('$L_1,L_2=(' + ','.join(map(str, condition['ls_pair'])) + ')$'
                     if 'ls_pair' in condition else f"$L={condition['ls']}$")
            cases.append(dict(label=label, color=color, **case))
        conditional.append(dict(title=title, cases=cases))
    observed = {c['condition']['id'] for p in conditional for c in p['cases']}
    expected = {key for key, value in data['models']['pooled_seed0']['cohorts'].items()
                if value['condition']['kind'] == 'conditional'}
    require(observed == expected and len(observed) == 20, 'Conditional selection is incomplete')
    return dict(schema='multifamily-figures-v1', passed=True,
                source_path=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
                validation_path=str(VALIDATION.relative_to(ROOT)), validation_sha256=sha(VALIDATION),
                script_sha256=sha(__file__), source_plan_id=data['source_plan_id'],
                models={key: {k: v for k, v in value.items() if k != 'cohorts'}
                        for key, value in data['models'].items()},
                main_panel=pooled_panel(), full_prior_next_token_regret=full_prior,
                conditional_cumulative_gaps=conditional,
                uncertainty='Plus/minus one standard error across independent sequences. Gap errors are paired before averaging.',
                no_training_seed_pooling=True, n_test_sequences=8192, n_tokens=1025,
                target_tokens='Cumulative losses predict tokens 2 through t+1. Next-token regret predicts token t+1.',
                connecting_lines='Straight visual guides between the six measured times; no new estimates.',
                figures=dict(prior_regrets=dict(size_inches=[5.5, 2.55], minimum_font_points=7),
                             conditional_gaps=dict(size_inches=[5.5, 4.65], minimum_font_points=7)),
                bayesian_references=data['bayesian_references'],
                new_inference=False, new_sequences=False, new_bayesian_computation=False)


def axes_style(ax):
    ax.set_xscale('log', base=2)
    ax.set_xlim(.85, 1300)
    ax.set_xticks([1, 16, 1024], ['1', '16', '1024'])
    ax.minorticks_off()
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#D8DEE5', linewidth=.45)


def draw_gaps(ax, cases, capsize=1.1):
    for case in cases:
        for values in case['curves']:
            ax.errorbar(values['times'], values['mean'], yerr=values['se'],
                        color=case['color'], linestyle='-' if values['seed'] == 0 else ':',
                        linewidth=1.05, marker=None, elinewidth=.65, capsize=capsize)
    ax.axhline(0, color='#767F87', linewidth=.65, zorder=0)
    axes_style(ax)
    ax.margins(y=.13)


def save(fig, out, stem):
    from matplotlib.text import Text
    fig.canvas.draw()
    require(all(x.get_fontsize() >= 7 for x in fig.findobj(Text) if x.get_text()),
            'Figure contains text below 7 points')
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            require(box.x0 >= -.5 and box.x1 <= fig.bbox.width + .5,
                    f'Text crosses horizontal figure boundary: {artist.get_text()}')
    fig.savefig(out / f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out / f'{stem}.png', dpi=260, metadata={'Software': 'multifamily_figures.py'})
    plt.close(fig)


def render(receipt, out):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.55))
        fig.subplots_adjust(left=.095, right=.975, bottom=.20, top=.70, wspace=.29)
        for index, (ax, panel) in enumerate(zip(axes, receipt['full_prior_next_token_regret'])):
            for curve in panel['curves']:
                name = curve['model']
                pooled = name.startswith('pooled')
                seed = 1 if name.endswith('seed1') else 0
                color = panel['color'] if pooled else '#333333'
                line = ('-' if seed == 0 else ':') if pooled else ('--' if seed == 0 else '-.')
                t, m, se = [np.asarray(curve[key]) for key in ['times', 'mean', 'se']]
                ax.fill_between(t, np.maximum(0, m-se), m+se, color=color, alpha=.1, linewidth=0)
                ax.plot(t, m, color=color, linestyle=line, linewidth=1.15)
            axes_style(ax)
            ax.set_ylim(bottom=0)
            ax.set_title(f"({chr(97+index)}) {panel['label']}", loc='left', pad=8)
        fig.legend([Line2D([], [], color='#333333', linestyle=s, linewidth=1.2) for s in ['-', ':']]
                   + [Line2D([], [], color='#333333', linestyle=s, linewidth=1.2) for s in ['--', '-.']],
                   ['Pooled seed 0', 'Pooled seed 1', 'Specialist seed 0', 'Specialist seed 1 (shared base)'],
                   loc='upper center', bbox_to_anchor=(.54, 1), frameon=False, ncol=2,
                   fontsize=7.5, columnspacing=1., handlelength=2.3)
        fig.supylabel('Expected next-token regret (bits)', x=.002, fontsize=8)
        fig.supxlabel('Observed tokens, $t$', y=.015, fontsize=8)
        save(fig, out, 'prior_regrets')

        fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.65))
        fig.subplots_adjust(left=.11, right=.975, bottom=.10, top=.84, wspace=.42, hspace=.87)
        for index, (ax, panel) in enumerate(zip(axes.flat, receipt['conditional_cumulative_gaps'])):
            draw_gaps(ax, panel['cases'])
            ax.set_title(f"({chr(97+index)}) {panel['title']}", loc='left', pad=31)
            labels = [case['label'].replace('$L_1,L_2=', '$') for case in panel['cases']]
            ax.legend([Line2D([], [], color=case['color'], linewidth=1.5) for case in panel['cases']],
                      labels, loc='lower left', bbox_to_anchor=(-.04, 1.015), frameon=False,
                      ncol=2, fontsize=7, columnspacing=.8, handletextpad=.3, handlelength=1,
                      labelspacing=.1, borderaxespad=0)
        axes.flat[-1].set_axis_off()
        axes.flat[-1].legend([Line2D([], [], color='#333333', linewidth=1.3, linestyle=s) for s in ['-', ':']],
                            ['Pooled seed 0', 'Pooled seed 1'], loc='upper left', frameon=False,
                            fontsize=8, handlelength=2)
        axes.flat[-1].text(.05, .54, 'Positive: higher loss\nNegative: lower loss\n\n256 sequences\nper condition\nBars: one standard error\nof paired differences',
                          transform=axes.flat[-1].transAxes, va='top', fontsize=7, linespacing=1.4)
        fig.supylabel('Cumulative loss: pooled minus specialist (bits)', x=.002, fontsize=8)
        fig.supxlabel('Observed tokens, $t$', y=.015, fontsize=8)
        save(fig, out, 'conditional_gaps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    receipt = collect()
    names = [f'{stem}.{ext}' for stem in ['prior_regrets', 'conditional_gaps'] for ext in ['pdf', 'png']]
    if args.check_only:
        old = json.loads((args.out / 'receipt.json').read_text())
        outputs = old.pop('outputs')
        require(old == receipt, 'Receipt differs from saved source values or plotting code')
        require(outputs == {name: sha(args.out / name) for name in names}, 'Figure outputs changed')
        print('Verified all displayed source values, all 20 conditional groups, and artifact hashes.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    render(receipt, args.out)
    receipt['outputs'] = {name: sha(args.out / name) for name in names}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()

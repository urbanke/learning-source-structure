"""Presentation of saved-data unseen-context tests, with failed Bayes means withheld."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/transformer/section4_unseen_context_2026-09-17'
PRODUCT = ROOT / 'results/transformer/product_context_confirmation_2026-09-17'
UNIFORM = ROOT / 'results/transformer/section4_unseen_uniform_control_2026-09-17'
DEPTHS, TIMES = (0, 1, 4, 8), (4, 16, 64)
BLUE, ORANGE, GRAY = '#1565a8', '#d77719', '#777777'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def groups(summary, depth):
    values = sorted((g for g in summary['groups'] if g['depth'] == depth),
                    key=lambda g: g['time'])
    require([g['time'] for g in values] == list(TIMES), 'Wrong observation times')
    return values


def accepted(group):
    require(group['bayes_released'] == group['subset_numerical_validation']['passed'],
            'Bayesian release flag disagrees with subset precision')
    return group['bayes_released']


def style(ax, selected, *, counts=True):
    ax.axhline(0, color='.6', lw=.65, zorder=-3)
    ax.set_xscale('log', base=4)
    ax.set_xlim(3, 88)
    labels = [str(t) + (f"\nn={g['n_selected']}" if counts else '')
              for t, g in zip(TIMES, selected)]
    ax.set_xticks(TIMES, labels)
    ax.minorticks_off()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.13)


def error_line(ax, selected, name, metric, **kwargs):
    values = [g['predictors'][name][metric] for g in selected]
    valid = [accepted(g) or name != 'bayes' for g in selected]
    means = np.array([v['mean'] if ok else np.nan for v, ok in zip(values, valid)])
    ses = np.array([v['standard_error'] if ok else np.nan for v, ok in zip(values, valid)])
    ax.errorbar(TIMES, means, yerr=ses, capsize=2.5, elinewidth=.8, lw=1.1,
                markersize=4, **kwargs)
    return means, ses


def create(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from product_context_figures import draw_forecast

    settings = {'font.family': 'DejaVu Sans', 'font.size': 8.5,
                'axes.titlesize': 9, 'axes.labelsize': 8.5,
                'axes.linewidth': .6, 'pdf.fonttype': 42}
    artifacts = []

    def save(fig, name):
        for ext in ('pdf', 'png'):
            filename = f'{name}.{ext}'
            fig.savefig(out / filename, dpi=240,
                        metadata={'Creator': 'section4_unseen_context_figures'})
            artifacts.append(filename)
        plt.close(fig)

    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.9), sharey=True)
        extrema = [0.]
        for ax, depth in zip(axes.flat, DEPTHS):
            selected = groups(summary, depth)
            for name, color, marker, line in (
                    ('bayes', ORANGE, 'o', '--'),
                    ('larger_seed0', BLUE, 's', '-'),
                    ('kt_unigram', GRAY, '^', ':')):
                mean, se = error_line(ax, selected, name, 'gain_over_uniform_bits',
                                     color=color, marker=marker, ls=line,
                                     markerfacecolor='none' if name == 'bayes' else color)
                extrema.extend((mean[np.isfinite(mean)] - se[np.isfinite(mean)]).tolist())
                extrema.extend((mean[np.isfinite(mean)] + se[np.isfinite(mean)]).tolist())
            style(ax, selected)
            ax.set_title(f'Depth $L={depth}$' + (' (iid control)' if depth == 0 else ''))
            missing = [str(g['time']) for g in selected if not accepted(g)]
            if missing:
                ax.text(.97, .96, 'Bayes pending at ' + ', '.join(missing),
                        transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color='.35')
        lo, hi = min(extrema), max(extrema)
        axes[0, 0].set_ylim(lo - .08 * (hi-lo), hi + .12 * (hi-lo))
        fig.legend(handles=[
            Line2D([], [], color=BLUE, marker='s', ms=4, label='Larger transformer, run 1'),
            Line2D([], [], color=ORANGE, marker='o', mfc='none', ms=4, ls='--', label='Informed Bayes'),
            Line2D([], [], color=GRAY, marker='^', ms=4, ls=':', label='Token counts')],
            loc='upper center', ncol=2, frameon=False, fontsize=8)
        fig.supxlabel('Observed tokens, $t$; selected sequences, $n$', fontsize=9)
        fig.supylabel('Expected next-token loss saved over uniform (bits)', fontsize=9)
        fig.tight_layout(rect=(.025, .035, 1, .905), h_pad=1.6, w_pad=1.3)
        save(fig, 'gains')

        # Separate training runs; every point is a paired same-sequence difference.
        fig, axes = plt.subplots(4, 2, figsize=(5.5, 7.3), sharex=True, sharey='row')
        for row, depth in enumerate(DEPTHS):
            selected = groups(summary, depth)
            for seed in (0, 1):
                ax = axes[row, seed]
                for name, color, line, marker in (
                        (f'transformer_seed{seed}', GRAY, ':', 'x'),
                        (f'larger_seed{seed}', BLUE, '-', 's'),
                        ('kt_unigram', ORANGE, '--', '^')):
                    vals = [g['paired_loss_bits'][name + '_minus_bayes'] for g in selected]
                    means = [v['mean'] if accepted(g) else np.nan for v, g in zip(vals, selected)]
                    ses = [v['standard_error'] if accepted(g) else np.nan for v, g in zip(vals, selected)]
                    ax.errorbar(TIMES, means, yerr=ses, color=color, ls=line, marker=marker,
                                ms=3.5, lw=1, elinewidth=.7, capsize=2)
                style(ax, selected, counts=False)
                ax.set_title(f'$L={depth}$; training run {seed+1}', fontsize=8.5)
                if any(not accepted(g) for g in selected):
                    ax.text(.97, .93, '64: Bayes pending', transform=ax.transAxes,
                            ha='right', va='top', fontsize=7, color='.4')
                ax.tick_params(labelbottom=True)
        fig.legend(handles=[
            Line2D([], [], color=GRAY, marker='x', ms=4, ls=':', label='Original transformer'),
            Line2D([], [], color=BLUE, marker='s', ms=4, label='Larger transformer'),
            Line2D([], [], color=ORANGE, marker='^', ms=4, ls='--', label='Token counts')],
            loc='upper center', ncol=2, frameon=False, fontsize=8)
        fig.supxlabel('Observed tokens, $t$', fontsize=9)
        fig.supylabel('Expected next-token loss above informed Bayes (bits)', fontsize=9)
        fig.tight_layout(rect=(.03, .025, 1, .92), h_pad=1.0, w_pad=1.2)
        save(fig, 'paired_runs')

        # Keep the completed product result independent of the old transfer panel.
        product = read(PRODUCT / 'summary.json')
        valid = read(PRODUCT / 'validation.json')
        require(valid['passed'] and valid['inputs'][str(PRODUCT / 'summary.json')] == sha(PRODUCT / 'summary.json'),
                'Product summary validation differs')
        fig, ax = plt.subplots(figsize=(3.65, 3.35))
        draw_forecast(ax, product)
        for text in ax.get_legend().get_texts():
            text.set_fontsize(8)
        ax.get_legend().get_title().set_fontsize(8)
        ax.tick_params(labelsize=8)
        fig.tight_layout(pad=.6)
        save(fig, 'context_forecast')

        uniform = read(UNIFORM / 'summary.json')
        def uniform_panel(ax, depth):
            selected = sorted((g for g in uniform['groups'] if g['depth'] == depth),
                              key=lambda g: g['time'])
            require([g['time'] for g in selected] == list(TIMES), 'Uniform times differ')
            for name, color, marker, line in (('transformer', BLUE, 's', '-'),
                                             ('counts', GRAY, '^', ':')):
                ax.errorbar(TIMES, [-g[name]['mean'] for g in selected],
                            yerr=[g[name]['se'] for g in selected],
                            color=color, marker=marker, ls=line, ms=4,
                            lw=1.1, elinewidth=.8, capsize=2.5)
            ax.plot(TIMES, [0]*len(TIMES), color=ORANGE, marker='o', mfc='none',
                    ms=5, ls='--', lw=1, zorder=5)
            style(ax, [dict(g, n_selected=g['n']) for g in selected])
            ax.set_title(f'Uniform base, $L={depth}$')
            ax.set_ylim(-.46, .045)

        uniform_handles = [
            Line2D([], [], color=BLUE, marker='s', ms=4, label='Uniform-base transformer'),
            Line2D([], [], color=ORANGE, marker='o', mfc='none', ls='--', ms=4, label='Informed Bayes'),
            Line2D([], [], color=GRAY, marker='^', ls=':', ms=4, label='Token counts')]
        fig, ax = plt.subplots(figsize=(3.65, 3.35))
        uniform_panel(ax, 1)
        ax.legend(handles=uniform_handles, fontsize=7.5, frameon=False, loc='lower left')
        ax.set_ylabel('Expected loss saved over uniform (bits)')
        ax.set_xlabel('Observed tokens, $t$; selected sequences, $n$')
        fig.tight_layout(pad=.65)
        save(fig, 'uniform_control')

        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8), sharey=True)
        for ax, depth in zip(axes.flat, DEPTHS):
            uniform_panel(ax, depth)
        fig.legend(handles=uniform_handles, loc='upper center', ncol=2,
                   frameon=False, fontsize=8)
        fig.supxlabel('Observed tokens, $t$; selected sequences, $n$', fontsize=9)
        fig.supylabel('Expected next-token loss saved over uniform (bits)', fontsize=9)
        fig.tight_layout(rect=(.03, .035, 1, .9), h_pad=1.5, w_pad=1.2)
        save(fig, 'uniform_all_depths')
    return artifacts


def paired_table(out, summary):
    lines = [r'\begin{tabular}{rrrccc}', r'\toprule',
             r'$L$ & $t$ & $n$ & Larger, run 1 & Larger, run 2 & Token counts \\',
             r'\midrule']
    for depth in DEPTHS:
        for group in groups(summary, depth):
            cells = [str(depth), str(group['time']), str(group['n_selected'])]
            for name in ('larger_seed0', 'larger_seed1', 'kt_unigram'):
                value = group['paired_loss_bits'][name + '_minus_bayes']
                cells.append(f"${value['mean']:.4f}\\pm{value['standard_error']:.4f}$"
                             if accepted(group) else r'---')
            lines.append(' & '.join(cells) + r' \\')
        if depth != DEPTHS[-1]:
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'paired_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    summary = read(args.out / 'summary.json')
    validation = read(args.out / 'validation.json')
    require(validation['passed'] and validation['files']['summary.json'] == sha(args.out / 'summary.json')
            and validation['files']['records.csv'] == sha(args.out / 'records.csv'),
            'Unseen-context data validation differs')
    uniform_validation = read(UNIFORM / 'validation.json')
    require(uniform_validation['passed'] and all(sha(UNIFORM / name) == digest
            for name, digest in uniform_validation['outputs'].items()),
            'Uniform-base control is not validated')
    require(len(summary['groups']) == 12, 'Need all twelve conditions, including any withheld means')
    for depth in DEPTHS:
        for group in groups(summary, depth):
            require(group['n_selected'] == len(group['selected_indices']) and group['n_original'] == 256,
                    'Selected counts differ')
    paths = [args.out / 'summary.json', args.out / 'validation.json', Path(__file__),
             PRODUCT / 'summary.json', PRODUCT / 'validation.json',
             UNIFORM / 'summary.json', UNIFORM / 'validation.json',
             Path(__file__).with_name('product_context_figures.py')]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if args.check_only:
        receipt = read(args.out / 'figure_receipt.json')
        require(receipt['inputs'] == inputs, 'Figure inputs changed')
        for name, digest in receipt['outputs'].items():
            require(sha(args.out / name) == digest, 'Figure output changed')
        print('Verified figure provenance; unvalidated Bayesian means remain withheld.')
        return
    artifacts = create(args.out, summary)
    paired_table(args.out, summary)
    artifacts.append('paired_table.tex')
    receipt = dict(inputs=inputs, outputs={n: sha(args.out / n) for n in artifacts},
                   bayes_withheld=[{'depth': g['depth'], 'time': g['time']}
                                   for g in summary['groups'] if not accepted(g)],
                   uncertainty='One standard error across independent selected sequences; paired differences use the same histories.',
                   main_training_run='larger_seed0; fixed run 1, no performance selection')
    (args.out / 'figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'outputs': artifacts, 'bayes_withheld': receipt['bayes_withheld']}))


if __name__ == '__main__':
    main()

"""Validate transferred paired scores and draw the copying-payoff figures.

Uses only saved evaluations; neither transformer inference nor Bayesian
sampling is performed. Run 1 is fixed in the main plot. Both runs and the
independent-token control remain in the appendix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from source import seed_of

MANIFEST = ROOT / 'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json'
PLAN = ROOT / 'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root):
    from shared_base_shuffle_eval import paired_summary

    manifest, plan = read(MANIFEST), read(PLAN)
    saved = read(root / 'comparison/summary.json')
    require(saved['scientific_results'] is True and saved['primary_seed'] == 0,
            'Wrong scientific result or primary run')
    require(saved['manifest_sha256'] == sha(MANIFEST), 'Cohort manifest changed')
    require(len(saved['groups']) == 6, 'Missing comparison group')
    groups, bindings = [], {}
    for seed in (0, 1):
        for condition in manifest['conditions']:
            ident, depth = condition['id'], condition['depth']
            loaded = {}
            for role in ('ordered', 'shuffled'):
                folder = root / 'eval' / f'{role}_seed{seed}'
                receipt, status = read(folder / 'protocol.json'), read(folder / 'status.json')
                require(status['complete'] and status['scientific_results'], 'Incomplete evaluation')
                require(receipt['manifest_sha256'] == sha(MANIFEST)
                        and receipt['training_plan_sha256'] == sha(PLAN)
                        and receipt['evaluation_code_sha256'] == sha(ROOT / 'code/transformer/shared_base_shuffle_eval.py')
                        and receipt['model_code_sha256'] == sha(ROOT / 'code/transformer/model.py')
                        and receipt['role'] == role and receipt['seed'] == seed,
                        'Evaluation provenance differs')
                pair = plan['ordered_runs'][str(seed)]['files_sha256']
                require(receipt['paired_ordered_files_sha256'] == pair, 'Training pairing differs')
                if role == 'ordered':
                    require(receipt['checkpoint_sha256'] == pair['model_100000.pt']
                            and receipt['config_sha256'] == pair['config.json'], 'Ordered weights differ')
                path = folder / (ident + '.npz')
                key = f'{role}_seed{seed}/{ident}'
                bindings[key] = sha(path)
                require(saved['inputs'][key] == bindings[key], 'Score file changed')
                with np.load(path, allow_pickle=False) as archive:
                    require(str(archive['protocol_sha256']) == sha(folder / 'protocol.json')
                            and str(archive['source_tokens_sha256']) == condition['arrays_sha256']['tokens'],
                            'Scores have different histories or model receipt')
                    require(np.array_equal(archive['times'], np.arange(1, 1025)), 'Time grid differs')
                    loaded[role] = {k: archive[k].copy() for k in ('expected_loss_bits', 'regret_bits')}
                require(all(v.shape == (256, 1024) and np.isfinite(v).all()
                            for v in loaded[role].values()), 'Invalid score arrays')
            matches = [g for g in saved['groups'] if g['seed'] == seed and g['condition'] == ident]
            require(len(matches) == 1, 'Missing or duplicated group')
            group = matches[0]
            require(group['depth'] == depth and group['n_sequences'] == 256, 'Group scope differs')
            gap = loaded['shuffled']['expected_loss_bits'] - loaded['ordered']['expected_loss_bits']
            rng = np.random.default_rng(seed_of('shared-base-shuffle-bootstrap', seed, depth))
            counts = rng.multinomial(256, np.full(256, 1/256), size=1000).astype(float)
            for name, values in (('instantaneous', gap), ('cumulative', gap.cumsum(1))):
                rebuilt = paired_summary(values, counts)
                for stat in ('mean', 'se', 'lower95', 'upper95'):
                    require(np.allclose(rebuilt[stat], group[name][stat], rtol=0, atol=2e-12),
                            f'{ident} run {seed + 1}: {name} {stat} differs')
                require(rebuilt['crossing'] == group[name]['crossing'], 'Crossing summary differs')
            for role in loaded:
                a = loaded[role]['regret_bits']
                for stat, value in [('mean', a.mean(0)), ('se', a.std(0, ddof=1)/16)]:
                    require(np.allclose(value, group['absolute_regret'][role][stat], rtol=0, atol=2e-12),
                            'Absolute regret summary differs')
            groups.append(group)
    return dict(schema='shared-base-shuffle-paper-results-v1', groups=groups,
                primary_seed=0, scientific_scope='Paired diagnostic reuse; fresh timing confirmation pending.',
                bayesian_timing_compared=False, sign=saved['sign'], uncertainty=saved['uncertainty'],
                first_token=saved['first_token'], manifest_sha256=sha(MANIFEST),
                training_plan_sha256=sha(PLAN), original_summary_sha256=sha(root / 'comparison/summary.json'),
                input_sha256=bindings)


def draw(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7, 'axes.linewidth': .6,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {0: '#777777', 2: '#2468A0', 3: '#8A4E98'}
    times = np.arange(1, 1025)

    def curve(ax, group, metric, label):
        stats = group[metric]
        color = colors[group['depth']]
        ax.fill_between(times, stats['lower95'], stats['upper95'], color=color, alpha=.16, lw=0)
        ax.plot(times, stats['mean'], color=color, lw=1.25, label=label)

    def style(ax, metric):
        ax.axhline(0, color='.3', lw=.7, zorder=1)
        ax.set_xscale('log', base=2)
        ax.set(xlabel='Observed tokens, $t$', xlim=(.9, 1180),
               ylabel=('Next-token advantage (bits)' if metric == 'instantaneous'
                       else 'Cumulative advantage (bits)'))
        ax.set_xticks([1, 16, 256, 1024], ['1', '16', '256', '1024'])
        ax.minorticks_off()
        ax.grid(axis='y', color='#D8DEE5', lw=.45)
        ax.tick_params(length=2.5, pad=2)

    def save(fig, stem):
        for extension in ('pdf', 'png'):
            fig.savefig(out / f'{stem}.{extension}', dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), layout='constrained')
    for ax, metric in zip(axes, ('instantaneous', 'cumulative')):
        for group in summary['groups']:
            if group['seed'] == 0 and group['depth'] in (2, 3):
                curve(ax, group, metric, f"$L={group['depth']}$")
        style(ax, metric)
    axes[0].set_title('(a) Instantaneous benefit', loc='left')
    axes[1].set_title('(b) Accumulated benefit', loc='left')
    axes[0].legend(frameon=False)
    save(fig, 'payoff')
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55), layout='constrained')
    for col, seed in enumerate((0, 1)):
        for row, metric in enumerate(('instantaneous', 'cumulative')):
            ax = axes[row, col]
            for group in summary['groups']:
                if group['seed'] == seed and group['depth'] in (2, 3):
                    stats = group[metric]
                    color = colors[group['depth']]
                    ax.fill_between(times[:16], stats['lower95'][:16], stats['upper95'][:16],
                                    color=color, alpha=.16, lw=0)
                    ax.plot(times[:16], stats['mean'][:16], color=color, lw=1.15,
                            label=f"$L={group['depth']}$")
            style(ax, metric)
            ax.set_xlim(.95, 16.5)
            ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
            if col:
                ax.set_ylabel('')
            panel = 2 * row + col
            description = 'Next-token benefit' if row == 0 else 'Cumulative benefit'
            ax.set_title(f"({chr(97 + panel)}) Run {seed + 1}: {description}", loc='left', pad=5)
            if row == 0:
                ax.set_xlabel('')
    axes[0, 0].legend(frameon=False)
    save(fig, 'payoff_early')
    for seed in (0, 1):
        fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True, layout='constrained')
        for col, depth in enumerate((0, 2, 3)):
            group = next(g for g in summary['groups'] if g['seed'] == seed and g['depth'] == depth)
            for row, metric in enumerate(('instantaneous', 'cumulative')):
                curve(axes[row, col], group, metric, None)
                style(axes[row, col], metric)
                if col:
                    axes[row, col].set_ylabel('')
                if row == 0:
                    axes[row, col].set_xlabel('')
                    axes[row, col].set_title(f'$L={depth}$')
        fig.suptitle(f'Training run {seed + 1}; positive values favour ordered training', fontsize=9)
        save(fig, f'payoff_run{seed + 1}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'output/shared_base_shuffle_download_2026-09-18')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/transformer/shared_base_shuffle_2026-09-18')
    args = parser.parse_args()
    summary = validate(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    draw(summary, args.out)
    artifacts = {p.name: sha(p) for p in args.out.iterdir() if p.suffix in ('.pdf', '.png')}
    (args.out / 'validation.json').write_text(json.dumps(dict(passed=True,
        summary_sha256=sha(args.out / 'summary.json'), script_sha256=sha(__file__),
        checks='All paired means, sequence SEs, bootstrap bands and crossing records independently replayed from saved scores.',
        artifacts=artifacts), indent=2) + '\n')
    for g in summary['groups']:
        print(json.dumps(dict(seed=g['seed'], depth=g['depth'],
            instantaneous=g['instantaneous']['crossing'], cumulative=g['cumulative']['crossing'],
            final_gain=g['instantaneous']['mean'][-1], final_cumulative=g['cumulative']['mean'][-1])))


if __name__ == '__main__':
    main()

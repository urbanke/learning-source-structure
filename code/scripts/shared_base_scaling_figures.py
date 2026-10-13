"""Plot the audited shared-base capacity comparison without new inference.

The main figure compares original and larger models from the first run. This appendix view keeps
the two training runs in separate columns and the four source depths in
separate rows. Original models and Bayesian references are cross-checked
against the existing complete-cohort releases before presentation.
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

from shared_base_conditional_figures import read_panels

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/transformer/shared_base_scaling_2026-09-17'
PLAN = ROOT / 'code/transformer/configs/shared_base_scaling_plan_2026-09-17.json'
TIME16 = ROOT / 'results/transformer/shared_base_conditional_curves/time16_summary.json'
TIME64 = ROOT / 'results/transformer/shared_base_time64_complete_2026-09-17/summary.json'
TIMES = [1, 4, 16, 64, 256, 1024]
BAYES_TIMES = [1, 4, 16, 64]
DEPTHS = [0, 1, 4, 8]
COLORS = {'baseline': '#2468A0', 'larger': '#288167', 'bayes': '#222222'}
ROW_SCALES = {0: (1.2, [0, .4, .8, 1.2]), 1: (1.8, [0, .6, 1.2, 1.8]),
              4: (3., [0, 1, 2, 3]), 8: (3.9, [0, 1, 2, 3])}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(rows, key):
    values = [r[key] for r in rows]
    require(all(v['n'] == 256 for v in values), 'Incomplete displayed cohort')
    mean = np.asarray([v['mean'] for v in values], dtype=float)
    se = np.asarray([v['standard_error'] for v in values], dtype=float)
    require(np.isfinite(mean).all() and np.isfinite(se).all()
            and np.all(mean >= 0) and np.all(se >= 0), 'Invalid displayed regret statistics')
    return dict(times=[r['time'] for r in rows], mean=mean.tolist(), se=se.tolist())


def same_curve(actual, original, label):
    require(actual['times'] == original['times'], f'{label}: times changed')
    require(all(np.allclose(actual[k], original[k], rtol=1e-10, atol=2e-12)
                for k in ['mean', 'se']), f'{label}: released numerical values changed')


def load_scaling(source=SOURCE, reference_panels=None):
    """Load both runs and bind original/Bayes curves to their prior releases."""
    source = Path(source)
    paths = [source / 'comparison.json', source / 'validation.json', PLAN]
    data, audit, plan = [json.loads(p.read_text()) for p in paths]
    require(data['schema'] == 'shared-base-scaling-evaluation-v1', 'Unexpected scaling summary')
    require(audit['schema'] == 'shared-base-scaling-independent-audit-v1'
            and audit['passed'] is True and audit['summary_sha256'] == sha(paths[0]),
            'Scaling summary lacks an unchanged passing independent audit')
    require(data['plan_sha256'] == audit['plan_sha256'] == sha(PLAN), 'Scaling plan changed')
    require(data['depths'] == DEPTHS and data['times'] == TIMES and data['seed_order'] == [0, 1]
            and data['n_sequences_per_depth'] == 256 and data['n_tokens_per_sequence'] == 1025
            and data['evaluated_conditions_hidden'] is True, 'Changed scaling test scope')
    require(data['training_prior'] == dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
            'Changed Bayesian or training prior')
    require(plan['config']['steps'] == 100000 and plan['config']['batch'] == 64
            and plan['completed_sequences_per_seed'] == 6400000 and plan['n_params'] == 3454464
            and plan['baseline']['completed_sequences_per_seed'] == 6400000
            and plan['baseline']['n_params'] == 544256, 'Changed capacity or training exposure')
    require(len(data['rows']) == 48 and {(r['depth'], r['seed'], r['time']) for r in data['rows']}
            == {(d, s, t) for d in DEPTHS for s in [0, 1] for t in TIMES},
            'Missing, repeated or selected comparison rows')
    require(all(r['alpha'] == 1 and r['bayes_available'] == (r['time'] in BAYES_TIMES)
                for r in data['rows']), 'Changed source conditions or Bayesian coverage')
    if reference_panels is None:
        reference_files, reference_panels = read_panels(TIME16, TIME64)
        paths.extend(reference_files)
    panels = []
    for depth in DEPTHS:
        original = next(p for p in reference_panels if p['depth'] == depth)
        for seed in [0, 1]:
            rows = sorted((r for r in data['rows'] if r['depth'] == depth and r['seed'] == seed),
                          key=lambda r: r['time'])
            baseline = curve(rows, 'baseline_regret')
            larger = curve(rows, 'larger_regret')
            bayes = curve([r for r in rows if r['bayes_available']], 'bayes_regret')
            name = 'd64' if seed == 0 else 'd64_seed1'
            same_curve(baseline, dict(times=original['times'], **original['transformers'][name]),
                       f'L{depth} run{seed+1} baseline')
            same_curve(bayes, original['bayes'], f'L{depth} Bayes')
            for row in rows:
                difference = row['baseline_regret']['mean'] - row['larger_regret']['mean']
                require(np.isclose(difference, row['improvement_over_baseline']['mean'],
                                   rtol=1e-10, atol=2e-12), 'Paired improvement mean differs')
                if row['bayes_available']:
                    for arm in ['baseline', 'larger']:
                        difference = row[f'{arm}_regret']['mean'] - row['bayes_regret']['mean']
                        require(np.isclose(difference, row[f'{arm}_minus_bayes']['mean'],
                                           rtol=1e-10, atol=2e-12), 'Paired Bayesian gap mean differs')
            panels.append(dict(depth=depth, training_seed=seed, n_sequences=256, n_tokens=1025,
                               baseline=baseline, larger=larger, bayes=bayes))
    return paths, data, panels


def render(panels, out):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8., 'axes.titlesize': 8.5,
            'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
            'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(4, 2, figsize=(5.5, 6.8), sharex=True, sharey='row')
        fig.subplots_adjust(left=.105, right=.985, bottom=.09, top=.90, wspace=.18, hspace=.35)
        for ax, panel in zip(axes.flat, panels):
            y_max, y_ticks = ROW_SCALES[panel['depth']]
            for arm in ['baseline', 'larger']:
                value = panel[arm]
                t, mean, se = [np.asarray(value[k]) for k in ['times', 'mean', 'se']]
                require(np.max(mean + se) < y_max, 'Transformer uncertainty clipped')
                ax.fill_between(t, np.maximum(0, mean-se), mean+se,
                                color=COLORS[arm], alpha=.11, linewidth=0)
                ax.plot(t, mean, color=COLORS[arm], linewidth=1.15, linestyle='-')
            b = panel['bayes']
            require(np.max(np.asarray(b['mean']) + b['se']) < y_max, 'Bayesian uncertainty clipped')
            ax.errorbar(b['times'], b['mean'], yerr=b['se'], color=COLORS['bayes'],
                        linestyle='none', marker='o', markersize=4., markerfacecolor='white',
                        markeredgewidth=.85, elinewidth=.75, capsize=1.6, zorder=5)
            ax.set_xscale('log', base=2); ax.set_xlim(.85, 1250)
            ax.set_xticks(TIMES, [str(t) for t in TIMES]); ax.minorticks_off()
            ax.set_ylim(0, y_max); ax.set_yticks(y_ticks)
            ax.tick_params(labelsize=7., length=2.5, pad=2)
            ax.grid(axis='y', color='#D8DEE5', linewidth=.45); ax.set_axisbelow(True)
            ax.set_title(f"$L={panel['depth']}$, training run {panel['training_seed']+1}",
                         loc='left', pad=5)
        handles = [Line2D([], [], color=COLORS['baseline'], linewidth=1.4),
                   Line2D([], [], color=COLORS['larger'], linewidth=1.4),
                   Line2D([], [], color=COLORS['bayes'], linestyle='none', marker='o',
                          markerfacecolor='white', markersize=4.)]
        fig.legend(handles, ['Original model', 'Larger model', 'Informed Bayes'],
                   loc='upper center', bbox_to_anchor=(.54, .982), ncol=3, frameon=False,
                   fontsize=8., handlelength=2., columnspacing=1.3)
        fig.supylabel('Expected next-token regret (bits)', x=.012, fontsize=8.)
        fig.supxlabel('Observed tokens, $t$', y=.043, fontsize=8.)
        fig.text(.54, .011, 'Same 256 test sequences per depth; shading and bars show ±1 standard error.',
                 ha='center', fontsize=7.)
        from matplotlib.text import Text
        require(all(t.get_fontsize() >= 7 for t in fig.findobj(Text) if t.get_text()),
                'Text below seven-point minimum')
        fig.savefig(out / 'capacity_comparison.pdf', metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(out / 'capacity_comparison.png', dpi=260,
                    metadata={'Software': 'shared_base_scaling_figures.py'})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=SOURCE / 'figures')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    paths, data, panels = load_scaling(args.source)
    receipt = dict(schema='shared-base-scaling-figure-v1', passed=True, script_sha256=sha(__file__),
        reference_helper_sha256=sha(ROOT / 'code/scripts/shared_base_conditional_figures.py'),
        inputs={str(p.resolve().relative_to(ROOT)): sha(p) for p in paths}, panels=panels,
        plan_sha256=data['plan_sha256'], test_role='development',
        uncertainty='One standard error across256 independent sequences per depth; separate training runs.',
        display=dict(size_inches=[5.5, 6.8], minimum_font_points=7, rows=DEPTHS,
            columns_training_seeds=[0, 1], arm_colors=COLORS, transformer_lines='solid without markers',
            bayes_style='Open circles only at times1,4,16,64; no connecting line or extrapolation.',
            y_range_by_depth={str(d): [0, s[0]] for d, s in ROW_SCALES.items()},
            same_y_scale_within_each_row=True, training_sequence_count_per_run=6400000,
            seed_pooling=False, smoothing=False), new_inference=False, new_bayesian_computation=False)
    names = ['capacity_comparison.pdf', 'capacity_comparison.png']
    if args.check_only:
        saved = json.loads((args.out / 'receipt.json').read_text()); outputs = saved.pop('outputs')
        require(saved == receipt and outputs == {n: sha(args.out / n) for n in names},
                'Shared-base scaling figure receipt or files changed')
        print('Verified audited capacity comparison, original references, separate runs and figure hashes.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    render(panels, args.out)
    receipt['outputs'] = {n: sha(args.out / n) for n in names}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(args.out / 'capacity_comparison.pdf')


if __name__ == '__main__':
    main()

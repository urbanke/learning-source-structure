"""Present the independently audited hidden-factor scaling comparison.

Uses saved full-prior means and standard errors only. No model inference,
Bayesian computation, source generation, or training-seed pooling is performed.
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
SOURCE = ROOT / 'results/transformer/hidden_factor_scaling_2026-09-17'
PLAN = ROOT / 'code/transformer/configs/hidden_factor_scaling_plan_2026-09-16.json'
BASELINE = ROOT / 'results/transformer/hidden_factor_pilot_2026-09-16/summary.json'
BASELINE_VALIDATION = BASELINE.with_name('validation.json')
TIMES = [1, 4, 16, 64, 256, 1024]
ARMS = ['baseline', 'longsmall', 'large']
LABELS = {'baseline': 'Baseline', 'longsmall': 'More training', 'large': 'Larger'}
COLORS = {'baseline': '#2468A0', 'longsmall': '#BF611C', 'large': '#288167'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_curve(value):
    mean, se = np.asarray(value['mean']), np.asarray(value['se'])
    require(value['n_seq'] == 256 and mean.shape == se.shape == (6,)
            and np.isfinite(mean).all() and np.isfinite(se).all()
            and np.all(mean >= 0) and np.all(se >= 0), 'Invalid saved prediction-regret curve')
    return dict(times=TIMES, mean=mean.tolist(), se=se.tolist())


def load_scaling(source=SOURCE):
    """Return the complete audited comparison, six separate curves, and inputs."""
    source = Path(source)
    paths = [source / 'comparison.json', source / 'validation.json', PLAN,
             BASELINE, BASELINE_VALIDATION]
    data, validation, plan, baseline, baseline_validation = [json.loads(p.read_text()) for p in paths]
    require(data['schema'] == 'hidden-factor-scaling-comparison-v1', 'Unexpected scaling comparison')
    require(validation['schema'] == 'hidden-factor-scaling-independent-audit-v1'
            and validation['passed'] is True and validation['summary_sha256'] == sha(paths[0]),
            'Scaling comparison lacks an unchanged passing independent audit')
    require(validation['plan_sha256'] == data['plan_sha256'] == sha(PLAN), 'Scaling plan changed')
    require(data['complete_for_requested_comparison'] is True and data['full_cohort_complete'] is False
            and data['evaluated_n_sequences'] == validation['evaluated_sequences'] == 256
            and data['planned_n_sequences'] == 4096, 'Incomplete or changed scaling comparison scope')
    test = plan['test']
    require(data['test_role'] == test['role'] == 'development'
            and data['cohort_id'] == test['cohort_id'] == baseline['cohort']['cohort_id']
            and data['cohort_data_sha256'] == test['data_sha256']
            and data['dimensions'] == plan['source_dimensions'] == [2, 8]
            and data['sequence_length'] == test['length'] == 1025
            and data['times'] == test['times'] == TIMES, 'Source/cohort identity changed')
    require(all(validation[k] == data[k] for k in ['cohort_id', 'cohort_data_sha256',
                'dimensions', 'sequence_length', 'times']), 'Independent audit source identity differs')
    require(data['new_bayesian_computation'] is False
            and data['reference_records_sha256'] == plan['baseline']['reference_records_sha256']
            and data['baseline'] == plan['baseline'] and data['primary'] == plan['primary'],
            'Scaling comparison changed its baseline, reference, or final-checkpoint rule')
    require(baseline_validation['passed'] is True
            and baseline_validation['summary_sha256'] == sha(BASELINE) == data['baseline']['summary_sha256'],
            'Original baseline audit or summary changed')
    groups = [g for g in data['groups'] if g['id'] == 'full_prior']
    require(len(groups) == 1 and groups[0] == validation['full_prior']
            and groups[0]['status'] == 'complete' and groups[0]['n_seq'] == 256,
            'Full-prior comparison has not been independently validated')
    predictors = groups[0]['predictors']
    expected = {f'{arm}_seed{seed}' for arm in ARMS for seed in [0, 1]}
    require(set(predictors) == expected | {'bayes_hidden'}, 'Missing or selected scaling curves')
    original = next(g for g in baseline['groups'] if g['id'] == 'full_prior')['predictors']
    require(predictors['bayes_hidden'] == original['bayes_hidden'], 'Bayesian reference changed')
    require(set(data['model_provenance']) == {f'{arm}_seed{seed}' for arm in ARMS[1:] for seed in [0, 1]},
            'Missing scaling checkpoint identities')
    cases = []
    for arm in ARMS:
        for seed in [0, 1]:
            name = f'{arm}_seed{seed}'
            if arm == 'baseline':
                require(predictors[name] == original[f'seed{seed}'], 'Baseline predictions changed')
                checkpoint = data['baseline']['checkpoint_sha256'][str(seed)]
                steps, n_params, sequences = 100000, 531968, 6400000
            else:
                provenance = data['model_provenance'][name]
                spec = plan['arms'][arm]
                steps = spec['config']['steps']; n_params = spec['n_params']
                sequences = steps * spec['config']['batch']
                require(provenance['arm'] == arm and provenance['name'] == f'seed{seed}'
                        and provenance['plan_sha256'] == data['plan_sha256']
                        and provenance['cohort_id'] == data['cohort_id']
                        and provenance['checkpoint_step'] == steps
                        and provenance['completed_sequences'] == sequences
                        and provenance['n_params'] == n_params
                        and provenance['reference_records_sha256'] == data['reference_records_sha256'],
                        'Unexpected intervention checkpoint or training exposure')
                checkpoint = provenance['checkpoint_sha256']
            cases.append(dict(label=LABELS[arm], arm=arm, color=COLORS[arm],
                condition=dict(dimensions=[2, 8], arm=arm, training_seed=seed),
                checkpoint_sha256=checkpoint, checkpoint_step=steps, n_params=n_params,
                training_sequences=sequences, n_sequences=256, n_tokens=1025,
                transformer=checked_curve(predictors[name]['next_kl_bits'])))
    require(len({c['checkpoint_sha256'] for c in cases}) == 6, 'Repeated checkpoint in scaling comparison')
    panel = dict(id='c', title='Product source: hidden encoding', cases=cases,
        bayes=checked_curve(predictors['bayes_hidden']['next_kl_bits']), bayes_style='dashed black',
        cohort_id=data['cohort_id'], pilot=True, test_role='development',
        evaluated_sequences=256, planned_sequences=4096,
        quantity='Full-prior expected next-token regret; same unknown encoding for Bayes and all six transformers.',
        seed_rule='Training runs1/2 correspond to saved seeds0/1, shown in separate appendix panels. Main figure shows fixed run1. No pooling or selection by performance.',
        checkpoint_scope='Final planned checkpoints; recorded checkpoint identities validated, raw weight files not transferred.')
    return paths, data, panel


def draw_curve(ax, value, color, marker=None, linestyle='-', linewidth=1.1, alpha=.10):
    t, mean, se = [np.asarray(value[k]) for k in ['times', 'mean', 'se']]
    ax.fill_between(t, np.maximum(0, mean-se), mean+se, color=color, alpha=alpha, linewidth=0)
    ax.plot(t, mean, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker,
            markersize=3.2, markerfacecolor='white', markeredgewidth=.75)


def render(panel, out):
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8., 'axes.titlesize':8.5,
            'axes.spines.top':False, 'axes.spines.right':False,
            'pdf.fonttype':42, 'ps.fonttype':42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
        fig.subplots_adjust(left=.11, right=.975, bottom=.22, top=.70, wspace=.17)
        for seed, ax in enumerate(axes):
            for case in panel['cases']:
                if case['condition']['training_seed'] != seed:
                    continue
                draw_curve(ax, case['transformer'], case['color'])
            draw_curve(ax, panel['bayes'], '#222222', marker='o', linestyle='--', alpha=.06)
            ax.set_xscale('log', base=2); ax.set_xlim(.85, 1250)
            ax.set_xticks([1, 4, 16, 64, 256, 1024], ['1', '4', '16', '64', '256', '1024'])
            ax.minorticks_off(); ax.set_ylim(0, .9); ax.set_yticks([0, .3, .6, .9])
            ax.tick_params(labelsize=7., length=2.5, pad=2)
            ax.grid(axis='y', color='#D8DEE5', linewidth=.5); ax.set_axisbelow(True)
            ax.set_title(f'({chr(97+seed)}) Training run {seed+1}', loc='left', pad=5)
        handles = [Line2D([], [], color=COLORS['baseline'], linewidth=1.4),
                   Line2D([], [], color=COLORS['longsmall'], linewidth=1.4),
                   Line2D([], [], color=COLORS['large'], linewidth=1.4),
                   Line2D([], [], color='#222222', linewidth=1.2, linestyle='--',
                          marker='o', markerfacecolor='white', markersize=3.2)]
        fig.legend(handles, ['Original', 'More training', 'Larger model', 'Informed Bayes'],
                   loc='upper center', bbox_to_anchor=(.54, 1.01), ncol=2, frameon=False,
                   fontsize=7.5, handlelength=2., columnspacing=1.5, labelspacing=.45)
        fig.supylabel('Expected next-token regret (bits)', x=.008, fontsize=8.)
        fig.supxlabel('Observed tokens, $t$', y=.095, fontsize=8.)
        fig.text(.54, .018, 'Same 256 test sequences; shading shows ±1 standard error.',
                 ha='center', fontsize=7.)
        from matplotlib.text import Text
        require(all(t.get_fontsize() >= 7 for t in fig.findobj(Text) if t.get_text()), 'Text below 7-point minimum')
        for ext in ['pdf', 'png']:
            kwargs = {'metadata':{'CreationDate':None, 'ModDate':None}} if ext == 'pdf' else {'dpi':260}
            fig.savefig(out / f'next_token_regret.{ext}', **kwargs)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=SOURCE / 'figures')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); paths, data, panel = load_scaling(args.source)
    receipt = dict(schema='hidden-factor-scaling-figure-v1', passed=True, script_sha256=sha(__file__),
        inputs={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths},
        full_prior=next(g for g in data['groups'] if g['id']=='full_prior'),
        panel=panel, test_role='development', uncertainty='±1 standard error across256 independent sequences; each training seed separate.',
        display=dict(size_inches=[5.5,2.8], minimum_font_points=7, transformer_lines='solid',
            source_title='Product source: hidden encoding',
            transformer_markers=False, training_seed_by_panel=[0,1], curves_per_panel=4,
            panel_titles=['Training run 1','Training run 2'], run_to_seed={'1':0,'2':1},
            common_bayes='one dashed black curve with open circles per panel',
            arm_colors=COLORS, arm_labels={'baseline':'Original','longsmall':'More training','large':'Larger model'},
            common_legend=['Original','More training','Larger model','Informed Bayes'], smoothing=False),
        new_inference=False, new_bayesian_computation=False)
    names=['next_token_regret.pdf','next_token_regret.png']
    if args.check_only:
        saved=json.loads((args.out/'receipt.json').read_text()); hashes=saved.pop('outputs')
        require(saved==receipt and hashes=={n:sha(args.out/n) for n in names}, 'Scaling figure receipt or files changed')
        print('Verified audited scaling data, separate seeds, figure scope and output hashes.'); return
    args.out.mkdir(parents=True,exist_ok=True); render(panel,args.out)
    receipt['outputs']={n:sha(args.out/n) for n in names}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(args.out/'next_token_regret.pdf')


if __name__=='__main__': main()

"""Compact Section 3 overview, using saved results and explicit source metadata.

This presentation step does not draw sequences, run models, or compute Bayes
references. Pooled long-context estimates are displayed for author review;
their incomplete numerical-validation status is retained in source metadata.
Run with --check-only to verify exact data, selection, code and output hashes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import shared_base_conditional_figures as shared_full
from shared_base_conditional_figures import read_panels
from hidden_factor_campaign_results import load_campaign as load_hidden_campaign
from shared_base_campaign_results import load_campaign as load_shared_campaign
from shared_base_full_cohort_results import load_full_cohort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'results/transformer/section3_compact'
DEPTHS = [0, 1, 4, 8]
COLORS = ['#2468A0', '#BF611C', '#288167', '#8A4E98']
DEPTH_COLORS = dict(zip(DEPTHS, COLORS))
UNSTRUCTURED = '#666666'
SIZE = (5.5, 4.35)
MIN_FONT = 7.0
MIXTURE_SOURCE = ROOT / 'results/transformer/multifamily_bayes_2026-09-16/next_token_regret.json'
MIXTURE_VALIDATION = MIXTURE_SOURCE.with_name('next_token_validation.json')
MIXTURE_LONG = ROOT / 'results/transformer/pooled_estimates_2026-09-24/points.json.gz'
SHARED_TIME64 = ROOT / 'results/transformer/shared_base_time64_complete_2026-09-17/summary.json'
LARGER = ROOT / 'results/transformer/larger_models_2026-09-23'


def larger_release():
    with gzip.open(LARGER / 'manifest.json.gz', 'rt') as f:
        manifest = json.load(f)
    require(manifest['schema'] == 'larger-transformer-release-v1' and manifest['passed'] is True,
            'Larger-model audit failed')
    paths = [LARGER / 'manifest.json.gz']
    for name, digest in manifest['files'].items():
        path = LARGER / name
        require(sha(path) == digest, 'Larger-model release changed')
        paths.append(path)
    audit_path = ROOT / 'code/scripts/release_large_transformer_results.py'
    with gzip.open(LARGER / 'provenance.json.gz', 'rt') as f:
        audit = json.load(f)
    require(audit['passed'] is True and audit['validation_script'] == audit_path.read_text(),
            'Larger-model validation implementation changed')
    paths.append(audit_path)
    with np.load(LARGER / 'curves.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return paths, arrays


def add_larger(case, data, family, group):
    require(np.array_equal(case['transformer']['times'], data[family + '_times']),
            'Larger-model time grid differs')
    require(np.allclose(case['transformer']['mean'], data[family + '_small_mean'][group],
                        atol=2e-12, rtol=0), 'Larger-model baseline cohort differs')
    require(np.allclose(case['bayes']['mean'], data[family + '_bayes_mean'][group],
                        atol=2e-12, rtol=0), 'Larger-model Bayesian cohort differs')
    require(case['n_sequences'] == data[family + '_counts'][group], 'Larger-model cohort size differs')
    case['original_transformer'] = case['transformer']
    case['larger_runs'] = [curve(data[family + '_times'], dict(
        mean=data[family + '_mean'][seed, group], se=data[family + '_se'][seed, group])) for seed in (0, 1)]
    case['transformer'] = case['larger_runs'][0]
    case['condition'].update(training_seed=0, larger_layers=4, larger_training_sequences=6400000)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def saved_summary(name):
    base = ROOT / 'results/transformer/conditional_accuracy' / name
    paths = [base / 'summary.json', base / 'receipt.json']
    data, receipt = [json.loads(p.read_text()) for p in paths]
    require(receipt['passed'] is True, f'{name}: validation receipt failed')
    if name == 'product':
        require(receipt['validation']['passed'] is True, 'Product numerical audit failed')
        digest = receipt['files_sha256'][relative(paths[0])]
    else:
        require(data['passed'] is True, f'{name}: summary validation failed')
        digest = receipt['outputs']['summary.json']
    require(sha(paths[0]) == digest, f'{name}: validated summary has changed')
    return paths, data


def curve(times, moments):
    t, mean, se = [np.asarray(x) for x in (times, moments['mean'], moments['se'])]
    require(t.ndim == mean.ndim == se.ndim == 1 and len(t) == len(mean) == len(se),
            'Inconsistent curve lengths')
    require(np.all(np.diff(t) > 0) and np.all(t >= 1), 'Invalid time grid')
    require(np.isfinite(mean).all() and np.isfinite(se).all(), 'Nonfinite displayed statistic')
    require(np.all(mean >= 0) and np.all(se >= 0), 'Negative regret or uncertainty')
    return dict(times=t.tolist(), mean=mean.tolist(), se=se.tolist())


def selected_case(case, label, color):
    require(case['n_sequences'] >= 100 and case['n_tokens'] in (257, 1025),
            'Unexpected cohort size or sequence length')
    require(case['times'][0] == 1 and case['times'][-1] == case['n_tokens'] - 1,
            'Incomplete displayed context range')
    return dict(label=label, color=color, n_sequences=case['n_sequences'],
                n_tokens=case['n_tokens'],
                transformer=curve(case['times'], case['transformer']),
                bayes=curve(case['times'], case['informed_bayes']))


def hidden_panel():
    return load_hidden_campaign()[2]


def pooled_panel():
    data = json.loads(MIXTURE_SOURCE.read_text())
    validation = json.loads(MIXTURE_VALIDATION.read_text())
    require(data['schema'] == 'pooled-next-token-regret-v1', 'Unexpected mixture schema')
    require(validation['passed'] is True, 'Mixture probability validation failed')
    require(data['validation_receipt']['sha256'] == sha(MIXTURE_VALIDATION),
            'Mixture validation receipt changed')
    payload = {k: v for k, v in data.items() if k != 'validation_receipt'}
    require(hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
            == validation['summary_payload_sha256'], 'Validated next-token summary changed')
    require(data['bayes_times'] == validation['reference_times'] == [1, 4],
            'Direct mixture coverage changed; update this presentation deliberately')
    require(data['transformer_times'] == [1, 4, 16, 64, 256, 1024], 'Pooled time grid changed')
    require(data['source_plan_id'] == validation['source_plan_id'], 'Mixture cohort mismatch')
    require(data['family_prior_weights'] == [1/3] * 3, 'Mixture prior changed')
    require(data['n_sequences'] == validation['independent_sequences'] == 3072
            and data['n_sequences_per_family'] == 1024 and data['n_tokens'] == 1025,
            'Mixture test count changed')
    def check_average(value, individual):
        require(value['n_seq'] == 3072 and all(v['n_seq'] == 1024 for v in individual),
                'Family cohort count changed')
        require(np.allclose(value['mean'], np.mean([c['mean'] for c in individual], axis=0),
                            atol=1e-12, rtol=0), 'Mixture mean differs from family averages')
        require(np.allclose(value['se'], np.sqrt(np.sum(np.square([c['se'] for c in individual]),
                            axis=0))/3, atol=1e-12, rtol=0), 'Mixture SE differs from family averages')
    check_average(data['bayes'], [c['bayes'] for c in data['cohorts'].values()])
    cases = []
    for seed in (0, 1):
        name = f'pooled_seed{seed}'
        value = data['models'][name]['next_token_regret']
        check_average(value, [c['models'][name]['next_token_regret'] for c in data['cohorts'].values()])
        cases.append(dict(label=f'Training run {seed+1}', color=COLORS[0],
                          condition=dict(training_seed=seed), n_sequences=value['n_seq'],
                          transformer=curve(data['transformer_times'], value)))
    with gzip.open(MIXTURE_LONG, 'rt') as handle:
        long = json.load(handle)
    require(long['author_review_display_authorized'] is True, 'Draft display not authorized')
    points = long['points']
    require([p['context'] for p in points] == [16, 64, 256, 1024], 'Long-context coverage changed')
    require(all(p['full_cohort'] == 192 and p['n_per_family'] == 64 for p in points),
            'Long-context estimates must retain the complete frozen cohort')
    bayes = curve(data['bayes_times'] + [p['context'] for p in points],
                  dict(mean=data['bayes']['mean'] + [p['mean_bits'] for p in points],
                       se=data['bayes']['se'] + [p['source_se_bits'] for p in points]))
    return dict(id='f', title='Pooled sources', cases=cases,
                bayes=bayes, bayes_style='points only',
                bayes_sequence_counts=[3072, 3072] + [p['full_cohort'] for p in points],
                pending_numerical_validation_times=[p['context'] for p in points
                                                   if not p['scientific_reference_validated']],
                quantity='Expected next-token regret: KL(true next-token distribution || predictor), in bits.',
                scored_tokens='At time t, predict token t+1 given tokens 1 through t.',
                family_weights='Posterior weights proportional to each family likelihood of the observed tokens.',
                family_prior_weights=data['family_prior_weights'],
                uncertainty='Equal-family mean; one stratified standard error across independent sequences.',
                long_context_bayesian_comparison='Frozen complete-cohort estimates; numerical validation tracked separately')


def collect():
    files, panels = [], []
    larger_paths, larger = larger_release(); files.extend(larger_paths)
    paths, coins = saved_summary('coins'); files.extend(paths)
    require(coins['schema'] == 'conditional-coin-accuracy-v1', 'Unexpected coin schema')
    selections = [('twopoint', .2, r'Prior $\{0.2,0.8\}$; test $\theta=0.2$'),
                  ('uniform', .1, r'Uniform $[0,1]$; test $\theta=0.1$'),
                  ('uniform', .5, r'Uniform $[0,1]$; test $\theta=0.5$')]
    cases = []
    for (prior, theta, label), color in zip(selections, COLORS):
        found = [c for c in coins['cases'] if c['prior'] == prior and c['theta'] == theta]
        require(len(found) == 1 and found[0]['validation']['passed'], 'Missing validated coin condition')
        case = selected_case(found[0], label, color)
        case['condition'] = dict(prior=prior, theta=theta)
        cases.append(case)
    panels.append(dict(id='a', title='Coins', cases=cases, bayes_style='points only'))

    paths, product = saved_summary('product'); files.extend(paths)
    require(product['schema'] == 'conditional-product-accuracy-v1', 'Unexpected product schema')
    pairs = [[1, 1], [1, 8], [4, 4], [8, 8]]
    require([c['depth_pair'] for c in product['cases']] == pairs, 'Product depth pairs changed')
    cases = []
    for group, (row, color) in enumerate(zip(product['cases'], COLORS)):
        a, b = row['depth_pair']
        case = selected_case(row, f'({a},{b})', color)
        case['condition'] = dict(depth_pair=[a, b])
        add_larger(case, larger, 'product', group)
        cases.append(case)
    panels.append(dict(id='b', title='Product source: known encoding', cases=cases, bayes_style='points only'))

    hidden_paths, _, hidden = load_hidden_campaign()
    # Bayes for the same 3840 sequences without the product structure: a 16-state
    # chain with independent rows under a uniform prior (flat_dirichlet in the
    # validated comparison file that load_hidden_campaign has already checked).
    require(hidden_paths[0].name == 'comparison.json', 'Unexpected hidden-factor input order')
    comparison = json.loads(hidden_paths[0].read_text())
    group = next(g for g in comparison['groups'] if g['id'] == 'full_prior')
    flat = group['predictors']['flat_dirichlet']['next_kl_bits']
    require(flat['n_seq'] == 3840 and comparison['times'] == [1, 4, 16, 64, 256, 1024],
            'Unexpected cohort or time grid of the unstructured Bayesian predictor')
    hidden['bayes_unstructured'] = curve(comparison['times'], flat)
    hidden['bayes_unstructured_definition'] = ('Bayesian predictor that ignores the factorization: '
        '16-state chain, independent rows, uniform (flat Dirichlet) prior; flat_dirichlet in comparison.json.')
    panels.append(hidden)
    files.extend(hidden_paths)

    source = ROOT / 'results/transformer/shared_base_conditional_curves'
    paths, shared = read_panels(source / 'time16_summary.json', SHARED_TIME64); files.extend(paths)
    # Reuse the complete independent time16/time64 cohort checks in read_panels, and
    # bind every numerical input to the receipt of the existing validated plot.
    existing_receipt = source / 'figures_t64_complete/receipt.json'
    receipt = json.loads(existing_receipt.read_text()); files.append(existing_receipt)
    require(receipt['passed'] is True, 'Shared-base figure receipt failed')
    for path in paths:
        require(receipt['inputs'][relative(path)] == sha(path), 'Shared-base validated input changed')
    # The capacity comparison retains these exact original curves and references;
    # show both model sizes from the fixed first run in the main panel.
    scaling_paths, scaling_data, scaling_panels = load_shared_campaign()
    files.extend(scaling_paths)
    require(all(p['larger_updates'] == 400000 and not p['larger_interim'] for p in scaling_panels),
            'Update the caption deliberately before showing another campaign checkpoint')
    full_paths, full_data, long_curves = load_full_cohort(
        ROOT / 'results/transformer/shared_base_full_cohort_2026-09-23')
    files.extend(full_paths)
    require({(int(d), t) for d, c in long_curves.items() for t in c['times']}
            == {(d, t) for d in DEPTHS for t in (256, 1024)},
            'Update the caption for new validated reference coverage')
    cases = []
    for row in shared:
        depth = row['depth']
        require(row['bayes']['times'] == ([1, 4, 16, 64] if 'time64_numerical_validation' in row else [1, 4, 16]),
                'Shared-base Bayesian coverage changed')
        bayes = curve(row['bayes']['times'], row['bayes'])
        extra = long_curves.get(depth, long_curves.get(str(depth)))
        if extra:
            require(all(t > 64 for t in extra['times']), 'Unexpected long-context reference')
            for key in ('times', 'mean', 'se'):
                bayes[key].extend(extra[key])
        cases.append(dict(label=f'$L={depth}$', color=DEPTH_COLORS[depth],
            condition=dict(alpha=1, depth=depth, training_seed=0, model='larger'),
            n_sequences=row['n_sequences'], n_tokens=row['n_tokens'],
            transformer=next(p['larger'] for p in scaling_panels
                             if p['depth'] == depth and p['training_seed'] == 0),
            original_transformer=curve(row['times'], row['transformers']['d64']),
            bayes=bayes))
    shared_panel = dict(id='e', title='Shared-base LSA', cases=cases,
                       bayes_style='points only',
                       capacity_comparison_plan_sha256=scaling_data['plan_sha256'],
                       training_sequences=25600000, n_params=25809920,
                       checkpoint_step=400000, final_checkpoint=True,
                       full_cohort_reference_schema=full_data['schema'],
                       bayes_pending_times_by_depth={str(c['condition']['depth']):
                           [t for t in [1, 4, 16, 64, 256, 1024] if t not in c['bayes']['times']]
                           for c in cases})
    time64_depths=[c['condition']['depth'] for c in cases if 64 in c['bayes']['times']]
    shared_coverage='Open circles only: times 1, 4, 16 at every depth'
    if time64_depths==DEPTHS:
        shared_coverage='Open circles only: times 1, 4, 16, 64 at every depth.'
    elif time64_depths:
        labels=[str(d) for d in time64_depths]
        shared_coverage+='; time 64 additionally at depths '+(', '.join(labels[:-1])+' and '+labels[-1] if len(labels)>1 else labels[0])+'.'
    else:
        shared_coverage+='.'
    shared_coverage += ' Complete validated 256-sequence groups add t256 and t1024 at every depth.'

    paths, uniform = saved_summary('uniform_base'); files.extend(paths)
    require(uniform['schema'] == 'conditional-uniform-accuracy-v1', 'Unexpected uniform schema')
    require(uniform['selected_depths'] == DEPTHS, 'Uniform selected depths changed')
    cases = []
    for depth in DEPTHS:
        found = [c for c in uniform['panels'] if c['details']['source_depth'] == depth]
        require(len(found) == 1, 'Missing uniform depth')
        case = selected_case(found[0], f'$L={depth}$', DEPTH_COLORS[depth])
        case['condition'] = dict(alpha=0, depth=depth)
        add_larger(case, larger, 'uniform_base', depth)
        cases.append(case)
    panels.append(dict(id='d', title='Uniform-base control', cases=cases,
                       bayes_style='points only'))
    panels.append(shared_panel)
    pooled = pooled_panel()
    for seed, case in enumerate(pooled['cases']):
        require(np.allclose(case['transformer']['mean'], larger['pooled_small_mean'][seed], atol=2e-12, rtol=0)
                and np.allclose(case['transformer']['se'], larger['pooled_small_se'][seed], atol=2e-12, rtol=0),
                'Pooled larger-model comparison uses different sequences')
        case['original_transformer'] = case['transformer']
        case['transformer'] = curve(larger['pooled_times'], dict(mean=larger['pooled_mean'][seed], se=larger['pooled_se'][seed]))
        case['condition'].update(larger_layers=4, larger_training_sequences=19200000)
    panels.append(pooled)
    files.extend([MIXTURE_SOURCE, MIXTURE_VALIDATION, MIXTURE_LONG])
    require([p['id'] for p in panels] == list('abcdef'), 'Panel ordering changed')
    require(sum(len(p['cases']) for p in panels[:5] if p['id'] != 'c') == 15,
            'Missing or additional existing source conditions')
    return dict(schema='section3-compact-figure-v1', passed=True,
        all_bayesian_points_numerically_validated=False, script_sha256=sha(__file__),
        helper_sha256=sha(ROOT / 'code/scripts/shared_base_conditional_figures.py'),
        hidden_helper_sha256=sha(ROOT / 'code/scripts/hidden_factor_campaign_results.py'),
        shared_scaling_helper_sha256=sha(ROOT / 'code/scripts/shared_base_campaign_results.py'),
        shared_full_helper_sha256=sha(ROOT / 'code/scripts/shared_base_full_cohort_results.py'),
        inputs={relative(p): sha(p) for p in files}, panels=panels,
        quantity_a_to_f='Expected next-token regret in bits: KL(true next-token distribution || predictor).',
        uncertainty='Plus/minus one standard error across independent complete test sequences. No training-seed pooling.',
        display=dict(size_inches=list(SIZE), minimum_font_points=MIN_FONT,
            transformer_style='Two-layer transformers solid; larger transformers dashed. Four layers in b,d,f; four/eight in c; eight in e.',
            x_ticks='1, 32, 1024 on the base-2 axes to 1024; 1, 16, 256 in panel a, whose data end at t=256',
            run_labels='"Training run 1" at the upper right of panels b through f; both larger runs separately in appendix.',
            dense_bayes_style='unconnected open circles with error bars',
            shared_base_bayes_style=shared_coverage,
            shared_base_transformer='Two-layer100k solid and final eight-layer400k dashed, fixed training run1 (seed0),6.4M and25.6M training sequences. Both training runs separately in appendix; no performance selection.',
            uniform_base_inset=False,
            hidden_factor_style='Training seed0 only: two-layer100k solid, final four-layer400k and final eight-layer400k dashed; the two larger models retain their distinct colors. Informed Bayes is shown by unconnected black open circles; the Bayesian predictor that ignores the factorization (legend: Unstructured Bayes) by a gray dash-dot curve. Same3840 previously examined comparison sequences; no fresh confirmation claimed. Both seeds retained separately.',
            hidden_factor_training_seed=0,
            hidden_factor_legend={'baseline': '2 layers: 6.4M', 'continue_large': '4 layers: 25.6M', 'xlarge': '8 layers: 25.6M'},
            depth_colors={str(k): v for k, v in DEPTH_COLORS.items()},
            pooled_panel='Two-layer solid and four-layer dashed, matched training run1 at all six times. Informed Bayesian mixture at times1,4,16,64,256,1024, black open circles only. Both runs separately in appendix. Transformer and short Bayes use3072 sequences; long Bayes uses192. Equal-family averages; one stratified source standard error.',
            pooled_training_seed=0,
            connecting_lines='Transformer estimates are joined by straight visual guides; Bayesian estimates are unconnected points.'),
        new_sequences=False, new_model_inference=False, new_bayesian_computation=False,
        time64_raw_audit_rehash=shared_full.RAW_AUDIT_REHASH)


def draw_curves(ax, cases, sparse=False):
    for case in cases:
        color = case['color']
        names = ['original_transformer', 'transformer', 'bayes'] if 'original_transformer' in case else ['transformer', 'bayes']
        for name in names:
            value = case[name]
            t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
            if name == 'bayes':
                ax.errorbar(t, mean, yerr=se, color=color, linestyle='none', marker='o',
                            markersize=4.0, markerfacecolor='white', markeredgewidth=.8,
                            elinewidth=.7, capsize=1.4, zorder=5)
            else:
                ax.fill_between(t, np.maximum(0, mean-se), mean+se, color=color,
                                alpha=.12 if name == 'transformer' else .06, linewidth=0)
                linestyle = '-' if name == 'original_transformer' else '--' if name == 'transformer' and 'original_transformer' in case else '-'
                ax.plot(t, mean, color=color, linestyle=linestyle,
                        linewidth=1.05, marker='o' if name == 'bayes' else None,
                        markersize=3., markerfacecolor='white',
                        markeredgewidth=.65, zorder=4 if name == 'bayes' else 3)


def draw_hidden(ax, panel):
    """Show the fixed seed0 comparison: three models plus common Bayes."""
    cases = [c for c in panel['cases'] if c['condition']['training_seed'] == 0]
    require([c['arm'] for c in cases] == ['baseline', 'continue_large', 'xlarge'],
            'Hidden-factor display must contain exactly three seed0 models')
    for case in cases:
        value = case['transformer']
        t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
        ax.fill_between(t, np.maximum(0, mean-se), mean+se, color=case['color'], alpha=.07, linewidth=0)
        linestyle = '-' if case['arm'] == 'baseline' else '--'
        ax.plot(t, mean, color=case['color'], linewidth=1.15, linestyle=linestyle)
    value = panel['bayes_unstructured']
    t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
    ax.fill_between(t, np.maximum(0, mean-se), mean+se, color=UNSTRUCTURED, alpha=.07, linewidth=0)
    ax.plot(t, mean, color=UNSTRUCTURED, linewidth=1.05, linestyle='-.', zorder=2)
    value = panel['bayes']
    t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
    ax.errorbar(t, mean, yerr=se, color='#333333', linestyle='none', marker='o',
                markersize=3., markerfacecolor='white', markeredgewidth=.65,
                elinewidth=.7, capsize=1.4, zorder=4)


def run_label(ax):
    """Name the training run in a corner that the curves leave free (upper right)."""
    ax.text(.97, .95, 'Training run 1', transform=ax.transAxes, fontsize=7., ha='right', va='top')


def axis_style(ax, end, ylim, yticks):
    ax.set_xscale('log', base=2)
    ax.set_xlim(.87, end * 1.35)
    ticks = [1, 16, 256] if end == 256 else [1, 32, 1024]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#D8DEE5', linewidth=.45)


def condition_legend(ax, panel):
    cases = panel['cases'] if panel['id'] != 'c' else [c for c in panel['cases'] if c['condition']['training_seed'] == 0]
    handles = [Line2D([], [], color=c['color'], linewidth=2,
                      linestyle='--' if panel['id'] == 'c' and c['arm'] != 'baseline' else '-')
               for c in cases]
    labels = [c['label'] for c in cases]
    if panel['id'] == 'c':
        labels = [{'baseline': '2 layers: 6.4M', 'continue_large': '4 layers: 25.6M',
                   'xlarge': '8 layers: 25.6M'}[c['arm']] for c in cases]
    # Labels occupy a fixed strip above the data, never the curves themselves.
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(-.03, 1.015),
              borderaxespad=0, frameon=False, ncol=1 if panel['id'] in ('a','c') else 2, fontsize=7.0,
              columnspacing=.7, handletextpad=.4, handlelength=.9, labelspacing=.12)


def render(receipt, out):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.,
            'axes.titlesize': 8., 'axes.labelsize': 8.,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(2, 3, figsize=SIZE)
        fig.subplots_adjust(left=.098, right=.965, bottom=.105, top=.78,
                            wspace=.42, hspace=.77)
        settings = [((0, .32), [0, .1, .2, .3]), ((0, 3.9), [0, 1, 2, 3]),
                    ((0, .9), [0, .3, .6, .9]), ((0, 3.9), [0, 1, 2, 3]),
                    ((0, 3.9), [0, 1, 2, 3]), ((0, 3.0), [0, 1, 2, 3])]
        for i, (ax, panel) in enumerate(zip(axes.flat, receipt['panels'])):
            title = {'b':'Product source\nKnown encoding',
                     'c':'Product source\nHidden encoding'}.get(panel['id'], panel['title'])
            ax.set_title(f"({panel['id']}) {title}", loc='left', pad=31)
            ylim, yticks = settings[i]
            end = 256 if panel['id'] == 'a' else 1024
            axis_style(ax, end, ylim, yticks)
            if panel['id'] == 'f':
                draw_pooled(ax, panel, 0)
                run_label(ax)
                continue
            if panel['id'] == 'c':
                values = [c['transformer'] for c in panel['cases']] + [panel['bayes'], panel['bayes_unstructured']]
                require(all(max(np.array(v['mean']) + v['se']) < ylim[1] for v in values),
                        'Hidden-factor panel clips uncertainty')
                draw_hidden(ax, panel)
                condition_legend(ax, panel)
                run_label(ax)
                continue
            for case in panel['cases']:
                predictors = ['transformer', 'bayes'] + (['original_transformer'] if 'original_transformer' in case else [])
                for predictor in predictors:
                    value = case[predictor]
                    require(max(np.array(value['mean']) + value['se']) < ylim[1], 'Figure clips uncertainty')
            sparse = panel['bayes_style'] == 'points only'
            draw_curves(ax, panel['cases'], sparse=sparse)
            condition_legend(ax, panel)
            if panel['id'] in ('b', 'd', 'e'):
                run_label(ax)
        predictor_handles = [Line2D([], [], color='#333333', linewidth=1.2),
            Line2D([], [], color='#333333', linewidth=1.2, linestyle='--'),
            Line2D([], [], color='#333333', linestyle='none', marker='o',
                   markersize=3.5, markerfacecolor='white'),
            Line2D([], [], color=UNSTRUCTURED, linewidth=1.2, linestyle='-.')]
        fig.legend(predictor_handles, ['Two-layer transformer', 'Larger transformers', 'Informed Bayes',
                                       'Unstructured Bayes'],
                   loc='upper center', bbox_to_anchor=(.5, .997), ncol=4,
                   frameon=False, fontsize=7.0, handlelength=1.7, columnspacing=1.0)
        fig.supylabel('Expected next-token regret (bits)', x=.006, fontsize=8.)
        fig.supxlabel('Observed tokens, $t$', y=.008, fontsize=8.)
        fig.canvas.draw()
        # Explicitly guard the nominal font floor; the PDF is included at its
        # native 5.5-inch width in the manuscript, not downscaled from a poster.
        from matplotlib.text import Text
        require(all(x.get_fontsize() >= MIN_FONT for x in fig.findobj(Text) if x.get_text()),
                'Figure contains text below the legibility floor')
        fig.savefig(out / 'overview.pdf', metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(out / 'overview.png', dpi=260, metadata={'Software': 'section3_compact_figure.py'})
        plt.close(fig)


def draw_pooled(ax, panel, seed):
    case = next(c for c in panel['cases'] if c['condition']['training_seed'] == seed)
    for name, style in (('original_transformer', '-'), ('transformer', '--')):
        value = case[name]
        t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
        ax.fill_between(t, mean-se, mean+se, color=case['color'], alpha=.08, linewidth=0)
        ax.plot(t, mean, color=case['color'], linestyle=style, linewidth=1.2)
    value = panel['bayes']
    t, mean, se = [np.asarray(value[k]) for k in ('times', 'mean', 'se')]
    ax.errorbar(t, mean, yerr=se, color='#222222', linestyle='none', marker='o',
                markersize=4., markerfacecolor='white', markeredgewidth=.85,
                elinewidth=.75, capsize=1.6, zorder=5)


def render_pooled_runs(receipt, out):
    panel = receipt['panels'][-1]
    require(panel['id'] == 'f', 'Missing pooled panel')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65), sharey=True)
        fig.subplots_adjust(left=.11, right=.98, bottom=.23, top=.75, wspace=.20)
        for seed, ax in enumerate(axes):
            axis_style(ax, 1024, (0, 3), [0, 1, 2, 3])
            ax.set_xticks([1, 4, 16, 64, 256, 1024], ['1', '4', '16', '64', '256', '1024'])
            draw_pooled(ax, panel, seed)
            ax.set_title(f'Training run {seed+1}', fontsize=9., pad=7)
        handles = [Line2D([], [], color=COLORS[0], linewidth=1.3),
                   Line2D([], [], color=COLORS[0], linewidth=1.3, linestyle='--'),
                   Line2D([], [], color='#222222', linestyle='none', marker='o',
                          markersize=4., markerfacecolor='white')]
        fig.legend(handles, ['2 layers: 19.2M', '4 layers: 19.2M', 'Informed Bayes'],
                   loc='upper center', bbox_to_anchor=(.54, .995), ncol=3,
                   frameon=False, fontsize=8., handlelength=1.8, columnspacing=1.5)
        fig.supylabel('Expected next-token regret (bits)', x=.012, fontsize=8.)
        fig.supxlabel('Observed tokens, $t$', y=.08, fontsize=8.)
        fig.text(.55, .015, '1024 histories/family; long-context Bayes: 64/family. Bars: ±1 source SE.',
                 ha='center', fontsize=7.)
        fig.savefig(out / 'pooled_runs.pdf', metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(out / 'pooled_runs.png', dpi=260, metadata={'Software': 'section3_compact_figure.py'})
        plt.close(fig)


def render_specialist_runs(receipt, out):
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8.,
            'axes.spines.top':False, 'axes.spines.right':False,
            'axes.linewidth':.6, 'pdf.fonttype':42, 'ps.fonttype':42}):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8))
        fig.subplots_adjust(left=.11, right=.98, bottom=.13, top=.83, wspace=.25, hspace=.65)
        for row, ident in enumerate(('b','d')):
            panel = next(p for p in receipt['panels'] if p['id']==ident)
            for seed, ax in enumerate(axes[row]):
                cases = [dict(c, transformer=c['larger_runs'][seed]) for c in panel['cases']]
                axis_style(ax, 1024, (0,3.9), [0,1,2,3])
                draw_curves(ax, cases)
                ax.set_title(('Known-encoding product' if ident=='b' else 'Uniform-base control')
                             + f', run {seed+1}', loc='left', fontsize=8., pad=28)
                condition_legend(ax, dict(panel, cases=cases))
        handles = [Line2D([],[],color='#333333',linewidth=1.2),
                   Line2D([],[],color='#333333',linewidth=1.2,linestyle='--'),
                   Line2D([],[],color='#333333',linestyle='none',marker='o',markerfacecolor='white')]
        fig.legend(handles,['2 layers: 6.4M','4 layers: 6.4M','Informed Bayes'],
                   loc='upper center',bbox_to_anchor=(.54,.995),ncol=3,frameon=False,fontsize=8.)
        fig.supylabel('Expected next-token regret (bits)',x=.012,fontsize=8.)
        fig.supxlabel('Observed tokens, $t$',y=.065,fontsize=8.)
        fig.text(.54,.015,'Both larger runs use the same two-layer baseline and test sequences.',
                 ha='center',fontsize=7.)
        for ext in ('pdf','png'):
            fig.savefig(out/f'specialist_runs.{ext}',dpi=260)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    receipt = collect()
    names = ['overview.pdf', 'overview.png', 'pooled_runs.pdf', 'pooled_runs.png',
             'specialist_runs.pdf', 'specialist_runs.png']
    if args.check_only:
        saved = json.loads((args.out / 'receipt.json').read_text())
        outputs = saved.pop('outputs')
        require(saved == receipt, 'Compact figure receipt differs from source data or plotting code')
        require(outputs == {name: sha(args.out / name) for name in names}, 'Figure outputs have changed')
        print('Verified saved source values, selected curves, recorded Bayes status and artifact hashes.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    render(receipt, args.out)
    render_pooled_runs(receipt, args.out)
    render_specialist_runs(receipt, args.out)
    receipt['outputs'] = {name: sha(args.out / name) for name in names}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(args.out / 'overview.pdf')


if __name__ == '__main__':
    main()

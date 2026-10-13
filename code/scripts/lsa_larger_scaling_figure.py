"""Plot larger-alphabet LSA mean-gap crossings from existing archived runs.

OPENBLAS_NUM_THREADS=1 python code/scripts/lsa_larger_scaling_figure.py
Smoke: python code/scripts/lsa_larger_scaling_figure.py --bootstrap 200 --out /tmp/lsa_larger_scaling.json --figure /tmp/lsa_larger_scaling.pdf

This reads 16 settings and 256 saved runs; it generates no source tables,
paths, Bayes costs, or new LSA evaluations. Crossings are roots of the mean
cumulative code gap on each setting's common checkpoint grid. Dashed fits
summarize the four available alphabet sizes, not an asymptotic law.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from crossing_count_prediction_audit import archived_paths, digest, root, summarize_roots


ROOT = Path(__file__).resolve().parents[2]
DEPTHS, ALPHAS, DS = (3, 6), (.75, 1.), (512, 1024, 2048, 4096)
GAMMA = .5772156649015329


def load_cell(L, alpha, d, args):
    stem = f'd{d}_a{alpha}_L{L}'
    path = args.archive/stem/f'summary_{stem}.json'
    if not path.exists():
        raise FileNotFoundError(f'Required archived setting is absent: {path}')
    metadata = json.loads(path.read_text())
    assert (metadata['d'], metadata['alpha'], metadata['L']) == (d, alpha, L)
    assert len(metadata['seeds']) == len(set(metadata['seeds'])) == 16
    assert metadata['n_censored'] == 0 and metadata['mean_lambda_crossing'] is not None
    lmax = round(2/(1-GAMMA)*math.log(d))
    cell = dict(L=L, alpha=alpha, d=d, l_max=lmax,
        archive_path=str(path.relative_to(ROOT)), archive_sha256=digest(path),
        archived_mean_curve_crossing=metadata['mean_lambda_crossing'])
    ns, gaps, hashes = archived_paths(cell)
    assert gaps.shape == (16, len(ns))
    horizons, estimators = [], set()
    for seed in metadata['seeds']:
        raw = json.loads((path.parent/f'{stem}_seed{seed}.json').read_text())
        assert raw['censored'] is False and raw['l_max'] == lmax
        assert raw['checkpoints'] == sorted(set(raw['checkpoints']))
        assert raw['checkpoints'][-1] == raw['horizon']
        horizons.append(raw['horizon'])
        estimators.add(raw['sequence_estimator'])
    assert len(estimators) == 1
    mean = gaps.mean(0)
    value, status, upcrossings = root(ns, mean)
    assert status == 'located'
    # A second arithmetic route checks that the reported root averages runs
    # before locating the crossing, rather than averaging their path roots.
    independently_averaged = np.array([math.fsum(col)/16 for col in gaps.T])
    independent_root = root(ns, independently_averaged)[0]
    assert math.isclose(value, independent_root, rel_tol=1e-12, abs_tol=1e-8)
    rng = np.random.default_rng([args.seed, L, round(100*alpha), d])
    weights = rng.multinomial(16, np.full(16, 1/16), size=args.bootstrap)/16
    roots = [root(ns, row) for row in weights@gaps]
    bootstrap = summarize_roots([r[0] for r in roots], [r[1] for r in roots])
    fraction = bootstrap['null']/args.bootstrap
    eligible = bootstrap['located']/args.bootstrap >= .95
    quantiles = bootstrap['conditional_on_located_quantiles']
    interval = [quantiles[0], quantiles[2]] if eligible else None
    return dict(**cell, repetitions=16, seeds=metadata['seeds'], source_files_sha256=hashes,
        source_sequence_estimator=next(iter(estimators)), censored_paths=0,
        stored_path_token_horizons=horizons, total_stored_tokens=sum(horizons),
        common_tokens=ns.tolist(), common_checkpoint_count=len(ns),
        mean_gap_bits=mean.tolist(), sample_SE_gap_bits=(gaps.std(0, ddof=1)/4).tolist(),
        mean_gap_crossing=value, observed_mean_upcrossings=upcrossings,
        bootstrap=bootstrap, unlocated_bootstrap_fraction=fraction,
        plotted_interval_95_percent=interval, interval_plotted=eligible)


def panel_summary(cells, L, alpha):
    selected = sorted((c for c in cells if (c['L'], c['alpha']) == (L, alpha)), key=lambda c:c['d'])
    assert [c['d'] for c in selected] == list(DS)
    d = np.array([c['d'] for c in selected])
    n = np.array([c['mean_gap_crossing'] for c in selected])
    x, y = np.log(d), np.log(n)
    slope = float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
    intercept = float(y.mean()-slope*x.mean())
    residual = y-intercept-slope*x
    return dict(L=L, alpha=alpha, d=d.tolist(), mean_gap_crossings=n.tolist(),
        endpoint_slope=float((y[-1]-y[0])/(x[-1]-x[0])),
        OLS_log_log_slope=slope, OLS_log_intercept=intercept,
        adjacent_doubling_slopes=(np.diff(y)/np.diff(x)).tolist(),
        fitted_crossings=np.exp(intercept+slope*x).tolist(),
        maximum_absolute_log_fit_residual=float(abs(residual).max()),
        maximum_absolute_relative_fit_residual=float(abs(np.expm1(residual)).max()),
        unlocated_bootstrap_fractions=[c['unlocated_bootstrap_fraction'] for c in selected])


def plot(data, figure, preview):
    plt.rcParams.update({'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
        'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.7))
    colors = ('#2369a0', '#b75d25')
    for i, L in enumerate(DEPTHS):
        for j, alpha in enumerate(ALPHAS):
            ax = axes[i, j]
            panel = next(p for p in data['panels'] if (p['L'], p['alpha']) == (L, alpha))
            cells = sorted((c for c in data['cells'] if (c['L'], c['alpha']) == (L, alpha)), key=lambda c:c['d'])
            x = np.geomspace(DS[0], DS[-1], 100)
            ax.plot(x, np.exp(panel['OLS_log_intercept'])*x**panel['OLS_log_log_slope'],
                    color='.45', ls='--', lw=1.)
            for cell in cells:
                d, value = cell['d'], cell['mean_gap_crossing']
                interval = cell['plotted_interval_95_percent']
                if interval is not None:
                    ax.vlines(d, interval[0], interval[1], color=colors[i], lw=1.)
                    ax.plot([d, d], interval, marker='_', ms=5, color=colors[i], ls='none')
                ax.plot(d, value, marker='o', ms=4, color=colors[i], mfc='white', mew=1.1)
                if interval is None:
                    ax.annotate('†', (d, value), xytext=(4, 2), textcoords='offset points')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlim(440, 4700)
            ax.set_xticks(DS, [str(d) for d in DS])
            all_y = [c['mean_gap_crossing'] for c in cells]
            all_y += [v for c in cells if c['plotted_interval_95_percent']
                      for v in c['plotted_interval_95_percent']]
            ax.set_ylim(min(all_y)*.72, max(all_y)*1.5)
            candidates = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])
            low, high = ax.get_ylim()
            ax.yaxis.set_major_locator(FixedLocator(candidates[(candidates >= low)&(candidates <= high)]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k' if value >= 1000 else f'{value:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(rf'$L={L},\quad\alpha={alpha:g}$')
            ax.text(.04, .94, f"Empirical slope {panel['OLS_log_log_slope']:.2f}",
                    va='top', transform=ax.transAxes, fontsize=7)
            ax.grid(alpha=.15, lw=.5, which='major')
            if j == 0:
                ax.set_ylabel('Mean-gap crossing (tokens)')
            if i == 1:
                ax.set_xlabel('Alphabet size $d$')
    fig.legend(handles=[Line2D([], [], marker='o', mfc='white', color='.2', ls='none',
                                  label='Root of the 16-run mean gap'),
                        Line2D([], [], color='.45', ls='--', label='Fit over these four alphabet sizes')],
               loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(.52, 1.))
    fig.text(.5, .018, 'Bars: 95% bootstrap intervals, conditional on a located root.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=.135, right=.985, bottom=.125, top=.86, wspace=.30, hspace=.37)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'output/takeover')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=202609073)
    parser.add_argument('--out', type=Path, default=ROOT/'results/mechanism/lsa_larger_scaling.json')
    parser.add_argument('--figure', type=Path, default=ROOT/'figures/lsa_larger_scaling.pdf')
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    assert args.bootstrap >= 100
    args.archive = args.archive.resolve()
    cells = [load_cell(L, a, d, args) for L, a, d in itertools.product(DEPTHS, ALPHAS, DS)]
    data = dict(schema_version=1, complete=True, source_depths=list(DEPTHS), alphas=list(ALPHAS), ds=list(DS),
        experiment='Archived symmetric depth-mixture LSA codes on actual Markov paths; no Bayes comparison.',
        crossing_definition='Last observed nonpositive-to-positive root of the mean cumulative code gap on the common grid, interpolated linearly in log token length; final mean must be positive.',
        repetitions_per_cell=16, cells_count=len(cells), total_source_path_draws=sum(c['repetitions'] for c in cells),
        bootstrap_draws=args.bootstrap, bootstrap_seed=args.seed,
        bootstrap_protocol='Within each setting, resample the 16 complete run curves, average first, then locate the root; default_rng([seed,L,round(100*alpha),d]).',
        interval_rule='Plot the 2.5th and97.5th percentiles of located bootstrap roots only if at least95% of resamples locate a root; preserve all null counts and reasons.',
        cells=cells, panels=[panel_summary(cells, L, a) for L, a in itertools.product(DEPTHS, ALPHAS)],
        source_script_sha256=digest(Path(__file__)),
        shared_audit_routines_sha256=digest(Path(__file__).with_name('crossing_count_prediction_audit.py')),
        qualifications=['No new source, path or estimator evaluation; every result comes from retained archived runs.',
            'Crossings of mean gaps are not means or medians of individual-path crossing times.',
            'All required settings have16 distinct runs and zero archived censored paths.',
            'Finite checkpoint windows do not certify a permanent crossing or exclude later reversals.',
            'Bootstrap intervals describe sampling on the retained grids, condition on locating a root, and exclude interpolation error or unseen later crossings.',
            'OLS and endpoint slopes summarize only d512 through4096; they are not theoretical or asymptotic exponents.',
            'Source depth L differs from the coding cap, which grows from30 to39 across these alphabets.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    plot(data, args.figure, args.preview)
    print(json.dumps(dict(cells=len(cells), paths=data['total_source_path_draws'],
        slopes=[{k:p[k] for k in ('L','alpha','endpoint_slope','OLS_log_log_slope','adjacent_doubling_slopes')} for p in data['panels']],
        unlocated=[dict(L=c['L'], alpha=c['alpha'], d=c['d'], fraction=c['unlocated_bootstrap_fraction'],
                        reasons=c['bootstrap']['status_counts']) for c in cells if c['bootstrap']['null']]), indent=2))


if __name__ == '__main__':
    main()

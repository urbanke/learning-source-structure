"""Compare matched Bayes costs, missing-knowledge bounds, and the LSA codes.

MPLCONFIGDIR=/tmp/la-mpl python code/scripts/optimal_lsa_matrix_figure.py
Smoke: python code/scripts/optimal_lsa_matrix_figure.py --main-only --out /tmp/optimal_lsa_matrix.pdf --summary /tmp/optimal_lsa_matrix_figures.json

The main PDF has six d=256 panels. The appendix PDF has two pages, repeating
the matrix at d=128 and d=512. No new source draws, paths, code evaluations,
or fitted model parameters are produced. Shading estimates bounds on the
cost of the common-prior Bayes code; it is not a confidence interval.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from crossing_count_prediction_audit import archived_paths, root


ROOT = Path(__file__).resolve().parents[2]
DEPTHS, ALPHAS, DS = (2, 3), (.75, 1., 1.25), (128, 256, 512)
BLUE, ORANGE, GRAY = '#22699c', '#c36b30', '#777777'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=5e-8, rtol=4e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), (a, b)
    return float(abs(a-b).max()) if a.size else 0.


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def read_cell(cell):
    d, alpha, L = cell['d'], cell['alpha'], cell['L']
    assert (d, alpha, L) in set(itertools.product(DS, ALPHAS, DEPTHS))
    assert cell['repetitions'] == 16 and cell.get('complete', True)
    paths = {}
    for key, hash_key in [('source', 'source_sha256'), ('source_summary', 'source_summary_sha256'),
                          ('raw_path', 'raw_sha256')]:
        path = resolve(cell[key])
        assert sha(path) == cell[hash_key], (key, path)
        paths[key] = path
    source = json.loads(paths['source_summary'].read_text())
    assert (source['L'], source['alpha'], source['d'], source['repetitions']) == (L, alpha, d, 16)
    ns = np.asarray(cell['tokens'], dtype=int)
    assert len(ns) == 10 and np.array_equal(ns, source['tokens'])
    costs = cell['costs']
    fields = ['bayes_full_cost', 'predicted_finite_full_cost',
        'lsa_context_full_cost', 'lsa_context_formula_full_cost',
        'lsa_frequency_full_cost', 'lsa_frequency_formula_full_cost',
        'bayes_plus_knowledge_allowance_full_cost']
    with np.load(paths['raw_path']) as z:
        raw = {key:z[key] for key in fields}
    maximum_summary_error = 0.
    for key in fields:
        assert raw[key].shape == (16, 10) and np.isfinite(raw[key]).all()
        maximum_summary_error = max(maximum_summary_error,
            close(raw[key].mean(0), costs[key]['mean']),
            close(raw[key].std(0, ddof=1)/4, costs[key]['SE']))
    allowance = math.lgamma(d+1)/math.log(2.)+math.log2(6.)
    close(allowance, cell['knowledge_allowance_bits'])
    close(raw['bayes_plus_knowledge_allowance_full_cost'], raw['bayes_full_cost']+allowance)
    close(costs['bayes_plus_knowledge_allowance_full_cost']['SE'], costs['bayes_full_cost']['SE'])
    roots = {}
    for name, first, second in [('actual', 'lsa_frequency_full_cost', 'lsa_context_full_cost'),
                                ('formula', 'lsa_frequency_formula_full_cost', 'lsa_context_formula_full_cost')]:
        gap = raw[first]-raw[second]
        value, status, number = root(ns, gap.mean(0))
        record = cell['lsa_crossings'][name]
        assert status == record['status'] == 'located'
        close(value, record['value'])
        source_record = source['chain' if name == 'actual' else 'count']['mean_curve_crossing']
        close(value, source_record['value'])
        roots[name] = dict(value=value, status=status, observed_upcrossings=number)
    # Reconstruct every archived mean root from its sixteen retained curves.
    acell = dict(d=d, alpha=alpha, L=L, l_max=source['l_max'],
        archive_path=source['archive_path'], archive_sha256=source['archive_sha256'],
        archived_mean_curve_crossing=source['archived_mean_curve_crossing'])
    ans, agaps, ahashes = archived_paths(acell)
    avalue, astatus, anumber = root(ans, agaps.mean(0))
    assert astatus == 'located'
    close(avalue, cell['lsa_crossings']['archive'])
    roots['archive'] = dict(value=avalue, status=astatus, observed_upcrossings=anumber,
        common_checkpoint_count=len(ans), source_files_sha256=ahashes)
    displayed = {key:dict(mean_per_token=(np.asarray(costs[key]['mean'])/ns).tolist(),
                         SE_per_token=(np.asarray(costs[key]['SE'])/ns).tolist()) for key in fields}
    low = np.asarray(displayed['bayes_full_cost']['mean_per_token'])
    upper = np.asarray(displayed['bayes_plus_knowledge_allowance_full_cost']['mean_per_token'])
    close(upper-low, allowance/ns)
    excess = raw['lsa_context_full_cost']-raw['bayes_full_cost']
    excess_upper = excess-allowance
    return dict(d=d, alpha=alpha, L=L, repetitions=16, tokens=ns.tolist(),
        source=cell['source'], source_sha256=cell['source_sha256'],
        source_summary=cell['source_summary'], source_summary_sha256=cell['source_summary_sha256'],
        raw_path=cell['raw_path'], raw_sha256=cell['raw_sha256'],
        coding_depth_cap=source['l_max'], knowledge_allowance_bits=allowance,
        displayed=displayed, envelope=dict(lower=low.tolist(), upper=upper.tolist(),
            definition='Actual matched-Bayes sample mean and that mean plus [log2(d!)+log2(6)]/n.',
            qualification='Estimated bounds on expected codelength, not a confidence interval and not a rigorous shift of the scalar approximation.'),
        LSA_crossings=roots, maximum_summary_recomputation_error=maximum_summary_error,
        lsa_context_minus_known_Bayes=dict(mean_bits=excess.mean(0).tolist(),
            paired_SE_bits=(excess.std(0, ddof=1)/4).tolist()),
        lsa_context_minus_missing_knowledge_upper=dict(mean_bits=excess_upper.mean(0).tolist(),
            paired_SE_bits=(excess_upper.std(0, ddof=1)/4).tolist()))


def sparse_ticks(ns):
    lo, hi = int(ns[0]), int(ns[-1])
    powers = 2.**np.arange(math.floor(math.log2(lo)), math.ceil(math.log2(hi))+1)
    candidates = powers[(powers >= lo)&(powers <= hi)]
    if len(candidates) > 3:
        candidates = candidates[np.linspace(0, len(candidates)-1, 3).round().astype(int)]
    return candidates


def draw_matrix(cells, d):
    selected = {(c['L'], c['alpha']):c for c in cells if c['d'] == d}
    assert set(selected) == set(itertools.product(DEPTHS, ALPHAS))
    plt.rcParams.update({'font.size':7.5, 'axes.titlesize':8.5, 'axes.labelsize':8,
        'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.8))
    for i, L in enumerate(DEPTHS):
        for j, alpha in enumerate(ALPHAS):
            ax = axes[i, j]
            cell = selected[L, alpha]
            n, values = np.asarray(cell['tokens']), cell['displayed']
            lower, upper = np.asarray(cell['envelope']['lower']), np.asarray(cell['envelope']['upper'])
            ax.fill_between(n, lower, upper, facecolor=BLUE, alpha=.10, lw=0, zorder=0)
            ax.plot(n, upper, color=BLUE, alpha=.35, lw=.55, ls=':', zorder=1)
            for formula, actual, color, marker, fill, width, linestyle in [
                ('predicted_finite_full_cost','bayes_full_cost',BLUE,'o','white',1.25,'-'),
                ('lsa_context_formula_full_cost','lsa_context_full_cost',ORANGE,'o',ORANGE,1.2,'-'),
                ('lsa_frequency_formula_full_cost','lsa_frequency_full_cost',GRAY,'x',GRAY,.8,':')]:
                prediction = np.asarray(values[formula]['mean_per_token'])
                mean = np.asarray(values[actual]['mean_per_token'])
                se = np.asarray(values[actual]['SE_per_token'])
                ax.plot(n, prediction, color=color, lw=width, ls=linestyle, zorder=3)
                ax.plot(n, mean, ls='none', marker=marker, ms=2.8 if marker=='o' else 2.6,
                        mfc=fill, mec=color, mew=.75, color=color, zorder=4)
                if marker == 'o':
                    at = np.array([0, 4, 9])
                    ax.errorbar(n[at], mean[at], yerr=2*se[at], fmt='none',
                                ecolor=color, elinewidth=.6, capsize=1.2, zorder=2)
            ax.set_xscale('log', base=2)
            ax.set_xlim(n[0]/1.06, n[-1]*1.06)
            ticks = sparse_ticks(n)
            ax.set_xticks(ticks, [str(int(t)) for t in ticks])
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            shown = [np.asarray(v['mean_per_token']) for v in values.values()]
            for actual in ('bayes_full_cost', 'lsa_context_full_cost'):
                at = np.array([0, 4, 9])
                mean = np.asarray(values[actual]['mean_per_token'])[at]
                se = np.asarray(values[actual]['SE_per_token'])[at]
                shown.extend([mean-2*se, mean+2*se])
            lo, hi = min(float(x.min()) for x in shown), max(float(x.max()) for x in shown)
            pad = max(.12, .06*(hi-lo))
            ax.set_ylim(lo-pad, hi+pad)
            ax.set_title(rf'$L={L},\quad\alpha={alpha:g}$')
            ax.grid(alpha=.13, lw=.5)
            if j == 0:
                ax.set_ylabel('Cumulative cost / tokens\n(bits per token)')
            if i == 1:
                ax.set_xlabel('Tokens $n$')
    handles = [
        Line2D([], [], color=BLUE, marker='o', mfc='white', ms=3, lw=1.2,
               label='Bayes: formula / measured (q,L known)'),
        Line2D([], [], color=ORANGE, marker='o', mfc=ORANGE, ms=3, lw=1.2,
               label='LSA context: formula / measured'),
        Line2D([], [], color=GRAY, marker='x', ms=3, lw=.8, ls=':',
               label='LSA frequency: formula / measured'),
        Patch(facecolor=BLUE, alpha=.10, edgecolor='none',
              label='Missing-information range (estimated)')]
    fig.legend(handles=handles, ncol=2, frameon=False, loc='upper center',
               bbox_to_anchor=(.525, .992), handlelength=2.0, columnspacing=1.5)
    fig.text(.525, .869, rf'Alphabet size $d={d}$; 16 saved table-and-path draws per panel',
             ha='center', fontsize=8)
    fig.text(.53, .018, 'Sparse bars: ±2 SE. Shading is not a confidence band.', ha='center', fontsize=7)
    fig.subplots_adjust(left=.10, right=.985, bottom=.11, top=.81, wspace=.34, hspace=.37)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/mechanism/optimal_lsa_matrix_check.json')
    parser.add_argument('--out', type=Path, default=ROOT/'figures/optimal_lsa_matrix.pdf')
    parser.add_argument('--appendix', type=Path, default=ROOT/'figures/optimal_lsa_matrix_alphabets.pdf')
    parser.add_argument('--summary', type=Path, default=ROOT/'results/mechanism/optimal_lsa_matrix_figures.json')
    parser.add_argument('--preview-dir', type=Path)
    parser.add_argument('--main-only', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    if not args.main_only:
        assert data['complete'], 'The full eighteen-setting producer has not finished.'
    assert data['script_sha256'] == sha(Path(__file__).with_name('optimal_lsa_matrix_check.py'))
    for filename, expected_hash in data['evaluator_hashes'].items():
        assert sha(Path(__file__).with_name(filename)) == expected_hash
    for scalar in data['scalar_curves'].values():
        assert sha(resolve(scalar['source'])) == scalar['source_sha256']
    need_ds = (256,) if args.main_only else DS
    expected = set(itertools.product(DEPTHS, ALPHAS, need_ds))
    supplied = {(c['L'], c['alpha'], c['d']) for c in data['cells'] if c['d'] in need_ds}
    assert supplied == expected, ('Missing or duplicate parameter settings', supplied, expected)
    records = [read_cell(c) for c in data['cells'] if c['d'] in need_ds]
    assert len(records) == len(expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_matrix(records, 256)
    fig.savefig(args.out, metadata={'CreationDate':None, 'ModDate':None})
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_dir/'optimal_lsa_matrix_d256.png', dpi=180)
    plt.close(fig)
    outputs = [dict(path=str(args.out), sha256=sha(args.out), pages=1, alphabet_sizes=[256])]
    if not args.main_only:
        args.appendix.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.appendix, metadata={'CreationDate':None, 'ModDate':None}) as pdf:
            for d in (128, 512):
                fig = draw_matrix(records, d)
                pdf.savefig(fig)
                if args.preview_dir:
                    fig.savefig(args.preview_dir/f'optimal_lsa_matrix_d{d}.png', dpi=180)
                plt.close(fig)
        outputs.append(dict(path=str(args.appendix), sha256=sha(args.appendix), pages=2, alphabet_sizes=[128, 512]))
    result = dict(schema_version=1, complete=not args.main_only,
        source=str(args.source.relative_to(ROOT)), source_sha256=sha(args.source),
        script_sha256=sha(Path(__file__)),
        plotting_versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        shared_audit_routines_sha256=sha(Path(__file__).with_name('crossing_count_prediction_audit.py')),
        cell_count=len(records), repetitions_per_cell=16, table_and_path_draws=16*len(records),
        checked_mean_gap_roots=3*len(records), cells=records, outputs=outputs,
        caption='Formula curves and measured mean cumulative costs per token. Blue is the matched Bayes code with q and source L known; orange is the symmetric LSA context code; gray is its frequency code. The shaded interval starts at the measured Bayes mean and adds [log2(d!)+log2(6)]/n. It estimates bounds for a Bayes code with a uniform unknown label ranking and a uniform prior over the six plotted (alpha,L) pairs. It is not a confidence band. Sparse error bars show two sample standard errors over the sixteen saved runs.',
        qualifications=['All displayed values come from existing saved source tables and paths; no new source/path draws or code evaluations in this plotter.',
            'Formula curves join ten numerical means on a logarithmic token-length axis; no crossing constants are fitted.',
            'Knowledge allowance is added to measured Bayes costs, never to the scalar approximation.',
            'Envelope endpoints are Monte Carlo estimates; the shading does not include their sampling uncertainty.',
            'Sparse bars show sampling errors for the measured Bayes and LSA context costs; other sampling errors remain in the recorded data.',
            'The optimal code uses a known fixed source depth; LSA mixes coding depths separately by context.',
            'The common-prior Bayes code itself has not been evaluated; shading gives bounds, not its measured curve.',
            'All three reported root types are roots of mean cumulative gaps, not path-crossing medians; their finite grids do not certify later signs.'])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['cell_count','table_and_path_draws','checked_mean_gap_roots','outputs']}, indent=2))


if __name__ == '__main__':
    main()

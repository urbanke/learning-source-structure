"""Audit and plot the saved early three-family comparison; no new inference.

Only the two retained prediction times (1 and 4) are displayed. The plotted
quantity subtracts the loss of Bayes given the generating family, sequence
by sequence, from the actual mixture or transformer loss.
"""
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
SOURCE = ROOT / 'results/transformer/multifamily_bayes_2026-09-16'
RAW = ROOT / 'output/multifamily_bayes_2026-09-16/short_comparison'
OUT = ROOT / 'results/transformer/section4_family_cost'
FAMILIES = ['product', 'shared_base', 'uniform_base']
MODELS = ['pooled_seed0', 'pooled_seed1']
LN2 = np.log(2.)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_path(value):
    """Resolve an archived absolute repository path in the current checkout."""
    path = Path(value)
    if path.is_file():
        return path
    if 'layered_architectures' in path.parts:
        index = path.parts.index('layered_architectures')
        path = ROOT.joinpath(*path.parts[index+1:])
    return path


def statistics(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (1024, 2) and np.isfinite(values).all(), 'Invalid paired observations')
    return dict(mean=values.mean(axis=0).tolist(),
                se=(values.std(axis=0, ddof=1)/np.sqrt(len(values))).tolist(), n_seq=len(values))


def balanced(values):
    return dict(mean=np.mean([v['mean'] for v in values], axis=0).tolist(),
                se=(np.linalg.norm([v['se'] for v in values], axis=0)/3).tolist(),
                n_seq=3072, family_weights=[1/3]*3)


def close(actual, expected, name, checks, tolerance=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name + ': shape or finiteness')
    error = float(np.max(np.abs(a-b)))
    require(error <= tolerance, name + f': replay error {error}')
    checks[name] = error


def moments_match(actual, expected, name, checks):
    require(actual['n_seq'] == expected['n_seq'], name + ': sequence count changed')
    for metric in ('mean', 'se'):
        close(actual[metric], expected[metric], name+'_'+metric, checks)


def audit(source=SOURCE, raw=RAW):
    source, raw = Path(source), Path(raw)
    old = json.loads((source/'short_comparison.json').read_bytes())
    validation_path = source/'short_validation.json'
    validation = json.loads(validation_path.read_bytes())
    require(old['schema'] == 'multifamily-short-mixture-comparison-v1'
            and validation['schema'] == 'multifamily-short-mixture-validation-v1'
            and validation['passed'] is True, 'Source comparison is not validated')
    require(old['validation_receipt']['sha256'] == sha(validation_path), 'Source validation binding changed')
    require(old['source_plan_id'] == validation['source_plan_id'], 'Source cohort identity changed')
    require(old['prediction_times'] == validation['direct_comparison_prediction_times'] == [1, 4]
            and old['direct_mixture_evidence_available_through_tokens'] == 5
            and validation['independent_sequences'] == 3072, 'Unsupported comparison scope')
    require(old['family_order'] == FAMILIES and old['family_prior_weights'] == [1/3]*3,
            'The saved experiment has a different family prior')
    hashes = {str(source/'short_comparison.json'): sha(source/'short_comparison.json'),
              str(validation_path): sha(validation_path)}
    for label in ('numerical_validations', 'verified_file_sha256', 'code_sha256', 'raw_output_sha256'):
        for name, digest in validation[label].items():
            path = recorded_path(name)
            require(path.is_file() and sha(path) == digest, 'Validated source file changed: '+str(path))
            hashes[str(path)] = digest
    checks, cohorts, observed_seeds = {}, {}, []
    for k, family in enumerate(FAMILIES):
        path = raw/f'{family}_prior.npz'
        expected = next(digest for name, digest in validation['raw_output_sha256'].items()
                        if Path(name).name == path.name)
        require(sha(path) == expected, 'Paired source arrays changed: '+str(path))
        hashes[str(path)] = expected
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        require(data['prefixes'].tolist() == [1, 4] and data['family_names'].tolist() == FAMILIES,
                'Source times or component order changed')
        require(data['sequence_seed'].shape == (1024, 4), 'Unexpected sequence identities')
        observed_seeds.extend(map(tuple, data['sequence_seed'].tolist()))
        logs = data['log_marginals_after_prediction']
        require(logs.shape == (1024, 2, 3) and np.isfinite(logs).all(), 'Invalid absolute family evidence')
        # Independent elementary log-sum-exp replay. First token is uniform
        # in all three families and was supplied, so remove its six bits.
        log_mix = np.logaddexp.reduce(logs, axis=2) - np.log(3.)
        component_loss = -(logs + np.log(64.))/LN2
        mixture_loss = -(log_mix + np.log(64.))/LN2
        family_loss = component_loss[:, :, k]
        cost = mixture_loss - family_loss
        close(component_loss, data['component_cumulative_loss_bits'], family+'_component_loss', checks)
        close(mixture_loss, data['mixture_cumulative_loss_bits'], family+'_mixture_loss', checks)
        close(cost, data['mixture_minus_generating_family_bits'], family+'_cost', checks)
        posterior = np.exp(logs[:, :, k] - np.log(3.) - log_mix)
        close(cost, np.log2(3.)+np.log2(posterior), family+'_posterior_identity', checks)
        require(float(cost.max()) <= np.log2(3.)+1e-10, 'Mixture coding bound violated')
        records = dict(bayes_mixture_minus_known_family_bits=statistics(cost), models={})
        moments_match(records['bayes_mixture_minus_known_family_bits'],
                      old['cohorts'][family]['mixture_minus_generating_family_bits'], family+'_old_cost', checks)
        for model in MODELS:
            tf = data[model+'_cumulative_loss_bits']
            residual = tf-mixture_loss
            close(residual, data[model+'_minus_mixture_bits'], family+'_'+model+'_residual', checks)
            row = dict(transformer_minus_known_family_bits=statistics(tf-family_loss),
                       transformer_minus_mixture_bits=statistics(residual))
            moments_match(row['transformer_minus_mixture_bits'],
                          old['cohorts'][family]['models'][model]['transformer_minus_mixture_bits'],
                          family+'_'+model+'_old_gap', checks)
            records['models'][model] = row
        cohorts[family] = records
    require(len(set(observed_seeds)) == 3072, 'Test sequences are not independently indexed')
    average = dict(bayes_mixture_minus_known_family_bits=balanced(
        [cohorts[f]['bayes_mixture_minus_known_family_bits'] for f in FAMILIES]), models={})
    moments_match(average['bayes_mixture_minus_known_family_bits'],
                  old['equal_family_average']['mixture_minus_generating_family_bits'], 'old_balanced_cost', checks)
    for model in MODELS:
        average['models'][model] = {metric: balanced([cohorts[f]['models'][model][metric] for f in FAMILIES])
            for metric in ('transformer_minus_known_family_bits', 'transformer_minus_mixture_bits')}
        moments_match(average['models'][model]['transformer_minus_mixture_bits'],
                      old['equal_family_average']['models'][model]['transformer_minus_mixture_bits'],
                      model+'_old_balanced_gap', checks)
    return dict(schema='section4-family-cost-v1', prediction_times=[1, 4], source_plan_id=old['source_plan_id'],
        source_families=FAMILIES, family_prior_weights=[1/3]*3, n_sequences_per_family=1024,
        n_sequences_total=3072, sequence_length=1025,
        scored_tokens='At time t, score tokens 2 through t+1; the first token is supplied.',
        reference='Family-informed Bayes is told the generating family, but not its sampled parameters.',
        scope='Saved early endpoints only; no full cost curve, later contexts or new transformer inference.',
        uncertainty='Paired sequence SE; balanced mean uses sqrt(sum(family_SE**2))/3. Two training seeds remain separate.',
        bound_bits=float(np.log2(3)), cohorts=cohorts, equal_family_average=average,
        checks=checks, verified_file_sha256=hashes, code_sha256=sha(__file__),
        no_new_inference=True, longer_context_claim=False)


def plot(report, out):
    styles = [('bayes', '#202020', 'D', 'Bayesian mixture'),
              ('pooled_seed0', '#2468A0', 'o', 'Transformer seed 0'),
              ('pooled_seed1', '#BF611C', 's', 'Transformer seed 1')]
    def metric(group, model):
        return (group['bayes_mixture_minus_known_family_bits'] if model == 'bayes'
                else group['models'][model]['transformer_minus_known_family_bits'])
    with plt.rc_context({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
        for j, (model, color, marker, label) in enumerate(styles):
            values = metric(report['equal_family_average'], model)
            axes[0].errorbar(np.array([1., 4.])+(j-1)*.07, values['mean'], yerr=values['se'],
                            color=color, marker=marker, ms=4.5, capsize=2.5, ls='none', label=label)
            family_values = [metric(report['cohorts'][f], model) for f in FAMILIES]
            axes[1].errorbar(np.arange(3)+(j-1)*.16, [v['mean'][1] for v in family_values],
                            yerr=[v['se'][1] for v in family_values], color=color, marker=marker,
                            ms=4.5, capsize=2.5, ls='none')
        axes[0].set_title('(a) Equal-family average')
        axes[0].set_xticks([1, 4])
        axes[0].set_xlim(.55, 4.45)
        axes[0].set_xlabel('Number of predictions')
        axes[0].set_ylabel('Extra cumulative loss (bits)')
        axes[1].set_title('(b) Four predictions, by source family')
        axes[1].set_xticks(range(3), ['Product', 'Shared-base\nLSA', 'Uniform-base\nLSA'])
        axes[1].set_xlim(-.5, 2.5)
        for ax in axes:
            ax.axhline(0, color='.55', lw=.7, zorder=0)
            ax.grid(axis='y', color='.9', lw=.65)
            ax.set_axisbelow(True)
            ax.set_ylim(-.035, .355)
        fig.suptitle('Additional loss relative to Bayes told the source family', fontsize=10, y=.985)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .075), frameon=False, fontsize=8)
        fig.text(.5, .017, 'Points: saved times only. Bars: ±1 paired SE. 1,024 independent test sequences per family.',
                 ha='center', fontsize=7.5)
        fig.subplots_adjust(left=.085, right=.99, bottom=.29, top=.83, wspace=.15)
        fig.savefig(out/'family_cost.png', dpi=220)
        fig.savefig(out/'family_cost.pdf', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    report = audit(args.source, args.raw)
    if args.check_only:
        require(json.loads((args.out/'summary.json').read_bytes()) == report, 'Saved family-cost summary changed')
        receipt = json.loads((args.out/'validation.json').read_bytes())
        require(receipt['passed'] is True and receipt['summary_sha256'] == sha(args.out/'summary.json'), 'Output receipt changed')
        for name, digest in receipt['outputs_sha256'].items():
            require(sha(args.out/name) == digest, 'Output artifact changed: '+name)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        plot(report, args.out)
        receipt = dict(schema='section4-family-cost-audit-v1', passed=True,
            summary_sha256=sha(args.out/'summary.json'), input_files_verified=len(report['verified_file_sha256']),
            replay_checks=len(report['checks']), maximum_replay_error_bits=max(report['checks'].values()),
            outputs_sha256={name: sha(args.out/name) for name in ('family_cost.png', 'family_cost.pdf')},
            no_new_inference=True, longer_context_claim=False)
        (args.out/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(passed=True, times=[1, 4], n_sequences=3072, output=str(args.out),
                         maximum_replay_error_bits=max(report['checks'].values()))))


if __name__ == '__main__':
    main()

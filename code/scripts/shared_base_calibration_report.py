"""Apply predeclared short-control gates and plot completed/partial pilots.

This report cannot certify a long-context or population Bayesian reference.
It keeps every supplied case, including failures and incomplete runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_shared_base_evidence import check_resume, fingerprint, normalizer_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(paths, protocol_path):
    # Atomic writers can replace live reports while this function runs. Bind
    # each input hash to the exact bytes parsed, not a later file version.
    protocol_bytes = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    gate = protocol['short_gate']
    rows, configurations, input_hashes = [], {}, {}
    for path in paths:
        raw = path.read_bytes()
        input_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        identity = data['identity']
        if data['schema'] != 'shared-base-evidence-calibration-v1':
            raise ValueError(f'Unexpected calibration schema: {path}')
        check_resume(data, identity)
        if identity['predeclared_protocol']['sha256'] != protocol_hash:
            raise ValueError('Numerical runs do not use this frozen protocol')
        descriptor = dict(algorithm=identity['algorithm'], n_tokens=identity['n_tokens'],
                          master_seed=identity['master_seed'], replicates=identity['replicates'],
                          prior=identity.get('prior'), code_sha256=identity.get('code_sha256'),
                          sealed_store_sha256=identity.get('sealed_store_sha256'),
                          numpy_version=identity.get('numpy_version'), scipy_version=identity.get('scipy_version'),
                          input_numerical_environment=identity.get('input_numerical_environment'),
                          reference_catalog_validation_id=fingerprint(identity.get('short_catalog_validation')))
        configuration_id = fingerprint(descriptor)
        configurations.setdefault(configuration_id, descriptor)
        for case in identity['cases']:
            records = [r for r in data['records'] if r['sequence_id'] == case['id']]
            if len({r['replica'] for r in records}) != len(records):
                raise ValueError('Repeated repetition identity')
            row = dict(run=str(path), case=case['name'], case_id=case['id'],
                       configuration_id=configuration_id,
                       input_identity_id=data['identity_id'], tokens=case['tokens'],
                       catalog_log_evidence=case.get('catalog_log_evidence'),
                       n_tokens=identity['n_tokens'], algorithm=identity['algorithm'],
                       completed_repetitions=len(records), requested_repetitions=identity['replicates'],
                       repetition_estimates=[dict(replica=r['replica'], seed=r['seed'],
                           log_evidence=r['estimate']['log_evidence'],
                           elapsed_seconds=r['estimate']['elapsed_seconds'])
                           for r in sorted(records, key=lambda r: r['replica'])],
                       short_control_passed=False, production_reference_validated=False)
            if not records:
                row.update(status='no completed estimates', checks={})
                rows.append(row)
                continue
            logs = np.array([r['estimate']['log_evidence'] for r in records])
            row['statistics'] = normalizer_summary(logs, case.get('catalog_log_evidence'))
            weights = np.exp(logs-logs.max()); weights /= weights.sum()
            row['effective_repetitions'] = float(1 / (weights @ weights))
            row['maximum_repetition_likelihood_share'] = float(weights.max())
            row['seconds_for_completed_repetitions'] = float(sum(r['estimate']['elapsed_seconds'] for r in records))
            row['final_initial_ancestors'] = [r['estimate']['initial_particle_ancestry_counts'][-1] for r in records]
            row['minimum_particle_ess'] = float(min(min(r['estimate']['effective_sample_sizes']) for r in records))
            stats = row['statistics']
            interval = stats['approximate_loss_ci95_bits']
            half = (max(abs(v-stats['loss_of_mean_evidence_bits']) for v in interval)
                    if interval is not None and all(v is not None for v in interval) else None)
            row['mc_halfwidth95_bits'] = half
            if 'catalog_log_evidence' in case:
                error = abs(stats['estimated_minus_catalog_loss_bits'])
                row['error_plus_mc_halfwidth_bits'] = None if half is None else float(error + half)
                checks = dict(
                    complete=len(records) == identity['replicates'],
                    sufficient_repetitions=len(records) >= gate['minimum_completed_replicates'],
                    exact_control_error_and_precision=half is not None and error + half <= gate['tolerance_bits'],
                    sufficient_effective_repetitions=row['effective_repetitions'] >= gate['minimum_effective_replica_count'],
                    no_dominant_repetition=row['maximum_repetition_likelihood_share'] <= gate['maximum_single_replica_likelihood_share'])
                row['checks'] = checks
                row['short_control_passed'] = bool(all(checks.values()))
                row['status'] = ('short control passes; independent refinement still required'
                                 if row['short_control_passed'] else 'short control does not pass')
            else:
                row.update(status='repeatability diagnostic only; no exact control', checks={})
            rows.append(row)
    coverage = []
    frozen = protocol['frozen_cases']
    for configuration_id, descriptor in configurations.items():
        selected = [row for row in rows if row['configuration_id'] == configuration_id]
        expected = [f"{name}_{frozen['sequence_index']}" for name in frozen['cohorts']]
        expected += ['pattern_' + '_'.join(map(str, pattern))
                     for pattern in frozen['additional_exact_controls']
                     if len(pattern) == descriptor['n_tokens']]
        names = [row['case'] for row in selected]
        missing = [name for name in expected if name not in names]
        unexpected = sorted(set(names) - set(expected))
        duplicated = sorted({name for name in names if names.count(name) > 1})
        incomplete = [row['case'] for row in selected
                      if row['completed_repetitions'] != row['requested_repetitions']]
        complete = not (missing or unexpected or duplicated or incomplete)
        all_passed = complete and descriptor['n_tokens'] == 5 and all(
            row['short_control_passed'] for row in selected)
        status = ('partial predeclared coverage' if missing or unexpected or duplicated else
                  'incomplete repetitions' if incomplete else
                  'all short controls pass; independent refinement still required' if all_passed else
                  'complete short controls; one or more accuracy gates fail' if descriptor['n_tokens'] == 5 else
                  'complete fixed-case repeatability only; no exact controls')
        coverage.append(dict(configuration_id=configuration_id, **descriptor,
                             expected_cases=expected, supplied_cases=names,
                             missing_cases=missing, unexpected_cases=unexpected,
                             duplicated_cases=duplicated, incomplete_cases=incomplete,
                             all_predeclared_cases_complete=complete,
                             all_short_controls_passed=all_passed, status=status))
    return dict(schema='shared-base-evidence-calibration-report-v1',
                production_reference_validated=False,
                scope='Numerical checks on fixed token strings only. Successful short cases do not validate longer sequences or a population comparison.',
                protocol_sha256=protocol_hash, input_sha256=input_hashes,
                tolerance_bits=gate['tolerance_bits'], rows=rows,
                configuration_coverage=coverage)


def render(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    cases = list(dict.fromkeys(r['case'] for r in report['rows'] if r['n_tokens'] == 5))
    if not cases:
        return
    extent = report['tolerance_bits']
    for row in report['rows']:
        if row['n_tokens'] != 5 or 'statistics' not in row:
            continue
        stats = row['statistics']
        extent = max(extent, abs(stats['estimated_minus_catalog_loss_bits']))
        interval = stats.get('approximate_loss_error_ci95_bits')
        if interval:
            extent = max(extent, *(abs(value) for value in interval if value is not None))
    # Shared limits make uncertainty directly comparable between controls.
    # Multiples of .02 give five evenly spaced, exact two-decimal ticks.
    limit = float(.02 * np.ceil(max(.02, 1.08 * extent) / .02))
    ticks = np.linspace(-limit, limit, 5)
    columns = 2
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots((len(cases)+1)//2, columns, figsize=(10, 2.5*((len(cases)+1)//2)), squeeze=False)
        for ax, name in zip(axes.flat, cases):
            rows = sorted(
                [r for r in report['rows'] if r['case'] == name and r['n_tokens'] == 5],
                key=lambda r: (r['algorithm']['sampler'] != 'original', r['algorithm']['particles']))
            ax.axvspan(-report['tolerance_bits'], report['tolerance_bits'], color='#DDE7DD', alpha=.65)
            ax.axvline(0, color='#444444', linewidth=.8)
            labels = []
            for i, r in enumerate(rows):
                a = r['algorithm']; method = 'Original' if a['sampler'] == 'original' else 'Row block'
                labels.append(f"{method}, N={a['particles']} ({r['completed_repetitions']}/{r['requested_repetitions']})")
                if 'statistics' not in r:
                    continue
                s = r['statistics']; value = s['estimated_minus_catalog_loss_bits']
                interval = s.get('approximate_loss_error_ci95_bits')
                color = '#2468A0' if a['sampler'] == 'original' else '#BF611C'
                if interval and all(v is not None for v in interval):
                    ax.errorbar(value, i, xerr=[[value-interval[0]], [interval[1]-value]],
                                color=color, fmt='o', capsize=3, markersize=4)
                else:
                    ax.plot(value, i, marker='x', color=color)
            ax.set_yticks(range(len(rows)), labels)
            ax.set_ylim(-.6, len(rows)-.4)
            ax.invert_yaxis()
            label = name.replace('_prior_0', ' sequence 0').replace('pattern_', 'Tokens ').replace('_', ' ')
            ax.set_title(label, loc='left', fontsize=10)
            ax.set_xlabel('Estimated codelength minus known value (bits)')
            ax.set_xlim(-limit, limit)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(axis='x', alpha=.18)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        fig.suptitle('Five-token likelihood calibration\n'
                     f"Intervals: approximate 95% Monte Carlo uncertainty; green band: ±{report['tolerance_bits']:g} bits",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .955))
        fig.savefig(out / 'short_calibration.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.runs, args.protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    render(report, args.out)
    report['report_code_sha256'] = sha(__file__)
    image = args.out / 'short_calibration.png'
    if image.exists():
        report['figure_sha256'] = sha(image)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps([dict(case=r['case'], sampler=r['algorithm']['sampler'], particles=r['algorithm']['particles'],
                           repetitions=r['completed_repetitions'], status=r['status'],
                           error_plus_halfwidth=r.get('error_plus_mc_halfwidth_bits')) for r in report['rows']], indent=2))


if __name__ == '__main__':
    main()

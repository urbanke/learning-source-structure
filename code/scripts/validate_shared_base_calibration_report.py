"""Independently verify a compact shared-base calibration report.

Recompute from retained log normalizers using scalar arithmetic; the SMC
runner/report statistical helpers are not imported. No raw experiment files
are needed. A failed numerical control is valid when correctly reported as
failed: this command verifies the report, not the success of the pilot.

Example:
  python code/scripts/validate_shared_base_calibration_report.py --summary PATH
Add --out PATH to save a validation receipt; the default is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scipy.special import stdtrit

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2)
SEED_NAMESPACE = 'shared-base-evidence-calibration-replicates-v1'
DESCRIPTOR_KEYS = ('algorithm', 'n_tokens', 'master_seed', 'replicates', 'prior',
                   'code_sha256', 'sealed_store_sha256', 'numpy_version', 'scipy_version',
                   'input_numerical_environment', 'reference_catalog_validation_id')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    if isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f'{label}[{i}]')
    elif expected is None or isinstance(expected, bool):
        require(actual is expected, label)
    elif isinstance(expected, str):
        require(actual == expected, label)
    else:
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=2e-10), label)


def independent_statistics(logs, reference):
    """Likelihood means and monotone t intervals, with stable scalar sums."""
    n = len(logs)
    require(n > 0 and all(math.isfinite(x) for x in logs), 'nonfinite log evidence')
    shift = max(logs)
    z = [math.exp(x-shift) for x in logs]
    total = math.fsum(z)
    mean = total/n
    logmean = shift + math.log(mean)
    loss = -logmean/LN2
    result = dict(repetitions=n, log_mean_evidence_nats=logmean,
        loss_of_mean_evidence_bits=loss, likelihood_scale_log_nats=shift,
        mean_scaled_likelihood=mean,
        arithmetic_mean_replicate_loss_bits_diagnostic_only=-math.fsum(logs)/n/LN2,
        replicate_loss_range_bits=(max(logs)-min(logs))/LN2,
        effective_replica_count=total*total/math.fsum(x*x for x in z),
        maximum_single_replica_likelihood_share=max(z)/total,
        relative_mc_se=None, loss_mc_se_bits_delta_method=None,
        approximate_loss_ci95_bits=None, unbounded_upper_loss_ci=False,
        approximate_loss_ci95_max_halfwidth_bits=None)
    if n > 1:
        se = math.sqrt(math.fsum((x-mean)**2 for x in z)/(n-1)/n)
        offset = float(stdtrit(n-1, .975))*se
        lower, upper = mean-offset, mean+offset
        interval = [-(shift+math.log(upper))/LN2,
                    None if lower <= 0 else -(shift+math.log(lower))/LN2]
        result.update(relative_mc_se=se/mean, loss_mc_se_bits_delta_method=se/mean/LN2,
                      approximate_loss_ci95_bits=interval,
                      unbounded_upper_loss_ci=lower <= 0)
        if lower > 0:
            result['approximate_loss_ci95_max_halfwidth_bits'] = max(abs(x-loss) for x in interval)
    if reference is not None:
        exact_loss = -reference/LN2
        error = loss-exact_loss
        half = result['approximate_loss_ci95_max_halfwidth_bits']
        interval = result['approximate_loss_ci95_bits']
        result.update(exact_catalog_loss_bits=exact_loss,
            estimated_minus_catalog_loss_bits=error,
            absolute_error_plus_ci95_max_halfwidth_bits=None if half is None else abs(error)+half,
            approximate_loss_error_ci95_bits=None if interval is None else
                [None if x is None else x-exact_loss for x in interval])
    return result


def known_catalog(path):
    """Direct ordered-sequence joint from the saved exhaustive length-four catalog."""
    if path is None or not path.is_file():
        return None, None
    raw = path.read_bytes()
    data = json.loads(raw)
    require(data.get('schema') == 'shared-base-shortprefix-reference-catalog-v1'
            and data.get('passed') is True and data.get('time') == 4, 'catalog validation/schema')
    require(data['prior'] == dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
            'catalog prior')
    entries = {tuple(x['pattern']): x for x in data['records']}
    require(len(entries) == 15, 'catalog pattern coverage')
    receipt_path = path.parent/'time4_summary.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_bytes())
        require(receipt['provenance']['reference_catalog_sha256'] == digest(raw), 'catalog receipt hash')
    return entries, digest(raw)


def validate(summary_path, protocol_path, catalog_path):
    raw, protocol_raw = summary_path.read_bytes(), protocol_path.read_bytes()
    report, protocol = json.loads(raw), json.loads(protocol_raw)
    require(report['schema'] == 'shared-base-evidence-calibration-report-v1', 'report schema')
    require(report['production_reference_validated'] is False, 'unsupported production-reference claim')
    require(report['protocol_sha256'] == digest(protocol_raw), 'protocol hash')
    gate, frozen = protocol['short_gate'], protocol['frozen_cases']
    equal(report['tolerance_bits'], gate['tolerance_bits'], 'gate tolerance')
    catalog, catalog_hash = known_catalog(catalog_path)
    groups = {x['configuration_id']: x for x in report['configuration_coverage']}
    require(len(groups) == len(report['configuration_coverage']), 'duplicate configuration IDs')
    for cid, group in groups.items():
        require(fingerprint({k: group[k] for k in DESCRIPTOR_KEYS}) == cid, 'configuration fingerprint')
    rows_checked, estimates_checked, catalog_checks = [], 0, 0
    for row in report['rows']:
        label = f"{row['case']} / {row['algorithm']['sampler']} N={row['algorithm']['particles']}"
        require(row['configuration_id'] in groups, label+' missing configuration')
        group = groups[row['configuration_id']]
        require(row['algorithm'] == group['algorithm'], label+' algorithm identity')
        require(row['n_tokens'] == group['n_tokens'] and len(row['tokens']) == row['n_tokens'], label+' token length')
        require(all(type(x) is int and 0 <= x < 64 for x in row['tokens']), label+' tokens')
        require(row['requested_repetitions'] == group['replicates'], label+' requested count')
        require(row['run'] in report['input_sha256'], label+' missing input provenance')
        for value in (row['input_identity_id'], report['input_sha256'][row['run']]):
            require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), label+' identity/hash format')
        require(row['production_reference_validated'] is False, label+' unsupported production claim')
        records = row['repetition_estimates']
        require(len(records) == row['completed_repetitions'], label+' completed count')
        indices = [r['replica'] for r in records]
        require(len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < group['replicates'] for i in indices), label+' replica indices')
        for record in records:
            value = [SEED_NAMESPACE, group['master_seed'], row['case_id'], record['replica']]
            expected = int.from_bytes(bytes.fromhex(fingerprint(value))[:8], 'little')
            require(record['seed'] == expected, label+' independent replica seed')
            require(math.isfinite(record['elapsed_seconds']) and record['elapsed_seconds'] >= 0, label+' elapsed time')
        reference = row['catalog_log_evidence']
        if catalog is not None and row['n_tokens'] == 5:
            seen, pattern = {}, []
            for token in row['tokens']:
                pattern.append(seen.setdefault(token, len(seen)))
            entry = catalog[tuple(pattern[:4])]
            known = math.log(entry['prefix_probability']) + math.log(entry['predictive_probs'][pattern[4]])
            equal(reference, known, label+' independent catalog joint')
            catalog_checks += 1
        estimates_checked += len(records)
        if not records:
            require(row['short_control_passed'] is False and row['checks'] == {}, label+' empty case claimed successful')
            require(row['status'] == 'no completed estimates', label+' empty status')
            continue
        stats = independent_statistics([r['log_evidence'] for r in records], reference)
        for key, value in stats.items():
            equal(row['statistics'][key], value, label+' '+key)
        equal(row['effective_repetitions'], stats['effective_replica_count'], label+' replica ESS')
        equal(row['maximum_repetition_likelihood_share'], stats['maximum_single_replica_likelihood_share'], label+' largest weight')
        equal(row['seconds_for_completed_repetitions'], math.fsum(r['elapsed_seconds'] for r in records), label+' completed elapsed time')
        half = stats['approximate_loss_ci95_max_halfwidth_bits']
        equal(row['mc_halfwidth95_bits'], half, label+' halfwidth')
        expected_pass = False
        if reference is not None:
            quantity = stats['absolute_error_plus_ci95_max_halfwidth_bits']
            equal(row['error_plus_mc_halfwidth_bits'], quantity, label+' gate quantity')
            checks = dict(complete=len(records) == group['replicates'],
                sufficient_repetitions=len(records) >= gate['minimum_completed_replicates'],
                exact_control_error_and_precision=quantity is not None and quantity <= gate['tolerance_bits'],
                sufficient_effective_repetitions=stats['effective_replica_count'] >= gate['minimum_effective_replica_count'],
                no_dominant_repetition=stats['maximum_single_replica_likelihood_share'] <= gate['maximum_single_replica_likelihood_share'])
            require(row['checks'] == checks, label+' gate decisions')
            expected_pass = all(checks.values())
            status = ('short control passes; independent refinement still required' if expected_pass else 'short control does not pass')
        else:
            require(row['checks'] == {}, label+' unsupported exact gate')
            status = 'repeatability diagnostic only; no exact control'
        require(row['short_control_passed'] is expected_pass and row['status'] == status, label+' pass/status')
        rows_checked.append(dict(case=row['case'], sampler=row['algorithm']['sampler'], particles=row['algorithm']['particles'],
                                 repetitions=len(records), short_control_passed=expected_pass))
    for cid, group in groups.items():
        selected = [r for r in report['rows'] if r['configuration_id'] == cid]
        require(bool(selected), 'configuration has no cases')
        expected = [f"{name}_{frozen['sequence_index']}" for name in frozen['cohorts']]
        expected += ['pattern_'+'_'.join(map(str, p)) for p in frozen['additional_exact_controls'] if len(p) == group['n_tokens']]
        names = [r['case'] for r in selected]
        missing = [x for x in expected if x not in names]
        unexpected = sorted(set(names)-set(expected))
        duplicates = sorted(k for k, n in Counter(names).items() if n > 1)
        incomplete = [r['case'] for r in selected if r['completed_repetitions'] != r['requested_repetitions']]
        complete = not (missing or unexpected or duplicates or incomplete)
        passed = complete and group['n_tokens'] == 5 and all(r['short_control_passed'] for r in selected)
        for key, value in dict(expected_cases=expected, supplied_cases=names, missing_cases=missing,
                              unexpected_cases=unexpected, duplicated_cases=duplicates, incomplete_cases=incomplete,
                              all_predeclared_cases_complete=complete, all_short_controls_passed=passed).items():
            equal(group[key], value, cid+' '+key)
        status = ('partial predeclared coverage' if missing or unexpected or duplicates else
                  'incomplete repetitions' if incomplete else
                  'all short controls pass; independent refinement still required' if passed else
                  'complete short controls; one or more accuracy gates fail' if group['n_tokens'] == 5 else
                  'complete fixed-case repeatability only; no exact controls')
        require(group['status'] == status, cid+' group status')
    return dict(schema='shared-base-calibration-report-independent-validation-v1', status='passed',
                scope='Arithmetic, saved replica identities, catalog anchors and declared gates reproduced; this is not a successful-pilot or production-reference claim.',
                summary_sha256=digest(raw), protocol_sha256=digest(protocol_raw), validator_sha256=digest(Path(__file__).read_bytes()),
                catalog_sha256=catalog_hash, catalog_joints_checked=catalog_checks,
                rows_recomputed=len(rows_checked), configurations_checked=len(groups), repetition_estimates_checked=estimates_checked,
                controls_passing=sum(r['short_control_passed'] for r in rows_checked), rows=rows_checked,
                production_reference_validated=False,
                limitations=['Intervals are approximate Monte Carlo diagnostics; missed-mode and quadrature errors are not bounded.',
                             'Source input identity/hash references are retained; raw files are not required or rehashed by this compact check.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'notes/reviews/shared_base_evidence_calibration_protocol_2026-09-16.json')
    parser.add_argument('--catalog', type=Path, default=ROOT/'results/transformer/shared_base_conditional_curves/shortprefix_reference.json')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.summary, args.protocol, args.catalog)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, OSError) as error:
        result = dict(schema='shared-base-calibration-report-independent-validation-v1', status='failed',
                      scope='Report consistency validation failed; this is separate from a correctly reported unsuccessful pilot.',
                      summary_sha256=digest(args.summary.read_bytes()) if args.summary.is_file() else None,
                      error=str(error))
        code = 1
    else:
        code = 0
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(code)


if __name__ == '__main__':
    main()

"""Publish selected unseen-row refinements without replacing historical exports.

Only the 66 depth-four and 57 depth-eight time-64 Bayesian predictions change.
Both integration rules retain the original complete-cohort numerical gates.
The original paired neural losses and every other selected group are preserved.
No sampling, inference or new confirmation cohort is performed here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

import unseen_context_refinement as refinement
from prepare_unseen_context_refinement import PLAN, RUNTIME_PYTHON, read_plan

# Original exporter imports the row evaluator transitively. Its environment is
# read at import time, so engage the frozen sealed-store mode first.
_IMPORT_PLAN, _ = read_plan(PLAN)
os.environ.update(refinement.environment(_IMPORT_PLAN))
import numpy as np
import scipy
import section4_unseen_context_extension as original

ROOT = original.ROOT
BASE = original.OUT
REFINEMENT = ROOT/'output/unseen_context_refinement_2026-09-19'
OUT = ROOT/'results/transformer/section4_unseen_context_refined_2026-09-20'
require, sha = refinement.require, refinement.sha
FIELDS = ['depth', 'time', 'index', 'source_id', 'query', 'entropy_bits', 'bayes_released']
FIELDS += [name+'_loss_bits' for name in original.PREDICTORS]
TARGETS = {(4, 64): 66, (8, 64): 57}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path, files):
    path = Path(path).resolve()
    digest = sha(path)
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    require(name not in files or files[name] == digest, 'Input changed during publication')
    files[name] = digest
    return digest


def load_baseline(folder):
    """Verify the existing compact release, without scanning unrelated raw chains."""
    folder = Path(folder)
    summary, validation = read(folder/'summary.json'), read(folder/'validation.json')
    require(validation['passed'] is True and summary['script_sha256'] == sha(original.__file__)
            and validation['script_sha256'] == sha(original.__file__),
            'Historical exporter or validation differs')
    require(set(validation['files']) == {'summary.json', 'records.csv'}
            and all(sha(folder/name) == digest for name, digest in validation['files'].items()),
            'Historical compact data changed')
    original.verify_export(folder)
    require(len(summary['groups']) == 12, 'All historical groups required')
    rows = []
    with (folder/'records.csv').open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, 'Historical loss-table schema changed')
        for raw in reader:
            row = dict(raw)
            for key in ('depth', 'time', 'index', 'query'):
                row[key] = int(row[key])
            require(row['bayes_released'] in ('True', 'False'), 'Invalid historical release flag')
            row['bayes_released'] = row['bayes_released'] == 'True'
            for key in ['entropy_bits']+[name+'_loss_bits' for name in original.PREDICTORS]:
                row[key] = float(row[key])
                require(math.isfinite(row[key]), 'Nonfinite historical score')
            rows.append(row)
    files = {}
    for path in (folder/'summary.json', folder/'records.csv', folder/'validation.json',
                 Path(original.__file__)):
        bind(path, files)
    return summary, rows, files


def refine_compact(baseline, baseline_rows, plan, rows, source_records):
    """Pure complete-cohort substitution, with unchanged precision decisions."""
    groups = {(g['depth'], g['time']): g for g in baseline['groups']}
    require(len(groups) == 12 and len(baseline_rows) == sum(g['n_selected'] for g in groups.values()),
            'Incomplete historical selected population')
    require(len({(r['depth'], r['time'], r['index']) for r in baseline_rows}) == len(baseline_rows),
            'Duplicate historical selected row')
    require({(g['depth'], g['time']): len(g['selected_indices']) for g in plan['groups']} == TARGETS,
            'Refinement target groups changed')
    for g in plan['groups']:
        key = g['depth'], g['time']
        require(g['selected_indices'] == groups[key]['selected_indices']
                and groups[key]['n_selected'] == TARGETS[key]
                and groups[key]['n_original'] == 256,
                'Refinement changed the original selected histories')
    aggregate = refinement.aggregate_groups(plan, rows)
    require(aggregate['complete'] and aggregate['expected_references'] == 246
            and aggregate['observed_references'] == 246, 'All 246 audited references are required')
    effective = {(r['task']['depth'], r['task']['index'], r['task']['order']): r for r in rows}
    require(len(effective) == 246, 'Duplicate effective reference')
    specs = {(s['task']['depth'], s['task']['index']): s for s in plan['tasks']}
    require(set(source_records) == set(specs), 'Incomplete selected source identities')
    replacements = {(g['depth'], g['time']): g for g in aggregate['groups']}
    result_rows = copy.deepcopy(baseline_rows)
    changed = 0
    maximum_loss_error = maximum_entropy_error = 0.
    for row in result_rows:
        key = row['depth'], row['time']
        if key not in TARGETS:
            continue
        source = source_records[row['depth'], row['index']]
        spec = specs[row['depth'], row['index']]
        tokens = np.asarray(source['tokens'])
        require(source['source_id'] == spec['source_id'] == row['source_id']
                and source['identity']['source_depth'] == row['depth']
                and source['identity']['source_index'] == row['index']
                and int(tokens[63]) == row['query'] and int(tokens[63]) not in tokens[:63],
                'Source, query or unseen event differs from original selected row')
        oracle = np.asarray(source['oracle']['transition_probs'])[tokens[63]]
        entropy = float(-oracle@np.log2(oracle))
        maximum_entropy_error = max(maximum_entropy_error, abs(entropy-row['entropy_bits']))
        require(abs(entropy-row['entropy_bits']) < 2e-12, 'Original oracle entropy differs')
        audit = effective[row['depth'], row['index'], 32]['corrected']['audit']
        precision = audit['record']['precision']
        loss = original.old.loss(precision['predictive_probs'], oracle)
        error = abs(loss-entropy-precision['bayes_regret_bits'])
        maximum_loss_error = max(maximum_loss_error, error)
        require(error < 2e-12, 'Refined raw prediction and audited regret disagree')
        row['bayes_loss_bits'] = loss
        row['bayes_released'] = replacements[key]['bayes_released']
        changed += 1
    require(changed == 123, 'Exactly 123 Bayesian rows must be replaced')
    allowed = {'bayes_loss_bits', 'bayes_released'}
    require(all(all(before[k] == after[k] for k in before if k not in allowed)
                for before, after in zip(baseline_rows, result_rows)),
            'Refinement changed neural scores, comparator scores or source identity')
    summary = copy.deepcopy(baseline)
    for group in summary['groups']:
        key = group['depth'], group['time']
        if key not in TARGETS:
            continue
        selected = [r for r in result_rows if (r['depth'], r['time']) == key]
        require([r['index'] for r in selected] == group['selected_indices'], 'Selected row order differs')
        replacement = replacements[key]
        for name in original.PREDICTORS:
            values = [r[name+'_loss_bits'] for r in selected]
            group['predictors'][name] = dict(loss_bits=original.moments(values),
                regret_bits=original.moments([value-r['entropy_bits'] for value, r in zip(values, selected)]),
                gain_over_uniform_bits=original.moments([6-value for value in values]),
                scientific_status='withheld_numerical_precision' if name == 'bayes'
                    and not replacement['bayes_released'] else 'released')
        group['paired_loss_bits'] = {name+'_minus_'+ref: original.moments(
            [r[name+'_loss_bits']-r[ref+'_loss_bits'] for r in selected]) for name, ref in original.PAIRS}
        group.update(subset_numerical_validation=replacement['numerical_validation'],
                     bayes_released=replacement['bayes_released'],
                     paired_bayes_comparisons_released=replacement['paired_bayes_comparisons_released'])
    summary.update(schema='section4-unseen-context-refined-publication-v1',
        status='Existing fixed selected histories with refined time64 references at depths4 and8; no fresh confirmation.',
        source_exporter_sha256=baseline['script_sha256'], script_sha256=sha(__file__),
        baseline_arithmetic_checks=baseline['arithmetic_checks'],
        arithmetic_checks=dict(replaced_bayesian_rows=changed,
            maximum_refined_loss_replay_error_bits=maximum_loss_error,
            maximum_original_entropy_difference_bits=maximum_entropy_error,
            neural_and_comparator_scores_unchanged=True),
        refinement=aggregate, sampling=False, training=False, model_inference=False)
    return original.native(summary), result_rows


def verify_export(folder):
    """Independent statistics replay against the newly written paired CSV."""
    folder = Path(folder)
    summary = read(folder/'summary.json')
    require(summary['script_sha256'] == sha(__file__)
            and summary['records_sha256'] == sha(folder/'records.csv'), 'Refined producer or table differs')
    with (folder/'records.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    count = 0
    for group in summary['groups']:
        selected = [r for r in rows if (int(r['depth']), int(r['time'])) == (group['depth'], group['time'])]
        require([int(r['index']) for r in selected] == group['selected_indices'], 'Selected CSV coverage differs')
        def check(values, saved):
            nonlocal count
            require(saved['n'] == len(values) == group['n_selected']
                    and math.isclose(statistics.mean(values), saved['mean'], rel_tol=1e-12, abs_tol=2e-14)
                    and math.isclose(statistics.stdev(values)/math.sqrt(len(values)), saved['standard_error'],
                                     rel_tol=1e-12, abs_tol=2e-14), 'Independent CSV statistics differ')
            count += 1
        for name in original.PREDICTORS:
            losses = [float(r[name+'_loss_bits']) for r in selected]
            for field, values in (('loss_bits', losses),
                                 ('regret_bits', [v-float(r['entropy_bits']) for v, r in zip(losses, selected)]),
                                 ('gain_over_uniform_bits', [6-v for v in losses])):
                check(values, group['predictors'][name][field])
        for name, ref in original.PAIRS:
            check([float(r[name+'_loss_bits'])-float(r[ref+'_loss_bits']) for r in selected],
                  group['paired_loss_bits'][name+'_minus_'+ref])
        released = group['bayes_released']
        require(released == group['subset_numerical_validation']['passed']
                == group['paired_bayes_comparisons_released']
                and all((r['bayes_released'] == 'True') == released for r in selected),
                'Refined scientific release flags differ')
        require((group['predictors']['bayes']['scientific_status'] == 'released') == released,
                'Refined predictor status differs')
    require(len(rows) == sum(g['n_selected'] for g in summary['groups']), 'Extra CSV rows')
    return dict(passed=True, statistics_recomputed=count, records=len(rows),
                method='Independent statistics.mean/stdev of complete paired CSV; numerical gates checked separately.')


def load_refinement(plan_path, root, recovery_root=None):
    """Replay only the 246 selected continuations and optional one-case overlay."""
    plan, digest = read_plan(plan_path)
    check_runtime(plan)
    if recovery_root is not None:
        import unseen_context_recovery as recovery
        require(Path(root).resolve() == recovery.PARENT.resolve(),
                'Recovery must overlay its own frozen parent refinement')
    require((Path(root)/'summary.json').exists() and not refinement.busy(Path(root)/'dispatcher.lock'),
            'Wait for the parent dispatcher and automatic merge')
    ctx = read(Path(root)/'context.json')
    require(ctx == dict(schema=refinement.SCHEMA, plan=str(Path(plan_path).resolve()),
                       plan_sha256=digest, output=str(Path(root).resolve())),
            'Refinement context differs')
    bindings = {}
    for path in (plan_path, Path(root)/'context.json', Path(root)/'frozen_manifest.json',
                 Path(root)/'summary.json'):
        bind(path, bindings)
    rows, source_records = [], {}
    old_env = dict(os.environ)
    try:
        os.environ.update(refinement.environment(plan))
        for position, spec in enumerate(plan['tasks']):
            paths = refinement.paths(root, spec)
            require(not refinement.busy(paths['task_lock']) and not refinement.busy(paths['sample_lock']),
                    'Selected refinement still running')
            require(paths['receipt'].exists(), 'Missing selected refinement receipt')
            _, row = refinement.checked_result(plan, ctx, spec, replay=True)
            rows.append(row)
            for path in (paths['receipt'], paths['audit'], paths['prepared']):
                bind(path, bindings)
            for path, expected in row['task']['file_sha256'].items():
                require(bind(path, bindings) == expected, 'Effective refined raw file changed')
            source = refinement.rooted(spec['source_files']['source']['path'])
            require(bind(source, bindings) == spec['source_files']['source']['sha256'],
                    'Selected source checksum changed')
            source_records[spec['task']['depth'], spec['task']['index']] = read(source)
            if position % 12 == 0:
                print(json.dumps(dict(event='selected_reference_checked', completed=position+1, total=246)), flush=True)
        if recovery_root is not None:
            rows, extra = apply_recovery(plan, rows, recovery_root)
            for path, expected in extra.items():
                require(bind(path, bindings) == expected, 'Recovery binding changed')
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return plan, rows, source_records, bindings


def apply_recovery(plan, rows, root):
    """Adapter supplied by the independently audited one-case recovery tool."""
    # The recovery worker owns sample-prefix preservation and unchanged identities.
    import unseen_context_recovery as recovery
    overlay = recovery.validated_overlay(root, replay=True)
    wanted = (overlay['task']['depth'], overlay['task']['index'], overlay['task']['time'], overlay['task']['order'])
    require(wanted == (4, 58, 64, 32), 'Only the declared one-case recovery is permitted')
    matches = [i for i, r in enumerate(rows) if tuple(r['task'][k] for k in ('depth', 'index', 'time', 'order')) == wanted]
    require(len(matches) == 1 and len(rows) == 246, 'Recovery cannot add or remove a selected reference')
    parent_files = read(Path(root)/'plan.json')['parent_files']
    old = rows[matches[0]]['task']
    require(all(parent_files.get(old[name]) == old['file_sha256'][old[name]]
                for name in ('reference', 'checkpoint', 'moments')),
            'Recovered reference does not continue the selected parent files')
    result = list(rows)
    result[matches[0]] = overlay
    bindings = dict(overlay['task']['file_sha256'])
    spec = next(s for s in plan['tasks'] if s['task']['id'] == recovery.TASK_ID)
    paths = refinement.paths(root, spec)
    for path in [Path(root)/name for name in ('plan.json', 'context.json', 'allocation.json')]+[
            paths['receipt'], paths['audit'], paths['prepared'], Path(recovery.__file__)]:
        bindings[str(path.resolve())] = sha(path)
    return result, bindings


def check_runtime(plan):
    require(Path(sys.prefix).resolve() == RUNTIME_PYTHON.parent.parent.resolve()
            and dict(python=list(sys.version_info[:3]), numpy=np.__version__, scipy=scipy.__version__)
            == plan['runtime'] and np.finfo(np.longdouble).nmant+1 == 53,
            'Use the frozen local refinement runtime and 53-bit accumulator')


def build(base=BASE, plan_path=PLAN, refinement_root=REFINEMENT, recovery_root=None):
    check_runtime(read_plan(plan_path)[0])
    baseline, old_rows, files = load_baseline(base)
    plan, rows, sources, added = load_refinement(plan_path, refinement_root, recovery_root)
    require(not (set(files) & set(added)) or all(files[k] == added[k] for k in set(files) & set(added)),
            'Baseline/refinement provenance conflicts')
    files.update(added)
    summary, table = refine_compact(baseline, old_rows, plan, rows, sources)
    summary['inherited_historical_input_sha256'] = summary.pop('input_sha256')
    summary['input_sha256'] = files
    summary['historical_source_export'] = str(Path(base).resolve())
    summary['historical_source_raw_arrays_replayed'] = False
    summary['selected_refinement_raw_arrays_replayed'] = True
    return summary, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--refinement', type=Path, default=REFINEMENT)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print('Unseen-context publication: complete selected cohorts; no sampling or inference.', flush=True)
    if args.verify_only:
        print(json.dumps(verify_export(args.out)))
        return
    require(not args.out.exists(), 'Preserve existing outputs; choose a fresh publication directory')
    summary, rows = build(args.base, args.plan, args.refinement, args.recovery)
    require(summary['refinement']['all_groups_passed'], 'Both unchanged selected-group numerical gates must pass')
    args.out.mkdir(parents=True)
    with (args.out/'records.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary['records_sha256'] = sha(args.out/'records.csv')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    replay = verify_export(args.out)
    validation = dict(schema='section4-unseen-context-refined-validation-v1', passed=True,
        all_subset_numerical_gates_passed=all(g['bayes_released'] for g in summary['groups']),
        withheld_bayes_groups=[dict(depth=g['depth'], time=g['time']) for g in summary['groups'] if not g['bayes_released']],
        groups=len(summary['groups']), selected_records=len(rows), independent_csv_replay=replay,
        arithmetic_checks=summary['arithmetic_checks'], training=False, sampling=False, model_inference=False,
        fresh_confirmation=False, historical_artifacts_modified=False, script_sha256=sha(__file__),
        files={name: sha(args.out/name) for name in ('summary.json', 'records.csv')})
    (args.out/'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, out=str(args.out), replaced_rows=123, fresh_confirmation=False)))


if __name__ == '__main__':
    main()

"""Independent release gate for the fixed copying precision campaign.

Read immutable inference layers, replay the common frequency banks, and retain
the original numerical gates. No sampler runs and no source output is changed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import copying_bayes_precision_chains as precision
import copying_bayes_precision_early_recovery as recovery
import copying_bayes_precision_early4 as extra_recovery
import copying_bayes_precision_banks as frequency
import merge_copying_payoff_parallel as early_rules

b, r = precision.b, precision.r
SCHEMA = 'copying-precision-independent-release-v1'
EXTRA_INDICES = {10804, 10920, 11115, 11210}


def codes():
    return {**precision.codes(), **recovery.codes(), **extra_recovery.codes(), **frequency.codes(),
            str(Path(__file__).relative_to(b.ROOT)): b.sha(__file__),
            'code/scripts/merge_copying_payoff_parallel.py': b.sha(early_rules.__file__)}


def validate_scope(original, replacement, data):
    """Bind all canonical references to every original history and both rules."""
    for g in b.CONDITIONS:
        b.require(data[g+'_tokens'].shape == (256, 17) and
                  data[g+'_rows'].shape == (256, 64, 64), 'Incomplete source cohort')
    references, uses = b.inventory_tokens(
        {g: data[g+'_tokens'] for g in b.CONDITIONS}, range(5, 17))
    expected = {key+f'_a{o}' for key in references for o in (16, 32)}
    tasks = original['tasks']
    found = {s['task']['id'] for s in tasks}
    b.require(len(tasks) == len(found) == 11223 and len(original['retained']) == 1 and
              not found.intersection(original['retained']) and
              found | set(original['retained']) == expected and len(expected) == 11224 and
              len(uses) == 768*12, 'Incomplete or duplicate canonical inference cohort')
    for spec in tasks:
        task = spec['task']; key = task['reference_key']
        b.require(task['tokens'] == references[key]['tokens'] and
                  task['time'] == references[key]['time'] and task['order'] in (16, 32) and
                  task['id'] == key+f"_a{task['order']}", 'Canonical source identity differs')
    assigned = [i for shard in original['shards'] for i in shard['tasks']]
    b.require(sorted(assigned) == list(range(len(tasks))), 'Shard assignment differs')
    ids = original['shards'][322]['tasks']
    b.require(len(ids) == len(replacement['tasks']) == 8 and
              {s['original_index'] for s in replacement['tasks']} == set(ids),
              'Recovery must replace exactly the eight declared early references')
    for spec in replacement['tasks']:
        source = tasks[spec['original_index']]
        b.require(spec['task'] == source['task'] and source['early_fresh'] and
                  spec['target_draws'] == source['target_draws'] == 1048576,
                  'Recovery changed source, seed or draw target')
    return references, sorted(expected)


def load_extra(folder, parent, original):
    p = b.read(folder/'plan.json')
    b.require(p['schema'] == 'copying-precision-early4-recovery-v1' and
              p['plan_id'] == b.fp({k: v for k, v in p.items() if k != 'plan_id'}),
              'Additional recovery plan changed')
    b.require(p['code_sha256'] == extra_recovery.codes(), 'Additional recovery implementation differs')
    for name, digest in p['code_sha256'].items():
        b.require(b.sha(b.ROOT/name) == digest, 'Additional recovery code changed')
    r.verify(p['parent_plan']); r.verify(p['submission'])
    b.require(Path(p['parent']).resolve() == parent.resolve() and
              p['parent_plan']['sha256'] == b.sha(parent/'plan.json') and
              p['runtime'] == original['runtime'] and p['terminal_verified'] is True and
              set(b.read(p['submission']['path'])['original_indices']) == EXTRA_INDICES,
              'Additional recovery parent or terminal scope differs')
    eligible = {original['tasks'][i]['task']['id']: i for i in EXTRA_INDICES}
    tasks = {s['task']['id']: s for s in p['tasks']}
    b.require(len(tasks) == len(p['tasks']) and not set(tasks).intersection(p['retained']) and
              set(tasks) | set(p['retained']) == set(eligible), 'Additional recovery scope differs')
    for key, spec in tasks.items():
        source = original['tasks'][eligible[key]]
        b.require(spec['original_index'] == eligible[key] and spec['task'] == source['task'] and
                  spec['target_draws'] in (source['target_draws'], 2*source['target_draws']),
                  'Additional recovery changed source, seed or declared draw target')
    for key, binding in p['retained'].items():
        r.verify(binding)
        source = original['tasks'][eligible[key]]
        b.require(Path(binding['path']) == b.accuracy.paths(parent, source['task'])['report'],
                  'Additional recovery retained unrelated reference')
        report = b.read(binding['path'])
        b.require(report['complete'] and report['audit_passed'] and
                  report['task'] == source and report['plan_id'] == original['plan_id'],
                  'Additional recovery retained incomplete reference')
    return p


def prepare(out, parent, recovered, banks, extra=None):
    b.require(not (out/'plan.json').exists(), 'Release already prepared')
    for protected in (parent, recovered, banks, *([extra] if extra else [])):
        a, z = out.resolve(), protected.resolve()
        b.require(a != z and a not in z.parents and z not in a.parents,
                  'Release output overlaps an input layer')
    original, _ = precision.load(parent)
    replacement, _ = recovery.load(recovered)
    bank_plan = frequency.load(banks)
    b.require(Path(replacement['parent']).resolve() == parent.resolve() and
              replacement['runtime'] == original['runtime'], 'Recovery parent differs')
    b.require(Path(bank_plan['parent']).resolve() == Path(original['parent']).resolve() and
              Path(original['banks']).resolve() == banks.resolve(), 'Frequency cohort differs')
    data = b.archive(Path(original['parent'])/'input.npz')
    _, expected = validate_scope(original, replacement, data)
    additional = load_extra(extra, parent, original) if extra else None
    folders = [('parent', parent), ('recovery', recovered), ('banks', banks)]
    if extra:
        folders.append(('extra', extra))
    plan = dict(schema=SCHEMA, parent=str(parent.resolve()), recovery=str(recovered.resolve()),
                banks=str(banks.resolve()), extra=str(extra.resolve()) if extra else None,
                code_sha256=codes(), input_plans={name: r.bind(folder/'plan.json') for name, folder in folders},
                source_input=r.bind(Path(original['parent'])/'input.npz'),
                expected_reference_ids=expected, original_histories=768,
                times=list(range(1, 17)), replacement_ids=sorted(
                    s['task']['id'] for s in replacement['tasks'] + (additional['tasks'] if additional else [])),
                numerical_allowance_bits=.01/3, source_se_fraction=.25,
                automatic_retries=False, sampling_performed=False)
    plan['plan_id'] = b.fp(plan)
    b.write(out/'plan.json', plan, immutable=True)
    print(json.dumps(dict(event='release_prepared', references=len(expected),
                         replacements=len(plan['replacement_ids']), plan_id=plan['plan_id'])), flush=True)


def load(out):
    p = b.read(out/'plan.json')
    b.require(p['schema'] == SCHEMA and
              p['plan_id'] == b.fp({k: v for k, v in p.items() if k != 'plan_id'}) and
              p['code_sha256'] == codes(), 'Release plan or code changed')
    for binding in (*p['input_plans'].values(), p['source_input']):
        r.verify(binding)
    original, _ = precision.load(Path(p['parent']))
    replacement, _ = recovery.load(Path(p['recovery']))
    bank_plan = frequency.load(Path(p['banks']))
    b.require(bank_plan['plan_id'] == b.read(p['input_plans']['banks']['path'])['plan_id'] and
              Path(bank_plan['parent']).resolve() == Path(original['parent']).resolve() and
              Path(replacement['parent']).resolve() == Path(p['parent']).resolve() and
              replacement['runtime'] == original['runtime'], 'Inference/frequency layers differ')
    data = b.archive(p['source_input']['path'])
    _, expected = validate_scope(original, replacement, data)
    additional = load_extra(Path(p['extra']), Path(p['parent']), original) if p['extra'] else None
    b.require(expected == p['expected_reference_ids'] and p['original_histories'] == 768 and
              p['times'] == list(range(1, 17)) and p['numerical_allowance_bits'] == .01/3 and
              p['source_se_fraction'] == .25, 'Release scope or numerical budget changed')
    b.require(p['replacement_ids'] == sorted(s['task']['id'] for s in
              replacement['tasks'] + (additional['tasks'] if additional else [])), 'Replacement list changed')
    return p, original, replacement, additional, bank_plan, data


def read_reference(folder, spec, plan, bindings, retained=None):
    task = spec['task'] if spec else None
    path = b.accuracy.paths(folder, task)['report'] if task else Path(retained['path'])
    if not path.exists():
        return None
    if retained:
        r.verify(retained)
    report = b.read(path)
    if task is None:
        task = report['task'].get('task', report['task'])
    paths = b.accuracy.paths(path.parents[2], task)
    b.require(paths['report'] == path, 'Reference path differs')
    if not report['complete']:
        return None
    ident, audit = b.read(paths['identity']), b.read(paths['audit'])
    b.require(ident['runtime'] == plan['runtime'] and ident['task'] == report['task'],
              'Reference runtime or task identity differs')
    if spec:
        b.require(report['task'] == spec and report['plan_id'] == ident['plan_id'] == plan['plan_id'],
                  'Reference plan/specification differs')
    for key in ('identity', 'audit', 'moments', 'checkpoint'):
        digest = b.sha(paths[key])
        b.require(digest == report[key+'_sha256'], 'Reference artifact changed: '+task['id'])
        bindings[str(paths[key])] = digest
    bindings[str(path)] = b.sha(path)
    b.require(audit['raw_checks'] and all(x['passed'] for x in audit['raw_checks']) and
              audit['raw_inputs']['canonical_accuracy:checkpoint']['sha256'] ==
              report['checkpoint_sha256'], 'Raw sample audit does not bind checkpoint')
    grids = audit['final_state_grid_checks']
    passed = bool(audit['passed'] and len(grids) == 4 and all(g['passed'] for g in grids))
    b.require(report['audit_passed'] == passed, 'Stored numerical gate differs')
    moments = b.archive(paths['moments'])
    draws = int(moments['draws'])
    if spec:
        b.require(draws == spec['target_draws'], 'Fixed draw target changed')
    precision.checked_moments(moments, report['estimate'], len(set(task['tokens'])), draws)
    if spec and spec['parent_files']:
        for item in spec['parent_files'].values():
            r.verify(item); bindings[item['path']] = item['sha256']
        r.preserve(spec['parent_files']['checkpoint']['path'], paths['checkpoint'], draws)
    return task, moments, passed


def frequency_arrays(plan, bank_plan, data, bindings):
    folder, parent = Path(plan['banks']), Path(bank_plan['parent'])
    path = folder/'frequency_summary.json'; report = b.read(path)
    b.require(report['schema'] == frequency.SCHEMA and report['plan_id'] == bank_plan['plan_id'] and
              report['complete'] and report['banks'] == 256 and
              report['retained_bank_parity_passed'], 'Frequency release is incomplete or unrelated')
    arrays_path = folder/'frequency_arrays.npz'
    b.require(b.sha(arrays_path) == report['arrays_sha256'], 'Frequency scores changed')
    bindings[str(path)] = b.sha(path); bindings[str(arrays_path)] = report['arrays_sha256']
    banks, expected = [], {}
    for i in range(256):
        root = (parent if i < 16 else folder)/'banks'/str(i)
        receipt, raw = root/'report.json', root/'moments.npz'
        entry = b.read(receipt)
        expected_plan = b.read(parent/'plan.json')['plan_id'] if i < 16 else bank_plan['plan_id']
        b.require(entry['complete'] and entry['index'] == i and entry['plan_id'] == expected_plan and
                  b.sha(raw) == entry['arrays_sha256'], 'Frequency bank provenance differs')
        if i < 16:
            frozen = bank_plan['existing_banks'][i]
            b.require(frozen['index'] == i and Path(frozen['path']).resolve() == root.resolve() and
                      b.sha(receipt) == frozen['report_sha256'] and
                      entry['arrays_sha256'] == frozen['moments_sha256'], 'Retained bank changed')
        expected[str(raw)] = entry['arrays_sha256']
        bindings[str(receipt)] = b.sha(receipt); bindings[str(raw)] = entry['arrays_sha256']
        banks.append(b.archive(raw))
    b.require(report['input_bindings'] == expected, 'Frequency bank inventory differs')
    arrays = b.archive(arrays_path)
    kinds = [(x[0], tuple(x[1])) for x in b.read(parent/'kinds.json')]
    for key, name, leave_name in (
            ('fine', 'stationary_loss', 'stationary_leave_one_bank_out_loss'),
            ('coarse', 'stationary_coarse_loss', 'stationary_coarse_leave_one_bank_out_loss')):
        joints = np.stack([bank[key] for bank in banks])
        old, generalized = b.score_frequency(joints[:16], data, kinds), frequency.score_frequency(joints[:16], data, kinds)
        for g in b.CONDITIONS:
            b.require(all(np.array_equal(a, z) for a, z in zip(old[g], generalized[g])),
                      'Original 16-bank estimator parity failed')
        replay = frequency.score_frequency(joints, data, kinds)
        for g in b.CONDITIONS:
            b.require(np.array_equal(replay[g][0], arrays[g+'_'+name]) and
                      np.array_equal(replay[g][1], arrays[g+'_'+leave_name]),
                      '256-bank scoring replay differs')
        print(json.dumps(dict(event='release_frequency_replayed', rule=key)), flush=True)
    return arrays


def paired_failures(refs):
    late = {o: {k: v for k, v in rows.items() if int(k[1:5]) >= 9} for o, rows in refs.items()}
    failed = r.paired_failures(late)
    for key, left in refs[16].items():
        n = int(key[1:5]); right = refs[32][key]
        if n == 5:
            check = b.accuracy.paired_order_check(left['mean'], left['mcse_probs'], right['mean'], right['mcse_probs'])
        elif n in (6, 7, 8):
            check = early_rules.paired_order_check(left, right)
        else:
            continue
        if not check['passed']:
            failed.append(key)
    return failed


def merge(out):
    p, original, replacement, additional, bank_plan, data = load(out)
    b.require(not (out/'summary.json').exists(), 'Release already attempted')
    result = dict(schema=SCHEMA, plan_id=p['plan_id'], complete=False,
                  population_reference_approved=False, aggregate_precision_passed=False,
                  publication_status='diagnostic_nonpublication', payoff_forecast_approved=False)
    bindings = {v['path']: v['sha256'] for v in (*p['input_plans'].values(), p['source_input'])}
    refs = {16: {}, 32: {}}; missing, failed, seeds = [], [], set()
    changed = {s['original_index']: (Path(p['recovery']), s, replacement, None) for s in replacement['tasks']}
    if additional:
        for spec in additional['tasks']:
            b.require(spec['original_index'] not in changed, 'Overlapping replacement layers')
            changed[spec['original_index']] = (Path(p['extra']), spec, additional, None)
        for binding in additional['retained'].values():
            bindings[binding['path']] = binding['sha256']
    entries = [changed[i] if i in changed else
               (Path(p['parent']), spec, original, None) for i, spec in enumerate(original['tasks'])]
    entries += [(None, None, original, binding) for binding in original['retained'].values()]
    try:
        for i, (folder, spec, layer, retained) in enumerate(entries):
            value = read_reference(folder, spec, layer, bindings, retained)
            if value is None:
                missing.append(spec['task']['id'] if spec else retained['path']); continue
            task, moments, passed = value
            b.require(task['seed'] not in seeds and task['reference_key'] not in refs[task['order']],
                      'Duplicate inference seed/reference')
            seeds.add(task['seed']); refs[task['order']][task['reference_key']] = moments
            if not passed:
                failed.append(task['id'])
            if i % 100 == 0:
                print(json.dumps(dict(event='release_reference_verified', completed=i+1, total=len(entries))), flush=True)
        result.update(complete=not missing, missing=missing, failed_audits=failed)
        if missing or failed:
            b.write(out/'summary.json', result); return 3
        b.require(len(refs[16]) == len(refs[32]) == 5612 and
                  set(refs[16]) == set(refs[32]) and
                  {key+f'_a{o}' for o in refs for key in refs[o]} == set(p['expected_reference_ids']),
                  'Complete paired coverage required')
        failed_pairs = paired_failures(refs); result['failed_paired_orders'] = failed_pairs
        if failed_pairs:
            b.write(out/'summary.json', result); return 3
        arrays = frequency_arrays(p, bank_plan, data, bindings)
        groups = []
        for g in b.CONDITIONS:
            losses, mcse, bias = {}, {}, {}
            for order in (16, 32):
                loss = np.empty((256, 16))
                loss[:, :4] = data[g+('_bayes_loss' if order == 32 else '_bayes_loss_a16')][:, :4]
                mcse[order], bias[order] = np.zeros(16), np.zeros(16)
                for n in range(5, 17):
                    oracles, keys = [], []
                    for i, row in enumerate(data[g+'_tokens']):
                        pattern, labels = b.canonicalize(row[:n]); key = b.reference_key(pattern)
                        oracle = data[g+'_rows'][i, row[n-1], b.alphabet_map(labels)]
                        loss[i, n-1] = -oracle@np.log2(refs[order][key]['mean'])
                        oracles.append(oracle); keys.append(key)
                    error = b.shared_error(np.asarray(oracles), keys, refs[order])
                    mcse[order][n-1], bias[order][n-1] = error['numerical_mcse_bits'], error['log_bias_proxy_bits']
                losses[order] = loss; arrays[g+f'_bayes_loss_a{order}'] = loss
            f, c, fl, cl = [arrays[g+'_'+k] for k in ('stationary_loss', 'stationary_coarse_loss',
                'stationary_leave_one_bank_out_loss', 'stationary_coarse_leave_one_bank_out_loss')]
            checks = {name: b.allowance(f, c, fl, cl, losses[32], losses[16], mcse[32], mcse[16], bias[32], bias[16], cumulative)
                      for name, cumulative in (('instantaneous', False), ('cumulative', True))}
            groups.append(dict(condition=g, **checks))
            for role in ('ordered', 'shuffled'):
                for seed in (0, 1):
                    arrays[g+f'_{role}_seed{seed}'] = data[g+f'_{role}_seed{seed}']
        approved = all(g[name]['passed'] for g in groups for name in ('instantaneous', 'cumulative'))
        b.save(out/'arrays.npz', **arrays); b.write(out/'input_bindings.json', bindings, immutable=True)
        result.update(groups=groups, aggregate_precision_passed=approved, population_reference_approved=approved,
            publication_status='approved' if approved else 'diagnostic_nonpublication',
            arrays_sha256=b.sha(out/'arrays.npz'), input_bindings_sha256=b.sha(out/'input_bindings.json'),
            original_histories=768, times=list(range(1, 17)), independent_frequency_replay_passed=True,
            reference_count=11224, retained_reference_count=1, replacement_reference_count=len(changed),
            comparator='Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.',
            uncertainty='Shared canonical gradients; common-bank delete-one covariance; independent-time MCMC seeds.',
            scope='Instantaneous and cumulative gains only; no payoff crossing bootstrap.')
        b.write(out/'summary.json', result)
        return 0 if approved else 3
    except Exception as error:
        result.update(population_reference_approved=False, aggregate_precision_passed=False,
                      publication_status='provenance_or_validation_failure', error=str(error))
        b.write(out/'summary.json', result)
        raise


def main():
    print('Copying release: full cohort, independent bank replay, unchanged numerical gates.', flush=True)
    b.require(os.environ.get('SLURM_JOB_ID'), 'Run the release audit on a compute allocation')
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage', choices=('prepare', 'merge')); a.add_argument('--out', type=Path, required=True)
    for name in ('parent', 'recovery', 'banks', 'extra'):
        a.add_argument('--'+name, type=Path)
    args = a.parse_args()
    if args.stage == 'prepare':
        prepare(args.out, args.parent, args.recovery, args.banks, args.extra); return 0
    return merge(args.out)


if __name__ == '__main__':
    raise SystemExit(main())

"""Frozen, bounded t16 pooled mean calculation. Existing campaigns are untouched.

All 192 histories and four independent rules are mandatory. A separate exact
short-history mean gate precedes population sampling. Nothing retries itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    print('Pooled mean protocol: bounded complete-cohort work; no retries or later contexts.', flush=True)
sys.path[:0] = [str(ROOT / 'code/scripts'), str(ROOT / 'code/transformer')]
import numpy as np
from numpy.polynomial.legendre import leggauss
import pooled_predictive_calibration as cal
import pooled_predictive_population as old
import pooled_mean_validation as validation
from shared_base_predictive_smc import estimate, component_prediction, combine_components
from shared_base_short_evidence import SharedBaseShortEvidence
from mixture_family_evidence import FamilyEvidence
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC

PLAN = ROOT / 'code/transformer/configs/pooled_mean_population_plan_2026-09-20.json'
ARMS = {
    'A': dict(particles=256, temperatures=64, moves=4, alpha_order=16,
              row_bounds=[-32., 32.], row_step=1/32),
    'B': dict(particles=1024, temperatures=64, moves=4, alpha_order=16,
              row_bounds=[-32., 32.], row_step=1/32),
    'C': dict(particles=1024, temperatures=64, moves=4, alpha_order=32,
              row_bounds=[-32., 32.], row_step=1/32),
    'D': dict(particles=1024, temperatures=64, moves=4, alpha_order=32,
              row_bounds=[-40., 40.], row_step=1/64),
}
require, sha, digest, save = cal.require, cal.sha, cal.digest, cal.atomic_json


def runtime():
    return dict(python=sys.version, numpy=np.__version__, scipy=cal.scipy.__version__)


def read_plan(path=PLAN):
    p = json.loads(Path(path).read_text())
    require(p['schema'] == 'pooled-mean-population-plan-v1', 'Wrong protocol')
    require(p['arms'] == ARMS and p['repetitions'] == 32 and p['observed'] == 16
            and p['term_cache_size'] == 4096 and p['publication_arm'] == 'D',
            'Scientific settings changed')
    require(p['indices'] == list(range(5, 69)) and p['families'] == list(old.FAMILIES)
            and p['family_prior'] == [1/3]*3, 'Cohort changed')
    require(p['maximum_attempts_per_task'] == 1 and p['automatic_retries'] is False,
            'Unbounded or automatic retry forbidden')
    require(p['criteria'] == validation.CRITERIA, 'Numerical acceptance criteria changed')
    require(sha(ROOT/p['parent_plan']['path']) == p['parent_plan']['sha256'],
            'Original selection plan changed')
    parent, _ = old.read_plan(ROOT/p['parent_plan']['path'])
    require(p['selection_digest'] == parent['selection_digest'], 'Selection changed')
    required = {'code/scripts/pooled_mean_population.py', 'code/scripts/pooled_mean_validation.py',
                'code/scripts/pooled_predictive_population.py',
                'code/transformer/scitas/pooled_mean_population.sbatch',
                'code/transformer/scitas/submit_pooled_mean_population.sh'}
    require(set(p['code_sha256']) == required, 'Incomplete implementation binding')
    for name, expected in p['code_sha256'].items():
        require(sha(ROOT/name) == expected, 'Frozen source changed: '+name)
    s = p['scheduler']
    require(s['calibration']['tasks'] == 52 and s['population']['tasks'] == 768
            and 1 <= s['concurrency'] <= 64 and s['maximum_array_size'] <= 768,
            'Scheduler coverage changed')
    for stage in ('calibration', 'population'):
        require(1 <= s[stage]['allocation_minutes'] <= 240, 'Unbounded allocation')
    require(p['worker_margin_seconds'] >= 90, 'Insufficient checkpoint margin')
    return p, sha(path), parent


def canonical_labels(tokens, d=64):
    seen = list(dict.fromkeys(map(int, tokens)))
    order = seen + [i for i in range(d) if i not in seen]
    inverse = {v: i for i, v in enumerate(order)}
    return [inverse[int(v)] for v in tokens], order


def reorder_component(component, order):
    old.checked_component(component)
    return dict(log_evidence=component['log_evidence'],
                predictive_probs=np.asarray(component['predictive_probs'])[order].tolist(),
                log_joint_next=np.asarray(component['log_joint_next'])[order].tolist())


def load_sources(raw, parent, indices=None):
    """Hash each frozen file once, then read each cohort once, never per task."""
    raw = Path(raw)
    selection = parent['selection']
    hashes = {name: sha(raw/name) for name in selection['input_sha256']}
    require(hashes == selection['input_sha256'], 'Frozen inputs changed')
    selected = set(parent['indices'] if indices is None else indices)
    result = []
    for family in old.FAMILIES:
        cohort = family+'_prior'
        receipt = json.loads((raw/f'heldout/{cohort}.json').read_text())
        require(receipt['plan_id'] == selection['source_plan_id']
                and receipt['data_sha256'] == hashes[f'heldout/{cohort}.npz'], 'Source receipt changed')
        with np.load(raw/f'heldout/{cohort}.npz', allow_pickle=False) as a:
            require(np.array_equal(a['prefixes'], old.ALL_TIMES), 'Wrong prediction times')
            source = {name: a[name] for name in ('tokens', 'sequence_seed', 'oracle_rows', 'transition_probs')}
        scores = {}
        for model in ('pooled_seed0', 'pooled_seed1'):
            r = json.loads((raw/f'eval/{model}/{cohort}.json').read_text())
            require(r['data_sha256'] == hashes[f'eval/{model}/{cohort}.npz']
                    and r['source_data_sha256'] == receipt['data_sha256']
                    and r['identity']['name'] == model
                    and r['identity']['source_plan_id'] == selection['source_plan_id'], 'Model receipt changed')
            with np.load(raw/f'eval/{model}/{cohort}.npz', allow_pickle=False) as a:
                require(np.array_equal(a['prefixes'], old.ALL_TIMES)
                        and np.array_equal(a['sequence_seed'], source['sequence_seed']), 'Model pairing changed')
                scores[model] = a['next_kl_bits']
        for case in selection['cases']:
            if case['family'] != family or case['index'] not in selected:
                continue
            i = case['index']
            require(np.array_equal(source['sequence_seed'][i], case['sequence_seed']), 'Case seed changed')
            for t in (1, 4, 16):
                col = old.ALL_TIMES.index(t)
                tokens = source['tokens'][i, :t].tolist()
                oracle = source['oracle_rows'][i, col]
                require(np.array_equal(oracle, source['transition_probs'][i, tokens[-1]]), 'Oracle off by one')
                old.checked_probs(oracle, 64)
                tf = {m: float(v[i, col]) for m, v in scores.items()}
                require(all(math.isfinite(v) for v in tf.values()), 'Invalid transformer score')
                result.append(dict(family=family, index=i, observed=t, tokens=tokens,
                                   oracle_probs=oracle.tolist(), transformer_regret_bits=tf))
    return result, hashes


def dependency_state(dependencies, *, full):
    dep = Path(dependencies)
    cp, _ = cal.read_plan(cal.PLAN)  # Explicit path also covers the former merge startup bug.
    store, src = dep/'tables/anchors_prod', dep/'src'
    require((store/'anchors.json').is_file(), 'A sealed production store is required')
    env = cal.numerical_environment(store, src)
    manifest = json.loads((ROOT/cp['dependencies']['manifest']).read_text())
    require(sha(ROOT/cp['dependencies']['manifest']) == cp['dependencies']['sha256'], 'Dependency manifest changed')
    if full:
        cal.dependencies(cp, store, src)
    fingerprints = {}
    for name, record in manifest['files'].items():
        path = dep/name
        require(path.is_file() and not path.is_symlink(), 'Missing or linked dependency: '+name)
        s = path.stat()
        require(s.st_size == record['size_bytes'], 'Dependency size changed: '+name)
        fingerprints[name] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns)
    return dict(manifest_sha256=cp['dependencies']['sha256'], files=fingerprints, environment=env)


def build_prepared(cases):
    engines = {rule: {f: FamilyEvidence(f, bounds=bounds, step=step)
                     for f in ('product', 'uniform_base')}
               for rule, bounds, step in (('default', (-30., 30.), 1/64),
                                          ('refined', (-40., 40.), 1/128))}
    catalog = SharedBaseShortEvidence()
    groups = {str(t): dict(references={}, histories=[], exact_rows=[]) for t in (1, 4, 16)}
    for number, case in enumerate(cases):
        t, tokens = case['observed'], case['tokens']
        if t <= 4:
            canonical, order = canonical_labels(tokens)
            key = 'n'+str(t)+'_'+''.join(map(str, canonical))
        else:
            canonical, order = tokens, list(range(64))
            key = case['family']+'_'+str(case['index'])
        group = groups[str(t)]
        group['references'][key] = dict(tokens=canonical, observed=t)
        fixed = {rule: [reorder_component(component_prediction(e, tokens), order)
                        for e in values.values()] for rule, values in engines.items()}
        oracle = np.asarray(case['oracle_probs'])[order].tolist()
        history = dict(family=case['family'], index=case['index'], reference_key=key,
                       original_tokens=tokens, canonical_to_original=order,
                       oracle_probs=oracle, fixed_components=fixed['default'],
                       refined_fixed_components=fixed['refined'],
                       transformer_regret_bits=case['transformer_regret_bits'])
        group['histories'].append(history)
        if t <= 4:
            exact = cal.exact_reference(catalog, canonical)
            parts = [fixed['refined'][0], exact, fixed['refined'][1]]
            mixture = combine_components([x['log_evidence'] for x in parts],
                                         [x['predictive_probs'] for x in parts])
            group['exact_rows'].append(dict(family=case['family'], index=case['index'],
                bayes_regret_bits=old.regret(oracle, mixture['predictive_probs']),
                transformer_regret_bits=case['transformer_regret_bits']))
        if number % 24 == 0:
            print(json.dumps(dict(event='preparing', completed_histories=number+1, total=len(cases))), flush=True)
    return groups


def prepare(plan_path, raw, dependencies, out, *, preflight=False):
    p, h, parent = read_plan(plan_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out/'prepared.json'
    with cal.writer_claim(path):
        require(not path.exists(), 'Prepared output already exists; inspect it, do not overwrite')
        dep = dependency_state(dependencies, full=True)
        cases, hashes = load_sources(raw, parent, indices=[5] if preflight else None)
        groups = build_prepared(cases)
        if not preflight:
            require([len(groups[str(t)]['histories']) for t in (1, 4, 16)] == [192]*3, 'Incomplete cohort')
            require(len(groups['1']['references'])+len(groups['4']['references']) == 13, 'Short pattern coverage changed')
        report = dict(schema='pooled-mean-prepared-v1', plan_sha256=h,
                      source_sha256=hashes, dependencies=dep, groups=groups,
                      runtime=runtime(), preflight_only=preflight, scientific_reference_validated=False)
        save(path, report)
        save(out/'plan.json', p)
    print(json.dumps(dict(event='prepared', preflight_only=preflight, path=str(path))), flush=True)
    return report


def read_prepared(p, h, out, dependencies, *, full=False):
    path = Path(out)/'prepared.json'
    r = json.loads(path.read_text())
    require(r['schema'] == 'pooled-mean-prepared-v1' and r['plan_sha256'] == h
            and r['preflight_only'] is False, 'Wrong or smoke-only prepared data')
    require(r['runtime'] == runtime(), 'Numerical runtime changed after preparation')
    require(r['dependencies'] == dependency_state(dependencies, full=full), 'Frozen dependencies changed')
    return r, sha(path)


def tasks(prepared, stage):
    require(stage in ('calibration', 'population'), 'Unknown stage')
    result = []
    for t in ((1, 4) if stage == 'calibration' else (16,)):
        for key, ref in sorted(prepared['groups'][str(t)]['references'].items()):
            for arm in ARMS:
                result.append(dict(task_index=len(result), stage=stage, observed=t,
                                   reference_key=key, arm=arm, tokens=ref['tokens']))
    return result


def seed_for(plan_hash, spec, repetition):
    return int(digest(['pooled-mean-v1', plan_hash, spec, repetition])[:16], 16)


def check_record(record, p, h, spec, repetition):
    settings = ARMS[spec['arm']]
    require(record['schema'] == 'shared-base-predictive-smc-repetition-v1'
            and record['seed'] == seed_for(h, spec, repetition)
            and record['n_observed'] == spec['observed']
            and record['resample_fraction'] == .75
            and all(record[k] == settings[k] for k in ('particles', 'temperatures', 'moves')),
            'Repetition identity changed')
    old.checked_component(record)
    require(0 <= record['maximum_conditional_normalization_error'] <= 1e-7,
            'Predictive normalization failed')
    ess = np.asarray(record['effective_sample_sizes'])
    require(ess.shape == (settings['temperatures'],) and np.isfinite(ess).all()
            and np.all(ess > 0), 'Invalid particle diagnostics')


def task_path(out, stage, index):
    return Path(out)/stage/'tasks'/f'task-{index:04d}.json'


def check_task(r, p, h, prepared_hash, spec):
    require(r['schema'] == 'pooled-mean-task-v1' and r['plan_sha256'] == h
            and r['prepared_sha256'] == prepared_hash and r['task'] == spec,
            'Task provenance changed')
    require(r['records_sha256'] == digest(r['records']), 'Records changed')
    require(len(r['records']) <= p['repetitions']
            and r['complete'] == (len(r['records']) == p['repetitions']), 'Coverage flag changed')
    for j, record in enumerate(r['records']):
        check_record(record, p, h, spec, j)


def calibration_gate(out, h, prepared_hash):
    path = Path(out)/'calibration'/'summary.json'
    r = json.loads(path.read_text())
    require(r['schema'] == 'pooled-mean-stage-v1' and r['stage'] == 'calibration'
            and r['plan_sha256'] == h and r['prepared_sha256'] == prepared_hash
            and r['complete'] is True and r['passed'] is True, 'Exact-short mean gate has not passed')
    for name, expected in r['task_sha256'].items():
        require(sha(Path(out)/name) == expected, 'Calibration task changed')
    require(len(r['task_sha256']) == 52, 'Calibration gate coverage changed')
    return sha(path)


def run_task(plan_path, stage, index, dependencies, out):
    p, h, _ = read_plan(plan_path)
    prepared, ph = read_prepared(p, h, out, dependencies)
    specs = tasks(prepared, stage)
    require(len(specs) == p['scheduler'][stage]['tasks'] and 0 <= index < len(specs), 'Task index outside plan')
    spec = specs[index]
    gate = calibration_gate(out, h, ph) if stage == 'population' else None
    path = task_path(out, stage, index); path.parent.mkdir(parents=True, exist_ok=True)
    with cal.writer_claim(path):
        require(not path.exists(), 'Task already attempted; preserve it and review before any continuation')
        settings = ARMS[spec['arm']]
        report = dict(schema='pooled-mean-task-v1', plan_sha256=h, prepared_sha256=ph,
                      task=spec, calibration_summary_sha256=gate, records=[], records_sha256=digest([]),
                      complete=False, status='starting', scientific_reference_validated=False)
        save(path, report)
        nodes, weights = leggauss(settings['alpha_order'])
        dep = Path(dependencies)
        kernel = RankedLSARowEvidence(64, 1+nodes/2, depths=range(9),
            store=dep/'tables/anchors_prod', src=dep/'src', u_lo=settings['row_bounds'][0],
            u_hi=settings['row_bounds'][1], step=settings['row_step'], term_cache_size=p['term_cache_size'])
        sampler = SharedBaseMCMC(64, kernel, weights/2)
        started = time.monotonic()
        limit = 60*p['scheduler'][stage]['allocation_minutes'] - p['worker_margin_seconds']
        def interrupted(sig, frame):
            raise InterruptedError('Signal '+signal.Signals(sig).name)
        handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
        try:
            for j in range(p['repetitions']):
                seconds = min(p['seconds_per_repetition'], limit-(time.monotonic()-started))
                if seconds <= 0:
                    raise TimeoutError('Bounded allocation exhausted')
                print(json.dumps(dict(event='repetition_started', stage=stage, task=index,
                                      arm=spec['arm'], repetition=j, seconds_limit=seconds)), flush=True)
                with old.heartbeat(path.with_suffix('.progress.json'), dict(stage=stage, task=index, completed_repetitions=j)):
                    record = estimate(sampler, spec['tokens'], **{k: settings[k] for k in ('particles', 'temperatures', 'moves')},
                                      seed=seed_for(h, spec, j), resample_fraction=.75, seconds_limit=seconds)
                check_record(record, p, h, spec, j)
                report['records'].append(record)
                report.update(records_sha256=digest(report['records']), complete=j+1 == p['repetitions'],
                              status='complete' if j+1 == p['repetitions'] else 'running',
                              elapsed_seconds=time.monotonic()-started)
                save(path, report)
                print(json.dumps(dict(event='repetition_complete', stage=stage, task=index, completed=j+1)), flush=True)
        except (TimeoutError, InterruptedError) as error:
            report.update(status='incomplete', error=str(error), elapsed_seconds=time.monotonic()-started)
            save(path, report)
            return 75
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
    return 0


def merge(plan_path, stage, raw, dependencies, out, *, write=True):
    p, h, parent = read_plan(plan_path)
    prepared, ph = read_prepared(p, h, out, dependencies, full=True)
    # Revalidate source hashes once on the merge compute node.
    require({name: sha(Path(raw)/name) for name in prepared['source_sha256']} == prepared['source_sha256'],
            'Publication inputs changed')
    specs = tasks(prepared, stage)
    require(len(specs) == p['scheduler'][stage]['tasks'], 'Prepared task coverage changed')
    gate = None
    if stage == 'population':
        gate = calibration_gate(out, h, ph)
        saved_gate = json.loads((Path(out)/'calibration'/'summary.json').read_text())
        replay = merge(plan_path, 'calibration', raw, dependencies, out, write=False)
        require(replay == saved_gate, 'Exact-short gate does not independently replay')
    missing, grouped, hashes = [], {}, {}
    for spec in specs:
        path = task_path(out, stage, spec['task_index'])
        if not path.exists():
            missing.append(spec['task_index']); continue
        r = json.loads(path.read_text()); check_task(r, p, h, ph, spec)
        require(r['calibration_summary_sha256'] == gate, 'Task calibration gate changed')
        hashes[str(path.relative_to(out))] = sha(path)
        if not r['complete']:
            missing.append(spec['task_index']); continue
        grouped.setdefault(str(spec['observed']), {}).setdefault(spec['arm'], {})[spec['reference_key']] = r['records']
    result = dict(schema='pooled-mean-stage-v1', stage=stage, plan_sha256=h,
                  prepared_sha256=ph, complete=not missing, passed=False, missing_tasks=missing,
                  task_sha256=hashes, scientific_reference_validated=False, by_context={})
    if not missing:
        for t, arm_records in grouped.items():
            group = prepared['groups'][t]
            analyses = {a: validation.analyze_arm(records, group['histories']) for a, records in arm_records.items()}
            refined_histories = [dict(row, fixed_components=row['refined_fixed_components']) for row in group['histories']]
            refined = validation.analyze_arm(arm_records['D'], refined_histories)
            check = validation.validate_campaign(analyses, cheap_refined=refined)
            if stage == 'calibration':
                check['exact_short'] = validation.exact_short_validation(analyses, group['exact_rows'], cheap_refined=refined)
                check['passed'] = check['passed'] and check['exact_short']['passed']
            result['by_context'][t] = dict(checks=check, passed=check['passed'],
                arms={a: v['summary'] for a, v in analyses.items()}, publication=refined)
        result['passed'] = all(v['passed'] for v in result['by_context'].values())
        if stage == 'population':
            result['calibration_summary_sha256'] = gate
            result['scientific_reference_validated'] = result['passed']
    if write:
        save(Path(out)/stage/'summary.json', result)
    print(json.dumps({k: result[k] for k in ('stage', 'complete', 'passed', 'scientific_reference_validated')}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'calibration', 'population', 'merge'))
    ap.add_argument('--plan', type=Path, default=PLAN)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--dependencies', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=('calibration', 'population'))
    ap.add_argument('--task-index', type=int)
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.plan, a.raw, a.dependencies, a.out, preflight=a.preflight); return 0
    require(not a.preflight, 'Preflight applies only to preparation')
    if a.command == 'merge':
        require(a.stage is not None, 'Merge needs --stage')
        return 0 if merge(a.plan, a.stage, a.raw, a.dependencies, a.out)['passed'] else 3
    require(a.task_index is not None, 'Worker needs --task-index')
    return run_task(a.plan, a.command, a.task_index, a.dependencies, a.out)


if __name__ == '__main__':
    raise SystemExit(main())

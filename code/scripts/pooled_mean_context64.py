"""Context-64 complete-cohort extension of the frozen pooled mean protocol.

Reuse the verified exact-short calibration. Never alter its records or the
context-16 campaign. Every population history, arm and repetition is required.
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
sys.path[:0] = [str(ROOT/'code/scripts'), str(ROOT/'code/transformer')]
import numpy as np
import pooled_mean_population as base
from pooled_mean_population_resume import verified_overlay

PLAN = ROOT/'code/transformer/configs/pooled_mean_context64_plan_2026-09-22.json'
require, sha, save = base.require, base.sha, base.save


def read_plan(path):
    p = json.loads(Path(path).read_text())
    require(p['schema'] == 'pooled-mean-context64-plan-v1'
            and p['observed'] == 64 and p['repetitions'] == 32
            and p['indices'] == list(range(5, 69))
            and p['arms'] == base.ARMS and p['criteria'] == base.validation.CRITERIA
            and p['publication_arm'] == 'D' and p['term_cache_size'] == 4096
            and p['automatic_retries'] is False, 'Context-64 protocol changed')
    require(p['resources'] == dict(tasks=768, concurrency=256, cpus=1, memory_gib=6,
            prepare_minutes=120, task_minutes=480, merge_minutes=60,
            worker_margin_seconds=300, seconds_per_repetition=7200,
            maximum_cpu_hours=6147, maximum_cost_chf=33.8085), 'Allocation bounds changed')
    required = {'code/scripts/pooled_mean_context64.py',
                'code/transformer/scitas/pooled_mean_context64.sbatch',
                'code/scripts/pooled_mean_population_resume.py',
                'code/scripts/recover_pooled_mean_calibration.py'}
    require(set(p['code_sha256']) == required, 'Incomplete implementation binding')
    for name, value in p['code_sha256'].items():
        require(sha(ROOT/name) == value, 'Frozen extension code changed: '+name)
    parent, parent_hash, selection = base.read_plan(base.PLAN)
    require(p['parent_plan_sha256'] == parent_hash
            and p['selection_digest'] == parent['selection_digest'], 'Original protocol changed')
    return p, sha(path), selection


def source_cases(raw, selection):
    """Read each frozen source/model array once; use its context-64 oracle row."""
    raw = Path(raw)
    hashes = {name: sha(raw/name) for name in selection['selection']['input_sha256']}
    require(hashes == selection['selection']['input_sha256'], 'Frozen sources changed')
    col = base.old.ALL_TIMES.index(64)
    cases = []
    for family in base.old.FAMILIES:
        cohort = family+'_prior'
        receipt = json.loads((raw/f'heldout/{cohort}.json').read_text())
        require(receipt['plan_id'] == selection['selection']['source_plan_id']
                and receipt['data_sha256'] == hashes[f'heldout/{cohort}.npz'], 'Source identity changed')
        with np.load(raw/f'heldout/{cohort}.npz', allow_pickle=False) as a:
            require(np.array_equal(a['prefixes'], base.old.ALL_TIMES), 'Prediction times changed')
            source = {k: a[k] for k in ('tokens', 'sequence_seed', 'oracle_rows', 'transition_probs')}
        scores = {}
        for model in ('pooled_seed0', 'pooled_seed1'):
            r = json.loads((raw/f'eval/{model}/{cohort}.json').read_text())
            require(r['data_sha256'] == hashes[f'eval/{model}/{cohort}.npz']
                    and r['source_data_sha256'] == receipt['data_sha256']
                    and r['identity']['name'] == model
                    and r['identity']['source_plan_id'] == selection['selection']['source_plan_id'],
                    'Transformer identity changed')
            with np.load(raw/f'eval/{model}/{cohort}.npz', allow_pickle=False) as a:
                require(np.array_equal(a['prefixes'], base.old.ALL_TIMES)
                        and np.array_equal(a['sequence_seed'], source['sequence_seed']), 'Pairing changed')
                scores[model] = a['next_kl_bits']
        for case in selection['selection']['cases']:
            if case['family'] != family or case['index'] not in range(5, 69):
                continue
            i = case['index']
            require(np.array_equal(source['sequence_seed'][i], case['sequence_seed']), 'Source seed changed')
            tokens = source['tokens'][i, :64].tolist()
            oracle = source['oracle_rows'][i, col]
            require(np.array_equal(oracle, source['transition_probs'][i, tokens[-1]]), 'Oracle off by one')
            base.old.checked_probs(oracle, 64)
            tf = {m: float(v[i, col]) for m, v in scores.items()}
            require(all(math.isfinite(v) for v in tf.values()), 'Nonfinite transformer score')
            cases.append(dict(family=family, index=i, tokens=tokens, oracle_probs=oracle.tolist(),
                              transformer_regret_bits=tf))
    require({(r['family'], r['index']) for r in cases} ==
            {(f, i) for f in base.old.FAMILIES for i in range(5, 69)} and len(cases) == 192,
            'Complete balanced cohort required')
    return cases, hashes


def specifications(prepared):
    result = []
    for row in sorted(prepared['histories'], key=lambda r: r['reference_key']):
        for arm in base.ARMS:
            result.append(dict(task_index=len(result), stage='population', observed=64,
                reference_key=row['reference_key'], arm=arm, tokens=row['original_tokens']))
    require(len(result) == 768, 'Population task coverage changed')
    return result


def prepare(args):
    p, h, selection = read_plan(args.plan)
    path = args.out/'prepared.json'
    with base.cal.writer_claim(path):
        require(not path.exists(), 'Preparation already attempted')
        _, calibration = verified_overlay(base.PLAN, args.raw, args.dependencies, args.recovery)
        dep = base.dependency_state(args.dependencies, full=True)
        cases, hashes = source_cases(args.raw, selection)
        engines = {rule: [base.FamilyEvidence(f, bounds=bounds, step=step)
            for f in ('product', 'uniform_base')] for rule, bounds, step in
            (('default', (-30., 30.), 1/64), ('refined', (-40., 40.), 1/128))}
        histories = []
        for i, case in enumerate(cases):
            fixed = {rule: [base.component_prediction(engine, case['tokens']) for engine in values]
                     for rule, values in engines.items()}
            histories.append(dict(family=case['family'], index=case['index'],
                reference_key=case['family']+'_'+str(case['index']), original_tokens=case['tokens'],
                canonical_to_original=list(range(64)), oracle_probs=case['oracle_probs'],
                fixed_components=fixed['default'], refined_fixed_components=fixed['refined'],
                transformer_regret_bits=case['transformer_regret_bits']))
            print(json.dumps(dict(event='context64_preparing', completed=i+1, total=192)), flush=True)
        report = dict(schema='pooled-mean-context64-prepared-v1', plan_sha256=h,
            source_sha256=hashes, dependencies=dep, runtime=base.runtime(), histories=histories,
            calibration=dict(recovery=str(args.recovery.resolve()), hashes=calibration),
            scientific_reference_validated=False)
        specifications(report)
        save(path, report)
        save(args.out/'plan.json', p)


def prepared(args, h, full=False):
    path = args.out/'prepared.json'
    r = json.loads(path.read_text())
    require(r['schema'] == 'pooled-mean-context64-prepared-v1' and r['plan_sha256'] == h
            and r['runtime'] == base.runtime()
            and r['dependencies'] == base.dependency_state(args.dependencies, full=full),
            'Prepared data or runtime changed')
    require(r['calibration']['recovery'] == str(args.recovery.resolve()), 'Calibration source changed')
    for name, expected in r['calibration']['hashes'].items():
        require(sha(args.recovery/'overlay'/name) == expected, 'Calibration binding changed')
    return r, sha(path)


def sample(args):
    p, h, _ = read_plan(args.plan)
    r, ph = prepared(args, h)
    require(0 <= args.index < 768, 'Task index outside plan')
    spec = specifications(r)[args.index]
    path = base.task_path(args.out, 'population', args.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with base.cal.writer_claim(path):
        require(not path.exists(), 'Task already attempted; preserve saved repetitions')
        report = dict(schema='pooled-mean-task-v1', plan_sha256=h, prepared_sha256=ph, task=spec,
            calibration_summary_sha256=r['calibration']['hashes']['calibration/summary.json'],
            records=[], records_sha256=base.digest([]), complete=False, status='starting',
            scientific_reference_validated=False)
        save(path, report)
        s = base.ARMS[spec['arm']]
        nodes, weights = base.leggauss(s['alpha_order'])
        kernel = base.RankedLSARowEvidence(64, 1+nodes/2, depths=range(9),
            store=args.dependencies/'tables/anchors_prod', src=args.dependencies/'src',
            u_lo=s['row_bounds'][0], u_hi=s['row_bounds'][1], step=s['row_step'], term_cache_size=4096)
        sampler = base.SharedBaseMCMC(64, kernel, weights/2)
        started = time.monotonic()
        limit = 480*60-300
        def interrupted(sig, frame):
            raise InterruptedError('Signal '+signal.Signals(sig).name)
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
        try:
            for j in range(32):
                seconds = min(7200, limit-(time.monotonic()-started))
                if seconds <= 0:
                    raise TimeoutError('Bounded allocation exhausted')
                print(json.dumps(dict(event='repetition_started', context=64, task=args.index, repetition=j)), flush=True)
                with base.old.heartbeat(path.with_suffix('.progress.json'), dict(context=64, task=args.index, completed=j)):
                    record = base.estimate(sampler, spec['tokens'],
                        **{k: s[k] for k in ('particles', 'temperatures', 'moves')},
                        seed=base.seed_for(h, spec, j), resample_fraction=.75, seconds_limit=seconds)
                base.check_record(record, p, h, spec, j)
                report['records'].append(record)
                report.update(records_sha256=base.digest(report['records']), complete=j == 31,
                    status='complete' if j == 31 else 'running', elapsed_seconds=time.monotonic()-started)
                save(path, report)
                print(json.dumps(dict(event='repetition_complete', context=64, task=args.index, completed=j+1)), flush=True)
        except (TimeoutError, InterruptedError) as error:
            report.update(status='incomplete', error=str(error), elapsed_seconds=time.monotonic()-started)
            save(path, report)
            return 75
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return 0


def merge(args):
    p, h, _ = read_plan(args.plan)
    r, ph = prepared(args, h, full=True)
    _, calibration = verified_overlay(base.PLAN, args.raw, args.dependencies, args.recovery)
    require(calibration == r['calibration']['hashes'], 'Calibration no longer replays')
    require({name: sha(args.raw/name) for name in r['source_sha256']} == r['source_sha256'], 'Source changed')
    missing, grouped, hashes = [], {arm: {} for arm in base.ARMS}, {}
    for spec in specifications(r):
        path = base.task_path(args.out, 'population', spec['task_index'])
        if not path.exists():
            missing.append(spec['task_index']); continue
        value = json.loads(path.read_text())
        base.check_task(value, p, h, ph, spec)
        require(value['calibration_summary_sha256'] == calibration['calibration/summary.json'], 'Task gate changed')
        hashes[str(path.relative_to(args.out))] = sha(path)
        if not value['complete']:
            missing.append(spec['task_index']); continue
        grouped[spec['arm']][spec['reference_key']] = value['records']
    result = dict(schema='pooled-mean-context64-summary-v1', observed=64, plan_sha256=h,
        prepared_sha256=ph, calibration=calibration, complete=not missing, missing_tasks=missing,
        task_sha256=hashes, passed=False, scientific_reference_validated=False)
    if not missing:
        analyses = {a: base.validation.analyze_arm(records, r['histories']) for a, records in grouped.items()}
        refined = base.validation.analyze_arm(grouped['D'],
            [dict(row, fixed_components=row['refined_fixed_components']) for row in r['histories']])
        checks = base.validation.validate_campaign(analyses, cheap_refined=refined)
        result.update(checks=checks, arms={a: v['summary'] for a, v in analyses.items()},
            publication=refined, passed=checks['passed'], scientific_reference_validated=checks['passed'])
    save(args.out/'population/summary.json', result)
    print(json.dumps({k: result[k] for k in ('complete', 'passed', 'scientific_reference_validated')}), flush=True)
    return 0 if result['passed'] else 3


def main():
    print('Pooled context64: full192 histories, all32 repetitions, unchanged acceptance gates.', flush=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'population', 'merge'))
    for name in ('plan', 'raw', 'dependencies', 'recovery', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--index', type=int, default=-1)
    args = ap.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Numerical work requires a compute allocation')
    return {'prepare': prepare, 'population': sample, 'merge': merge}[args.stage](args)


if __name__ == '__main__':
    sys.exit(main())

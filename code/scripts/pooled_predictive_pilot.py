"""Freeze a balanced pooled cohort; run ONE bounded predictive-SMC case.

No command here submits or loops over the population. All saved transformer
scores and the two inexpensive references are reused on identical histories.
Every completed independent repetition is saved atomically and reused on
restart. An interrupted repetition restarts from its original deterministic
seed; this prototype does not claim within-repetition checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/transformer'))
from mixture_family_evidence import FamilyEvidence
from ranked_lsa import RankedLSARowEvidence
from shared_base_mcmc import SharedBaseMCMC
from shared_base_predictive_smc import (estimate, pool_repetitions, combine_components,
                                       component_prediction, checked_probs, mixture_regret_mcse)
from source import seed_of
from calibrate_shared_base_evidence import normalizer_summary

FAMILIES = ('product', 'shared_base', 'uniform_base')
TIMES = (1, 4, 16, 64, 256, 1024)
INDICES = tuple(range(5, 69))
CODE = ('code/scripts/pooled_predictive_pilot.py',
        'code/transformer/shared_base_predictive_smc.py',
        'code/transformer/shared_base_evidence_smc.py',
        'code/transformer/shared_base_mcmc.py',
        'code/transformer/ranked_lsa.py', 'code/transformer/weighted_lsa.py',
        'code/transformer/mixture_family_evidence.py',
        'code/transformer/validate_shared_depth.py', 'code/transformer/phi_source.py',
        'code/transformer/lsa_seq.py', 'code/transformer/shared_depth.py',
        'code/transformer/source.py', 'code/scripts/calibrate_shared_base_evidence.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_plan(raw, path):
    raw, path = Path(raw), Path(path)
    require(not path.exists(), 'plan already exists; do not change a frozen selection')
    manifest = json.loads((raw/'heldout/manifest.json').read_text())
    files, cases, development_cases = {}, [], []
    for family in FAMILIES:
        cohort = family+'_prior'
        source = json.loads((raw/f'heldout/{cohort}.json').read_text())
        data_path = raw/f'heldout/{cohort}.npz'
        require(source['data_sha256'] == sha(data_path) and source['plan_id'] == manifest['plan_id'],
                'source cohort identity mismatch')
        with np.load(data_path, allow_pickle=False) as saved:
            require(saved['tokens'].shape == (1024, 1025)
                    and np.array_equal(saved['prefixes'], TIMES), 'unexpected full-prior cohort')
            seeds = saved['sequence_seed'].copy()
            for index in INDICES:
                cases.append(dict(family=family, index=index, sequence_seed=seeds[index].tolist()))
            for index in range(1, 5):
                development_cases.append(dict(family=family, index=index, sequence_seed=seeds[index].tolist()))
        names = [f'heldout/{cohort}.json', f'heldout/{cohort}.npz']
        for model in ('pooled_seed0', 'pooled_seed1'):
            model_path = raw/f'eval/{model}/{cohort}.npz'
            model_record = json.loads(model_path.with_suffix('.json').read_text())
            require(model_record['data_sha256'] == sha(model_path)
                    and model_record['source_data_sha256'] == source['data_sha256']
                    and model_record['identity']['name'] == model
                    and model_record['identity']['source_plan_id'] == manifest['plan_id'],
                    'saved model provenance mismatch')
            with np.load(model_path, allow_pickle=False) as scores:
                require(np.array_equal(scores['sequence_seed'], seeds)
                        and np.array_equal(scores['prefixes'], TIMES)
                        and scores['next_kl_bits'].shape == (1024, 6)
                        and np.isfinite(scores['next_kl_bits']).all(), 'saved model scores mismatch')
            names += [f'eval/{model}/{cohort}.npz', f'eval/{model}/{cohort}.json']
        files.update({name: sha(raw/name) for name in names})
    files['heldout/manifest.json'] = sha(raw/'heldout/manifest.json')
    plan = dict(schema='pooled-predictive-balanced-selection-v1',
        source_plan_id=manifest['plan_id'], cases=cases,
        n_per_family=64, n_total=192, indices=list(INDICES),
        observation_times=[16, 64], family_order=list(FAMILIES), family_prior=[1/3]*3,
        input_sha256=files,
        selection='Existing independent full-prior draws, fixed indices5..68 in every family. No outcome-dependent selection.',
        development_indices=[1, 2, 3, 4], development_cases=development_cases,
        scope='Selection only. No production sampler effort or scientific acceptance certified. First benchmark16, then64. No automatic population launch.',
        source_generation=False, transformer_inference=False)
    atomic_json(path, plan)
    return plan


@contextmanager
def exclusive(path):
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_case(raw, plan, family, index, observed, *, development=False):
    require(plan['schema'] == 'pooled-predictive-balanced-selection-v1'
            and plan['family_order'] == list(FAMILIES) and plan['indices'] == list(INDICES)
            and plan['observation_times'] == [16, 64]
            and plan['family_prior'] == [1/3]*3 and plan['n_per_family'] == 64
            and plan['n_total'] == 192, 'frozen plan changed')
    cases = plan['development_cases'] if development else plan['cases']
    wanted_indices = list(range(1, 5)) if development else list(INDICES)
    require(len(cases) == 3*len(wanted_indices)
            and all(sorted(c['index'] for c in cases if c['family'] == f) == wanted_indices
                    for f in FAMILIES), 'frozen case coverage changed')
    selected = [c for c in cases if c['family'] == family and c['index'] == index]
    require(len(selected) == 1 and observed in plan['observation_times'], 'case outside frozen selection')
    cohort = family+'_prior'
    names = ['heldout/manifest.json', f'heldout/{cohort}.json', f'heldout/{cohort}.npz']
    names += [f'eval/{model}/{cohort}.{suffix}'
              for model in ('pooled_seed0', 'pooled_seed1') for suffix in ('json', 'npz')]
    hashes = {name: sha(raw/name) for name in names}
    require(all(value == plan['input_sha256'][name] for name, value in hashes.items()), 'frozen file changed')
    column = TIMES.index(observed)
    with np.load(raw/f'heldout/{cohort}.npz', allow_pickle=False) as a:
        tokens = a['tokens'][index, :observed].copy()
        require(np.array_equal(a['sequence_seed'][index], selected[0]['sequence_seed']), 'sequence changed')
        oracle = a['oracle_rows'][index, column].copy()
        require(np.array_equal(oracle, a['transition_probs'][index, tokens[-1]]), 'oracle/time mismatch')
        checked_probs(oracle)
    scores = {}
    for model in ('pooled_seed0', 'pooled_seed1'):
        with np.load(raw/f'eval/{model}/{cohort}.npz', allow_pickle=False) as a:
            require(np.array_equal(a['sequence_seed'][index], selected[0]['sequence_seed']), 'model sequence changed')
            scores[model] = float(a['next_kl_bits'][index, column])
    return tokens, oracle, scores, hashes


def regret(oracle, prediction):
    return float(np.dot(oracle, np.log2(oracle/checked_probs(prediction, len(oracle)))))


def run(args):
    raw = args.raw.resolve()
    plan = json.loads(args.plan.read_text())
    tokens, oracle, scores, hashes = load_case(raw, plan, args.family, args.index, args.observed,
                                             development=args.development)
    require(2 <= args.particles <= 16384 and 1 <= args.replicates <= 32
            and 1 <= args.temperatures <= 256 and 1 <= args.moves <= 16
            and 0 < args.seconds_per_replica <= 3600, 'prototype budget outside bounded limits')
    require(args.alpha_order in (8, 16, 32, 64), 'unsupported integration order')
    store, src = args.store.resolve(), args.src.resolve()
    store_names = ['manifest.json', 'anchors.json'] + [f'level_{d:02d}.{suffix}'
                  for d in range(2, 9) for suffix in ('index.json', 'bin')]
    config = dict(schema='pooled-predictive-single-case-prototype-v1',
        plan_sha256=sha(args.plan), family=args.family, sequence_index=args.index,
        development_case=args.development,
        n_observed=args.observed, tokens=tokens.tolist(), input_sha256=hashes,
        particles=args.particles, temperatures=args.temperatures, moves=args.moves,
        replicates=args.replicates, alpha_order=args.alpha_order, seed=args.seed,
        code_sha256={name: sha(ROOT/name) for name in CODE},
        store_sha256={name: sha(store/name) for name in store_names},
        backend_sha256=sha(src/'product_model_with_memory/universal_tables.py'),
        backend_source_sha256={str(p.relative_to(src)): sha(p)
            for p in sorted((src/'product_model_with_memory').iterdir())
            if p.is_file() and (p.suffix == '.py' or p.name.startswith('_kernel'))},
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        shared_grid=dict(bounds=[-32., 32.], step=1/32),
        cheap_grid=dict(bounds=[-30., 30.], step=1/64), scientific_reference_validated=False)
    folder = args.out/identity(config)[:20]
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder/'.lock'):
        path = folder/'config.json'
        if path.exists():
            require(json.loads(path.read_text()) == config, 'existing result configuration differs')
        else:
            atomic_json(path, config)
        previous_path = folder/'result.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
        if previous is not None:
            require(previous['config_sha256'] == sha(path), 'saved result configuration changed')
            for name, expected in previous['replica_sha256'].items():
                require(Path(name).name == name and sha(folder/name) == expected,
                        'previously saved replica changed')
            require(previous['cheap_components_sha256'] == sha(folder/'cheap_components.json'),
                    'previously saved inexpensive components changed')
        os.environ.update(LSA_STORE=str(store), PMM_SRC=str(src), PMM_UNIVERSAL_TABLES=str(store),
            PMM_PHI_LADDER_EVERY='1', PMM_PHI_LADDER_DEGREE='11', PMM_PHI_SADDLE_MIN_L='54')
        nodes, weights = leggauss(args.alpha_order)
        sampler = SharedBaseMCMC(64, RankedLSARowEvidence(64, 1+nodes/2, store=store, src=src), weights/2)
        cheap_path = folder/'cheap_components.json'
        if cheap_path.exists():
            cheap = json.loads(cheap_path.read_text())
        else:
            cheap = {family: component_prediction(FamilyEvidence(family), tokens)
                     for family in ('product', 'uniform_base')}
            atomic_json(cheap_path, cheap)
        records = []
        for replica in range(args.replicates):
            path = folder/f'replica{replica:04d}.json'
            seed = seed_of('pooled-predictive-smc-v1', args.seed, args.family, args.index,
                           args.observed, args.alpha_order, args.particles, args.temperatures, args.moves, replica)
            if path.exists():
                record = json.loads(path.read_text())
                require(record['seed'] == seed and record['n_observed'] == len(tokens), 'replica identity mismatch')
                checked_probs(record['predictive_probs'], 64)
            else:
                print(json.dumps(dict(event='replica_started', replica=replica, folder=str(folder))), flush=True)
                try:
                    record = estimate(sampler, tokens, particles=args.particles, temperatures=args.temperatures,
                        moves=args.moves, seed=seed, seconds_limit=args.seconds_per_replica)
                except TimeoutError:
                    atomic_json(folder/'status.json', dict(status='time_limit', complete=False,
                        completed_replicates=len(records), requested_replicates=args.replicates,
                        interrupted_replica=replica, saved_replicates_preserved=True,
                        within_replica_checkpoint=False, scientific_reference_validated=False))
                    raise
                atomic_json(path, record)
            records.append(record)
            pooled = pool_repetitions(records)
            components = [cheap['product'], pooled, cheap['uniform_base']]
            mixture = combine_components([r['log_evidence'] for r in components],
                                         [r['predictive_probs'] for r in components])
            mixture_regret = regret(oracle, mixture['predictive_probs'])
            result = dict(schema='pooled-predictive-single-case-result-v1', config_sha256=sha(folder/'config.json'),
                cheap_components_sha256=sha(cheap_path),
                complete=len(records) == args.replicates, scientific_reference_validated=False,
                shared_base=pooled, shared_base_normalizer_diagnostics=normalizer_summary([r['log_evidence'] for r in records]),
                mixture=mixture, mixture_next_token_regret_bits=mixture_regret,
                mixture_regret_mcse_bits_delta_method=mixture_regret_mcse(records,
                    [cheap['product'], cheap['uniform_base']], oracle),
                transformer_next_token_regret_bits=scores,
                transformer_minus_mixture_regret_bits={name: value-mixture_regret for name, value in scores.items()},
                replica_sha256={f'replica{j:04d}.json': sha(folder/f'replica{j:04d}.json') for j in range(len(records))},
                timings=dict(sampler_seconds=sum(r['sampler_seconds'] for r in records),
                             prediction_seconds=sum(r['prediction_seconds'] for r in records)),
                scope='One bounded case; development flag distinguishes existing pilot histories from the fixed balanced test subset. No population result or accuracy approval.')
            atomic_json(folder/'result.json', result)
            atomic_json(folder/'status.json', dict(status='completed_prototype' if result['complete'] else 'running',
                complete=result['complete'], completed_replicates=len(records), requested_replicates=args.replicates,
                scientific_reference_validated=False))
            print(json.dumps(dict(event='replica_saved', replica=replica, complete=result['complete'],
                sampler_seconds=record['sampler_seconds'], prediction_seconds=record['prediction_seconds'])), flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--raw', type=Path, required=True)
    freeze.add_argument('--out', type=Path, required=True)
    case = sub.add_parser('case')
    case.add_argument('--raw', type=Path, required=True)
    case.add_argument('--plan', type=Path, required=True)
    case.add_argument('--out', type=Path, required=True)
    case.add_argument('--family', choices=FAMILIES, required=True)
    case.add_argument('--index', type=int, required=True)
    case.add_argument('--development', action='store_true',
                      help='Use existing pilot index1..4; otherwise fixed test indices5..68 only')
    case.add_argument('--observed', type=int, choices=(16, 64), required=True)
    case.add_argument('--store', type=Path, required=True)
    case.add_argument('--src', type=Path, required=True)
    case.add_argument('--particles', type=int, default=32)
    case.add_argument('--temperatures', type=int, default=8)
    case.add_argument('--moves', type=int, default=1)
    case.add_argument('--replicates', type=int, default=2)
    case.add_argument('--alpha-order', type=int, default=16)
    case.add_argument('--seed', type=int, default=2026091801)
    case.add_argument('--seconds-per-replica', type=float, default=300)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze_plan(args.raw, args.out)
        print(json.dumps(dict(plan=str(args.out), sequences=192, times=[16, 64], computation_started=False)))
    else:
        print(json.dumps(dict(folder=str(run(args)), scientific_reference_validated=False)))


if __name__ == '__main__':
    main()

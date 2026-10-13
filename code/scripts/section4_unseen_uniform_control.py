"""Uniform-base negative control after a token's first occurrence.

Read previously validated probabilities only. No source generation, network
inference, Bayesian integration or training. All declared sequences are
checked before selection by the observed history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
from source import seed_of

SOURCE = ROOT / 'output/copier_time/run_2026-09-15'
ORIGINAL = ROOT / 'results/transformer/copier_time/summary.json'
OUT = ROOT / 'results/transformer/section4_unseen_uniform_control_2026-09-17'
DEPTHS = (0, 1, 4, 8)
TIMES = (4, 16, 64)
FIELDS = ('index', 'depth', 'time', 'transformer_excess_bits', 'counts_excess_bits')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def moments(values):
    values = np.asarray(values, dtype=float)
    require(len(values) >= 2 and np.isfinite(values).all(), 'Invalid selected cohort')
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1) / math.sqrt(len(values))))


def collect():
    original = json.loads(ORIGINAL.read_text())
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    require(original['passed'] and original['publication_cohort'] and original['n_sequences'] == 1024,
            'Original full-cohort validation is unavailable')
    require(original['spec'] == protocol and original['spec_id'] == fingerprint(protocol),
            'Original protocol differs')
    require(protocol['n_seq'] == 1024 and protocol['n_ctx'] == 1024 and protocol['d'] == 64
            and protocol['alpha'] == [0., 0.] and protocol['source_depths'] == list(range(9))
            and protocol['source_relabelled'], 'Wrong source prior or dimensions')
    require(sorted(p.name for p in SOURCE.glob('seq_*.npz')) ==
            [f'seq_{i:06d}.npz' for i in range(1024)], 'Incomplete source cohort')
    inputs = {str(ORIGINAL.relative_to(ROOT)): sha(ORIGINAL),
              str((SOURCE / 'protocol.json').relative_to(ROOT)): sha(SOURCE / 'protocol.json')}
    # Bind the source generator, predictor implementation and saved checkpoint.
    # The expensive historical quadrature is not used: Bayes is analytic here.
    for key in ('source.py', 'copier_time_source.py', 'model.py', 'eval_copier_time.py',
                'natural_circuit_probe.py', 'circuit_probe.py', 'checkpoint', 'config'):
        binding = protocol['provenance'][key]
        path = Path(binding['path'])
        digest = sha(path)
        require(digest == binding['sha256'], f'Changed provenance: {key}')
        inputs[str(path.relative_to(ROOT))] = digest
    config = json.loads(Path(protocol['provenance']['config']['path']).read_text())
    metadata = protocol['model_metadata']
    require(config['model'] == metadata['architecture'] and config['alpha'] == [0., 0.]
            and config['ls'] == [0, 8] and config['seed'] == 0 and config['steps'] == 100000
            and config['batch'] == 64 and config['n_params'] == 544256
            and metadata['step'] == 100000 and metadata['parameter_values_preserved'],
            'Wrong uniform-base model')
    numerical = ROOT / 'results/transformer/copier_time/numerical_validation.json'
    require(json.loads(numerical.read_text())['passed'], 'Original Bayes audit is unavailable')
    inputs[str(numerical.relative_to(ROOT))] = sha(numerical)
    previous = {r['index']: r for r in original['inputs']}
    require(set(previous) == set(range(1024)), 'Original record identities are incomplete')
    maxima = {}

    def check(name, value, tolerance):
        value = float(value)
        require(math.isfinite(value) and value <= tolerance, f'{name}: {value} > {tolerance}')
        maxima[name] = max(maxima.get(name, 0.), value)

    rows, sources, depth_counts = [], [], Counter()
    keys = ('tokens', 'prefix_lengths', 'source_depth', 'initial_probs', 'transition_probs',
            'permutation', 'intact_probs', 'bayes_probs')
    for i in range(1024):
        path = SOURCE / f'seq_{i:06d}.npz'
        expected = previous[i]
        digest = sha(path)
        require(digest == expected['sha256'], f'Changed raw source/predictions: {i}')
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata_json']))
            arrays = {name: archive[name] for name in keys}
        require(meta['spec'] == protocol and meta['spec_id'] == original['spec_id']
                and meta['index'] == i and meta['sequence_seed'] == expected['sequence_seed']
                == seed_of(protocol['schema'], protocol['master_seed'], i)
                and fingerprint(meta) == expected['metadata_sha256'], 'Changed source/model identity')
        require(meta['network_checks']['passed'] and
                meta['network_checks']['original_validation']['passed'], 'Original-model check failed')
        for key, value in arrays.items():
            require(array_hash(value) == meta['array_hashes'][key], f'Corrupt {i}/{key}')
        x, depth = arrays['tokens'], int(arrays['source_depth'])
        require(x.shape == (1025,) and x.dtype.kind in 'iu' and np.all((x >= 0) & (x < 64))
                and depth == expected['source_depth'] and depth in range(9)
                and np.array_equal(arrays['prefix_lengths'], np.arange(1, 1025)), 'Wrong source/time alignment')
        require(np.array_equal(np.sort(arrays['permutation']), np.arange(64)), 'Invalid relabeling')
        require(arrays['initial_probs'].shape == (64,) and arrays['transition_probs'].shape == (64, 64),
                'Wrong source distributions')
        for name, values in (('source', arrays['transition_probs']), ('transformer', arrays['intact_probs']),
                             ('bayes', arrays['bayes_probs'])):
            require(np.isfinite(values).all() and np.min(values) > 0, f'Invalid {name} probabilities')
            if name != 'source':
                require(values.shape == (1024, 64), f'Wrong {name} time dimensions')
            check(name + '_normalization', np.max(np.abs(values.sum(-1) - 1)), 1e-7 if name == 'bayes' else 1e-12)
        check('initial_uniform_error', np.max(np.abs(arrays['initial_probs'] - 1/64)), 1e-12)
        if depth == 0:
            check('depth_zero_uniform_error', np.max(np.abs(arrays['transition_probs'] - 1/64)), 1e-12)
        depth_counts[depth] += 1
        sources.append(dict(index=i, depth=depth, path=str(path.relative_to(ROOT)), sha256=digest,
                            tokens_sha256=array_hash(x), metadata_sha256=fingerprint(meta)))
        if depth not in DEPTHS:
            continue
        for t in TIMES:
            observed = x[:t]
            selected = not np.any(observed[:-1] == observed[-1])
            # Independently restate eligibility as an outgoing-transition count.
            row_count = sum(int(a) == int(observed[-1]) for a, _ in zip(observed[:-1], observed[1:]))
            require(selected == (row_count == 0), 'Event-selection disagreement')
            if not selected:
                continue
            bayes = arrays['bayes_probs'][t-1]
            check('selected_bayes_uniform_error', np.max(np.abs(bayes - 1/64)), 1e-7)
            probabilities = arrays['intact_probs'][t-1]
            kt = (np.bincount(observed, minlength=64) + .5) / (t + 32)
            values = []
            for name, q in (('transformer', probabilities), ('counts', kt)):
                value = float(np.mean(np.log2((1/64) / q)))
                # Independent scalar cross-entropy calculation, including a
                # separately constructed count distribution for the baseline.
                raw_counts = Counter(int(token) for token in observed)
                independent_q = q if name == 'transformer' else [(raw_counts[b]+.5)/(t+32) for b in range(64)]
                independent = math.fsum(-math.log2(float(p)) / 64 for p in independent_q) - 6
                check(name + '_independent_loss_identity_bits', abs(value-independent), 3e-14)
                require(value >= -1e-12, 'Negative excess loss beyond rounding')
                values.append(value)
            rows.append(dict(index=i, depth=depth, time=t,
                             transformer_excess_bits=values[0], counts_excess_bits=values[1]))
        if (i+1) % 256 == 0:
            print(f'Checked {i+1}/1024 saved sources', flush=True)
    require({str(k):v for k,v in sorted(depth_counts.items())} == original['source_depth_counts'],
            'Changed source-depth counts')
    groups = []
    for depth in DEPTHS:
        for t in TIMES:
            selected = [r for r in rows if r['depth'] == depth and r['time'] == t]
            groups.append(dict(depth=depth, time=t, n=len(selected), candidates=depth_counts[depth],
                               transformer=moments([r['transformer_excess_bits'] for r in selected]),
                               counts=moments([r['counts_excess_bits'] for r in selected]),
                               counts_minus_transformer=moments([r['counts_excess_bits'] - r['transformer_excess_bits'] for r in selected])))
    return dict(schema='unseen-uniform-control-v1', passed=True, d=64, times=list(TIMES),
                depths=list(DEPTHS), n_original_sequences=1024, sequence_length=1025,
                depth_counts={str(k):v for k,v in sorted(depth_counts.items())},
                selection='At time t, token t has no occurrence at times 1 through t-1; all eligible histories at the specified depths/times are retained.',
                quantity='KL(U64 || predictor), in bits; exactly the conditional expected next-token excess loss over Bayes on this uniform-base unseen-row event.',
                bayes='Exactly U64 for every selected history, at every source depth. No numerical integration is needed.',
                model=dict(label='Existing uniform-base transformer, training run 1', training_seed=0,
                           parameters=config['n_params'], architecture=config['model'], training_sequences=6400000,
                           steps=100000, checkpoint=protocol['provenance']['checkpoint'],
                           distinct_from='The larger shared-base models; this control uses their separate uniform-base predecessor.'),
                counts_predictor='(N_b + 1/2)/(t + 32); symmetric Dirichlet-1/2 token counts, not the LSA unigram.',
                uncertainty='Mean +/- one standard error across independent selected sequences within each depth/time. Times overlap in sequences and are not independent replicates.',
                inference=False, sampling=False, training=False, new_source_draws=False,
                groups=groups, maximum_check_errors=maxima, inputs=inputs, source_records=sources,
                script_sha256=sha(__file__)), rows


def render_csv(rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def verify_records(summary, csv_text):
    rows = list(csv.DictReader(StringIO(csv_text)))
    for r in rows:
        for key in ('index', 'depth', 'time'):
            r[key] = int(r[key])
        for key in FIELDS[3:]:
            r[key] = float(r[key])
    require(len({(r['index'],r['time']) for r in rows}) == len(rows), 'Duplicate selected records')
    largest = 0.
    for group in summary['groups']:
        selected = [r for r in rows if (r['depth'],r['time']) == (group['depth'],group['time'])]
        require(len(selected) == group['n'], 'Selected CSV count differs')
        for name in ('transformer','counts','counts_minus_transformer'):
            values = [r[name+'_excess_bits'] if name != 'counts_minus_transformer' else
                      r['counts_excess_bits']-r['transformer_excess_bits'] for r in selected]
            expected = dict(mean=statistics.fmean(values), se=statistics.stdev(values)/math.sqrt(len(values)))
            largest = max(largest, *(abs(expected[key]-group[name][key]) for key in expected))
    require(largest < 2e-14, 'Independent CSV aggregation differs')
    return dict(passed=True, selected_records=len(rows), independent_aggregation_max_error_bits=largest,
                independent_scalar_loss_max_error_bits=max(summary['maximum_check_errors'][key]
                    for key in ('transformer_independent_loss_identity_bits','counts_independent_loss_identity_bits')),
                all_original_files_hash_checked=1024, all_declared_selected_histories_retained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    summary, rows = collect()
    csv_text = render_csv(rows)
    validation = verify_records(summary, csv_text)
    if args.check_only:
        require(json.loads((args.out/'summary.json').read_text()) == summary, 'Saved summary differs from raw data')
        require((args.out/'records.csv').read_text() == csv_text, 'Saved records differ from raw data')
        stored = json.loads((args.out/'validation.json').read_text())
        require(stored['checks'] == validation, 'Saved validation differs')
        for name, value in stored['outputs'].items():
            require(sha(args.out/name) == value, 'Saved output hash differs')
        print(json.dumps(dict(passed=True, groups=len(summary['groups']), records=len(rows))))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (args.out/'records.csv').write_text(csv_text)
    (args.out/'validation.json').write_text(json.dumps(dict(schema='unseen-uniform-control-validation-v1',
        passed=True, checks=validation, outputs={name:sha(args.out/name) for name in ('summary.json','records.csv')},
        script_sha256=sha(__file__)), indent=2)+'\n')
    for g in summary['groups']:
        print(json.dumps({k:g[k] for k in ('depth','time','n','candidates','transformer','counts')}))


if __name__ == '__main__':
    main()

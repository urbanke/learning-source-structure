"""Independently audit and release the completed shared-base capacity comparison.

Recomputes KL directly from saved model probabilities and source transition
rows, then independently aggregates paired scores. No model inference or
Bayesian sampling. Existing numerical Bayesian acceptance criteria are unchanged;
their separately audited releases are hash-bound, not re-certified here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPTHS = [0, 1, 4, 8]
TIMES = [1, 4, 16, 64, 256, 1024]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def moments(values):
    # Independent aggregation: direct sums instead of the evaluator's np.std.
    n = len(values)
    mu = float(sum(map(float, values)) / n)
    se = float(np.sqrt(sum((float(x)-mu)**2 for x in values)/(n*(n-1))))
    return dict(n=n, mean=mu, standard_error=se)


def check_moments(values, saved):
    actual = moments(values)
    require(actual['n'] == saved['n'], 'Sample counts differ')
    error = max(abs(actual[k]-saved[k]) for k in ('mean', 'standard_error'))
    require(error < 1e-11, 'Independent paired aggregation differs')
    return error


def audit(summary_path, plan_path, runs_root):
    out = summary_path.parent
    s, p = read(summary_path), read(plan_path)
    require(s['schema'] == 'shared-base-scaling-evaluation-v1'
            and s['plan_sha256'] == sha(plan_path), 'Summary/plan identity differs')
    require(s['depths'] == DEPTHS and s['times'] == TIMES and s['seed_order'] == [0, 1]
            and s['n_sequences_per_depth'] == 256 and s['n_tokens_per_sequence'] == 1025,
            'Unexpected evaluation population')
    require(len(s['rows']) == 48 and len(s['provenance']) == 4
            and {r['depth'] for r in s['provenance']} == set(DEPTHS), 'Incomplete comparison')
    require(s['evaluated_conditions_hidden'] is True and s['training_prior'] ==
            dict(alpha=[.5, 1.5], depths=list(range(9)), ranks='uniform permutation'),
            'Predictors must retain the full training prior')
    bound = {str(summary_path): sha(summary_path), str(plan_path): sha(plan_path)}
    for name, digest in p['evaluation_files_sha256'].items():
        require(sha(ROOT/name) == digest, 'Frozen evaluation file changed: '+name)
        bound[str(ROOT/name)] = digest
    require(s['evaluator_sha256'] == sha(ROOT/'code/transformer/shared_base_scaling_eval.py'),
            'Evaluator code changed')
    for extra in s['additional_evaluation_releases']:
        require(extra['raw_files_rehashed_before_evaluation'] is True
                and extra['mandatory_replacements_checked'] is True
                and extra['depths'] == DEPTHS and extra['times'] == [64],
                'Complete time64 import lacks the released scope/checks')
        for name, digest in extra['files_sha256'].items():
            require(sha(name) == digest, 'Supplement file changed: '+name)
            bound[name] = digest

    max_moment_error = max_raw_error = max_probability_error = 0.
    readbacks, time64, scores, raw_receipts, training_metadata = 0, [], [], [], []
    for prov in s['provenance']:
        d = prov['depth']
        score_path = Path(prov['scores_path'])
        require(sha(score_path) == prov['scores_sha256'], 'Paired score hash differs')
        scores.append(dict(depth=d, source_path=str(score_path),
                           filename=f'paired_scores_L{d}.npz', sha256=prov['scores_sha256']))
        with np.load(score_path, allow_pickle=False) as a:
            require(np.array_equal(a['source_indices'], np.arange(256))
                    and a['times'].tolist() == TIMES and a['bayes_times'].tolist() == [1, 4, 16, 64],
                    'Paired score coordinates differ')
            old, new, bayes = [a[k].copy() for k in
                              ('baseline_regret_bits', 'larger_regret_bits', 'bayes_regret_bits')]
        require(old.shape == new.shape == (256, 6, 2) and bayes.shape == (256, 4)
                and all(np.isfinite(x).all() for x in (old, new, bayes)), 'Invalid score arrays')
        cohort_path = ROOT/f'output/shared_base_conditional_curves_2026-09-15/a1_L{d}/cohort.json'
        cohort = read(cohort_path)
        for seed in (0, 1):
            training = prov['models'][seed]['training']
            run = runs_root/f'seed{seed}'
            for field, basename in [('checkpoint_sha256', 'model_100000.pt'),
                                    ('config_sha256', 'config.json'), ('status_sha256', 'status.json')]:
                require(sha(run/basename) == training[field], 'Training file hash differs')
                bound[str(run/basename)] = training[field]
            cfg, status = read(run/'config.json'), read(run/'status.json')
            require(cfg['seed'] == seed and status['complete'] is True and status['reason'] == 'completed'
                    and status['step'] == 100000 and status['completed_sequences'] == 6400000
                    and cfg['n_params'] == 3454464 and cfg['plan_sha256'] == sha(plan_path)
                    and cfg['implementation_hashes'] == p['implementation_sha256'],
                    'Training scope or provenance differs')
            require(cfg['model'] == dict(vocab=64, n_ctx=1024, n_layer=4, n_head=8,
                                        d_model=256, mlp=True, dropout=0.), 'Architecture differs')
            runtime = read(run/'scaling_runtime.json')
            require(runtime['return_code'] == 0 and runtime['step'] == 100000
                    and runtime['seed'] == seed and runtime['plan_sha256'] == sha(plan_path),
                    'Training runtime completion differs')
            log = [json.loads(line) for line in (run/'log.jsonl').read_text().splitlines() if line]
            last_train = [row for row in log if row.get('event') == 'train'][-1]
            require(last_train['step'] == 100000 and last_train['completed_sequences'] == 6400000,
                    'Training log does not reach planned exposure')
            bound[str(run/'log.jsonl')] = sha(run/'log.jsonl')
            if d == DEPTHS[0]:
                for basename in ('config.json', 'status.json', 'scaling_runtime.json'):
                    training_metadata.append(dict(seed=seed, source_path=str(run/basename),
                        filename=f'training_seed{seed}/{basename}', sha256=sha(run/basename)))
                    bound[str(run/basename)] = sha(run/basename)
            for key, value in p['config'].items():
                require(cfg['config'][key] == value, 'Planned training option differs: '+key)
            identity_receipts = []
            for i in range(256):
                f = out/f'a1_L{d}/models/seed{seed}/seq_{i:06d}.json'
                r = read(f)
                require(r['sequence_index'] == i and r['prefixes'] == TIMES
                        and r['identity']['model']['checkpoint_sha256'] == training['checkpoint_sha256']
                        and r['identity']['model']['config'] == cfg and r['cache_id'] == fingerprint(r['identity'])
                        and r['checks']['alignment']['passed'] is True, 'Model record identity/alignment differs')
                require(sha(f.with_suffix('.npz')) == r['arrays_sha256'], 'Model probability hash differs')
                source_path = Path(r['source_path'])
                source_digest = sha(source_path)
                require(source_digest == r['identity']['source_file_sha256'] == cohort['records'][i]['file_sha256'],
                        'Source is not the frozen matched sequence')
                source = read(source_path)
                tokens = np.asarray(source['tokens'])
                oracle = np.asarray(source['oracle']['transition_probs'])[tokens[np.asarray(TIMES)-1]]
                with np.load(f.with_suffix('.npz'), allow_pickle=False) as a:
                    q = a['predictive_probs']
                    require(q.shape == oracle.shape == (6, 64) and np.array_equal(oracle, a['oracle_probs'])
                            and np.array_equal(a['targets'], tokens[TIMES]) and np.isfinite(q).all()
                            and np.all(q > 0) and np.max(np.abs(q.sum(axis=1)-1)) < 1e-12,
                            'Probability, target or oracle coordinates differ')
                    # Rebuild softmax independently from saved raw logits.
                    logits = a['logits'].astype(float)
                    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
                    q_from_logits = exp/exp.sum(axis=1, keepdims=True)
                    probability_error = float(np.max(np.abs(q_from_logits-q)))
                    require(probability_error < 1e-12, 'Saved probabilities differ from raw logits')
                    max_probability_error = max(max_probability_error, probability_error)
                    # Direct log2 ratios, independently of scipy.special.rel_entr.
                    raw = np.sum(oracle*np.log2(oracle/q), axis=1)
                    error = float(np.max(np.abs(raw-new[i, :, seed])))
                    require(error < 1e-11, 'Independent raw KL computation differs')
                    max_raw_error = max(max_raw_error, error)
                identity_receipts.append(dict(record_sha256=sha(f), arrays_sha256=r['arrays_sha256']))
                raw_receipts.append(dict(path=str(f), report_sha256=sha(f), arrays_sha256=r['arrays_sha256'],
                                         source_sha256=source_digest))
                readbacks += 1
            require(fingerprint(identity_receipts) == prov['models'][seed]['prediction_digest'],
                    'Model prediction population differs')
            for j, t in enumerate(TIMES):
                rows = [x for x in s['rows'] if (x['depth'], x['time'], x['seed']) == (d, t, seed)]
                require(len(rows) == 1, 'Duplicate or missing summary row')
                row = rows[0]
                pairs = [(old[:, j, seed], row['baseline_regret']), (new[:, j, seed], row['larger_regret']),
                         (old[:, j, seed]-new[:, j, seed], row['improvement_over_baseline'])]
                require(row['bayes_available'] == (j < 4), 'Missing Bayesian times were filled')
                if j < 4:
                    b = bayes[:, j]
                    pairs.extend([(b, row['bayes_regret']), (old[:, j, seed]-b, row['baseline_minus_bayes']),
                                  (new[:, j, seed]-b, row['larger_minus_bayes'])])
                for values, saved in pairs:
                    max_moment_error = max(max_moment_error, check_moments(values, saved))
                if t == 64:
                    time64.append(dict(depth=d, seed=seed, bayes=row['bayes_regret'],
                                       old_gap=row['baseline_minus_bayes'], new_gap=row['larger_minus_bayes'],
                                       improvement=row['improvement_over_baseline']))
    return dict(schema='shared-base-scaling-independent-audit-v1', passed=True,
                summary_sha256=sha(summary_path), plan_sha256=sha(plan_path),
                n_sequences_per_depth=256, depths=DEPTHS, times=TIMES, seed_order=[0, 1],
                model_records_recomputed=readbacks, summary_rows=48,
                max_summary_difference_bits=max_moment_error, max_raw_kl_difference_bits=max_raw_error,
                max_logits_probability_difference=max_probability_error,
                paired_scores=scores, training_metadata=training_metadata, inputs_sha256=bound,
                raw_records_manifest_sha256=fingerprint(raw_receipts),
                validator_sha256=sha(__file__), time64=time64,
                limits='Development cohorts; two training runs reported separately. Fixed exposure and joint capacity change. Bayes available only at 1,4,16,64; its numerical diagnostics do not establish global convergence.',
                inference_performed=False, bayesian_sampling_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'output/shared_base_scaling_evaluation_2026-09-17/comparison.json')
    parser.add_argument('--plan', type=Path, default=ROOT/'code/transformer/configs/shared_base_scaling_plan_2026-09-17.json')
    parser.add_argument('--runs-root', type=Path, default=ROOT/'output/shared_base_scaling_download_2026-09-17')
    parser.add_argument('--out', type=Path, required=True, help='Curated result directory; originals are retained')
    args = parser.parse_args()
    result = audit(args.summary.resolve(), args.plan.resolve(), args.runs_root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    for row in result['paired_scores']+result['training_metadata']:
        destination = args.out/row['filename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == row['sha256'], 'Refuse to replace a different curated score file')
        else:
            shutil.copyfile(row['source_path'], destination)
    destination = args.out/'comparison.json'
    if destination.exists():
        require(sha(destination) == result['summary_sha256'], 'Refuse to replace a different curated comparison')
    else:
        shutil.copyfile(args.summary, destination)
    (args.out/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('passed', 'model_records_recomputed', 'summary_rows',
                                            'max_summary_difference_bits', 'max_raw_kl_difference_bits')}))


if __name__ == '__main__':
    main()

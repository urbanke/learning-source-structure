"""Reuse saved shared-base predictions after a token first appears.

Adds the completed larger-model comparison and time 64 to the historical
time-4/time-16 analysis. No training, inference, new sequences or sampling.
Numerical acceptance is recomputed on exactly the event-selected subset;
failed subsets retain candidate values for audit but are withheld scientifically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import t as student_t

import section4_unseen_row_transfer as old
import export_shared_base_time64 as time64
from shared_base_conditional_cohort import inputs
from shared_base_precision_cohort import validate_cohort
from eval_shared_base_models import fingerprint

ROOT = old.ROOT
OUT = ROOT/'results/transformer/section4_unseen_context_2026-09-17'
SCALING = ROOT/'results/transformer/shared_base_scaling_2026-09-17'
RAW_SCALING = ROOT/'output/shared_base_scaling_evaluation_2026-09-17'
TIMES = (4, 16, 64)
PREDICTORS = ('bayes', 'transformer_seed0', 'transformer_seed1', 'larger_seed0',
              'larger_seed1', 'kt_unigram', 'one_token_bayes', 'uniform')
PAIRS = [(name, ref) for name in ('transformer_seed0', 'transformer_seed1',
          'larger_seed0', 'larger_seed1') for ref in ('bayes', 'kt_unigram')]
PAIRS += [('bayes', 'kt_unigram'), ('kt_unigram', 'bayes'),
          ('larger_seed0', 'transformer_seed0'), ('larger_seed1', 'transformer_seed1')]
require, sha, native = old.require, old.sha, old.native


def moments(values):
    result = old.moments(values)
    half = float(student_t.ppf(.975, result['n']-1))*result['standard_error']
    result['ci95_pointwise'] = [result['mean']-half, result['mean']+half]
    return result


def build():
    # Reuse its identity checks, deterministic mapping and selected t16 gates.
    previous, rows = old.build()
    tracked = old.Provenance()
    tracked.files.update(previous['input_sha256'])
    tracked.read(old.__file__, previous['script_sha256'])
    previous_by_case = {(r['depth'], r['time'], r['index']): r for r in rows}
    numerical = {(g['depth'], g['time']): g['subset_numerical_validation']
                 for g in previous['groups']}
    release_path = ROOT/'results/transformer/shared_base_time64_complete_2026-09-17/summary.json'
    original = tracked.json(old.RELEASE/'summary.json')
    files, release = time64.validate_export(release_path, original)
    for path in files:
        tracked.read(path)
    release_validation = tracked.json(files[1])
    environment = dict(os.environ)
    try:
        source_summary, plan, receipts, bound = time64.source_rows(release['source_summary'])
    finally:
        for key in set(os.environ)-set(environment):
            del os.environ[key]
        os.environ.update(environment)
    require(source_summary['schema'] == release['source_summary_schema']
            and plan['plan_id'] == release['source_plan_id'], 'Time64 source plan differs')
    require(all(release_validation['inputs'].get(path) == digest for path, digest in bound.items()),
            'Time64 source inputs differ from released provenance')
    raw_files = {path: digest for r in receipts for path, digest in r['task']['file_sha256'].items()}
    require(raw_files == release_validation['raw_file_sha256']
            and release_validation['raw_files_rehashed_at_export'], 'Raw time64 coverage differs')
    # source_rows rehashes all reports, chains and moments, including required replacements.
    tracked.files.update(bound)
    checked64 = {(r['task']['depth'], r['task']['index'], r['task']['order']): r for r in receipts}
    require(set(checked64) == {(d, i, o) for d in old.DEPTHS for i in range(256) for o in (16,32)},
            'Incomplete effective time64 population')

    scaling_validation = tracked.json(SCALING/'validation.json')
    scaling = tracked.json(SCALING/'comparison.json', scaling_validation['summary_sha256'])
    require(scaling_validation['passed'] and scaling_validation['model_records_recomputed'] == 2048
            and scaling_validation['validator_sha256'] == sha(ROOT/'code/scripts/validate_shared_base_scaling_results.py')
            and scaling['times'] == [1,4,16,64,256,1024]
            and scaling['seed_order'] == [0,1] and scaling['n_sequences_per_depth'] == 256
            and scaling['evaluated_conditions_hidden'], 'Larger-model comparison lacks verified coverage')
    tracked.read(ROOT/'code/scripts/validate_shared_base_scaling_results.py', scaling_validation['validator_sha256'])
    first = tracked.json(ROOT/'results/transformer/conditional_shared_base/first_step.json')
    first_probs = np.asarray(first['reports'][-1]['probs'])
    checkpoint_hashes = dict(previous['model_checkpoint_sha256'])
    max_logits_difference = max_kl_difference = max_old_loss_difference = 0.
    selected_model_records = 0
    groups = []
    extended_rows = []
    for depth in old.DEPTHS:
        folder = old.RAW/f'a1_L{depth}'
        cohort = tracked.json(folder/'cohort.json')
        declared = next(c for c in release['cohorts'] if c['depth'] == depth)
        require(sha(folder/'cohort.json') == declared['provenance']['cohort_sha256']
                and sha(folder/'protocol.json') == declared['provenance']['protocol_sha256']
                and cohort['depth'] == depth and cohort['n_sequences'] == 256
                and [r['index'] for r in cohort['records']] == list(range(256)), 'Wrong frozen sources')
        sources = [tracked.json(folder/'data'/f'seq_{i:06d}.json', cohort['records'][i]['file_sha256'])
                   for i in range(256)]
        prov = next(r for r in scaling['provenance'] if r['depth'] == depth)
        # Bind each selected raw record to the already audited full-population
        # digest, rather than trusting a filename or a newly read self-hash.
        for seed in (0,1):
            identities = []
            for index in range(256):
                path = RAW_SCALING/f'a1_L{depth}/models/seed{seed}/seq_{index:06d}.json'
                saved = tracked.json(path)
                require(saved['sequence_index'] == index, 'Full model population index differs')
                identities.append(dict(record_sha256=sha(path), arrays_sha256=saved['arrays_sha256']))
            require(fingerprint(identities) == prov['models'][seed]['prediction_digest'],
                    'Saved larger-model records differ from the independently audited population')
        paired = tracked.arrays(SCALING/f'paired_scores_L{depth}.npz', prov['scores_sha256'])
        require(paired['times'].tolist() == scaling['times'] and paired['bayes_times'].tolist() == [1,4,16,64]
                and np.array_equal(paired['source_indices'], np.arange(256)), 'Paired score coordinates differ')
        large_cache = {}
        for time in TIMES:
            selected = [i for i, source in enumerate(sources) if old.unseen_row(source['tokens'][:time])]
            require(len(selected) >= 2, 'Too few selected source sequences')
            if time == 64:
                numerical[depth,time] = native(validate_cohort(
                    [checked64[depth,i,16]['corrected']['audit']['record'] for i in selected],
                    [checked64[depth,i,32]['corrected']['audit']['record'] for i in selected], selected))
            num = numerical[depth,time]
            group_rows = []
            for index in selected:
                source = sources[index]
                tokens = np.asarray(source['tokens'])
                observed, oracle, models, identity = inputs(folder, index, time)
                require(np.array_equal(observed, tokens[:time]) and old.unseen_row(observed)
                        and np.array_equal(oracle, np.asarray(source['oracle']['transition_probs'])[tokens[time-1]]),
                        'Selection or oracle alignment differs')
                if time != 64:
                    record = json.loads(json.dumps(previous_by_case[depth,time,index]))
                else:
                    refined = checked64[depth,index,32]
                    reference = tracked.json(refined['task']['reference'], refined['task']['file_sha256'][refined['task']['reference']])
                    require(reference['identity']['inputs'] == identity
                            and reference['reference_id'] == refined['task']['reference_id'], 'Time64 source/model pairing differs')
                    p = np.asarray(refined['corrected']['audit']['record']['precision']['predictive_probs'])
                    count = (np.bincount(observed, minlength=64)+.5)/(time+32)
                    one = np.full(64, first_probs[1]); one[int(observed[-1])] = first_probs[0]
                    predictions = dict(bayes=p, transformer_seed0=models['d64']['prediction'],
                        transformer_seed1=models['d64_seed1']['prediction'], kt_unigram=count,
                        one_token_bayes=one, uniform=np.ones(64)/64)
                    losses = {name: old.loss(value,oracle) for name,value in predictions.items()}
                    record = dict(depth=depth,time=time,index=index,source_id=source['source_id'],
                        query=int(observed[-1]),n_observed_tokens=time,prior_outgoing_transitions=0,
                        entropy_bits=float(-oracle@np.log2(oracle)),loss_bits=losses)
                # All selected original models remain tied to their original checkpoint and file hashes.
                for name, model in models.items():
                    metadata = tracked.json(model['path'], model['sha256'])
                    tracked.read(Path(model['path']).with_suffix('.npz'), metadata['arrays_sha256'])
                    require(metadata['identity']['model']['checkpoint_sha256'] == checkpoint_hashes[name],
                            'Original checkpoint changed')
                    seed = int(name.endswith('seed1'))
                    j = scaling['times'].index(time)
                    error = abs(record['loss_bits'][f'transformer_seed{seed}']-record['entropy_bits']
                                -paired['baseline_regret_bits'][index,j,seed])
                    require(error < 2e-12, 'Original score differs from independently audited comparison')
                    max_old_loss_difference = max(max_old_loss_difference,error)
                for seed in (0,1):
                    if (index,seed) not in large_cache:
                        path = RAW_SCALING/f'a1_L{depth}/models/seed{seed}/seq_{index:06d}.json'
                        model = tracked.json(path)
                        arrays = tracked.arrays(path.with_suffix('.npz'),model['arrays_sha256'])
                        training = prov['models'][seed]['training']
                        require(model['sequence_index'] == index and model['prefixes'] == scaling['times']
                            and model['cache_id'] == fingerprint(model['identity'])
                            and model['identity']['model']['checkpoint_sha256'] == training['checkpoint_sha256']
                            and model['identity']['source_file_sha256'] == cohort['records'][index]['file_sha256']
                            and Path(model['source_path']).resolve() == (folder/'data'/f'seq_{index:06d}.json').resolve()
                            and model['checks']['alignment']['passed'], 'Larger prediction identity differs')
                        checkpoint_hashes[f'larger_seed{seed}'] = training['checkpoint_sha256']
                        expected_oracle = np.asarray(source['oracle']['transition_probs'])[tokens[np.asarray(scaling['times'])-1]]
                        require(np.array_equal(arrays['oracle_probs'],expected_oracle)
                                and np.array_equal(arrays['targets'],tokens[scaling['times']]), 'Larger prediction target/oracle order differs')
                        logits = arrays['logits'].astype(float)
                        exponentials = np.exp(logits-logits.max(axis=1,keepdims=True))
                        probabilities = exponentials/exponentials.sum(axis=1,keepdims=True)
                        error = float(np.max(np.abs(probabilities-arrays['predictive_probs'])))
                        require(error < 1e-12, 'Larger probability reconstruction differs')
                        max_logits_difference = max(max_logits_difference,error)
                        scores = np.sum(expected_oracle*np.log2(expected_oracle/probabilities),axis=1)
                        error = float(np.max(np.abs(scores-paired['larger_regret_bits'][index,:,seed])))
                        require(error < 1e-11, 'Larger raw KL differs from independently audited comparison')
                        max_kl_difference = max(max_kl_difference,error)
                        large_cache[index,seed] = probabilities
                        selected_model_records += 1
                    record['loss_bits'][f'larger_seed{seed}'] = old.loss(large_cache[index,seed][scaling['times'].index(time)],oracle)
                bayes_error = abs(record['loss_bits']['bayes']-record['entropy_bits']
                                  -paired['bayes_regret_bits'][index,[1,4,16,64].index(time)])
                require(bayes_error < 2e-12, 'Selected Bayesian value differs from released complete comparison')
                record['bayes_released'] = bool(num['passed'])
                group_rows.append(record); extended_rows.append(record)
            predictors = {}
            for name in PREDICTORS:
                values = [r['loss_bits'][name] for r in group_rows]
                predictors[name] = dict(loss_bits=moments(values),
                    regret_bits=moments([v-r['entropy_bits'] for v,r in zip(values,group_rows)]),
                    gain_over_uniform_bits=moments([6-v for v in values]),
                    scientific_status='withheld_numerical_precision' if name == 'bayes' and not num['passed'] else 'released')
            differences = {f'{name}_minus_{ref}': moments([r['loss_bits'][name]-r['loss_bits'][ref] for r in group_rows])
                           for name,ref in PAIRS}
            groups.append(dict(depth=depth,time=time,n_selected=len(selected),n_original=256,
                selected_indices=selected,predictors=predictors,paired_loss_bits=differences,
                subset_numerical_validation=num,bayes_released=bool(num['passed']),
                paired_bayes_comparisons_released=bool(num['passed']),
                selection_rate=len(selected)/256,
                note='All selected cases retained. Bayes and every Bayes-paired comparison are candidates only when bayes_released is false.'))
    summary = dict(schema='section4-unseen-context-extension-v1',alpha=1.,d=64,depths=list(old.DEPTHS),times=list(TIMES),
        status='Exploratory analysis of existing frozen cohorts, with larger saved model predictions and completed time64 references.',
        selection='E_t = {X_t not in {X_1,...,X_(t-1)}}: the last of t observed tokens appears for the first time. No completed outgoing transition from X_t occurs in the observed sequence.',
        score='Expected next-token log loss against the source-specific true transition row, in bits; mean across independent sequences satisfying E_t at fixed depth and time.',
        uncertainty='One SE and approximate 95% pointwise Student-t intervals across independent selected sequences. Intervals are descriptive, not simultaneous; Bayesian numerical accuracy is checked separately on the selected subset.',
        comparators=previous['comparators'],model_checkpoint_sha256=checkpoint_hashes,groups=groups,
        interpretation=['Selection depends only on observed tokens and is applied identically to every predictor.',
            'The predictors retain their full training prior; true depth and alpha are not supplied at prediction time.',
            'Subsets differ across times; connected means do not follow a fixed selected population.',
            'Gain over uniform establishes informative predictions without observed transitions from the current token, not identification of an internal circuit.',
            'Token counts have a different prior and are a descriptive comparator, not a matched structural ablation.',
            'L=0 is iid and serves as a control. It does not demonstrate transfer among distinct transition rows.',
            'Time64 Bayesian candidates at a failed selected-subset numerical gate are withheld even if the whole-cohort release passed.'],
        sampling=False,training=False,model_inference=False,input_sha256=tracked.files,
        source_time64_raw_manifest_sha256=fingerprint(raw_files),source_time64_raw_files_rehashed=len(raw_files),
        old_exporter_sha256=sha(old.__file__),script_sha256=sha(__file__),
        arithmetic_checks=dict(selected_larger_model_records_reconstructed=selected_model_records,
            maximum_softmax_probability_difference=max_logits_difference,maximum_larger_kl_difference_bits=max_kl_difference,
            maximum_original_loss_difference_bits=max_old_loss_difference))
    return native(summary),extended_rows


def verify_export(out):
    """Replay every aggregate independently from the compact paired loss table."""
    summary = json.loads((out/'summary.json').read_text())
    require(summary['script_sha256'] == sha(__file__) and summary['records_sha256'] == sha(out/'records.csv'),
            'Exporter or compact records changed')
    with (out/'records.csv').open() as stream:
        records = list(csv.DictReader(stream))
    count = 0
    for group in summary['groups']:
        selected = [r for r in records if int(r['depth']) == group['depth'] and int(r['time']) == group['time']]
        require([int(r['index']) for r in selected] == group['selected_indices'], 'Selected index ordering differs')
        def check(values,saved):
            nonlocal count
            mean = statistics.mean(values); se = statistics.stdev(values)/math.sqrt(len(values))
            require(saved['n'] == len(values) == group['n_selected']
                    and math.isclose(mean,saved['mean'],abs_tol=2e-14,rel_tol=1e-12)
                    and math.isclose(se,saved['standard_error'],abs_tol=2e-14,rel_tol=1e-12), 'Independent statistic replay differs')
            count += 1
        for name in PREDICTORS:
            values = [float(r[f'{name}_loss_bits']) for r in selected]
            for field,data in [('loss_bits',values),('regret_bits',[v-float(r['entropy_bits']) for v,r in zip(values,selected)]),
                               ('gain_over_uniform_bits',[6-v for v in values])]:
                check(data,group['predictors'][name][field])
        for name,ref in PAIRS:
            check([float(r[f'{name}_loss_bits'])-float(r[f'{ref}_loss_bits']) for r in selected],
                  group['paired_loss_bits'][f'{name}_minus_{ref}'])
        require(group['bayes_released'] == group['subset_numerical_validation']['passed']
                and all((r['bayes_released'] == 'True') == group['bayes_released'] for r in selected),
                'Scientific withholding flag differs')
    return dict(passed=True,statistics_recomputed=count,records=len(records),
                method='Independent Python statistics.mean/stdev aggregation of whole-sequence paired loss CSV.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_export(args.out))); return
    summary,records = build()
    args.out.mkdir(parents=True,exist_ok=True)
    fields = ['depth','time','index','source_id','query','entropy_bits','bayes_released']
    with (args.out/'records.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields+[f'{name}_loss_bits' for name in PREDICTORS],lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({k:row[k] for k in fields}|{f'{name}_loss_bits':row['loss_bits'][name] for name in PREDICTORS})
    summary['records_sha256'] = sha(args.out/'records.csv')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    failures = [dict(depth=g['depth'],time=g['time'],checks=[r['name'] for r in g['subset_numerical_validation'].get('checks',[]) if not r['passed']])
                for g in summary['groups'] if not g['bayes_released']]
    validation = dict(schema='section4-unseen-context-export-validation-v1',passed=True,
        meaning_of_passed='Provenance, saved-value arithmetic and withholding flags pass; not a declaration that all Bayesian subsets are scientifically released.',
        all_subset_numerical_gates_passed=not failures,withheld_bayes_groups=failures,
        groups=len(summary['groups']),selected_records=len(records),independent_csv_replay=verify_export(args.out),
        arithmetic_checks=summary['arithmetic_checks'],training=False,sampling=False,model_inference=False,
        script_sha256=sha(__file__),files={name:sha(args.out/name) for name in ('summary.json','records.csv')})
    (args.out/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out),groups=len(summary['groups']),records=len(records),withheld_bayes_groups=failures)))


if __name__ == '__main__':
    main()

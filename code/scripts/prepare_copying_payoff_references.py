"""Read-only inventory of canonical full-Bayes prefixes for copying payoff.

All 768 frozen original histories are retained. Exact alphabet relabeling
deduplicates ordered prefixes; counts alone never identify a full-Bayes task.
This script neither computes predictions nor launches samplers. Existing
references are candidates for a subsequent raw audit, never accepted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/transformer'))
MANIFEST = ROOT / 'code/transformer/configs/shared_base_shuffle_evaluation_cohort_2026-09-17.json'
COHORTS = ROOT / 'output/shared_base_shuffle_download_2026-09-18/cohorts'
OUT = ROOT / 'output/copying_payoff_reference_readiness_2026-09-19.json'
REFERENCE_ROOTS = tuple(ROOT / 'output' / name for name in (
    'shared_base_curve_references_2026-09-15',
    'shared_base_curve_continuation_2026-09-16/references',
    'shared_base_local_finish_2026-09-17/references',
    'shared_base_accuracy_cost_pilot_2026-09-15',
    'shared_base_accuracy_cost_pilot_v2_2026-09-15',
    'shared_base_accuracy_time16_extension_2026-09-15',
    'shared_base_surrogate_context_2026-09-15',
    'shared_base_surrogate_refined_2026-09-15',
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonicalize(tokens):
    """Return the complete ordered equality pattern and its observed-label map."""
    labels = {}
    pattern = []
    for value in tokens:
        value = int(value)
        require(0 <= value < 64, 'Token outside the declared alphabet')
        pattern.append(labels.setdefault(value, len(labels)))
    require(pattern, 'A prediction requires at least one observed token')
    return tuple(pattern), tuple(labels)


def alphabet_map(observed_labels, d=64):
    """Canonical index -> original symbol; unseen symbols follow sorted order."""
    labels = tuple(map(int, observed_labels))
    require(len(set(labels)) == len(labels) and all(0 <= x < d for x in labels),
            'Invalid observed-label map')
    return np.asarray(labels + tuple(x for x in range(d) if x not in labels), dtype=int)


def map_prediction(canonical_prediction, observed_labels):
    prediction = np.asarray(canonical_prediction, dtype=float)
    require(prediction.shape == (64,), 'Need all 64 predictive probabilities')
    result = np.empty(64)
    result[alphabet_map(observed_labels)] = prediction
    return result


def canonical_oracle(transition, tokens, observed_labels):
    """Preserve the queried original last context while permuting its successors."""
    return np.asarray(transition, dtype=float)[int(tokens[-1]), alphabet_map(observed_labels)]


def reference_key(pattern):
    return f't{len(pattern):04d}_' + fingerprint(list(pattern))[:20]


def inventory_tokens(cohorts, times=range(5, 17)):
    references, uses = {}, []
    for condition, tokens in sorted(cohorts.items()):
        for index, sequence in enumerate(tokens):
            for observed in times:
                pattern, labels = canonicalize(sequence[:observed])
                key = reference_key(pattern)
                if key not in references:
                    references[key] = dict(reference_key=key, time=observed, tokens=list(pattern),
                        token_pattern_sha256=fingerprint(list(pattern)), n_seen=len(labels),
                        queried_canonical_label=pattern[-1], uses_by_condition={}, use_count=0,
                        existing_reference_candidates=[])
                ref = references[key]
                require(ref['tokens'] == list(pattern), 'Canonical reference key collision')
                ref['use_count'] += 1
                ref['uses_by_condition'][condition] = ref['uses_by_condition'].get(condition, 0) + 1
                uses.append(dict(condition=condition, index=index, time=observed, reference_key=key,
                    canonical_to_original_seen=list(labels), queried_original_label=int(sequence[observed-1]),
                    queried_canonical_label=pattern[-1], original_prefix_sha256=hashlib.sha256(
                        np.asarray(sequence[:observed], dtype='<i8').tobytes()).hexdigest()))
    return references, uses


def relocate(path):
    path = Path(path)
    if path.exists():
        return path
    require('output' in path.parts, 'Original source path is unavailable')
    return ROOT.joinpath(*path.parts[path.parts.index('output'):])


def candidate_reference(path):
    report = json.loads(Path(path).read_text())
    identity = report['identity']
    require(report['reference_id'] == fingerprint(identity), 'Candidate reference identity hash mismatch')
    target = identity.get('target', identity.get('numerical_problem'))
    require(target is not None and target['d'] == 64 and target['depths'] == list(range(9))
            and np.allclose(target['depth_prior'], np.ones(9)/9, rtol=0, atol=1e-15),
            'Candidate does not use the full shared-base depth prior')
    order = len(target['alpha_nodes'])
    require(order in (16, 32), 'Candidate integration order outside declared pair')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    require(np.allclose(target['alpha_nodes'], 1+nodes/2, rtol=0, atol=2e-15)
            and np.allclose(target['alpha_weights'], weights/2, rtol=0, atol=2e-15),
            'Candidate exponent prior differs')
    inputs = identity['inputs']
    model = next(iter(inputs['model_files'].values()))
    folder = relocate(model['path']).parents[2]
    index = inputs.get('index', inputs.get('sequence_index'))
    source_path = folder / 'data' / f'seq_{index:06d}.json'
    require(source_path.is_file() and sha(source_path) == inputs['source_file_sha256'],
            'Candidate original source missing or changed')
    source = json.loads(source_path.read_text())
    require(source['source_id'] == inputs['source_id'], 'Candidate source identity mismatch')
    observed = target['prefix_length']
    tokens = np.asarray(source['tokens'][:observed], dtype='<i8')
    digest = hashlib.sha256(tokens.dtype.str.encode())
    digest.update(json.dumps(tokens.shape).encode()); digest.update(tokens.tobytes())
    require(digest.hexdigest() == target['token_hash'], 'Candidate prefix hash mismatch')
    pattern, labels = canonicalize(tokens)
    return reference_key(pattern), dict(path=str(Path(path).resolve()), sha256=sha(path),
        reference_id=report['reference_id'], order=order, source_path=str(source_path.resolve()),
        source_sha256=inputs['source_file_sha256'], canonical_to_original_seen=list(labels),
        reported_ready=report.get('ready_for_cohort_check', report.get('local_validation_passed')),
        accepted_for_reuse=False, raw_integrity_rechecked=False,
        note='Exact ordered-prefix match only. Raw-array and numerical acceptance for the new aggregate remain required.')


def scan_candidates(references, roots=REFERENCE_ROOTS, maximum_reports=10000):
    considered, rejected, matched = 0, [], 0
    seen = set()
    for root in roots:
        if not Path(root).exists():
            continue
        for path in sorted(Path(root).rglob('seq_*_t*.json')):
            match = re.fullmatch(r'seq_\d+_t(\d+)\.json', path.name)
            if match is None or not 5 <= int(match.group(1)) <= 16 or path.resolve() in seen:
                continue
            seen.add(path.resolve()); considered += 1
            require(considered <= maximum_reports, 'Candidate report scan exceeded its explicit bound')
            try:
                key, candidate = candidate_reference(path)
            except (ValueError, KeyError, OSError, StopIteration) as error:
                rejected.append(dict(path=str(path), reason=str(error)))
                continue
            if key in references:
                references[key]['existing_reference_candidates'].append(candidate)
                matched += 1
    return dict(roots=[str(Path(p).resolve()) for p in roots], reports_examined=considered,
                matching_report_count=matched, rejected_reports=rejected,
                scope='Only listed reference trees; no raw chain arrays opened or validated.')


def counts_by_horizon(references, uses, horizons=(4, 8, 16)):
    result = []
    for horizon in horizons:
        refs = [r for r in references.values() if r['time'] <= horizon]
        candidate_pairs = {(r['reference_key'], c['order']) for r in refs
                           for c in r['existing_reference_candidates']}
        result.append(dict(horizon=horizon, existing_exact_prefix_times=list(range(1, min(4, horizon)+1)),
            new_prefix_times=list(range(5, horizon+1)), selected_original_histories=768,
            original_history_time_pairs=sum(u['time'] <= horizon for u in uses),
            unique_ordered_prefixes=len(refs), reference_tasks_both_orders=2*len(refs),
            candidate_matched_reference_tasks=len(candidate_pairs),
            candidate_unmatched_reference_tasks=2*len(refs)-len(candidate_pairs),
            accepted_reused_reference_tasks=0,
            tasks_requiring_new_sampling_or_independent_reuse_audit=2*len(refs)))
    return result


def pilot_plan(references):
    """Twelve timing tasks selected solely by prefix length and label structure."""
    selected = []
    for observed in (5, 8, 16):
        refs = [r for r in references.values() if r['time'] == observed]
        if not refs:
            continue
        for name, reverse in [('fewest_seen', False), ('most_seen', True)]:
            target = min(refs, key=lambda r: ((-1 if reverse else 1)*r['n_seen'], tuple(r['tokens'])))
            if target['reference_key'] in {s['reference_key'] for s in selected}:
                continue
            selected.append(dict(reference_key=target['reference_key'], time=observed,
                                 tokens=target['tokens'], n_seen=target['n_seen'], selection=name))
    return dict(status='proposal_only_not_submitted', representative_patterns=selected,
        task_count=2*len(selected), orders=[16, 32], chains=4, burn=16384, thin=1,
        initial_draws_per_chain=4096, maximum_draws_per_chain=4096,
        maximum_workers=1, task_wall_seconds=600, total_worker_seconds=7200,
        output=str(ROOT/'output/copying_payoff_canonical_bayes_pilot_2026-09-19'),
        implementation='New small wrapper around unchanged RankedLSARowEvidence, SharedBaseSurrogateMCMC and CurveChainRun; selected_grid_checks for final states.',
        limitation='Existing cohort CLI requires source/oracle/model files and is not a generic canonical-token CLI. Timing pilot cannot certify population accuracy or initiate production.',
        followup='Measure normalization, probability mixing, integration agreement, time and memory before freezing a scientific draw budget. Include a t4 catalog reproduction check.')


def build(manifest_path=MANIFEST, cohorts_path=COHORTS, roots=REFERENCE_ROOTS):
    from shared_base_short_evidence import SharedBaseShortEvidence
    catalog = SharedBaseShortEvidence()
    manifest = json.loads(Path(manifest_path).read_text())
    require((manifest['d'], manifest['n_tokens'], manifest['master_seed']) == (64, 1025, 20260930),
            'Unexpected frozen cohort protocol')
    require(sorted(c['id'] for c in manifest['conditions']) == ['a1_L0', 'a1_L2', 'a1_L3'],
            'Need exactly the existing three source conditions')
    cohorts, bindings = {}, {}
    for condition in manifest['conditions']:
        path = Path(cohorts_path) / (condition['id']+'.npz')
        with np.load(path, allow_pickle=False) as archive:
            tokens = archive['tokens']
            require(tokens.shape == (256, 1025) and tokens.dtype.kind in 'iu', 'All 256 original histories required')
            token_hash = hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()
            require(token_hash == condition['arrays_sha256']['tokens'], 'Frozen token array changed')
            for name in ('base', 'transition'):
                require(hashlib.sha256(np.asarray(archive[name], dtype='<f8').tobytes()).hexdigest()
                        == condition['arrays_sha256'][name], f'Frozen oracle array changed: {name}')
            require(np.array_equal(archive['indices'], np.arange(256)), 'Original cohort index order changed')
            require(str(archive['manifest_sha256']) == sha(manifest_path), 'Cohort manifest binding changed')
        cohorts[condition['id']] = tokens
        bindings[condition['id']] = dict(path=str(path.resolve()), sha256=sha(path),
            tokens_sha256=token_hash, original_arrays_sha256=condition['arrays_sha256'])
    references, uses = inventory_tokens(cohorts)
    scan = scan_candidates(references, roots)
    by_time = [dict(time=t, unique_ordered_prefixes=sum(r['time']==t for r in references.values()),
                   original_history_time_pairs=sum(u['time']==t for u in uses)) for t in range(5,17)]
    return dict(schema='copying-payoff-canonical-reference-readiness-v1', manifest_sha256=sha(manifest_path),
        script_sha256=sha(__file__), source_bindings=bindings, exact_early_catalog_validation=catalog.validation,
        depths=[0,2,3], alpha=1, histories_per_condition=256, histories_total=768,
        canonicalization='First-occurrence alphabet relabeling of the complete ordered prefix; never collapse different transition orders to one count vector.',
        prior=dict(alpha=[.5,1.5],depths=list(range(9)),ranks='uniform permutation',initial_token='q'),
        predictive_mapping='canonical_to_original_seen followed by sorted unseen labels gives a full64 permutation; p_original[permutation]=p_canonical. Oracle row is T[X_t, permutation].',
        numerical_covariance='A reused canonical reference induces common Monte Carlo error across every mapped history. Aggregate oracle-weighted loss gradients per canonical task before computing numerical error; never count mapped uses as independent reference draws.',
        unseen_probability='The exact posterior is exchangeable over unseen labels. Any Rao-Blackwell averaging of their MCMC probabilities requires its own declared and validated postprocessing; original arrays remain unchanged.',
        horizons=counts_by_horizon(references,uses), by_time=by_time,
        references=sorted(references.values(),key=lambda r:(r['time'],tuple(r['tokens']))), uses=uses,
        candidate_scan=scan, pilot=pilot_plan(references), sampling=False, training=False,
        model_inference=False, scientific_results_released=False,
        interpretation='Readiness inventory only; source identities, all histories, queried contexts and probability mappings are fixed without inspecting loss gaps.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--cohorts',type=Path,default=COHORTS)
    parser.add_argument('--reference-root',type=Path,action='append')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args(argv)
    report=build(args.manifest,args.cohorts,tuple(args.reference_root or REFERENCE_ROOTS))
    if args.write:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        serialized=json.dumps(report,indent=2,allow_nan=False)+'\n'
        if args.out.exists():
            require(args.out.read_text()==serialized,'Frozen readiness output differs')
        else:
            with args.out.open('x') as stream:stream.write(serialized)
    print(json.dumps(dict(written=args.write,horizons=report['horizons'],by_time=report['by_time'],
                         candidate_scan={k:v for k,v in report['candidate_scan'].items() if k!='rejected_reports'},
                         rejected_candidate_reports=len(report['candidate_scan']['rejected_reports']),
                         pilot=report['pilot']),indent=2))


if __name__=='__main__':
    main()

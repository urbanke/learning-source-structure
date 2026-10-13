"""Provisional t128 display only; the accepted copying readers remain strict.

The compact producer must bind a complete fixed first32 cohort and the complete
one-draw-per-depth frequency calculation. This reader replays paired gain/source
SE. It does not certify the reduced latent integration or posterior sampling.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

CONDITIONS = ('a1_L0', 'a1_L2', 'a1_L3')
SCORES = ('stationary_loss', 'bayes_loss_a32', 'ordered_seed0',
          'shuffled_seed0', 'ordered_seed1', 'shuffled_seed1')
FULL_PLAN_SHA = 'b9a1e6b020f3ff81e3f95d895becc371254b5f09f2f71e0d0281e6f3c18c3934'
PARENT_FREQUENCY_PLAN_SHA = '345a9adbf4bbe7a2311108093145ae14cc86253612bdbfc3d182f55a5d5786be'
INPUT_SHA = '894397465e35bb72ed0ada3469d9a5a152bbb806afd5e94b01efc44dc7eb1617'
BANK_SHA = '20856e7c096d9d1fbd41058bfcf27d0ac417b24f652812a792125861821cf428'
ROW_INDICES = [129*j+k for j in range(32) for k in (0, *[1+16*(d-1) for d in range(1,9)])]
COMPARATOR = 'Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.'
LEGACY_ADAPTER_SHA = '96996415ee1b22e71ffa8ef301059490656d137b4e3c1464577609a0f544140e'
EXPANDED_FREQUENCY_SHA = 'a448b3f783aae7d2ee23d38e554a5c888064cdf22840b030201940239b189c33'
EXPANDED_FULL_SHA = '7806bb905a0b085a3e1b2303d13520c0fc6fe66a9473dba98780e293874dcc09'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def validate_metadata(p):
    require(p['schema'] == 'copying-endpoint128-provisional-figure-input-v1'
        and p['complete_cohort'] is True and p['provisional'] is True
        and p['scientific_reference_validated'] is False
        and p['numerical_acceptance_passed'] is False,
        'Only an explicitly unaccepted complete-cohort preview belongs here')
    auth = p['display_authorization']
    require(auth['authorized'] is True and isinstance(auth['verification_remark'], str)
        and auth['verification_remark'].strip(), 'Explicit provisional display authorization and remark required')
    require(p['times'] == [128] and p['conditions'] == list(CONDITIONS)
        and p['original_histories'] == 768 and p['selected_histories'] == 96
        and p['histories_per_condition'] == 32 and p['indices'] == list(range(32))
        and p['source_pairing_replayed'] is True and p['cumulative_claim'] is False
        and p['comparator'] == COMPARATOR, 'Fixed source/predictor pairing or endpoint scope changed')
    bindings = p['source_bindings']
    require(bindings['full_plan_sha256'] == FULL_PLAN_SHA
        and bindings['parent_frequency_plan_sha256'] == PARENT_FREQUENCY_PLAN_SHA
        and bindings['input_sha256'] == INPUT_SHA
        and all(hash_string(bindings[k]) for k in
            ('frequency_preview_plan_sha256', 'frequency_preview_summary_sha256',
             'frequency_preview_arrays_sha256', 'full_bayes_capture_sha256')),
        'Provisional source and computation bindings required')
    f = p['frequency']
    require(f['complete'] is True and f['types'] == 1105 and f['histories'] == 96
        and f['bank'] == 0 and f['original_bank_sha256'] == BANK_SHA
        and f['fine_alpha_order'] == 32 and f['original_rows'] == 4128
        and f['selected_rows'] == 288 and f['row_indices'] == ROW_INDICES
        and f['positive_depth_draw_index'] == 0 and f['positive_depth_weight_multiplier'] == 16
        and f['depth_zero_weight_unchanged'] is True
        and f['joint_integration_precedes_predictive_ratios'] is True,
        'Incomplete or different reduced latent integral')
    mass = np.asarray(f['depth_weight_mass'], dtype=float)
    residual = f['maximum_predictive_normalization_error']
    require(mass.shape == (9,) and np.isfinite(mass).all()
        and np.max(abs(mass-1/9)) < 2e-13
        and isinstance(residual, (int,float)) and np.isfinite(residual)
        and 0 <= residual < 1e-10, 'Invalid prior weights or predictive normalization')
    audit = p['bayes_audit']
    require(audit['fine_order'] == 32 and audit['complete_fine_references'] == 96
        and audit['source_reports_retained'] is True
        and isinstance(audit['failed_audits'], list)
        and isinstance(audit['paired_order_status'], str),
        'Preserve the complete fine posterior and its actual audit status')
    require(p['uncertainty_scope'] == 'Paired source-sequence SE only; numerical integration and posterior uncertainty are not certified.',
        'Source SE must not be represented as a numerical certificate')


def replay(p, arrays):
    validate_metadata(p)
    require(set(arrays) == {'times'} | {c+'_'+s for c in CONDITIONS for s in SCORES}
        and np.array_equal(arrays['times'], [128]), 'Wrong preview score keys or time')
    require(all(arrays[c+'_'+s].shape == (32,1)
        and np.isfinite(arrays[c+'_'+s]).all() for c in CONDITIONS for s in SCORES),
        'All32 paired histories in each of the three conditions are required')
    result = {}
    for condition in CONDITIONS:
        gains = {'bayes': arrays[condition+'_stationary_loss'] - arrays[condition+'_bayes_loss_a32']}
        gains.update({f'neural_seed{s}': arrays[condition+f'_shuffled_seed{s}']
            - arrays[condition+f'_ordered_seed{s}'] for s in (0,1)})
        result[int(condition.split('L')[-1])] = {
            key: value for name,g in gains.items() for key,value in
            ((name+'_mean', float(g.mean())),
             (name+'_source_se', float(g.std(ddof=1)/np.sqrt(32))))}
    # Structural checks only. These are deliberately not numerical acceptance.
    return result, {'copying_endpoint128_provisional_complete_cohort': True,
                    'copying_endpoint128_provisional_prior_mass_max_error': float(np.max(abs(np.asarray(p['frequency']['depth_weight_mass'])-1/9))),
                    'copying_endpoint128_provisional_normalization_max_error': float(p['frequency']['maximum_predictive_normalization_error'])}


def load(source):
    source = Path(source)
    metadata = source/'provenance.json.gz'
    if not metadata.exists():
        metadata = source/'provenance.json'
    opener = gzip.open if metadata.suffix == '.gz' else open
    with opener(metadata, 'rt') as handle:
        p = json.load(handle)
    score_file = source/'paired_scores.npz'
    expanded = p['schema'] == 'copying-endpoint128-expanded-provisional-figure-input-v1'
    require(p['paired_scores_sha256'] == sha(score_file)
        and p['adapter_sha256'] in ({sha(__file__)} if expanded else {sha(__file__), LEGACY_ADAPTER_SHA}),
        'Provisional score arrays or reader changed')
    with np.load(score_file, allow_pickle=False) as arrays:
        if expanded:
            original = source/'original_paired_scores.npz'
            require(sha(original) == p['original_preview']['paired_scores_sha256']
                and p['original_preview']['adapter_sha256'] == LEGACY_ADAPTER_SHA,
                'Original preview binding changed')
            with np.load(original, allow_pickle=False) as old:
                result, checks = expanded_replay(p, arrays, old)
        else:
            result, checks = replay(p, arrays)
    return result, checks, [score_file, metadata] + ([original] if expanded else [])


def expanded_replay(p, arrays, old):
    replay(p['original_preview'], old)
    groups = p['expanded_groups']
    require(set(groups) in ({'a1_L2'}, {'a1_L2', 'a1_L3'})
        and p['complete_cohort'] is True and p['provisional'] is True
        and p['scientific_reference_validated'] is False and p['numerical_acceptance_passed'] is False
        and p['display_authorization']['authorized'] is True
        and p['display_authorization']['verification_remark'].strip()
        and p['times'] == [128] and p['comparator'] == COMPARATOR
        and p['frequency_plan_sha256'] == EXPANDED_FREQUENCY_SHA
        and p['full_plan_sha256'] == EXPANDED_FULL_SHA,
        'Only complete fixed-depth provisional expansions are supported')
    require(set(arrays) == set(old) and np.array_equal(arrays['times'], [128]), 'Score keys changed')
    result = {}
    for c in CONDITIONS:
        n = 256 if c in groups else 32
        require(all(arrays[c+'_'+s].shape == (n,1) and np.isfinite(arrays[c+'_'+s]).all()
            and np.array_equal(arrays[c+'_'+s][:32], old[c+'_'+s]) for s in SCORES),
            'Incomplete expanded cohort or changed retained sources')
        if c in groups:
            g = groups[c]
            require(g['condition'] == c and g['histories'] == g['posterior_references'] == 256
                and g['indices'] == list(range(256)) and g['frequency_complete'] is True
                and g['posterior_complete'] is True and g['source_pairing_replayed'] is True
                and isinstance(g['failed_audits'], list)
                and 0 <= g['maximum_normalization_error'] < 1e-10
                and all(hash_string(g[k]) for k in ('frequency_summary_sha256', 'frequency_arrays_sha256',
                    'full_summary_sha256', 'full_arrays_sha256', 'input_sha256')),
                'Expanded computation or pairing proof missing')
        gains = {'bayes': arrays[c+'_stationary_loss']-arrays[c+'_bayes_loss_a32']}
        gains.update({f'neural_seed{s}': arrays[c+f'_shuffled_seed{s}']-arrays[c+f'_ordered_seed{s}'] for s in (0,1)})
        result[int(c.split('L')[-1])] = {k:v for name,g in gains.items() for k,v in
            ((name+'_mean', float(g.mean())), (name+'_source_se', float(g.std(ddof=1)/np.sqrt(n))))}
    return result, {'copying_endpoint128_expanded_complete_cohorts': True}

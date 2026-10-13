"""Summarize the general-source copying diagnostics without fitting components."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'output/general_lsa_copying_2026-09-17'
OUT = ROOT / 'results/transformer/general_lsa_copying_2026-09-17'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1) / np.sqrt(len(values))), n=len(values))


def main():
    selection = json.loads((RAW / 'discovery/selection.json').read_text())
    candidate, control = [selection[k] for k in ('candidate_label', 'control_label')]
    confirmation = json.loads((RAW / 'confirmation/summary.json').read_text())
    groups, arrays, sources = [], {}, []
    for depth in (0, 2, 3):
        path = RAW / 'natural' / f'a1_L{depth}.npz'
        with np.load(path, allow_pickle=False) as a:
            names = list(a['names'])
            ce, kl = a['ce_bits'], a['kl_bits']
            probabilities, q = a['probabilities'], a['base']
            times = a['times']
            assert ce.shape == (256, 9, 1024) and np.isfinite(ce).all()
            assert a['max_forward_error'] < 1e-10 and a['max_causal_error'] < 1e-10
            assert np.max(np.abs(probabilities.sum(-1) - 1)) < 1e-12
            results = {}
            for name in ('intact', 'off_' + candidate, 'off_' + control):
                j = names.index(name)
                difference = ce[:, j] - ce[:, 0]
                cumulative = difference.cumsum(1)
                results[name] = dict(
                    total_loss_change_bits=stats(cumulative[:, -1]),
                    time1_loss_change_bits=stats(difference[:, 0]),
                    at_times={str(int(t)): dict(
                        regret_bits=stats(kl[:, j, t-1]),
                        loss_change_bits=stats(difference[:, t-1]),
                        cumulative_loss_change_bits=stats(cumulative[:, t-1]),
                        distance_to_base=stats(.5 * np.abs(probabilities[:, j, k] - q).sum(-1)))
                              for k, t in enumerate(times)})
                for label, values in (('benefit', difference), ('cumulative_benefit', cumulative), ('regret', kl[:, j])):
                    arrays[f'L{depth}_{name}_{label}_mean'] = values.mean(0)
                    arrays[f'L{depth}_{name}_{label}_se'] = values.std(0, ddof=1) / np.sqrt(len(values))
            groups.append(dict(depth=depth, n_sequences=256, predictors=results))
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    key_groups = []
    for depth in (0, 2, 3):
        path = RAW / 'key_path_natural' / f'a1_L{depth}.npz'
        with np.load(path, allow_pickle=False) as a:
            ce, kl, intact = a['ce_bits'], a['kl_bits'], a['intact_ce_bits']
            assert np.max(np.abs(ce[:, :, 0] - intact[:, None, 0])) < 1e-12
            results = {}
            for j, name in enumerate(a['names']):
                difference = ce[:, j] - intact
                cumulative = difference.cumsum(1)
                eligible = a['eligible']
                ineligible_counts = (~eligible).sum(1)
                tv = a['probability_tv'][:, j]
                results[str(name)] = dict(
                    total_loss_change_bits=stats(cumulative[:, -1]),
                    time1_loss_change_bits=stats(difference[:, 0]),
                    no_previous_continuation_tv=stats(np.sum(tv * ~eligible, 1) / ineligible_counts),
                    no_previous_continuation_max_tv=float(tv[~eligible].max()),
                    at_times={str(int(t)):dict(regret_bits=stats(kl[:, j, t-1]),
                        loss_change_bits=stats(difference[:, t-1]),
                        cumulative_loss_change_bits=stats(cumulative[:, t-1])) for t in a['times']})
                for label, values in (('benefit', difference), ('cumulative_benefit', cumulative), ('regret', kl[:, j])):
                    arrays[f'L{depth}_{name}_{label}_mean'] = values.mean(0)
                    arrays[f'L{depth}_{name}_{label}_se'] = values.std(0, ddof=1) / np.sqrt(len(values))
            key_groups.append(dict(depth=depth, n_sequences=256, predictors=results))
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'head_curves.npz', **arrays)
    controlled = {name: confirmation['routing']['conditions'][name]['query_specific'] for name in
                  ('intact', 'zero_candidate', 'swap_successors', 'swap_contexts', 'sham')}
    report = dict(
        schema='general-lsa-copying-diagnostics-v1', d=64, alpha=1, source_depths=[0, 2, 3],
        n_tokens=1025, checkpoint='results/transformer/runs/d64/model_100000.pt',
        checkpoint_sha256=sha(ROOT / 'results/transformer/runs/d64/model_100000.pt'),
        trained_source_prior=dict(alpha=[.5, 1.5], depth=list(range(9)), permutation='uniform'),
        candidate=candidate, control=control, controlled_confirmation=controlled,
        key_path_controlled=json.loads((RAW / 'key_path_controlled/summary.json').read_text()),
        interpretation=[
            'Independent controlled successor-site swaps support a copying computation through the candidate.',
            'The L=0 control checks whether head deletion leaves the ability to learn a nonuniform common base.',
            'Deleting the entire head changes the first prediction, when no past continuation exists. Its full loss effect is not a pure copying effect.',
            'The narrower key-path intervention preserves the first prediction and removes controlled copying responses, but also changes natural predictions with no earlier completed continuation of the current token.',
            'Neither intervention is identified with the archived LSA unigram or an informed Bayesian unigram.',
            'Natural loss changes are conditional expected next-token log losses, averaged across independent sequences before accumulation.',
            'No shared-base informed-Bayes cumulative curve is provided by this diagnostic.',
            'Archived unigram/copier results in archived/ concern distinct statistical predictors.'],
        groups=groups, key_path_groups=key_groups, inputs=sources,
        controlled_inputs={name: sha(RAW / name) for name in
                           ('discovery/selection.json', 'discovery/summary.json', 'confirmation/summary.json', 'confirmation/records.npz')},
        code_sha256=sha(__file__), curves_sha256=sha(OUT / 'head_curves.npz'))
    (OUT / 'head_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(candidate=candidate, control=control, controlled=controlled,
                          cumulative_loss_changes={str(g['depth']):g['predictors']['off_' + candidate]['total_loss_change_bits'] for g in groups})))


if __name__ == '__main__':
    main()

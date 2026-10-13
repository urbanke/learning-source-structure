"""Check cohort alignment and fail-closed release without running a sampler."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts')]
import pooled_mean_context64 as run
import numpy as np


class Context64Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.selection = dict(selection=dict(source_plan_id='fixture', input_sha256={}, cases=[]))
        times = run.base.old.ALL_TIMES
        self.expected = {}
        for fi, family in enumerate(run.base.old.FAMILIES):
            name = family+'_prior'
            seeds = np.arange(69)+10000*fi
            tokens = (np.arange(1025)[None, :]+np.arange(69)[:, None])%64
            transitions = np.broadcast_to(.5*np.eye(64)+.5/64, (69, 64, 64)).copy()
            rows = np.array([[transitions[i, tokens[i, t-1]] for t in times] for i in range(69)])
            source = self.root/f'heldout/{name}.npz'
            source.parent.mkdir(parents=True, exist_ok=True)
            np.savez(source, prefixes=times, tokens=tokens, sequence_seed=seeds,
                     oracle_rows=rows, transition_probs=transitions)
            receipt = dict(plan_id='fixture', data_sha256=run.sha(source))
            source.with_suffix('.json').write_text(json.dumps(receipt))
            for mi, model in enumerate(('pooled_seed0', 'pooled_seed1')):
                path = self.root/f'eval/{model}/{name}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                scores = np.arange(69)[:, None]+np.arange(len(times))[None, :]/10+mi/100
                np.savez(path, prefixes=times, sequence_seed=seeds, next_kl_bits=scores)
                path.with_suffix('.json').write_text(json.dumps(dict(data_sha256=run.sha(path),
                    source_data_sha256=receipt['data_sha256'],
                    identity=dict(name=model, source_plan_id='fixture'))))
            for i in range(5, 69):
                self.selection['selection']['cases'].append(dict(family=family, index=i, sequence_seed=int(seeds[i])))
                self.expected[family, i] = dict(tokens=tokens[i, :64].tolist(), oracle=rows[i, times.index(64)].tolist())
        for path in self.root.rglob('*'):
            if path.is_file():
                self.selection['selection']['input_sha256'][str(path.relative_to(self.root))] = run.sha(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_context_and_frozen_pairing(self):
        cases, _ = run.source_cases(self.root, self.selection)
        self.assertEqual(len(cases), 192)
        for case in cases:
            expected = self.expected[case['family'], case['index']]
            self.assertEqual(case['tokens'], expected['tokens'])
            self.assertEqual(case['oracle_probs'], expected['oracle'])
            self.assertAlmostEqual(case['transformer_regret_bits']['pooled_seed1'],
                case['index']+run.base.old.ALL_TIMES.index(64)/10+.01)

    def test_source_seed_mismatch_is_rejected(self):
        self.selection['selection']['cases'][0]['sequence_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed changed'):
            run.source_cases(self.root, self.selection)

    def test_missing_history_is_not_an_available_case_mean(self):
        self.selection['selection']['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'Complete balanced cohort'):
            run.source_cases(self.root, self.selection)

    def test_source_mutation_is_rejected(self):
        path = self.root/'heldout/product_prior.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Frozen sources'):
            run.source_cases(self.root, self.selection)

    def test_incomplete_population_cannot_publish(self):
        cases, _ = run.source_cases(self.root, self.selection)
        histories = [dict(c, original_tokens=c['tokens'], reference_key=c['family']+'_'+str(c['index'])) for c in cases]
        r = dict(histories=histories, source_sha256={}, calibration=dict(hashes={}))
        args = types.SimpleNamespace(plan=self.root/'plan', out=self.root/'out', raw=self.root,
                                     dependencies=self.root, recovery=self.root)
        with patch.object(run, 'read_plan', return_value=({}, 'p', {})), \
             patch.object(run, 'prepared', return_value=(r, 'prepared')), \
             patch.object(run, 'verified_overlay', return_value=(self.root, {})), \
             patch.object(run.base.validation, 'analyze_arm', side_effect=AssertionError('Must not average')):
            self.assertEqual(run.merge(args), 3)
        report = json.loads((args.out/'population/summary.json').read_text())
        self.assertFalse(report['complete'])
        self.assertFalse(report['scientific_reference_validated'])
        self.assertEqual(len(report['missing_tasks']), 768)
        self.assertNotIn('publication', report)


if __name__ == '__main__':
    unittest.main()

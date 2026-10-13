"""Context alignment and complete-cohort gates for the two long populations."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pooled_mean_long_context as run
from test_pooled_mean_context64 import Context64Tests

class LongContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Context64Tests();self.fixture.setUp()
        self.root,self.selection=self.fixture.root,self.fixture.selection
    def tearDown(self):self.fixture.tearDown()

    def test_both_contexts_use_actual_history_and_score_column(self):
        for n in (256,1024):
            cases,_=run.source_cases(self.root,self.selection,n)
            self.assertEqual(len(cases),192)
            for row in cases:
                with np.load(self.root/f"heldout/{row['family']}_prior.npz") as z:
                    tokens=z['tokens'][row['index'],:n]
                    np.testing.assert_array_equal(row['tokens'],tokens)
                    np.testing.assert_array_equal(row['oracle_probs'],z['transition_probs'][row['index'],tokens[-1]])
                self.assertAlmostEqual(row['transformer_regret_bits']['pooled_seed1'],row['index']+run.base.old.ALL_TIMES.index(n)/10+.01)

    def test_missing_population_blocks_average_for_each_context(self):
        for n in (256,1024):
            cases,_=run.source_cases(self.root,self.selection,n)
            histories=[dict(c,original_tokens=c['tokens'],reference_key=c['family']+'_'+str(c['index'])) for c in cases]
            r=dict(observed=n,histories=histories,source_sha256={},calibration=dict(hashes={}))
            args=types.SimpleNamespace(plan=self.root/'plan',out=self.root/str(n),raw=self.root,dependencies=self.root,recovery=self.root)
            with patch.object(run,'read_plan',return_value=({'observed':n},'p',{})),patch.object(run,'prepared',return_value=(r,'prepared')),patch.object(run,'verified_overlay',return_value=(self.root,{})),patch.object(run.base.validation,'analyze_arm',side_effect=AssertionError('Must not average incomplete population')):
                self.assertEqual(run.merge(args),3)
            result=json.loads((args.out/'population/summary.json').read_text())
            self.assertEqual(len(result['missing_tasks']),768)
            self.assertFalse(result['scientific_reference_validated'])
            self.assertNotIn('publication',result)

    def test_partial_deterministic_preparation_is_not_released(self):
        args=types.SimpleNamespace(plan=self.root/'plan',out=self.root/'missing')
        rows=[dict(family=f,index=i) for f in run.base.old.FAMILIES for i in range(5,69)]
        with patch.object(run,'read_plan',return_value=({'observed':256},'p',{})),patch.object(run,'sources',return_value=({'cases':rows},'sources')):
            with self.assertRaises(FileNotFoundError):run.preparation_merge(args)
        self.assertFalse((args.out/'prepared.json').exists())

    def test_missing_source_history_rejected(self):
        self.selection['selection']['cases'].pop()
        with self.assertRaisesRegex(ValueError,'Complete balanced cohort'):
            run.source_cases(self.root,self.selection,1024)

    def test_parallel_preparation_binds_every_history(self):
        args=types.SimpleNamespace(plan=self.root/'plan.json',out=self.root/'prepared',raw=self.root,
            dependencies=self.root,recovery=self.root,index=-1)
        args.plan.write_text(json.dumps({'observed':256}))
        with patch.object(run,'read_plan',return_value=({'observed':256},'p',self.selection)), \
             patch.object(run,'verified_overlay',return_value=(self.root,{})), \
             patch.object(run.base,'dependency_state',return_value={'sealed':True}), \
             patch.object(run.base,'FamilyEvidence',return_value=object()), \
             patch.object(run.base,'component_prediction',return_value={'predictive_probs':[1/64]*64}), \
             patch('builtins.print'):
            # The test plan is deliberately outside the frozen source inventory.
            run.source_prepare(args)
            for i in range(192):args.index=i;run.history_prepare(args)
            run.preparation_merge(args)
            report,_=run.prepared(args,'p',full=True)
            self.assertEqual(len(report['histories']),192)
            tasks=run.specifications(report)
            self.assertEqual(len({(x['reference_key'],x['arm']) for x in tasks}),768)
            self.assertTrue(all(x['observed']==256 and len(x['tokens'])==256 for x in tasks))
            path=args.out/'preparation/history-000.json'
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Deterministic preparation changed'):
                run.prepared(args,'p',full=True)

if __name__=='__main__':unittest.main()

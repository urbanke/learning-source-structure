"""The publication adapter rejects unapproved, partial or altered releases."""
import copy,gzip,json,sys
from pathlib import Path
import numpy as np
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import copying_precision_figure_input as a
SOURCE=Path(__file__).resolve().parents[2]/'results/transformer/copying_bayes_precision_2026-09-24'

class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        with gzip.open(SOURCE/'provenance.json.gz','rt') as f:self.p=json.load(f)
        with np.load(SOURCE/'paired_scores.npz',allow_pickle=False) as f:self.arrays={k:f[k].copy() for k in f.files}

    def test_complete_release_replays_all_three_cohorts(self):
        p,arrays=self.p,self.arrays
        a.check_release(p['source_summary'],p['source_plan'],p['source_selection'])
        result,checks=a.replay(p['source_summary'],arrays)
        self.assertEqual(set(result),{0,2,3});self.assertEqual(len(checks),6);self.assertLess(max(checks.values()),2e-12)
        self.assertTrue(all(len(result[d][m]['bayes_mean'])==16 for d in result for m in result[d]))

    def test_incomplete_scientific_gates_rejected(self):
        p=self.p
        for flag in ('population_reference_approved','aggregate_precision_passed','independent_frequency_replay_passed'):
            with self.subTest(flag=flag):
                s=copy.deepcopy(p['source_summary']);s[flag]=False
                with self.assertRaises(ValueError):a.check_release(s,p['source_plan'],p['source_selection'])

    def test_missing_repair_cannot_publish(self):
        p=self.p;s=copy.deepcopy(p['source_selection']);s['selected_repair_ids'].pop()
        with self.assertRaises(ValueError):a.check_release(p['source_summary'],p['source_plan'],s)

    def test_changed_cohort_rejected(self):
        self.arrays['a1_L2_stationary_loss']=self.arrays['a1_L2_stationary_loss'][:255]
        with self.assertRaises(ValueError):a.replay(self.p['source_summary'],self.arrays)

    def test_numerical_allowance_cannot_be_relaxed(self):
        s=copy.deepcopy(self.p['source_summary']);s['groups'][0]['cumulative']['numerical_budget_bits'][-1]=.004
        with self.assertRaises(ValueError):a.replay(s,self.arrays)

    def test_changed_mean_cannot_publish(self):
        self.arrays['a1_L3_bayes_loss_a32'][:,15]+=.001
        with self.assertRaises(ValueError):a.replay(self.p['source_summary'],self.arrays)

    def test_compact_array_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'provenance.json.gz').write_bytes((SOURCE/'provenance.json.gz').read_bytes())
            (out/'paired_scores.npz').write_bytes((SOURCE/'paired_scores.npz').read_bytes()+b'changed')
            with self.assertRaises(ValueError):a.load(out)

if __name__=='__main__':unittest.main()

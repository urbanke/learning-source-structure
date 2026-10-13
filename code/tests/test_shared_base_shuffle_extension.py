import importlib.util
from pathlib import Path
import numpy as np
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/extend_shared_base_shuffle.py'
spec = importlib.util.spec_from_file_location('shuffle_extension', path)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class ExtensionTests(unittest.TestCase):
    def test_retained_source_identity_rejects_reordering(self):
        a = dict(tokens=np.arange(12).reshape(4, 3), base=np.arange(16).reshape(4, 4),
                 transition=np.arange(64).reshape(4, 4, 4))
        parent = dict(arrays_sha256=extension.old.checksums({k:v[:2] for k,v in a.items()}))
        self.assertEqual(extension.check_retained(a, parent, 2), extension.old.checksums(a))
        altered = {k:v.copy() for k,v in a.items()}
        altered['tokens'][[0,1]] = altered['tokens'][[1,0]]
        with self.assertRaisesRegex(ValueError, 'source arrays'):
            extension.check_retained(altered, parent, 2)


    def test_previous_scores_are_required_equal_for_all_metrics(self):
        old = {k:np.arange(8).reshape(2,4).astype(float) for k in extension.METRICS}
        new = {k:np.concatenate([v,np.zeros_like(v)]) for k,v in old.items()}
        self.assertEqual(max(extension.check_scores(new,old,2,5e-10).values()), 0)
        new['expected_loss_bits'][1,3] += 1e-8
        with self.assertRaisesRegex(ValueError, 'expected_loss_bits'):
            extension.check_scores(new,old,2,5e-10)


    def test_frozen_plan_dependencies(self):
        plan = extension.validate_plan(extension.PLAN)
        self.assertEqual(plan['n_sequences']-plan['retained_sequences'], 768)
        self.assertIs(plan['scheduler']['automatic_retries'], False)


if __name__ == '__main__':
    unittest.main()

"""Preservation, exact stream partitioning and bounded execution invariants."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('large_shuffle',ROOT/'code/scripts/extend_shared_base_shuffle_large.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class LargerCopying(unittest.TestCase):
    def test_only_new_indices_are_assigned_exactly_once(self):
        p=m.validate_plan(m.PLAN);tasks=m.tasks(p)
        self.assertEqual(len(tasks),720)
        for d in (0,2,3):
            indices=[i for depth,left,right in tasks if depth==d for i in range(left,right)]
            self.assertEqual(indices,list(range(1024,16384)))
        self.assertEqual(m.tasks(p,True),[(d,1024,1088) for d in (0,2,3)])

    def test_retained1024_scores_are_exact_not_only_first256(self):
        old={k:np.arange(1024*1024,dtype=float).reshape(1024,1024) for k in m.METRICS}
        new={k:np.full((2,1024),-3.) for k in m.METRICS}
        result=m.score_pair(old,new,1024)
        for k in m.METRICS:
            np.testing.assert_array_equal(result[k][:1024],old[k])
            np.testing.assert_array_equal(result[k][1024:],new[k])
        with self.assertRaisesRegex(ValueError,'Incomplete retained'):
            m.score_pair({k:v[:256] for k,v in old.items()},new,1024)

    def test_source_stream_does_not_depend_on_chunk_boundaries(self):
        for depth in (0,2,3):
            allrows=m.old.source_arrays(depth,range(1024,1028),20260930,d=4,n_tokens=17)
            first=m.old.source_arrays(depth,range(1024,1026),20260930,d=4,n_tokens=17)
            last=m.old.source_arrays(depth,range(1026,1028),20260930,d=4,n_tokens=17)
            for k in allrows:np.testing.assert_array_equal(allrows[k],np.concatenate((first[k],last[k])))

    def test_measured_projection_stops_over_budget(self):
        p=m.validate_plan(m.PLAN)
        small=m.projection(p,[{'seconds':1.}]*3,[{'seconds':1.}]*6)
        self.assertTrue(small['passed']);self.assertEqual(small['maximum_cost_chf'],2.07)
        self.assertAlmostEqual(small['projected_total_seconds_with_margin'],2940.)
        slow=m.projection(p,[{'seconds':1.}]*3,[{'seconds':60.}]*6)
        self.assertFalse(slow['passed'])

    def test_checked_chunk_rejects_unbound_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'chunk.npz';p.write_bytes(b'retained chunk')
            identity=dict(plan_sha256='fixed',task=[2,1024,1088])
            m.old.write_json(p.with_suffix('.json'),dict(identity=identity,sha256=m.old.sha(p)))
            self.assertEqual(m.check_chunk(p,identity)['identity'],identity)
            with self.assertRaises(ValueError):m.check_chunk(p,dict(identity,task=[2,1088,1152]))
            p.write_bytes(b'changed chunk')
            with self.assertRaises(ValueError):m.check_chunk(p,identity)


if __name__=='__main__':unittest.main()

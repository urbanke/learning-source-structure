import sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
import copying_bayes_dense16_final as f

class Overlay(unittest.TestCase):
    def test_only_declared_passed_recovery_layer_is_retained(self):
        spec={'task':{'id':'recovered'},'parent_files':{},'target_draws':65536}
        prior=f.r.parent_record;observed=[]
        def merge(out):
            observed.append(f.r.parent_record('base',{},dict(id='recovered')))
            observed.append(f.r.parent_record('base',{},dict(id='original')))
            return 3
        with patch.object(f,'load',return_value=({},Path('/first'),{'tasks':[spec]})),patch.object(f,'check_first',return_value=({'audit_passed':True},'moments',{})),patch.object(f,'ORIGINAL_READER',return_value='original'),patch.object(f.r,'merge',side_effect=merge):
            self.assertEqual(f.merge(Path('/out')),3)
        self.assertEqual(observed,[({'audit_passed':True},'moments',{}),'original'])
        self.assertIs(f.r.parent_record,prior)

    def test_unpassed_recovery_is_rejected_and_resolver_restored(self):
        spec={'task':{'id':'failed'},'parent_files':{},'target_draws':65536};prior=f.r.parent_record
        with patch.object(f,'load',return_value=({},Path('/first'),{'tasks':[spec]})),patch.object(f,'check_first',return_value=({'audit_passed':False},'moments',{})),patch.object(f.r,'merge',side_effect=lambda out:f.r.parent_record('base',{},dict(id='failed'))):
            with self.assertRaisesRegex(ValueError,'Unpassed'):f.merge(Path('/out'))
        self.assertIs(f.r.parent_record,prior)
if __name__=='__main__':unittest.main()

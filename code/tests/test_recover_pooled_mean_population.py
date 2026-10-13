import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[1]/'scripts/recover_pooled_mean_population.py'
spec=importlib.util.spec_from_file_location('recovery',p); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

class RecoveryTests(unittest.TestCase):
    def test_preserve_all_existing_repetitions_and_complete_suffix(self):
        old=dict(records=[{'seed':1,'x':[2]}, {'seed':2,'x':[3]}],error='timeout',status='incomplete')
        result=r.complete_report(old,{2:{'seed':3}},3)
        assert result['records']==[{'seed':1,'x':[2]},{'seed':2,'x':[3]},{'seed':3}]
        assert old['records']==[{'seed':1,'x':[2]},{'seed':2,'x':[3]}]
        assert result['complete'] and 'error' not in result
        assert not result['scientific_reference_validated']

    def test_no_partial_average_or_duplicate_prefix(self):
        old={'records':[{'seed':1}]}
        for additions in ({},{0:{'seed':1},1:{'seed':2}},{2:{'seed':3}}):
            with self.assertRaises((ValueError,RuntimeError)): r.complete_report(old,additions,2)

    def test_missing_manifest_keeps_task_identity_and_repetition_indices(self):
        specs=[dict(task_index=i,arm=a,tokens=[5,6]) for i,a in enumerate(('A','D'))]
        reports={0:dict(status='complete',records=[0,1,2]),1:dict(status='incomplete',records=[0])}
        assert r.missing_repetitions(specs,reports,3)==[
            dict(index=0,task=specs[1],repetition=1),dict(index=1,task=specs[1],repetition=2)]
        reports[1]['status']='running'
        with self.assertRaises((ValueError,RuntimeError)): r.missing_repetitions(specs,reports,3)

if __name__=="__main__": unittest.main()

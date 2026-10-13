import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
import copying_bayes_dense16_recovery as r
from shared_base_chain_runner import _hash_array

class Recovery(unittest.TestCase):
    def test_explicit_draw_budget_preserves_other_moment_checks(self):
        p=np.full(64,1/64);zero=np.zeros((64,64));m=dict(mean=p,chain_means=np.tile(p,(4,1)),chains=np.array(4),
            draws=np.array(65536),cov_batch=zero,cov_chain=zero,mcse_probs=np.zeros(64))
        est=dict(predictive_probs=p,diagnostics=dict(draws=65536))
        r.moments_checked(m,est,3,65536)
        with self.assertRaises(ValueError):r.moments_checked(m,est,3,16384)
        m['cov_chain']=zero.copy();m['cov_chain'][5,5]=-1e-4
        with self.assertRaises(ValueError):r.moments_checked(m,est,3,65536)

    def test_pair_discrepancy_not_hidden_by_recovery_selection(self):
        a=np.full(64,1/64);b=a.copy();b[0]+=3e-7;b[1]-=3e-7
        refs={16:{'x':dict(mean=a,mcse_probs=np.zeros(64))},32:{'x':dict(mean=b,mcse_probs=np.zeros(64))}}
        self.assertEqual(r.paired_failures(refs),['x'])

    def test_continuation_preserves_every_retained_chain_prefix(self):
        def checkpoint(path,n,changed=False):
            arrays={'burn_done':np.ones(4,dtype=bool)}
            for chain in range(4):
                for kind in ('predictions','depths','scalars','alpha_indices'):
                    a=np.arange(n,dtype=np.int64)+(10*chain)
                    if changed and chain==2 and kind=='predictions':a[1]+=1
                    arrays[f'{kind}_{chain}']=a
            metadata=dict(format='fixture',fingerprint={'prior':'fixed'},config={'seed':7},initializations=['fixed'],proposal_descriptor={'same':True},
                          array_hashes={k:_hash_array(v) for k,v in arrays.items()})
            text=json.dumps(metadata);np.savez(path,**arrays,metadata=text,metadata_sha256=hashlib.sha256(text.encode()).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'parent.npz',Path(folder)/'child.npz'
            checkpoint(a,2);checkpoint(b,4);r.preserve(a,b,4)
            checkpoint(b,4,True)
            with self.assertRaisesRegex(ValueError,'Retained samples changed'):r.preserve(a,b,4)
if __name__=='__main__':unittest.main()

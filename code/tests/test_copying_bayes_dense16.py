"""Independent invariants for the portable, finite copying extension."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
import copying_bayes_dense16 as d

class Dense16(unittest.TestCase):
    def test_unique_shard_ownership(self):
        tasks=list(range(10172));parts=[d.assigned(tasks,i,256) for i in range(256)]
        self.assertEqual(sorted(x for p in parts for x in p),tasks)
        self.assertLessEqual(max(map(len,parts)),40)
        with self.assertRaises(ValueError):d.assigned(tasks,256,256)

    def test_shared_bank_cumulative_covariance(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(256,16));zero=np.zeros(16)
        perturb=np.linspace(-.01,.01,16)[:,None,None]*np.ones((1,256,16))
        loo=x[None]+perturb
        inst=d.allowance(x,x,loo,loo,x,x,zero,zero,zero,zero)
        cum=d.allowance(x,x,loo,loo,x,x,zero,zero,zero,zero,True)
        self.assertAlmostEqual(cum['numerical_allowance_bits'][-1],16*inst['numerical_allowance_bits'][0],places=11)
        self.assertFalse(cum['passed'])

    def test_independent_time_mcmc(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(256,16));zero=np.zeros(16);m=np.ones(16)*.001
        loo=np.repeat(x[None],16,axis=0)
        r=d.allowance(x,x,loo,loo,x,x,m,zero,zero,zero,True)
        self.assertAlmostEqual(r['numerical_allowance_bits'][-1],5*.001*4,places=12)

    def test_replays_approved_gate(self):
        bundle=ROOT/'output/copying_bayes_dense16_bundle_2026-09-22/input.npz'
        if not bundle.exists():self.skipTest('Local approved fixture unavailable')
        data=d.archive(bundle)
        report=d.read(ROOT/'output/copying_payoff_alpha_refined_assessment_2026-09-21/summary.json')
        for group in report['groups']:
            g=group['condition'];a=lambda name:data[g+'_'+name]
            for scope,cum in (('instantaneous',False),('cumulative',True)):
                r=d.allowance(a('stationary_loss'),a('stationary_coarse_loss'),a('stationary_leave_one_bank_out_loss'),
                    a('stationary_coarse_leave_one_bank_out_loss'),a('bayes_loss'),a('bayes_loss_a16'),
                    a('old_32_numerical_mcse_bits'),a('old_16_numerical_mcse_bits'),
                    a('old_32_log_bias_proxy_bits'),a('old_16_log_bias_proxy_bits'),cum)
                np.testing.assert_allclose(r['numerical_allowance_bits'],[x['total_numerical_allowance_bits'] for x in group[scope]],rtol=1e-10,atol=2e-14)
                self.assertTrue(r['passed'])
if __name__=='__main__':unittest.main()

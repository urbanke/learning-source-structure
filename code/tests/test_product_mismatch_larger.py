"""Frozen cohort, checkpoint identity, and paired scoring checks without Torch."""
import copy
from pathlib import Path
import sys

import numpy as np
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import product_mismatch_larger as experiment


def make_data():
    n,t = 4,len(experiment.TIMES)
    rng = np.random.default_rng(723)
    p = rng.dirichlet(np.ones(64),size=(n,t))
    b = rng.dirichlet(np.ones(64),size=(n,t))
    s = rng.dirichlet(np.ones(64),size=(n,t))
    return dict(tokens=rng.integers(0,64,size=(n,1025)),true_probs=p,bayes_probs=b,
                small_probs=s,times=np.array(experiment.TIMES),depths=np.array([1,1,4,4]),
                indices=np.array([0,1,0,1]))


def config(seed=0):
    return dict(family="product:8x8",d=64,n_ctx=1024,ls=[1,8],alpha=[.5,1.5],
                no_permute=False,steps=100000,batch=64,seed=seed,n_layer=4,n_head=8,d_model=256,
                model=dict(vocab=64,n_ctx=1024,n_layer=4,n_head=8,d_model=256,mlp=True,dropout=0.))


class TestFrozenMismatch(unittest.TestCase):
    def setUp(self):
        cohort_size = patch.object(experiment, "N_PER_DEPTH", 2)
        cohort_size.start()
        self.addCleanup(cohort_size.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_complete_original_row_order_is_required(self):
        data = make_data()
        experiment.check_pairing(data)
        for name,replacement in (("indices",[0,0,0,1]),("indices",[1,0,0,1]),
                                 ("depths",[1,1,1,4]),("times",list(reversed(experiment.TIMES)))):
            changed = dict(data,**{name:np.array(replacement)})
            with self.assertRaises(ValueError):
                experiment.check_pairing(changed)
        with self.assertRaises(ValueError):
            experiment.check_pairing({k:v[:-1] if k != "times" else v for k,v in data.items()})


    def test_paired_differences_are_scored_before_standard_error(self):
        data = make_data()
        # A common Bayes predictor removes any ambiguity about the sign of the gap.
        logp = np.log(data["bayes_probs"])
        values = experiment.quantities(data,logp)
        expected = np.sum(data["true_probs"]*np.log(data["small_probs"]/data["bayes_probs"]),axis=-1)/np.log(2)
        np.testing.assert_allclose(values["gap_difference_larger_minus_small_bits"],expected,atol=2e-15,rtol=0)
        np.testing.assert_allclose(values["larger_minus_bayes_bits"],0,atol=0,rtol=0)
        np.testing.assert_allclose(values["kl_bayes_to_larger_bits"],0,atol=0,rtol=0)
        np.testing.assert_allclose(values["agreement_difference_larger_minus_small_bits"],
                                   -values["kl_bayes_to_small_bits"],atol=2e-15,rtol=0)
        groups = experiment.aggregate(data,values)
        for index,group in enumerate(groups):
            selected = expected[2*index:2*index+2]
            reported = group["metrics"]["gap_difference_larger_minus_small_bits"]
            np.testing.assert_allclose(reported["mean"],selected.mean(0),atol=2e-15,rtol=0)
            np.testing.assert_allclose(reported["se"],selected.std(0,ddof=1)/np.sqrt(2),atol=2e-15,rtol=0)


    def test_no_renormalization_or_probability_clipping(self):
        data = make_data()
        p = data["true_probs"].copy()
        p[:,:,0] = 0
        p /= p.sum(-1,keepdims=True)
        data["true_probs"] = p
        experiment.check_pairing(data)
        result = experiment.quantities(data,np.log(data["small_probs"]))
        np.testing.assert_array_equal(result["gap_difference_larger_minus_small_bits"],np.zeros_like(result["small_regret_bits"]))
        broken = copy.deepcopy(data)
        broken["small_probs"] *= .99
        with self.assertRaisesRegex(ValueError,"Unnormalized"):
            experiment.check_pairing(broken)
        with self.assertRaisesRegex(ValueError,"Unnormalized"):
            experiment.quantities(data,np.log(data["small_probs"]*.99))


    def test_checkpoint_requires_existing_family_seed_step_and_hash(self):
        tmp_path = self.tmp_path
        path = tmp_path/"model_100000.pt"
        path.write_bytes(b"stand-in checkpoint bytes")
        saved = dict(step=100000,cfg=config())
        binding = dict(config=config(),checkpoint_sha256=experiment.sha(path))
        experiment.check_checkpoint(saved,path,binding,0)
        for key,value in (("seed",1),("family","hidden_product:8x8"),("ls",[1,4]),("n_layer",2)):
            modified = copy.deepcopy(saved)
            modified["cfg"][key] = value
            with self.assertRaises(ValueError):
                experiment.check_checkpoint(modified,path,binding,0)
        with self.assertRaisesRegex(ValueError,"step/configuration"):
            experiment.check_checkpoint(dict(saved,step=95000),path,binding,0)
        path.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError,"checksum"):
            experiment.check_checkpoint(saved,path,binding,0)


    def test_manifest_and_cohort_payload_are_hash_bound(self):
        tmp_path, data = self.tmp_path, make_data()
        experiment.save(tmp_path/"cohort.npz",**data)
        manifest = dict(schema=experiment.SCHEMA,source_plan_id=experiment.SOURCE_PLAN,
            adapter_sha256=experiment.sha(experiment.__file__),model_code_sha256=experiment.MODEL_SHA,
            cohort_sha256=experiment.sha(tmp_path/"cohort.npz"))
        manifest["manifest_id"] = experiment.fingerprint(manifest)
        experiment.write(tmp_path/"manifest.json",manifest)
        experiment.load_cohort(tmp_path)
        altered = dict(manifest,condition="stationary")
        experiment.write(tmp_path/"manifest.json",altered)
        with self.assertRaisesRegex(ValueError,"Manifest identity"):
            experiment.load_cohort(tmp_path)
        experiment.write(tmp_path/"manifest.json",manifest)
        (tmp_path/"cohort.npz").write_bytes(b"changed payload")
        with self.assertRaisesRegex(ValueError,"cohort checksum"):
            experiment.load_cohort(tmp_path)


    def test_prepare_refuses_to_overwrite_existing_output(self):
        tmp_path = self.tmp_path
        sentinel = tmp_path/"summary.json"
        sentinel.write_text("preserve existing scientific result")
        with self.assertRaisesRegex(ValueError,"fresh output"):
            experiment.prepare(tmp_path/"absent-source",tmp_path/"absent-receipts",tmp_path)
        assert sentinel.read_text() == "preserve existing scientific result"


if __name__ == "__main__":
    unittest.main()

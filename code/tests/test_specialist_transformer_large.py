"""Paid jobs require complete timing gates and finite duplicate-free allocations."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("specialist",ROOT/"code/scripts/specialist_transformer_large.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
spec_release=importlib.util.spec_from_file_location("release",ROOT/"code/scripts/specialist_transformer_release.py")
release=importlib.util.module_from_spec(spec_release);spec_release.loader.exec_module(release)


class SubmissionTests(unittest.TestCase):
    def run_case(self,projection=2,missing=False,repeat=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();out=root/"output/fresh";plan=root/"plan.json"
            plan.write_text("{}")
            for name in ("code/scripts/specialist_transformer_large.py","code/transformer/kuma/specialist_transformer_large.sbatch"):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("frozen")
            (out/"cohorts").mkdir(parents=True);(out/"cohorts/manifest.json").write_text("{}")
            for family in driver.FAMILIES:
                if missing and family=="uniform_base":continue
                driver.write(out/f"benchmark/{family}/benchmark.json",dict(passed=True,plan_sha256=driver.sha(plan),projected_training_hours=projection))
            calls=[]
            def fake(cmd,**kw):
                calls.append(cmd);return subprocess.CompletedProcess(cmd,0,"12345" if cmd[0]=="sbatch" else "","")
            a=SimpleNamespace(out=out,plan=plan,stage="train",hours=4,submit=True)
            with patch.object(driver,"ROOT",root),patch.object(driver.subprocess,"run",fake),contextlib.redirect_stdout(io.StringIO()):
                if missing or projection>=3.5:
                    with self.assertRaises((ValueError,FileNotFoundError)):driver.submit(a)
                    self.assertFalse(calls);return
                driver.submit(a)
                if repeat:
                    with self.assertRaises(FileExistsError):driver.submit(a)
            jobs=[c for c in calls if c[0]=="sbatch"]
            self.assertEqual(len(jobs),1)
            self.assertIn("--array=0-3%4",jobs[0]);self.assertIn("--time=04:00:00",jobs[0])

    def test_both_family_benchmarks_required(self):self.run_case(missing=True)
    def test_evaluation_time_reserved(self):self.run_case(projection=3.6)
    def test_exact_four_bounded_jobs_and_duplicate_refused(self):self.run_case(repeat=True)


class ReleaseTests(unittest.TestCase):
    def check(self,mutation=None):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/"output/fresh";plan=root/"plan.json"
            for name in release.NAMES+("numerical.py",):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("frozen")
            driver.write(plan,dict(implementation_sha256={"numerical.py":driver.sha(root/"numerical.py")}))
            driver.write(out/"cohorts/manifest.json",{})
            binding=dict(plan_sha256=driver.sha(plan),cohort_manifest_sha256=driver.sha(out/"cohorts/manifest.json"))
            benchmark=dict(binding,job_id="12345",status="submitted",implementation_sha256={k:driver.sha(root/k) for k in release.NAMES[:2]})
            training=dict(binding,parent_benchmark_job="12345",implementation_sha256={k:driver.sha(root/k) for k in release.NAMES})
            driver.write(out/"benchmark_submission.json",benchmark);driver.write(out/"train_submission.json",training)
            for family in driver.FAMILIES:
                driver.write(out/f"benchmark/{family}/benchmark.json",dict(passed=True,plan_sha256=driver.sha(plan),family=family,seed=0,completed_steps=1000,launch_binding=benchmark,projected_training_hours=2))
            with patch.object(release,"ROOT",root):
                if mutation:
                    mutation(root,out)
                    with self.assertRaises((ValueError,FileNotFoundError)):release.gate(out,plan)
                else:self.assertTrue(release.gate(out,plan)["passed"])

    def test_both_matching_finite_measurements_release(self):self.check()
    def test_missing_family_blocks(self):
        self.check(lambda root,out:(out/"benchmark/uniform_base/benchmark.json").unlink())
    def test_changed_numerical_code_blocks(self):
        self.check(lambda root,out:(root/"numerical.py").write_text("changed"))
    def test_overlong_measurement_blocks(self):
        def change(root,out):
            p=out/"benchmark/product/benchmark.json";d=driver.read(p);d["projected_training_hours"]=3.6;driver.write(p,d)
        self.check(change)
    def test_foreign_benchmark_receipt_blocks(self):
        def change(root,out):
            p=out/"benchmark/product/benchmark.json";d=driver.read(p);d["launch_binding"]["job_id"]="99999";driver.write(p,d)
        self.check(change)


if __name__=="__main__":unittest.main()

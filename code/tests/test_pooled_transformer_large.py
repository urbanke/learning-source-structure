"""Check campaign isolation, preserved training data and bounded submission."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code/scripts" / (name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CampaignTests(unittest.TestCase):
    def test_only_architecture_and_experiment_change(self):
        worker = module("pooled_transformer_large")
        plan = worker.read(ROOT/"code/transformer/configs/pooled_transformer_large_plan_2026-09-22.json")
        for seed in (0, 1):
            old = worker.read(ROOT/f"code/transformer/configs/multifamily_seed{seed}.json")
            new = worker.configuration(plan, seed)
            self.assertEqual({k for k in old if old[k] != new[k]},
                             {"n_layer", "n_head", "d_model", "experiment"})
            self.assertEqual(new["steps"]*new["batch"]//3, 6400000)
        broken = copy.deepcopy(plan)
        broken["implementation_sha256"]["mixture_source.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            worker.configuration(broken, 0)

    def test_training_refuses_insufficient_wall_budget(self):
        self.launch_case(projection=12, wall=12, expect_failure=True)

    def test_training_has_two_finite_allocations_no_automatic_retries(self):
        command = self.launch_case(projection=6, wall=12)
        self.assertIn("--array=0-1%2", command)
        self.assertIn("--time=12:00:00", command)

    def test_second_submission_cannot_duplicate_first(self):
        self.launch_case(projection=6, wall=12, repeat=True)

    def launch_case(self, projection, wall, expect_failure=False, repeat=False):
        launcher = module("submit_pooled_transformer_large")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = root/"plan.json"
            plan.write_text("{}")
            (root/"original_cohort").mkdir()
            (root/"original_cohort/manifest.json").write_text("{}")
            for relative in ("code/scripts/pooled_transformer_large.py", "code/scripts/submit_pooled_transformer_large.py",
                             "code/transformer/kuma/pooled_transformer_large.sbatch"):
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("frozen code")
            out = root/"output/new_run"
            (out/"benchmark").mkdir(parents=True)
            (out/"benchmark/benchmark.json").write_text(json.dumps(dict(passed=True,
                plan_sha256=launcher.hashlib.sha256(plan.read_bytes()).hexdigest(),
                projected_training_hours_including_monitoring=projection)))
            calls = []
            def fake(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "12345\n" if command[0]=="sbatch" else "", "")
            arguments = ["submit", "train", "--plan", str(plan), "--out", str(out),
                         "--sources", str(root/"original_cohort"), "--hours-per-seed", str(wall), "--submit"]
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", fake), \
                 patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                if expect_failure:
                    with self.assertRaises(SystemExit):
                        launcher.main()
                    self.assertEqual(calls, [])
                    return
                launcher.main()
                if repeat:
                    with self.assertRaises(FileExistsError):
                        launcher.main()
            submitted = [c for c in calls if c[0]=="sbatch"]
            self.assertEqual(len(submitted), 1)
            self.assertEqual(json.loads((out/"train_submission.json").read_text())["job_id"], "12345")
            return submitted[0]


if __name__ == "__main__":
    unittest.main()

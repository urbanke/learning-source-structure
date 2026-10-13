"""Reject partial, mispaired or numerically unapproved copying figure inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import copying_dense_figure_input as adapter


class DenseFigureInput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.old, self.neural = (root / name for name in ("source", "old", "neural"))
        self.source.mkdir(); self.old.mkdir()
        self.arrays, old_arrays, groups = {}, {}, []
        rng = np.random.default_rng(823)
        for condition in adapter.CONDITIONS:
            stationary = rng.normal(3, .1, (256, 16))
            bayes = stationary - rng.normal(.02, .01, (256, 16))
            ordered = rng.normal(3, .1, (256, 16))
            shuffled = ordered + rng.normal(.02, .01, (256, 16))
            for name, values in (("stationary_loss", stationary), ("bayes_loss_a32", bayes),
                                 ("ordered_seed0", ordered), ("shuffled_seed0", shuffled)):
                self.arrays[condition + "_" + name] = values
                old_name = "bayes_loss" if name == "bayes_loss_a32" else name
                old_arrays[condition + "_" + old_name] = values[:, :8]
            for role, values in (("ordered", ordered), ("shuffled", shuffled)):
                folder = self.neural / (role + "_seed0"); folder.mkdir(exist_ok=True, parents=True)
                full = np.zeros((256, 1024)); full[:, :16] = values
                np.savez_compressed(folder / (condition + ".npz"), expected_loss_bits=full,
                                    times=np.arange(1, 1025))
            group = {"condition": condition}
            for metric in ("instantaneous", "cumulative"):
                gain = stationary - bayes
                if metric == "cumulative": gain = gain.cumsum(1)
                se = gain.std(0, ddof=1) / 16
                group[metric] = dict(passed=True, mean_bits=gain.mean(0).tolist(), source_se_bits=se.tolist(),
                    numerical_allowance_bits=[0.0] * 16, numerical_budget_bits=np.minimum(.01 / 3, .25 * se).tolist())
            groups.append(group)
        np.savez_compressed(self.old / "arrays.npz", **old_arrays)
        self.summary = dict(schema="copying-dense16-recovery-v1", complete=True,
            aggregate_precision_passed=True, population_reference_approved=True,
            publication_status="approved", missing=[], failed_audits=[], failed_paired_orders=[],
            original_histories=768, times=list(range(1, 17)), groups=groups,
            payoff_forecast_approved=False,
            comparator="Source-induced stationary-frequency surrogate; not exact finite-shuffle Bayes.")
        (self.source / "input_bindings.json").write_text("{}")
        self.write()

    def write(self):
        np.savez_compressed(self.source / "arrays.npz", **self.arrays)
        self.summary.update(arrays_sha256=adapter.sha(self.source / "arrays.npz"),
                            input_bindings_sha256=adapter.sha(self.source / "input_bindings.json"))
        (self.source / "summary.json").write_text(json.dumps(self.summary))
        validation = dict(schema="copying-dense16-release-validation-v1", passed=True,
            complete_original_cohorts=True, retained_early_scores_exact=True,
            old_arrays_sha256=adapter.sha(self.old / "arrays.npz"))
        for name in ("summary", "arrays", "input_bindings"):
            validation[name + "_sha256"] = adapter.sha(self.source / (name + (".npz" if name == "arrays" else ".json")))
        (self.source / "validation.json").write_text(json.dumps(validation))

    def load(self):
        return adapter.load_dense_copying(self.source, self.old, self.neural)

    def test_complete_paired_curves_and_cumulative_covariance(self):
        curves, checks, paths = self.load()
        self.assertEqual(set(curves), {0, 2, 3})
        self.assertEqual(len(checks), 6)
        gain = self.arrays["a1_L2_shuffled_seed0"] - self.arrays["a1_L2_ordered_seed0"]
        np.testing.assert_array_equal(curves[2]["cumulative"]["neural_seed0_source_se"],
                                      gain.cumsum(1).std(0, ddof=1) / 16)
        self.assertGreater(len(paths), 4)

    def test_missing_reference_cannot_be_plotted(self):
        self.summary["missing"] = ["n16_0123_a32"]; self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete"): self.load()

    def test_failed_control_condition_blocks_all(self):
        self.summary["groups"][0]["cumulative"]["passed"] = False; self.write()
        with self.assertRaisesRegex(ValueError, "gate failed"): self.load()

    def test_1024_histories_cannot_replace_the_paired_256(self):
        self.arrays["a1_L2_ordered_seed0"] = np.tile(self.arrays["a1_L2_ordered_seed0"], (4, 1)); self.write()
        with self.assertRaisesRegex(ValueError, "exactly 256"): self.load()

    def test_changed_early_scores_rejected_even_with_updated_receipt(self):
        self.arrays["a1_L2_bayes_loss_a32"][0, 0] += .01; self.write()
        with self.assertRaisesRegex(ValueError, "retained early"): self.load()

    def test_neural_pairing_checked_beyond_first_eight(self):
        self.arrays["a1_L2_ordered_seed0"][0, 15] += .01; self.write()
        with self.assertRaisesRegex(ValueError, "original paired"): self.load()

    def test_relaxed_numerical_budget_rejected(self):
        self.summary["groups"][1]["instantaneous"]["numerical_budget_bits"] = [.01] * 16; self.write()
        with self.assertRaisesRegex(ValueError, "numerical gate differs"): self.load()

    def test_modified_artifact_without_receipt_rejected(self):
        (self.source / "input_bindings.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "release changed"): self.load()


if __name__ == "__main__": unittest.main()

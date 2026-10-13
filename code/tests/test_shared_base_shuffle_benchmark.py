"""Reject incomplete or mismatched throughput measurements before submission."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('sb_benchmark', ROOT / 'code/scripts/shared_base_shuffle_benchmark.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.plan_path = self.run / 'plan.json'
        self.plan = json.loads((ROOT / 'code/transformer/configs/shared_base_shuffle_plan_2026-09-17.json').read_text())
        self.write('plan.json', self.plan)
        cfg = dict(schema='shared-base-shuffle-training-v1', **self.plan['config'],
                   **self.plan['source'], seed=0)
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        receipt = dict(config=cfg, device='cuda', plan_sha256=BENCH.sha(self.plan_path),
                       precision='float32', runtime=dict(float32_matmul_precision='highest',
                       cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False),
                       n_params=self.plan['n_params'], implementation_hashes=self.plan['implementation_sha256'],
                       config_hash=cfg_hash, model=dict(vocab=64, n_ctx=1024, n_layer=4,
                                                      n_head=8, d_model=256, mlp=True, dropout=0.0))
        receipt['control']='independent-tail-position-shuffle'
        receipt['ordered_training']=dict(
            original_plan_sha256=self.plan['ordered_plan']['sha256'],
            probes=self.plan['pairing_probes']['0'],
            original_files_sha256=self.plan['ordered_runs']['0']['files_sha256'])
        state = dict(step=1000, production_steps=100000, completed_sequences=64000,
                     complete=False, reason='max_steps', config_hash=cfg_hash)
        runtime = dict(plan_sha256=BENCH.sha(self.plan_path), seed=0, step=1000,
                       device='cuda', return_code=75, cuda_device='test GPU',
                       max_memory_allocated_bytes=2**30, max_memory_reserved_bytes=2**31,
                       total_memory_bytes=2**36)
        self.write('config.json', receipt)
        self.write('status.json', state)
        self.write('shuffle_runtime.json', runtime)
        self.events = [dict(event='train', step=100, run_elapsed_seconds=10, loss_bits=5),
                       dict(event='train', step=1000, run_elapsed_seconds=100, loss_bits=4)]
        self.write_events()

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value) + '\n')

    def write_events(self):
        (self.run / 'log.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.events))

    def collect(self):
        return BENCH.collect(self.plan_path, self.run)

    def test_valid_measurement(self):
        result = self.collect()
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['seconds_per_step'], .1)
        self.assertAlmostEqual(result['projected_training_hours'], 10000 / 3600)

    def test_reject_corrupt_receipts(self):
        variants = [('status.json', 'complete', True), ('status.json', 'completed_sequences', 63999),
                    ('config.json', 'device', 'cpu'), ('config.json', 'plan_sha256', 'wrong'),
                    ('config.json', 'n_params', 544256), ('config.json','control','wrong-control'), ('shuffle_runtime.json', 'return_code', 0),
                    ('shuffle_runtime.json', 'max_memory_allocated_bytes', 0)]
        for name, key, bad in variants:
            with self.subTest(name=name, key=key):
                original = json.loads((self.run / name).read_text())
                modified = copy.deepcopy(original)
                modified[key] = bad
                self.write(name, modified)
                with self.assertRaises(ValueError):
                    self.collect()
                self.write(name, original)

    def test_reject_restart_and_nonfinite_loss(self):
        for key, value in [('run_elapsed_seconds', 9), ('step', 999), ('loss_bits', float('nan'))]:
            with self.subTest(key=key):
                original = self.events[-1][key]
                self.events[-1][key] = value
                self.write_events()
                with self.assertRaises(ValueError):
                    self.collect()
                self.events[-1][key] = original

    def test_reject_changed_model_or_schedule(self):
        for key, value in [('n_layer', 2), ('steps', 1000), ('alpha', [1, 1])]:
            with self.subTest(key=key):
                original = json.loads((self.run / 'config.json').read_text())
                changed = copy.deepcopy(original)
                changed['config'][key] = value
                self.write('config.json', changed)
                with self.assertRaises(ValueError):
                    self.collect()
                self.write('config.json', original)


if __name__ == '__main__':
    unittest.main()

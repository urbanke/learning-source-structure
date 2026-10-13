"""Independent replay and resume checks for the shared-base cohort extension."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
import eval_shared_base_models as frozen
import extend_shared_base_models as extension
from source import Source


def test_extension_preserves_pilot_bytes_and_batch_generator_draws(tmp_path):
    pilot, out = tmp_path / "pilot", tmp_path / "extension"
    old = frozen.sources(pilot, 2, 20260922, create=True)
    before = [entry["path"].read_bytes() for entry in old]
    first = extension.prepare_cohort(out, 3, 20260922, pilot)
    second = extension.prepare_cohort(out, 4, 20260922, pilot)
    assert first["records"] == second["records"][:3]
    assert before == [entry["path"].read_bytes() for entry in old]
    assert before == [(out / "data" / f"seq_{i:06d}.json").read_bytes() for i in range(2)]
    for entry in second["records"]:
        source = extension.read_source(Path(entry["path"]), entry["index"], 20260922)["record"]
        seed = source["identity"]["sequence_seed"]
        actual, params = Source(64, 1025, alpha=(0.5, 1.5), ls=(0, 8), permute=True).batch(1, np.random.default_rng(seed))
        np.testing.assert_array_equal(actual[0], source["tokens"])
        observation = json.loads(Path(entry["observations_path"]).read_text())
        assert observation == {"tokens": source["tokens"]}
        assert source["generated_source"]["parameters"]["ls"] == params["ls"][0]
        assert source["generated_source"]["parameters"]["alpha"] == params["alpha"][0]
    with pytest.raises(ValueError, match="inputs changed"):
        extension.prepare_cohort(out, 4, 20260923, pilot)


def test_streamed_models_match_direct_prefix_predictions_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from model import GPT
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    run, out = tmp_path / "test_run", tmp_path / "extension"
    run.mkdir()
    cfg = dict(d=64, n_ctx=1024, alpha=[0.5, 1.5], ls=[0, 8], no_permute=False, seed=0,
               model=dict(vocab=64, n_ctx=1024, n_layer=1, n_head=2, d_model=16, mlp=True, dropout=0.0))
    model = GPT(**cfg["model"]).eval()
    (run / "config.json").write_text(json.dumps(cfg))
    torch.save(dict(step=100000, cfg=cfg, model=model.state_dict()), run / "model_100000.pt")
    extension.prepare_cohort(out, 3, 20260922, tmp_path / "absent_pilot")
    first = extension.evaluate_run(out, 2, 20260922, run, batch=2)
    paths = [out / "models" / "test_run" / f"seq_{i:06d}.npz" for i in range(3)]
    before = [path.read_bytes() for path in paths[:2]]
    second = extension.evaluate_run(out, 3, 20260922, run, batch=1)
    assert first["new_records"] == 2
    assert second["new_records"] == 1 and second["reused_records"] == 2
    assert before == [path.read_bytes() for path in paths[:2]]
    for i, path in enumerate(paths):
        source = json.loads((out / "data" / f"seq_{i:06d}.json").read_text())
        tokens = np.asarray(source["tokens"])
        matrix = np.asarray(source["oracle"]["transition_probs"])
        with np.load(path) as saved:
            for j, t in enumerate(extension.PREFIXES):
                with torch.no_grad():
                    direct = model(torch.tensor(tokens[None, :t]))[0, -1].numpy()
                np.testing.assert_allclose(saved["logits"][j], direct, atol=2e-6, rtol=0)
                q = matrix[tokens[t - 1]]
                p = saved["predictive_probs"][j]
                np.testing.assert_array_equal(saved["oracle_probs"][j], q)
                assert saved["targets"][j] == tokens[t]
                assert saved["expected_regret_bits"][j] == pytest.approx(sum(float(q[k]) * np.log2(float(q[k]) / float(p[k])) for k in range(64)), abs=1e-14)
                assert saved["selected_loss_bits"][j] == pytest.approx(-np.log2(p[tokens[t]]), abs=1e-14)
    paths[1].write_bytes(b"corrupted array")
    with pytest.raises(ValueError, match="array checksum failed"):
        extension.evaluate_run(out, 3, 20260922, run)

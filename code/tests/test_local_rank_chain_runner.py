"""Independent resumption and checkpoint-separation checks for the variant."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from eval_shared_base_local_rank import SEED_TAG, main, parser
from local_rank_chain_runner import LocalRankChainRun, proposal_descriptor
from ranked_lsa import RankedLSARowEvidence
from shared_base_chain_runner import ChainRun
from shared_base_local_rank import MOVE_NAMES, PROPOSAL_SPEC, SharedBaseLocalRankMCMC
from shared_base_mcmc import SharedBaseMCMC
from source import seed_of


TOKENS = np.array([0, 2, 0, 1, 0, 2, 0, 0, 2])


def sampler():
    return SharedBaseLocalRankMCMC(3, RankedLSARowEvidence(3, [0.5, 1.5], [0, 1]), [0.3, 0.7])


def quiet(_):
    pass


def same_reports(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if key == "elapsed_seconds":
            continue
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, expected[key])
        else:
            assert value == expected[key]


def test_chunked_saved_resumption_matches_variant_sample_bit_for_bit(tmp_path):
    config = dict(chains=4, burn=19, thin=3, seed=20260925)
    expected = sampler().sample(TOKENS, draws=48, retain_samples=True, **config)
    samples = expected.pop("samples")
    run = LocalRankChainRun(sampler(), TOKENS, **config)
    run.extend(16, block_draws=7, progress=quiet)
    path = tmp_path / "variant.npz"
    run.save(path)
    resumed = LocalRankChainRun.load(sampler(), TOKENS, path)
    actual = resumed.extend(48, block_draws=13, progress=quiet)
    same_reports(actual, expected)
    for name, core in (("predictive_probs", "predictive_probs"), ("posterior_depth", "posterior_depth"),
                       ("alpha_indices", "alpha_index"), ("alpha_and_loglik", "alpha_and_loglik")):
        np.testing.assert_array_equal(resumed.samples[name], samples[core])
    assert actual["diagnostics"]["move_names"] == MOVE_NAMES
    assert actual["diagnostics"]["proposal_spec"] == PROPOSAL_SPEC
    assert resumed.initializations == expected["diagnostics"]["initializations"]


def test_partial_checkpoint_resumes_unequal_chain_lengths_exactly(tmp_path):
    config = dict(chains=2, burn=7, thin=2, seed=11)
    reference = sampler().sample(TOKENS, draws=32, **config)
    run = LocalRankChainRun(sampler(), TOKENS, **config)
    path = tmp_path / "partial.npz"

    class SavedInterruption(Exception):
        pass

    def interrupt(event):
        if event["phase"] == "draws":
            run.save(path)
            raise SavedInterruption()

    with pytest.raises(SavedInterruption):
        run.extend(32, block_draws=5, progress=interrupt)
    np.testing.assert_array_equal(run.draw_counts, [5, 0])
    resumed = LocalRankChainRun.load(sampler(), TOKENS, path)
    same_reports(resumed.extend(32, block_draws=3, progress=quiet), reference)
    assert np.all(resumed._burn_done == 7)


def rewrite_metadata(path, change):
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    change(metadata)
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    arrays["metadata"] = np.asarray(serialized)
    arrays["metadata_sha256"] = np.asarray(hashlib.sha256(serialized.encode()).hexdigest())
    with path.open("wb") as target:
        np.savez(target, **arrays)


@pytest.mark.parametrize("change", ["missing", "spec", "sampler_hash", "wrapper_hash"])
def test_changed_or_missing_descriptor_is_rejected_even_with_valid_checksum(tmp_path, change):
    run = LocalRankChainRun(sampler(), TOKENS, chains=2, burn=0, thin=1, seed=4)
    path = tmp_path / "changed.npz"
    run.save(path)

    def alter(metadata):
        if change == "missing":
            del metadata["proposal_descriptor"]
        elif change == "spec":
            metadata["proposal_descriptor"]["proposal_spec"]["id"] = "different-proposal"
        else:
            filename = "shared_base_local_rank.py" if change == "sampler_hash" else "local_rank_chain_runner.py"
            metadata["proposal_descriptor"]["code_sha256"][filename] = "0" * 64

    rewrite_metadata(path, alter)
    with pytest.raises(ValueError, match="proposal descriptor"):
        LocalRankChainRun.load(sampler(), TOKENS, path)


def test_original_untagged_checkpoint_and_original_sampler_are_rejected(tmp_path):
    old = SharedBaseMCMC(3, RankedLSARowEvidence(3, [0.5, 1.5], [0, 1]), [0.3, 0.7])
    run = ChainRun(old, TOKENS, chains=2, burn=0, thin=1, seed=4)
    path = tmp_path / "original.npz"
    run.save(path)
    with pytest.raises(ValueError, match="proposal descriptor"):
        LocalRankChainRun.load(sampler(), TOKENS, path)
    with pytest.raises(TypeError, match="SharedBaseLocalRankMCMC"):
        LocalRankChainRun(old, TOKENS)


def test_wrapper_preserves_target_fingerprint_array_hashes_and_initializations(tmp_path):
    run = LocalRankChainRun(sampler(), TOKENS, chains=2, burn=3, thin=2, seed=19)
    run.extend(16, progress=quiet)
    plain, tagged = tmp_path / "plain.npz", tmp_path / "tagged.npz"
    ChainRun.save(run, plain)
    run.save(tagged)
    with np.load(plain, allow_pickle=False) as before, np.load(tagged, allow_pickle=False) as after:
        bmeta, ameta = (json.loads(str(item["metadata"].item())) for item in (before, after))
        assert ameta["proposal_descriptor"] == proposal_descriptor()
        assert bmeta["fingerprint"] == ameta["fingerprint"] == run.fingerprint
        assert bmeta["array_hashes"] == ameta["array_hashes"]
        assert bmeta["initializations"] == ameta["initializations"] == run.initializations
        for key in bmeta["array_hashes"]:
            np.testing.assert_array_equal(before[key], after[key])
    assert not list(tmp_path.glob(".local-rank-checkpoint-*"))


def test_variant_cli_has_separate_paths_seed_protocol_and_cache_default():
    args = parser().parse_args(["--index", "0"])
    assert args.input == Path("output/shared_base_full_context")
    assert args.out == Path("output/shared_base_local_rank_context")
    assert args.term_cache_size == 16384
    assert seed_of(SEED_TAG, 20260924, 0, 1024, 16) != seed_of(
        "shared-base-full-context-reference-v1", 20260924, 0, 1024, 16)


def test_variant_cli_rejects_original_input_output_alias_before_running(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["eval_shared_base_local_rank.py", "--index", "0",
                                    "--input", str(tmp_path), "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2

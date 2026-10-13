"""Independent semantic checks of the count-matched copying intervention."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from copier_probes import ARRAY_KEYS, DISCOVERY_SEED, HELDOUT_SEED, make_probes, validate_probes


def test_swaps_preserve_counts_and_isolate_one_repeated_query_context():
    out = make_probes(32, seed=19)
    assert set(out) == set(ARRAY_KEYS)
    for index in range(32):
        clean, cf, neutral, neutral_cf = (out[key][index] for key in ARRAY_KEYS[:4])
        a, d, b, c, e = (int(out[key][index]) for key in
                         ("context_a", "context_d", "target_b", "target_c", "control_e"))
        ib, ic, iq = (int(out[key][index]) for key in ("source_b_pos", "source_c_pos", "query_pos"))
        assert len({a, d, b, c, e}) == 5
        assert Counter(clean) == Counter(cf)
        assert Counter(neutral) == Counter(neutral_cf)
        assert Counter(clean)[a] == 2 and len(set(clean)) == 63
        assert len(set(clean[:-1])) == 63 and e not in clean
        assert len(set(neutral)) == 64
        assert list(clean[ib-1:ib+1]) == [a, b]
        assert list(clean[ic-1:ic+1]) == [d, c]
        assert list(cf[ib-1:ib+1]) == [a, c]
        assert list(cf[ic-1:ic+1]) == [d, b]
        assert clean[iq] == cf[iq] == a
        assert neutral[iq] == neutral_cf[iq] == e
        assert set(np.flatnonzero(clean != cf)) == {ib, ic}
        assert set(np.flatnonzero(clean != neutral)) == {iq}
        assert abs(ib-ic) >= 3 and min(ib, ic) >= 2 and max(ib, ic) <= 61


def test_minimum_length_supports_both_temporal_orders_without_overlap():
    out = make_probes(24, length=8, vocab=8, seed=71)
    orders = set(zip(out["source_b_pos"], out["source_c_pos"]))
    assert orders == {(2, 5), (5, 2)}
    for row in out["neutral_clean"]:
        assert sorted(row) == list(range(8))


def test_streams_are_reproducible_distinct_and_stable_under_appending_draws():
    first = make_probes(5, seed=DISCOVERY_SEED)
    repeated = make_probes(5, seed=DISCOVERY_SEED)
    extended = make_probes(8, seed=DISCOVERY_SEED)
    heldout = make_probes(5, seed=HELDOUT_SEED)
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(first[key], repeated[key])
        np.testing.assert_array_equal(first[key], extended[key][:5])
    assert not np.array_equal(first["clean"], heldout["clean"])


@pytest.mark.parametrize("kwargs", [
    dict(n=0), dict(n=True), dict(n=1.5), dict(n=1, length=7),
    dict(n=1, length=65, vocab=64), dict(n=1, seed=-1),
])
def test_invalid_probe_sizes_and_seeds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_probes(**kwargs)


def test_validator_rejects_count_matched_changes_outside_successor_swap():
    out = make_probes(2, seed=23)
    original_counts = Counter(out["counterfactual"][0])
    # Swapping the two context tokens preserves the entire token histogram
    # but invalidates the intended intervention.
    first = out["source_b_pos"][0]-1
    second = out["source_c_pos"][0]-1
    out["counterfactual"][0, [first, second]] = out["counterfactual"][0, [second, first]]
    assert Counter(out["counterfactual"][0]) == original_counts
    with pytest.raises(ValueError, match="only the two successor"):
        validate_probes(out, vocab=64)


def test_validator_accepts_global_token_relabeling():
    out = make_probes(3, seed=29)
    permutation = np.random.default_rng(311).permutation(64)
    token_keys = ARRAY_KEYS[:4] + ("target_b", "target_c", "context_a", "context_d", "control_e")
    for key in token_keys:
        out[key] = permutation[out[key]]
    assert validate_probes(out, vocab=64)

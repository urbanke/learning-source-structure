"""Independent arithmetic checks for the read-only depth diagnostic."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from depth_rhat_statistics import coordinate_summary
from stable_chain_statistics import stable_chain_summary


def textbook_decimal(values):
    """Scalar classic split R-hat, without NumPy reductions or a cutoff."""
    with localcontext() as context:
        context.prec = 90
        half = values.shape[1] // 2
        pieces = [[Decimal.from_float(float(v)) for v in part]
                  for chain in values for part in (chain[:half], chain[-half:])]
        means = [sum(part)/half for part in pieces]
        within = sum(sum((v-mean)**2 for v in part)/(half-1)
                     for part, mean in zip(pieces, means))/len(pieces)
        overall = sum(means)/len(means)
        between = half*sum((mean-overall)**2 for mean in means)/(len(means)-1)
        result = (((half-1)*within+between)/(half*within)).sqrt()
        return float(result), within, between


def test_odd_split_matches_independent_decimal_and_range_summaries():
    values = np.random.default_rng(29).uniform(.05, .95, (4, 67))
    result = coordinate_summary(values)
    expected, within, between = textbook_decimal(values)
    assert result["rescaled"]["rhat"] == pytest.approx(expected, rel=3e-15)
    assert result["original"]["rhat"] == pytest.approx(expected, rel=3e-15)
    assert result["omitted_middle_draws_per_chain"] == 1
    assert result["mean"] == pytest.approx(float(sum(map(Decimal.from_float, values.ravel()))/values.size))
    assert result["chain_summaries"][2]["max"] == values[2].max()
    assert result["half_chain_summaries"][5]["min"] == values[2, -33:].min()
    assert float(result["rescaled"]["within_variance_original_units"]) == pytest.approx(float(within), rel=3e-15)
    assert float(result["rescaled"]["between_variance_original_units"]) == pytest.approx(float(between), rel=3e-15)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scale", [1., 1e-17, 1e-200, 1e-300])
def test_rescaled_rhat_matches_scale_invariant_decimal_definition(scale):
    base = np.random.default_rng(61).uniform(.1, .8, (4, 129))
    values = base*scale
    result = coordinate_summary(values)
    expected, within, between = textbook_decimal(values)
    assert result["rescaled"]["status"] == "finite"
    assert result["rescaled"]["rhat"] == pytest.approx(expected, rel=4e-15)
    assert result["rescaled"]["rhat"] == pytest.approx(coordinate_summary(base)["rescaled"]["rhat"], rel=4e-15)
    for field, reference in (("within", within), ("between", between)):
        saved = Decimal(result["rescaled"][f"{field}_variance_original_units"])
        assert abs(saved/reference-1) < Decimal("5e-15")
        assert saved > 0  # Original-unit strings survive float64 square underflow.
    json.dumps(result, allow_nan=False)


def test_absolute_cutoff_can_report_infinity_for_a_finite_small_rhat():
    base = np.tile([.4, .6], (4, 128)) + .02*np.arange(4)[:, None]
    values = base*9e-15
    result = coordinate_summary(values)
    assert result["old_cutoff_trigger"] is True
    assert result["original"]["rhat"] is None
    assert result["original"]["status"] == "cutoff_infinity"
    assert np.isinf(stable_chain_summary(values[:, :, None])["rhat"][0])
    assert 1 < result["rescaled"]["rhat"] < 1.05
    assert result["rescaled"]["rhat"] == pytest.approx(textbook_decimal(values)[0], rel=3e-15)


@pytest.mark.parametrize("value", [0., .125, 1e-200])
def test_shared_constant_is_explicitly_uninformative(value):
    result = coordinate_summary(np.full((4, 32), value))
    assert result["rescaled"]["status"] == "constant_uninformative"
    assert result["rescaled"]["rhat"] is None
    assert result["original"]["status"] == "constant_uninformative"
    assert result["original"]["rhat"] == 1.
    assert result["zero_fraction"] == (1. if value == 0 else 0.)
    assert not result["old_cutoff_trigger"]


@pytest.mark.parametrize("scale", [1., 1e-17, 1e-200])
@pytest.mark.parametrize("draws", [32, 67])
def test_chains_stuck_at_different_constants_remain_infinite(scale, draws):
    values = np.broadcast_to(np.array([.1, .2, .3, .4])[:, None]*scale, (4, draws)).copy()
    result = coordinate_summary(values)
    assert result["rescaled"]["status"] == "different_constant_chains"
    assert result["rescaled"]["rhat"] is None
    assert result["rescaled"]["within_variance"] == 0
    assert result["rescaled"]["between_variance"] > 0


def test_omitted_middle_draw_does_not_make_constant_halves_informative():
    values = np.full((4, 67), .123)
    values[:, 33] = [.2, .3, .4, .5]
    result = coordinate_summary(values)
    assert result["rescaled"]["status"] == "constant_uninformative"
    assert result["rescaled"]["rhat"] is None


def test_diagnostic_accepts_read_only_arrays_and_preserves_every_sample():
    values = np.random.default_rng(91).uniform(0., 1e-17, (4, 32))
    before = values.tobytes()
    values.flags.writeable = False
    coordinate_summary(values)
    assert values.tobytes() == before


@pytest.mark.parametrize("values", [np.ones((1, 32)), np.ones((2, 15)),
    np.ones((2, 32), dtype=np.float32), np.full((2, 32), -1.),
    np.full((2, 32), 1.1), np.full((2, 32), np.nan), np.full((2, 32), np.inf)])
def test_invalid_saved_weights_are_rejected(values):
    with pytest.raises(ValueError, match="float64 weights"):
        coordinate_summary(values)

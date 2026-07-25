import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physics.dbi import (
    assess_scar,
    classify_dbi,
    compute_dbi,
    dam_height_m,
    scar_volume_m3,
)


def test_worked_example_volume():
    v = scar_volume_m3(2e5)
    assert v == pytest.approx(1.6e6, rel=0.06)


def test_worked_example_dam_height():
    v = scar_volume_m3(2e5)
    h = dam_height_m(v)
    assert h == pytest.approx(117, rel=0.03)


def test_worked_example_dbi_and_verdict():
    result = assess_scar(scar_id="scar-worked-example", area_m2=2e5, upstream_area_km2=250)
    assert result["dbi"] == pytest.approx(4.26, rel=0.02)
    assert result["verdict"] == "breach_risk"
    assert result["scar_id"] == "scar-worked-example"
    assert result["area_m2"] == 2e5
    assert result["graph_node_id"] is None


def test_dam_height_capped_by_local_relief():
    v = scar_volume_m3(2e5)
    h = dam_height_m(v, local_relief_m=50.0)
    assert h == 50.0


def test_dam_height_uncapped_when_relief_is_generous():
    v = scar_volume_m3(2e5)
    uncapped = dam_height_m(v)
    generous = dam_height_m(v, local_relief_m=1000.0)
    assert generous == uncapped


def test_capping_lowers_dbi_and_can_flip_verdict():
    uncapped = assess_scar(scar_id="s1", area_m2=2e5, upstream_area_km2=250)
    capped = assess_scar(
        scar_id="s1", area_m2=2e5, upstream_area_km2=250, local_relief_m=20.0
    )
    assert capped["dam_height_m"] == 20.0
    assert capped["dbi"] < uncapped["dbi"]


@pytest.mark.parametrize(
    "dbi,expected",
    [
        (0.0, "stable"),
        (2.74, "stable"),
        (2.75, "uncertain"),
        (3.0, "uncertain"),
        (3.08, "uncertain"),
        (3.09, "breach_risk"),
        (5.0, "breach_risk"),
    ],
)
def test_classify_dbi_thresholds(dbi, expected):
    assert classify_dbi(dbi) == expected


def test_compute_dbi_matches_manual_formula():
    area_m2 = 2e5
    v = scar_volume_m3(area_m2)
    h = dam_height_m(v)
    ab_km2 = 250.0
    expected = math.log10(ab_km2 * h / (v / 1e6))
    assert compute_dbi(ab_km2, h, v) == pytest.approx(expected)


@pytest.mark.parametrize("bad_area", [0, -1, -100.0])
def test_scar_volume_rejects_non_positive_area(bad_area):
    with pytest.raises(ValueError):
        scar_volume_m3(bad_area)


@pytest.mark.parametrize("bad_upstream", [0, -5])
def test_compute_dbi_rejects_non_positive_upstream_area(bad_upstream):
    v = scar_volume_m3(2e5)
    h = dam_height_m(v)
    with pytest.raises(ValueError):
        compute_dbi(bad_upstream, h, v)

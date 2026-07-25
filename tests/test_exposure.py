import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from physics.exposure import LATERAL_BUFFER_KM, propagate_exposure, rank_assets
from study_areas.registry import paths as study_area_paths


def _osm_data_exists(region: str) -> bool:
    area_paths = study_area_paths(region)
    return area_paths["roads"].exists() or area_paths["pois"].exists()


def test_propagate_exposure_requires_region_and_coordinates():
    scar = {"verdict": "breach_risk"}
    with pytest.raises(TypeError):
        propagate_exposure(scar)


def test_real_exposure_scales_with_verdict():
    if not _osm_data_exists("sunkoshi"):
        pytest.skip("Sunkoshi OSM extract not built - run extract_osm.py")
    kwargs = dict(region="sunkoshi", lat=27.76767, lon=85.87099)
    stable = propagate_exposure({"verdict": "stable"}, **kwargs)
    breach = propagate_exposure({"verdict": "breach_risk"}, **kwargs)
    for s, b in zip(stable, breach):
        assert s["exposure_score"] < b["exposure_score"]


@pytest.mark.skipif(not _osm_data_exists("sunkoshi"), reason="Sunkoshi OSM extract not built - run extract_osm.py")
def test_real_exposure_at_jure_finds_real_nearby_assets():
    scar = {"verdict": "breach_risk"}
    exposure = propagate_exposure(scar, region="sunkoshi", lat=27.76767, lon=85.87099)
    assert len(exposure) > 0
    assert all(a["distance_km"] >= 0 for a in exposure)
    distances = [a["distance_km"] for a in exposure]
    assert distances == sorted(distances)
    assert all(str(a["name"]).lower() != "nan" for a in exposure)


@pytest.mark.skipif(not _osm_data_exists("sunkoshi"), reason="Sunkoshi OSM extract not built - run extract_osm.py")
def test_real_exposure_respects_max_radius():
    scar = {"verdict": "breach_risk"}
    exposure = propagate_exposure(scar, region="sunkoshi", lat=27.76767, lon=85.87099, max_radius_km=2.0)
    assert all(a["distance_km"] <= 2.0 for a in exposure)


@pytest.mark.skipif(not _osm_data_exists("sunkoshi"), reason="Sunkoshi OSM extract not built - run extract_osm.py")
def test_real_exposure_only_includes_assets_near_the_real_channel():
    """Every result must be a real asset near the actual downstream flow
    path (within LATERAL_BUFFER_KM), not just physically nearby the scar in
    any direction - the fix for the straight-line-radius limitation."""
    scar = {"verdict": "breach_risk"}
    exposure = propagate_exposure(scar, region="sunkoshi", lat=27.76767, lon=85.87099)
    assert len(exposure) > 0
    assert all("lateral_offset_km" in a for a in exposure)
    assert all(a["lateral_offset_km"] <= LATERAL_BUFFER_KM for a in exposure)


@pytest.mark.skipif(not _osm_data_exists("sunkoshi"), reason="Sunkoshi OSM extract not built - run extract_osm.py")
def test_real_exposure_returns_empty_at_the_graph_outlet():
    from terrain.lookup import get_downstream_path

    path = get_downstream_path("sunkoshi", 27.76767, 85.87099, max_distance_km=200.0)
    outlet = path[-1]
    scar = {"verdict": "breach_risk"}
    exposure = propagate_exposure(scar, region="sunkoshi", lat=outlet["lat"], lon=outlet["lon"])
    assert exposure == []


def test_rank_assets_orders_by_risk_score_descending():
    exposure = [
        {"name": "a", "type": "building", "distance_km": 1.0, "exposure_score": 0.5},
        {"name": "b", "type": "bridge", "distance_km": 1.0, "exposure_score": 0.5},
    ]
    ranked = rank_assets(exposure)
    assert ranked[0]["type"] == "bridge"
    assert ranked[0]["risk_score"] >= ranked[1]["risk_score"]


def test_rank_assets_unknown_type_gets_default_weight():
    exposure = [{"name": "x", "type": "mystery_type", "distance_km": 1.0, "exposure_score": 0.5}]
    ranked = rank_assets(exposure)
    assert ranked[0]["weight"] == 0.5

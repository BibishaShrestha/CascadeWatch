import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry


def test_list_study_areas_finds_trishuli_and_sunkoshi():
    areas = registry.list_study_areas()
    assert "trishuli" in areas
    assert "sunkoshi" in areas


def test_get_bbox_returns_real_trishuli_extent():
    lon_min, lat_min, lon_max, lat_max = registry.get_bbox("trishuli")
    assert lon_min == pytest.approx(84.4997577)
    assert lat_max == pytest.approx(28.5009389)


def test_get_crs_is_utm_45n_for_both_registered_areas():
    assert registry.get_crs("trishuli") == "EPSG:32645"
    assert registry.get_crs("sunkoshi") == "EPSG:32645"


def test_get_osm_pbf_path_falls_back_to_shared_nepal_extract():
    assert registry.get_osm_pbf_path("trishuli") == registry.SHARED_NEPAL_OSM_PBF
    assert registry.get_osm_pbf_path("sunkoshi") == registry.SHARED_NEPAL_OSM_PBF


def test_locate_study_area_inside_trishuli():
    lon_min, lat_min, lon_max, lat_max = registry.get_bbox("trishuli")
    assert registry.locate_study_area((lat_min + lat_max) / 2, (lon_min + lon_max) / 2) == "trishuli"


def test_locate_study_area_outside_any_registered_area_returns_none():
    assert registry.locate_study_area(0.0, 0.0) is None


def test_locate_study_area_none_coordinates_returns_none():
    assert registry.locate_study_area(None, None) is None


def test_unknown_study_area_raises_clear_error():
    with pytest.raises(KeyError):
        registry.get_bbox("nonexistent_area")
    with pytest.raises(KeyError):
        registry.paths("nonexistent_area")


def test_paths_returns_expected_keys_and_real_files_exist():
    p = registry.paths("trishuli")
    assert set(p.keys()) == {"base", "config", "dem", "processed", "channel_mask", "drainage_graph", "roads", "pois", "validation"}
    assert p["config"].exists()
    assert p["dem"].exists()
    assert p["drainage_graph"].exists()

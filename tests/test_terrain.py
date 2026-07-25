import pickle
import sys
from pathlib import Path

import networkx as nx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas.registry import get_bbox, paths

REGIONS = ["trishuli", "sunkoshi"]


def _skip_if_missing(region):
    if not paths(region)["drainage_graph"].exists():
        pytest.skip(f"{region} drainage graph not built - run src/terrain/process_dem.py first")


def _skip_if_dem_missing(region):
    if not paths(region)["dem"].exists():
        pytest.skip(f"{region} DEM not downloaded - run src/data_ingestion/download_dem.py first")


@pytest.mark.parametrize("region", REGIONS)
def test_drainage_graph_is_a_dag(region):
    _skip_if_missing(region)
    with open(paths(region)["drainage_graph"], "rb") as f:
        graph = pickle.load(f)
    assert nx.is_directed_acyclic_graph(graph)


@pytest.mark.parametrize("region", REGIONS)
def test_drainage_graph_upstream_areas_are_positive(region):
    _skip_if_missing(region)
    with open(paths(region)["drainage_graph"], "rb") as f:
        graph = pickle.load(f)
    areas = nx.get_node_attributes(graph, "upstream_area_km2")
    assert len(areas) == graph.number_of_nodes()
    assert all(a > 0 for a in areas.values())


@pytest.mark.parametrize("region", REGIONS)
def test_lookup_terrain_at_returns_real_varying_areas(region):
    _skip_if_missing(region)
    from terrain.lookup import lookup_terrain_at

    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    center = lookup_terrain_at(region, (lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    corner = lookup_terrain_at(region, lat_min + 0.01, lon_min + 0.01)

    assert center["upstream_area_km2"] > 0
    assert corner["upstream_area_km2"] > 0
    assert center["distance_to_channel_km"] >= 0
    assert center["nearest_channel_node"] != corner["nearest_channel_node"]


@pytest.mark.parametrize("region", REGIONS)
def test_get_local_relief_m_is_real_and_geographically_plausible(region):
    _skip_if_missing(region)
    _skip_if_dem_missing(region)
    from terrain.lookup import get_local_relief_m

    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    relief = get_local_relief_m(region, (lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    assert relief is not None
    assert 10.0 < relief < 3000.0


def test_get_local_relief_m_returns_none_outside_dem_extent():
    from terrain.lookup import get_local_relief_m

    assert get_local_relief_m("trishuli", 0.0, 0.0) is None


@pytest.mark.parametrize("region", REGIONS)
def test_lookup_terrain_at_uses_real_local_relief_when_dem_available(region):
    _skip_if_missing(region)
    _skip_if_dem_missing(region)
    from terrain.lookup import lookup_terrain_at

    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    result = lookup_terrain_at(region, (lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    assert result["local_relief_is_real"] is True
    assert result["local_relief_m"] > 0


@pytest.mark.parametrize("region", REGIONS)
def test_get_downstream_path_never_exceeds_requested_distance(region):
    """Regression test: the path used to overshoot by up to one full edge
    (~15-40m for this DEM's resolution) because the walk loop didn't clip
    the final segment - found via physics/exposure.py's max_radius_km test
    failing by exactly 0.015km. Now interpolates the exact cutoff point."""
    _skip_if_missing(region)
    from terrain.lookup import get_downstream_path

    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2

    for max_km in [0.5, 2.0, 5.0]:
        path = get_downstream_path(region, center_lat, center_lon, max_distance_km=max_km)
        assert path[-1]["distance_km"] <= max_km + 1e-9
        distances = [p["distance_km"] for p in path]
        assert distances == sorted(distances)

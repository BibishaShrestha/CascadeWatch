"""Real terrain lookups against a registered study area's drainage graph and DEM.

Given a study area name (see src/study_areas/registry.py - any folder under
data/study_areas/ with a config.json) and a real (lat, lon), returns: the
real upstream drainage area from the nearest node in that area's channel
graph, the real local relief (max-min elevation in a 1km window, read
directly from the GLO-30 DEM), and the real downstream flow path
(get_downstream_path). Used exclusively by the Study Area Flood page - the
core Landslide Risk page has no region/coordinate concept at all and never
calls into this module.
"""
from __future__ import annotations

import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry

LOCAL_RELIEF_WINDOW_KM = 1.0
METERS_PER_DEG = 111_320


@lru_cache(maxsize=None)
def _load_graph(region: str):
    path = registry.paths(region)["drainage_graph"]
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def _load_kdtree(region: str):
    graph = _load_graph(region)
    nodes = list(graph.nodes(data=True))
    coords = np.array([[data["lat"], data["lon"]] for _, data in nodes])
    tree = cKDTree(coords)
    return tree, nodes


def _nearest_node(region: str, lat: float, lon: float):
    tree, nodes = _load_kdtree(region)
    _, idx = tree.query([lat, lon])
    return nodes[idx]


def nearest_channel_distance_km(region: str, lat: float, lon: float) -> float:
    """Great-circle-ish (equirectangular approx, fine at this scale) distance
    in km from (lat, lon) to the nearest real channel-network node."""
    _, data = _nearest_node(region, lat, lon)
    dlat = lat - data["lat"]
    dlon = (lon - data["lon"]) * np.cos(np.radians(lat))
    return float(np.hypot(dlat, dlon) * 111.32)


def get_downstream_path(region: str, lat: float, lon: float, max_distance_km: float = 20.0) -> list[dict]:
    """Walk downstream from the nearest channel node to (lat, lon), following
    the D8 flow-direction edges built by process_dem.py, up to max_distance_km.

    Each channel node has at most one outgoing edge (single downhill
    direction per pysheds' D8 model), so this is a simple walk, not a
    graph search - but it's distance along the actual channel, not a
    straight line, which is what physics/exposure.py needs to tell "this
    asset is downstream of the scar" apart from "this asset is nearby but
    on the other side of a ridge or upstream".

    Returns a list of {"lat", "lon", "distance_km"} points ordered from the
    scar's nearest channel node (distance_km=0) to wherever the walk stops
    (max_distance_km reached, or the channel runs off this region's DEM
    extent with no further downstream node).
    """
    graph = _load_graph(region)
    node_id, data = _nearest_node(region, lat, lon)
    max_distance_m = max_distance_km * 1000

    path = [{"lat": data["lat"], "lon": data["lon"], "distance_km": 0.0}]
    current = node_id
    cumulative_m = 0.0
    visited = {current}
    while cumulative_m < max_distance_m:
        successors = list(graph.successors(current))
        if not successors:
            break
        next_node = successors[0]
        if next_node in visited:
            break
        edge = graph.edges[current, next_node]
        next_data = graph.nodes[next_node]

        if cumulative_m + edge["distance_m"] > max_distance_m:
            remaining_m = max_distance_m - cumulative_m
            frac = remaining_m / edge["distance_m"]
            cur_data = graph.nodes[current]
            interp_lat = cur_data["lat"] + frac * (next_data["lat"] - cur_data["lat"])
            interp_lon = cur_data["lon"] + frac * (next_data["lon"] - cur_data["lon"])
            path.append({"lat": interp_lat, "lon": interp_lon, "distance_km": max_distance_km})
            break

        cumulative_m += edge["distance_m"]
        path.append({"lat": next_data["lat"], "lon": next_data["lon"], "distance_km": cumulative_m / 1000})
        visited.add(next_node)
        current = next_node
    return path


def get_local_relief_m(region: str, lat: float, lon: float, window_km: float = LOCAL_RELIEF_WINDOW_KM) -> float | None:
    """Local relief (max - min elevation) in a window_km-wide window
    around (lat, lon), read directly from the GLO-30 DEM.

    Returns None if the DEM file isn't present (caller should fall back to
    the study area's configured local_relief_fallback_m in that case) or
    the point falls outside the DEM's extent.
    """
    dem_path = registry.paths(region)["dem"]
    if not dem_path.exists():
        return None

    with rasterio.open(dem_path) as dem:
        row, col = dem.index(lon, lat)
        if not (0 <= row < dem.height and 0 <= col < dem.width):
            return None

        px_size_deg = abs(dem.transform.a)
        px_size_m = px_size_deg * METERS_PER_DEG
        half_window_px = max(1, round((window_km * 1000 / 2) / px_size_m))

        row_min, row_max = max(0, row - half_window_px), min(dem.height, row + half_window_px + 1)
        col_min, col_max = max(0, col - half_window_px), min(dem.width, col + half_window_px + 1)
        window = dem.read(1, window=((row_min, row_max), (col_min, col_max)))

    if window.size == 0:
        return None
    return float(window.max() - window.min())


def lookup_terrain_at(region: str, lat: float, lon: float) -> dict:
    """Real upstream_area_km2 (from the drainage graph) and real
    local_relief_m (max-min elevation in a 1km window around (lat, lon),
    read directly from the GLO-30 DEM) at a real coordinate.

    local_relief_m falls back to the study area's configured
    local_relief_fallback_m only if the DEM file is missing or the point
    falls outside its extent - both should be rare in practice since
    (lat, lon) is already expected to be inside a registered study area by
    the time this is called.
    """
    node_id, data = _nearest_node(region, lat, lon)
    local_relief_m = get_local_relief_m(region, lat, lon)
    return {
        "upstream_area_km2": data["upstream_area_km2"],
        "local_relief_m": local_relief_m if local_relief_m is not None else registry.get_local_relief_fallback_m(region),
        "local_relief_is_real": local_relief_m is not None,
        "nearest_channel_node": node_id,
        "distance_to_channel_km": nearest_channel_distance_km(region, lat, lon),
    }

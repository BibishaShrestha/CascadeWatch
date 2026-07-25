"""Downstream exposure propagation and asset risk ranking.

Only used by the Study Area Flood page (app/pages/2_Study_Area_Flood.py),
which always has a real registered study area + real (lat, lon) by
construction - the core Landslide Risk page has no region/exposure concept
at all. So this module requires real inputs rather than offering a stub
fallback; there is no longer a caller that could need one.

Walks the actual drainage graph downstream from the scar
(terrain.lookup.get_downstream_path - real distance along real D8
flow-direction edges, not a straight line), then matches real OSM assets
(the study area's processed roads.geojson/pois.geojson) against that real
channel path: an asset only counts as exposed if it falls within a lateral
buffer of the actual flow path (i.e. genuinely near the river the scar
would affect), and its exposure decays with real distance measured *along*
that path from the scar, not straight-line distance to the scar itself.
This is the fix for a known limitation from an earlier version: a
straight-line radius can't distinguish "downstream and at risk" from
"physically nearby but upstream, or on the other side of a ridge, or on an
unrelated tributary."

Remaining known simplification: the lateral buffer is a flat constant
(LATERAL_BUFFER_KM), not derived from the DEM's actual valley width/local
relief at each point - a real floodplain-extent model would need that.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import LineString
from shapely.ops import nearest_points
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry
from terrain.lookup import get_downstream_path


def _clean_name(value, fallback: str) -> str:
    return fallback if pd.isna(value) else str(value)


ASSET_TYPE_WEIGHTS = {
    "bridge": 0.95,
    "health": 0.90,
    "power": 0.90,
    "school": 0.85,
    "settlement": 0.80,
    "road": 0.60,
    "building": 0.50,
}

_VERDICT_BASE_SCORE = {"stable": 0.05, "uncertain": 0.4, "breach_risk": 1.0}

LATERAL_BUFFER_KM = 1.0


@lru_cache(maxsize=None)
def _load_asset_index(region: str):
    """Real OSM assets for a study area, reprojected to meters, spatially indexed."""
    area_paths = registry.paths(region)
    crs = registry.get_crs(region)

    records = []

    if area_paths["roads"].exists():
        roads = gpd.read_file(area_paths["roads"]).to_crs(crs)
        for _, row in roads.iterrows():
            asset_type = "bridge" if not pd.isna(row.get("bridge")) else "road"
            name = _clean_name(row.get("name"), f"unnamed {_clean_name(row.get('highway'), 'road')}")
            records.append((row.geometry, name, asset_type))

    if area_paths["pois"].exists():
        pois = gpd.read_file(area_paths["pois"]).to_crs(crs)
        for _, row in pois.iterrows():
            name = _clean_name(row.get("name"), f"unnamed {row['asset_type']}")
            records.append((row.geometry, name, row["asset_type"]))

    tree = STRtree([r[0] for r in records])
    return tree, records


def propagate_exposure(
    scar_result: dict,
    region: str,
    lat: float,
    lon: float,
    decay_km: float = 8.0,
    max_radius_km: float = 10.0,
) -> list[dict]:
    """Distance-decayed exposure score for real assets near the real
    downstream flow path from (lat, lon) within a registered study area.

    Walks the real drainage graph downstream up to max_radius_km, then only
    counts assets within LATERAL_BUFFER_KM of that real channel path - not
    just physically nearby the scar in any direction. distance_km in the
    result is measured along the channel, not a straight line to the scar.
    """
    downstream_path = get_downstream_path(region, lat, lon, max_distance_km=max_radius_km)

    crs = registry.get_crs(region)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    path_xy = [transformer.transform(p["lon"], p["lat"]) for p in downstream_path]
    if len(path_xy) < 2:
        return []
    line = LineString(path_xy)

    tree, records = _load_asset_index(region)
    candidate_idxs = tree.query(line.buffer(LATERAL_BUFFER_KM * 1000))

    base = _VERDICT_BASE_SCORE[scar_result["verdict"]]
    exposure = []
    seen = set()
    for idx in candidate_idxs:
        geom, name, asset_type = records[idx]
        lateral_km = geom.distance(line) / 1000
        if lateral_km > LATERAL_BUFFER_KM:
            continue
        nearest_on_line, _ = nearest_points(line, geom)
        along_path_km = line.project(nearest_on_line) / 1000
        key = (name, asset_type, round(along_path_km, 2))
        if key in seen:
            continue
        seen.add(key)
        decay = math.exp(-along_path_km / decay_km)
        exposure.append({
            "name": name,
            "type": asset_type,
            "distance_km": round(along_path_km, 3),
            "lateral_offset_km": round(lateral_km, 3),
            "exposure_score": round(base * decay, 4),
        })

    exposure.sort(key=lambda a: a["distance_km"])
    return exposure[:50]


def rank_assets(exposure: list[dict]) -> list[dict]:
    """Rank exposed assets by exposure_score * asset-type weight, descending."""
    ranked = []
    for asset in exposure:
        weight = ASSET_TYPE_WEIGHTS.get(asset["type"], 0.5)
        risk_score = round(asset["exposure_score"] * weight, 4)
        ranked.append({**asset, "weight": weight, "risk_score": risk_score})
    ranked.sort(key=lambda a: a["risk_score"], reverse=True)
    return ranked

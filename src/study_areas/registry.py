"""Study-area registry: the single source of truth for which real-world
corridors have real DEM/drainage-graph/OSM data behind them.

Adding a new study area is a filesystem operation, not a code change: drop
a new `data/study_areas/<name>/` folder with a config.json and it's picked
up automatically by list_study_areas() - no Python edit required.

Replaces three previously-scattered hardcoded dicts (physics/region.py's
REGION_BBOXES and REGION_TERRAIN_STUB, physics/exposure.py's
REGION_UTM_CRS) with one config.json per area.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_AREAS_DIR = REPO_ROOT / "data" / "study_areas"
SHARED_NEPAL_OSM_PBF = REPO_ROOT / "data" / "raw" / "osm" / "nepal-latest.osm.pbf"


@lru_cache(maxsize=None)
def _scan() -> dict[str, dict]:
    if not STUDY_AREAS_DIR.exists():
        return {}
    return {
        p.parent.name: json.loads(p.read_text())
        for p in sorted(STUDY_AREAS_DIR.glob("*/config.json"))
    }


def list_study_areas() -> list[str]:
    return sorted(_scan().keys())


def _config(name: str) -> dict:
    configs = _scan()
    if name not in configs:
        raise KeyError(f"Unknown study area {name!r}. Registered: {list_study_areas()}")
    return configs[name]


def get_bbox(name: str) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max)."""
    return tuple(_config(name)["bbox"])


def get_crs(name: str) -> str:
    return _config(name)["utm_crs"]


def get_local_relief_fallback_m(name: str) -> float:
    return _config(name)["local_relief_fallback_m"]


def get_osm_pbf_path(name: str) -> Path:
    override = _config(name).get("osm_pbf_path")
    return Path(override) if override else SHARED_NEPAL_OSM_PBF


def locate_study_area(lat: float | None, lon: float | None) -> str | None:
    """Return the study area name whose bbox contains (lat, lon), else None."""
    if lat is None or lon is None:
        return None
    for name in list_study_areas():
        lon_min, lat_min, lon_max, lat_max = get_bbox(name)
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name
    return None


def paths(name: str) -> dict[str, Path]:
    _config(name)
    base = STUDY_AREAS_DIR / name
    processed = base / "processed"
    return {
        "base": base,
        "config": base / "config.json",
        "dem": base / "dem_glo30.tif",
        "processed": processed,
        "channel_mask": processed / "channel_mask.tif",
        "drainage_graph": processed / "drainage_graph.gpickle",
        "roads": processed / "roads.geojson",
        "pois": processed / "pois.geojson",
        "validation": base / "validation",
    }

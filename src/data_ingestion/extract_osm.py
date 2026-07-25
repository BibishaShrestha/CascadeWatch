"""Clip an OSM extract to a registered study area's bbox and pull out
roads (with highway class + bridge flag), settlements, health facilities,
and schools.

Uses the study area's config.json `osm_pbf_path` if set, else falls back to
the shared Nepal-wide Geofabrik extract at data/raw/osm/nepal-latest.osm.pbf
(see download instructions in this file's __main__ or just:
curl -L -o data/raw/osm/nepal-latest.osm.pbf
https://download.geofabrik.de/asia/nepal-latest.osm.pbf) - a study area
outside Nepal would set its own osm_pbf_path to a different country/region
extract.

Usage:
    python src/data_ingestion/extract_osm.py [--region <name>|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyrosm import OSM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry

POI_FILTER = {
    "amenity": ["hospital", "clinic", "health_post", "doctors", "school"],
    "place": ["city", "town", "village", "hamlet"],
}


def classify_poi_type(row) -> str:
    amenity = row.get("amenity")
    place = row.get("place")
    if amenity == "school":
        return "school"
    if amenity in ("hospital", "clinic", "health_post", "doctors"):
        return "health"
    if place in ("city", "town", "village", "hamlet"):
        return "settlement"
    return "other"


def extract_region(region: str) -> dict:
    bbox = registry.get_bbox(region)
    pbf_path = registry.get_osm_pbf_path(region)
    osm = OSM(str(pbf_path), bounding_box=list(bbox))

    roads = osm.get_network(network_type="driving")
    pois = osm.get_pois(custom_filter=POI_FILTER)

    out_dir = registry.paths(region)["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_roads = n_bridges = n_pois = 0

    if roads is not None and len(roads):
        keep_cols = [c for c in ["highway", "bridge", "name", "geometry"] if c in roads.columns]
        roads = roads[keep_cols]
        roads.to_file(out_dir / "roads.geojson", driver="GeoJSON")
        n_roads = len(roads)
        if "bridge" in roads.columns:
            n_bridges = int(roads["bridge"].notna().sum())

    if pois is not None and len(pois):
        pois = pois.copy()
        pois["asset_type"] = pois.apply(classify_poi_type, axis=1)
        pois = pois[pois["asset_type"] != "other"]
        keep_cols = [c for c in ["name", "amenity", "place", "asset_type", "geometry"] if c in pois.columns]
        pois = pois[keep_cols]
        pois["geometry"] = pois.geometry.apply(lambda g: g if g.geom_type == "Point" else g.representative_point())
        pois.to_file(out_dir / "pois.geojson", driver="GeoJSON")
        n_pois = len(pois)

    return {"region": region, "n_roads": n_roads, "n_bridges": n_bridges, "n_pois": n_pois}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=registry.list_study_areas() + ["all"], default="all")
    args = parser.parse_args()

    regions = registry.list_study_areas() if args.region == "all" else [args.region]
    for region in regions:
        pbf_path = registry.get_osm_pbf_path(region)
        if not pbf_path.exists():
            print(f"ERROR: [{region}] {pbf_path} not found. Download it first (e.g. for the shared "
                  "Nepal-wide extract):", file=sys.stderr)
            print("  curl -L -o data/raw/osm/nepal-latest.osm.pbf "
                  "https://download.geofabrik.de/asia/nepal-latest.osm.pbf", file=sys.stderr)
            sys.exit(1)
        stats = extract_region(region)
        print(f"[{region}] roads={stats['n_roads']} (bridges={stats['n_bridges']}), pois={stats['n_pois']}")


if __name__ == "__main__":
    main()

"""Pull historical Landsat 8 imagery for validation: the Jure/Sunkoshi 2014
breach case, and a non-breach control case.

Both events predate or nearly predate Sentinel-2 (launched 2015-06-23), so
this uses Landsat 8 Collection 2 Level 2 (surface reflectance) throughout,
for a consistent sensor across both cases.

Coordinates and areas come from existing inventories: the ASM monsoon
inventory (BGS/University of Plymouth) for Jure, and the USGS Gorkha
earthquake inventory for the control case, since no documented non-breach
landslide-dam case in either corridor could be found - the control is a
large in-corridor landslide with no reported breach, labeled "breach
outcome unconfirmed" rather than a verified non-breach ground truth.

Usage:
    earthengine authenticate   # once, if no cached credentials
    python src/data_ingestion/pull_validation_imagery.py --project <ee-project-id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry

COLLECTION = "LANDSAT/LC08/C02/T1_L2"
BANDS = ["SR_B4", "SR_B3", "SR_B2", "SR_B5"]
CHIP_METERS = 3000
SCALE_M = 30

CASES = {
    "jure": {
        "lat": 27.76767,
        "lon": 85.87099,
        "area_m2": 684783,
        "region": "sunkoshi",
        "stages": {
            "pre_landslide": ("2014-05-01", 30),
            "post_landslide": ("2014-08-15", 25),
            "post_breach": ("2014-09-20", 25),
        },
    },
    "control": {
        "lat": 28.21898,
        "lon": 85.48355,
        "area_m2": 809459,
        "region": "trishuli",
        "stages": {
            "pre_event": ("2015-02-01", 30),
            "post_event": ("2015-05-15", 30),
        },
    },
}


def least_cloudy_scene(point: "ee.Geometry.Point", target_date: str, window_days: int):
    start = ee.Date(target_date).advance(-window_days, "day")
    end = ee.Date(target_date).advance(window_days, "day")
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .sort("CLOUD_COVER")
    )
    return collection.first()


def download_chip(image: "ee.Image", region: "ee.Geometry", out_path: Path) -> dict:
    scaled = image.select(BANDS).multiply(0.0000275).add(-0.2)
    url = scaled.getDownloadURL({
        "region": region,
        "scale": SCALE_M,
        "format": "GEO_TIFF",
    })
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return {"path": str(out_path), "size_bytes": len(resp.content)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Earth Engine Cloud project id")
    parser.add_argument("--case", choices=["jure", "control", "all"], default="all")
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    cases = ["jure", "control"] if args.case == "all" else [args.case]
    for case_name in cases:
        case = CASES[case_name]
        point = ee.Geometry.Point([case["lon"], case["lat"]])
        region = point.buffer(CHIP_METERS / 2).bounds()

        for stage_name, (target_date, window_days) in case["stages"].items():
            image = least_cloudy_scene(point, target_date, window_days)
            info = image.getInfo()
            if info is None:
                print(f"[{case_name}/{stage_name}] NO SCENE FOUND in +/-{window_days}d of {target_date}")
                continue
            actual_date = info["properties"]["DATE_ACQUIRED"]
            cloud_pct = info["properties"]["CLOUD_COVER"]

            out_path = registry.paths(case["region"])["validation"] / case_name / f"{stage_name}.tif"
            result = download_chip(image, region, out_path)
            print(f"[{case_name}/{stage_name}] target={target_date} actual={actual_date} "
                  f"cloud={cloud_pct:.1f}% -> {result['path']} ({result['size_bytes']/1e3:.0f} KB)")


if __name__ == "__main__":
    main()

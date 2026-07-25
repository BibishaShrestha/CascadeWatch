"""Download Copernicus GLO-30 DEM tiles for a registered study area via the
OpenTopography Global DEM API.

Requires OPENTOPOGRAPHY_API_KEY (see src/data_ingestion/config.py - env var
or gitignored .env at repo root), and the study area's config.json to
already exist.

Usage:
    python src/data_ingestion/download_dem.py [--region <name>|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_ingestion.config import get_opentopography_key
from study_areas import registry

API_URL = "https://portal.opentopography.org/API/globaldem"
DEM_TYPE = "COP30"


def download_dem(region: str, out_path: Path, api_key: str) -> None:
    lon_min, lat_min, lon_max, lat_max = registry.get_bbox(region)
    params = {
        "demtype": DEM_TYPE,
        "south": lat_min,
        "north": lat_max,
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    print(f"[{region}] requesting DEM for bbox ({lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f})...")
    resp = requests.get(API_URL, params=params, timeout=180)
    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or "image" not in content_type and "octet-stream" not in content_type and "tiff" not in content_type.lower():
        raise RuntimeError(
            f"[{region}] DEM download failed (status={resp.status_code}, "
            f"content-type={content_type}): {resp.text[:500]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[{region}] saved {out_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=registry.list_study_areas() + ["all"], default="all")
    args = parser.parse_args()

    api_key = get_opentopography_key()
    if not api_key:
        print("ERROR: OPENTOPOGRAPHY_API_KEY not found (env var or .env). Aborting.", file=sys.stderr)
        sys.exit(1)

    regions = registry.list_study_areas() if args.region == "all" else [args.region]
    for region in regions:
        out_path = registry.paths(region)["dem"]
        if out_path.exists():
            print(f"[{region}] already downloaded at {out_path}, skipping (delete to re-fetch)")
            continue
        download_dem(region, out_path, api_key)


if __name__ == "__main__":
    main()

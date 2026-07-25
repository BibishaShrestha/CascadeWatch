"""Curated real-coordinate Sentinel-2 example chips for the Streamlit demo.

Source (ids "12xxx"/"11xxx"): an Earth Engine pull (post-event previews,
output/previews/post_<id>.png) that recorded each chip's (lat, lon) in a
manifest. Picked the highest-area_m2 chip per bucket so each example likely
shows a visible scar. Copied into datasets/geolocated_examples/ since these
are demo assets, not training data.

Source (ids "nepal_xxx"): Sen12Landslides Sentinel-2 Nepal patches with
embedded CRS (EPSG:32645). Center coordinates read directly from each
GeoTIFF's geotransform. area_m2 computed from the provided segmentation
mask at its 10m/px GSD (also read from the GeoTIFF). n_landslides for these
"nepal_xxx" entries means connected-component count in that mask
(scipy.ndimage.label) - not the same provenance as the "12xxx"/"11xxx"
entries' n_landslides, which comes from the original GEE pull's inventory
record; don't compare the two fields directly across sources. Two of this
batch's six images (nepal_155, nepal_218) are deliberately not included
here - held out for a separate Model A evaluation instead.

Unlike datasets/landslide_seg and datasets/nepal_landslide_seg (no
georeferencing at all), these do have coordinates, so the app can
auto-detect study-area membership via
study_areas.registry.locate_study_area() instead of requiring a manual guess.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOLOCATED_DIR = REPO_ROOT / "datasets" / "geolocated_examples"

GEOLOCATED_CHIPS = {
    "12389": (28.14092536, 84.90094091, 193953, 1),
    "12383": (28.12556605, 85.26330618, 118804, 6),
    "12822": (28.13267304, 85.05631087, 94665, 3),
    "12803": (28.04539317, 85.19660372, 80869, 3),
    "12844": (28.23130379, 84.85760970, 71422, 3),
    "12284": (27.85081256, 85.73842469, 62231, 1),
    "11505": (27.78445221, 85.55881468, 56990, 5),
    "12217": (27.86957892, 85.98450184, 41788, 3),
    "11816": (26.78114208, 87.42823159, 320634, 27),
    "11346": (27.47330947, 86.27088172, 235657, 1),
    "nepal_240": (28.090554, 85.239522, 25800, 3),
    "nepal_241": (28.102103, 85.239333, 22600, 1),
    "nepal_262": (28.102270, 85.252358, 30600, 1),
    "nepal_263": (28.113819, 85.252171, 23800, 1),
}


def image_path(chip_id: str) -> Path:
    return GEOLOCATED_DIR / f"post_{chip_id}.png"


def all_examples() -> list[dict]:
    return [
        {
            "id": chip_id,
            "path": image_path(chip_id),
            "lat": lat,
            "lon": lon,
            "area_m2": area_m2,
            "n_landslides": n_landslides,
        }
        for chip_id, (lat, lon, area_m2, n_landslides) in GEOLOCATED_CHIPS.items()
    ]

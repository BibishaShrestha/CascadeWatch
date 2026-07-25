import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

from common import annotate_detections

REAL_GEOTIFF = REPO_ROOT / "nepal-dataset-with-crs" / "tiff-images-filtered" / "nepal_241.tif"
SAMPLE_PNG = REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_100.png"


def test_annotate_detections_on_png_with_no_detections():
    if not SAMPLE_PNG.exists():
        pytest.skip("landslide_seg dataset not present")
    img = annotate_detections(str(SAMPLE_PNG), [])
    assert img.size == (128 * 4, 128 * 4)


def test_annotate_detections_on_real_geotiff_does_not_crash():
    if not REAL_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    detections = [{"mask_polygon": [[10, 10], [50, 10], [50, 50], [10, 50]], "confidence": 0.5, "class": "landslide"}]
    img = annotate_detections(str(REAL_GEOTIFF), detections)
    assert img.size == (128 * 4, 128 * 4)


def test_annotate_detections_accepts_path_object_not_just_str():
    if not SAMPLE_PNG.exists():
        pytest.skip("landslide_seg dataset not present")
    img = annotate_detections(SAMPLE_PNG, [])
    assert img.size == (128 * 4, 128 * 4)

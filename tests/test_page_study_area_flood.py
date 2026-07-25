"""Permanent regression coverage for app/pages/2_Study_Area_Flood.py
(ADD-ON module - only meaningful once a study area is registered).

The page requires an upload before the study-area/coordinate widgets even
render (it auto-fills them from the upload when possible), so every test
here uploads first, then interacts with the widgets - unlike the other
pages, there's no "widgets exist before upload" state to test against.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_PATH = str(REPO_ROOT / "app" / "Home.py")
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from study_areas.registry import get_bbox, list_study_areas

pytestmark = pytest.mark.skipif(
    not list_study_areas(), reason="No study areas registered under data/study_areas/"
)

DEFAULT_PNG = REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_825.png"
REAL_TRISHULI_GEOTIFF = REPO_ROOT / "nepal-dataset-with-crs" / "tiff-images-filtered" / "nepal_241.tif"


def _fresh_page() -> AppTest:
    at = AppTest.from_file(HOME_PATH, default_timeout=90)
    at.run()
    at.switch_page("pages/2_Study_Area_Flood.py")
    at.run()
    assert not at.exception, at.exception
    return at


def _upload(at: AppTest, path: Path = DEFAULT_PNG, filename: str = "my_tile.png", mimetype: str = "image/png") -> AppTest:
    img_bytes = path.read_bytes()
    at.get("file_uploader")[0].upload(filename, img_bytes, mimetype).run()
    assert not at.exception, at.exception
    return at


def test_page_loads_without_exception():
    _fresh_page()


def test_study_area_selectbox_lists_registered_areas():
    at = _upload(_fresh_page())
    assert set(at.selectbox[0].options) == set(list_study_areas())


def test_full_chain_uses_real_terrain_and_exposure():
    at = _upload(_fresh_page())
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    final_success = [s.value for s in at.success if "ran end to end" in s.value]
    assert final_success
    assert any("real DEM-derived upstream area" in s.value for s in at.success)
    assert len(at.dataframe) >= 1


def test_switching_study_area_updates_bbox_caption():
    at = _upload(_fresh_page())
    areas = sorted(list_study_areas())
    if len(areas) < 2:
        pytest.skip("Need at least 2 registered study areas to test switching")
    at.selectbox[0].set_value(areas[1]).run()
    assert not at.exception, at.exception
    assert any(areas[1] in c.value for c in at.caption)


def test_zero_detection_upload_shows_info_not_crash():
    at = _fresh_page()
    _upload(at, REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_1177.png", "empty_tile.png")
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    assert any("No landslide scars detected" in i.value for i in at.info)


def test_real_geotiff_upload_autofills_region_and_coordinates():
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    at = _fresh_page()
    _upload(at, REAL_TRISHULI_GEOTIFF, "nepal_241.tif", "image/tiff")
    assert at.selectbox[0].value == "trishuli"
    assert at.number_input[0].value == pytest.approx(28.1021, abs=1e-3)
    assert at.number_input[1].value == pytest.approx(85.2393, abs=1e-3)
    assert any("auto-filled from this upload" in c.value for c in at.caption)


def test_non_georeferenced_upload_defaults_to_region_bbox_center():
    at = _fresh_page()
    _upload(at)
    region = at.selectbox[0].value
    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    assert at.number_input[0].value == pytest.approx(round((lat_min + lat_max) / 2, 4))
    assert at.number_input[1].value == pytest.approx(round((lon_min + lon_max) / 2, 4))
    assert any("no embedded coordinates" in c.value.lower() for c in at.caption)


def test_manually_switching_region_after_autofill_resets_coordinates_safely():
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    if "sunkoshi" not in list_study_areas() or "trishuli" not in list_study_areas():
        pytest.skip("Need both trishuli and sunkoshi registered")
    at = _fresh_page()
    _upload(at, REAL_TRISHULI_GEOTIFF, "nepal_241.tif", "image/tiff")
    assert at.selectbox[0].value == "trishuli"
    at.selectbox[0].set_value("sunkoshi").run()
    assert not at.exception, at.exception
    lon_min, lat_min, lon_max, lat_max = get_bbox("sunkoshi")
    assert lat_min <= at.number_input[0].value <= lat_max
    assert lon_min <= at.number_input[1].value <= lon_max


def test_wrong_region_selection_shows_location_mismatch_error():
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    if "sunkoshi" not in list_study_areas() or "trishuli" not in list_study_areas():
        pytest.skip("Need both trishuli and sunkoshi registered")
    at = _fresh_page()
    _upload(at, REAL_TRISHULI_GEOTIFF, "nepal_241.tif", "image/tiff")
    at.selectbox[0].set_value("sunkoshi").run()
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    assert any("Location mismatch" in e.value and "trishuli" in e.value for e in at.error)


def test_correct_region_and_coordinate_shows_location_confirmed():
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    at = _fresh_page()
    _upload(at, REAL_TRISHULI_GEOTIFF, "nepal_241.tif", "image/tiff")
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    assert not list(at.error)
    assert not list(at.warning)
    assert any("confirm it's really inside" in s.value for s in at.success)


def test_manually_reverting_to_default_coordinate_shows_point_mismatch_warning():
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    if "trishuli" not in list_study_areas():
        pytest.skip("Need trishuli registered")
    at = _fresh_page()
    _upload(at, REAL_TRISHULI_GEOTIFF, "nepal_241.tif", "image/tiff")
    lon_min, lat_min, lon_max, lat_max = get_bbox("trishuli")
    at.number_input[0].set_value(round((lat_min + lat_max) / 2, 4)).run()
    at.number_input[1].set_value(round((lon_min + lon_max) / 2, 4)).run()
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    assert not list(at.error)
    assert any("km away" in w.value and "not updated to match this image" in w.value for w in at.warning)


def test_non_georeferenced_upload_shows_no_mismatch_banner():
    at = _upload(_fresh_page())
    at.button[0].click().run(timeout=90)
    assert not at.exception, at.exception
    assert not list(at.error)
    assert not any("embedded coordinates" in w.value for w in at.warning)

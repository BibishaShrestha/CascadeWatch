import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.gsd import get_center_latlon, get_gsd_m_per_px

REAL_DEM = REPO_ROOT / "data" / "study_areas" / "trishuli" / "dem_glo30.tif"
REAL_TRISHULI_GEOTIFF = REPO_ROOT / "nepal-dataset-with-crs" / "tiff-images-filtered" / "nepal_241.tif"


def test_real_geotiff_with_crs_returns_geodesic_gsd():
    if not REAL_DEM.exists():
        import pytest
        pytest.skip("Trishuli DEM not downloaded")
    value, source = get_gsd_m_per_px(REAL_DEM, fallback_gsd=10.0)
    assert source == "geotiff_crs"
    assert 20.0 < value < 40.0


def test_non_georeferenced_png_falls_back():
    png = REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_100.png"
    if not png.exists():
        import pytest
        pytest.skip("landslide_seg dataset not present")
    value, source = get_gsd_m_per_px(png, fallback_gsd=10.0)
    assert (value, source) == (10.0, "fallback")


def test_missing_file_falls_back_without_crashing():
    value, source = get_gsd_m_per_px("does/not/exist.tif", fallback_gsd=12.5)
    assert (value, source) == (12.5, "fallback")


def test_non_raster_extension_falls_back_without_opening_file():
    value, source = get_gsd_m_per_px("some_image.jpg", fallback_gsd=7.0)
    assert (value, source) == (7.0, "fallback")


def test_get_center_latlon_on_real_geotiff_matches_known_coordinate():
    import pytest
    if not REAL_TRISHULI_GEOTIFF.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    result = get_center_latlon(REAL_TRISHULI_GEOTIFF)
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(28.102103, abs=1e-3)
    assert lon == pytest.approx(85.239333, abs=1e-3)


def test_get_center_latlon_on_plain_png_returns_none():
    png = REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_100.png"
    if not png.exists():
        import pytest
        pytest.skip("landslide_seg dataset not present")
    assert get_center_latlon(png) is None


def test_get_center_latlon_on_missing_file_returns_none_not_crash():
    assert get_center_latlon("does/not/exist.tif") is None

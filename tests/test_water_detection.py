import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.water_detection import (
    MissingBandError,
    compute_ndwi,
    detect_water,
    read_green_nir,
)

REAL_TIF = REPO_ROOT / "data" / "study_areas" / "sunkoshi" / "validation" / "jure" / "post_breach.tif"


def test_compute_ndwi_pure_water_pixel_is_positive():
    green = np.array([[0.3]])
    nir = np.array([[0.05]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] > 0


def test_compute_ndwi_vegetation_like_pixel_is_negative():
    green = np.array([[0.1]])
    nir = np.array([[0.4]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] < 0


def test_compute_ndwi_handles_zero_denominator_without_crashing():
    green = np.array([[0.0]])
    nir = np.array([[0.0]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] == 0.0


def test_compute_ndwi_matches_formula():
    green = np.array([[0.4, 0.2]])
    nir = np.array([[0.1, 0.3]])
    ndwi = compute_ndwi(green, nir)
    expected = np.array([[(0.4 - 0.1) / (0.4 + 0.1), (0.2 - 0.3) / (0.2 + 0.3)]])
    np.testing.assert_allclose(ndwi, expected)


@pytest.mark.skipif(not REAL_TIF.exists(), reason="Real Landsat validation imagery not pulled")
def test_read_green_nir_on_real_tif_returns_real_arrays():
    green, nir = read_green_nir(REAL_TIF)
    assert green.shape == nir.shape
    assert green.shape[0] > 0


def test_read_green_nir_raises_on_insufficient_bands(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    single_band_tif = tmp_path / "one_band.tif"
    with rasterio.open(
        single_band_tif, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="float64", crs="EPSG:4326", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(np.zeros((4, 4)), 1)

    with pytest.raises(MissingBandError):
        read_green_nir(single_band_tif)


@pytest.mark.skipif(not REAL_TIF.exists(), reason="Real Landsat validation imagery not pulled")
def test_detect_water_on_real_jure_post_breach_finds_the_river():
    detections = detect_water(REAL_TIF)
    assert len(detections) >= 1
    for d in detections:
        assert set(d.keys()) == {"mask_polygon", "confidence", "class"}
        assert d["class"] == "flood"
        assert 0.0 <= d["confidence"] <= 1.0
        assert len(d["mask_polygon"]) >= 3


@pytest.mark.skipif(not REAL_TIF.exists(), reason="Real Landsat validation imagery not pulled")
def test_detect_water_threshold_is_tunable():
    strict = detect_water(REAL_TIF, threshold=0.2)
    lenient = detect_water(REAL_TIF, threshold=-0.2)
    strict_area = sum(len(d["mask_polygon"]) for d in strict)
    lenient_area = sum(len(d["mask_polygon"]) for d in lenient)
    assert lenient_area >= strict_area or len(lenient) >= len(strict)

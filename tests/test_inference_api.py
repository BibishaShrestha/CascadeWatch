import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import models.inference_api as inference_api
from models.inference_api import (
    default_gsd_m_per_px,
    polygon_area_m2,
    polygon_area_px,
    predict_flood,
    predict_landslide,
)

SAMPLE_IMAGE = REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_100.png"


def _assert_matches_frozen_interface(detections):
    assert isinstance(detections, list)
    for det in detections:
        assert set(det.keys()) == {"mask_polygon", "confidence", "class"}
        assert isinstance(det["mask_polygon"], list)
        assert all(len(pt) == 2 for pt in det["mask_polygon"])
        assert isinstance(det["confidence"], float)
        assert isinstance(det["class"], str)


def test_polygon_area_px_unit_square():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_area_px(square) == pytest.approx(100.0)


def test_polygon_area_px_triangle():
    triangle = [[0, 0], [4, 0], [0, 3]]
    assert polygon_area_px(triangle) == pytest.approx(6.0)


def test_polygon_area_px_degenerate_line_is_zero():
    line = [[0, 0], [1, 0], [2, 0]]
    assert polygon_area_px(line) == 0.0


def test_polygon_area_px_fewer_than_three_points_is_zero():
    assert polygon_area_px([[0, 0], [1, 1]]) == 0.0
    assert polygon_area_px([]) == 0.0


def test_polygon_area_px_is_orientation_independent():
    clockwise = [[0, 0], [0, 10], [10, 10], [10, 0]]
    counterclockwise = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_area_px(clockwise) == pytest.approx(polygon_area_px(counterclockwise))


def test_default_gsd_reads_real_trained_models_config():
    assert inference_api.LANDSLIDE_MODEL_WEIGHTS is not None, (
        "expected trained weights to be present in this checkout"
    )
    assert default_gsd_m_per_px() == pytest.approx(10.0)


def test_default_gsd_falls_back_when_no_weights_loaded(monkeypatch):
    monkeypatch.setattr(inference_api, "LANDSLIDE_MODEL_WEIGHTS", None)
    assert default_gsd_m_per_px() == pytest.approx(inference_api.DEFAULT_GSD_M_PER_PX)


def test_default_gsd_falls_back_when_weights_have_no_model_config(monkeypatch, tmp_path):
    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"")
    monkeypatch.setattr(inference_api, "LANDSLIDE_MODEL_WEIGHTS", str(fake_weights))
    assert default_gsd_m_per_px() == pytest.approx(inference_api.DEFAULT_GSD_M_PER_PX)
    assert inference_api.DEFAULT_GSD_M_PER_PX == pytest.approx(10.0)


def test_polygon_area_m2_default_gsd_matches_loaded_models_real_gsd():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    expected = 100.0 * default_gsd_m_per_px() ** 2
    assert polygon_area_m2(square) == pytest.approx(expected)


def test_polygon_area_m2_custom_gsd_scales_quadratically():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    area_10m = polygon_area_m2(square, gsd_m_per_px=10.0)
    area_30m = polygon_area_m2(square, gsd_m_per_px=30.0)
    assert area_30m == pytest.approx(area_10m * 9)


def test_predict_landslide_real_matches_frozen_interface():
    assert inference_api.LANDSLIDE_MODEL_WEIGHTS is not None, (
        "expected trained weights to be present in this checkout - "
        "if this fails, Model A training weights are missing"
    )
    detections = predict_landslide(str(SAMPLE_IMAGE))
    _assert_matches_frozen_interface(detections)


REAL_GEOTIFF_UPLOAD = REPO_ROOT / "nepal-dataset-with-crs" / "tiff-images-filtered" / "nepal_241.tif"
REAL_GEOTIFF_PNG_EQUIVALENT = REPO_ROOT / "datasets" / "geolocated_examples" / "post_nepal_241.png"


def test_predict_landslide_on_real_geotiff_does_not_crash():
    if not REAL_GEOTIFF_UPLOAD.exists():
        pytest.skip("nepal-dataset-with-crs/ not present in this checkout")
    detections = predict_landslide(str(REAL_GEOTIFF_UPLOAD))
    _assert_matches_frozen_interface(detections)


def test_predict_landslide_bgr_fix_matches_known_real_confidence():
    if not REAL_GEOTIFF_PNG_EQUIVALENT.exists():
        pytest.skip("Sen12Landslides demo assets not present in this checkout")
    detections = predict_landslide(str(REAL_GEOTIFF_PNG_EQUIVALENT))
    assert len(detections) == 1
    assert detections[0]["confidence"] == pytest.approx(0.4068, abs=1e-3)


def test_predict_landslide_stub_matches_frozen_interface(monkeypatch):
    monkeypatch.setattr(inference_api, "LANDSLIDE_MODEL_WEIGHTS", None)
    detections = predict_landslide(str(SAMPLE_IMAGE))
    _assert_matches_frozen_interface(detections)
    assert len(detections) == 1
    assert detections[0]["class"] == "landslide"


def test_predict_landslide_stub_is_deterministic(monkeypatch):
    monkeypatch.setattr(inference_api, "LANDSLIDE_MODEL_WEIGHTS", None)
    first = predict_landslide(str(SAMPLE_IMAGE))
    second = predict_landslide(str(SAMPLE_IMAGE))
    assert first == second


REAL_MULTIBAND_TIF = REPO_ROOT / "data" / "study_areas" / "sunkoshi" / "validation" / "jure" / "post_breach.tif"


def test_predict_flood_on_real_multiband_tif_matches_frozen_interface():
    if not REAL_MULTIBAND_TIF.exists():
        pytest.skip("Real Landsat validation imagery not pulled - run pull_validation_imagery.py")
    detections = predict_flood(str(REAL_MULTIBAND_TIF))
    _assert_matches_frozen_interface(detections)
    assert len(detections) >= 1
    assert detections[0]["class"] == "flood"
    assert 0.0 <= detections[0]["confidence"] <= 1.0


def test_predict_flood_on_non_raster_input_returns_empty_not_fake():
    detections = predict_flood(str(SAMPLE_IMAGE))
    assert detections == []


def test_predict_flood_raises_if_weights_ever_set_without_real_trained_impl(monkeypatch):
    monkeypatch.setattr(inference_api, "FLOOD_MODEL_WEIGHTS", "some/path.pt")
    with pytest.raises(NotImplementedError):
        predict_flood(str(SAMPLE_IMAGE))

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from validation.validate_jure import CASES, polygon_iou, run_case


def test_polygon_iou_identical_squares_is_one():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_iou(square, square) == pytest.approx(1.0)


def test_polygon_iou_disjoint_squares_is_zero():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[100, 100], [110, 100], [110, 110], [100, 110]]
    assert polygon_iou(a, b) == 0.0


def test_polygon_iou_half_overlap():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[5, 0], [15, 0], [15, 10], [5, 10]]
    assert polygon_iou(a, b) == pytest.approx(50 / 150)


@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_run_case_does_not_crash(case_name):
    case = CASES[case_name]
    if not case["post_landslide_img"].exists():
        pytest.skip(f"{case_name} validation imagery not pulled - run src/data_ingestion/pull_validation_imagery.py first")
    result = run_case(case_name)
    assert result["case"] == case_name
    assert isinstance(result["n_scars"], int)


def test_jure_flood_detections_use_real_ndwi_not_a_fake_fixed_polygon():
    case = CASES["jure"]
    if not case["post_breach_tif"].exists():
        pytest.skip("Real Landsat post_breach.tif not pulled - run pull_validation_imagery.py")
    result = run_case("jure")
    assert len(result["flood_detections"]) >= 1
    for det in result["flood_detections"]:
        assert det["class"] == "flood"
        assert 0.0 <= det["confidence"] <= 1.0
        assert det["confidence"] != 0.79
        assert len(det["mask_polygon"]) >= 3


def test_control_case_has_no_flood_detections_since_it_has_no_post_breach_tif():
    case = CASES["control"]
    assert case["post_breach_tif"] is None
    if not case["post_landslide_img"].exists():
        pytest.skip("Control validation imagery not pulled - run pull_validation_imagery.py")
    result = run_case("control")
    assert result["flood_detections"] == []
    assert result["iou"] is None

"""Permanent regression coverage for app/pages/1_Landslide_Risk.py (CORE
module - no region/coordinate concept, terrain is plain sliders)."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_PATH = str(REPO_ROOT / "app" / "Home.py")
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fresh_page() -> AppTest:
    at = AppTest.from_file(HOME_PATH, default_timeout=90)
    at.run()
    at.switch_page("pages/1_Landslide_Risk.py")
    at.run()
    assert not at.exception, at.exception
    return at


def _select_example(at: AppTest, filename_fragment: str) -> AppTest:
    labels = at.selectbox[0].options
    idx = next(i for i, label in enumerate(labels) if filename_fragment in label)
    at.selectbox[0].set_value(at.selectbox[0].options[idx]).run()
    assert not at.exception, at.exception
    return at


def test_page_loads_without_exception():
    _fresh_page()


def test_curated_example_dropdown_has_34_tagged_entries():
    at = _fresh_page()
    labels = at.selectbox[0].options
    nepal = [l for l in labels if l.startswith("[Nepal]")]
    global_ = [l for l in labels if l.startswith("[Global]")]
    real_gee = [l for l in labels if l.startswith("[Real-GEE]")]
    assert len(nepal) == 10
    assert len(global_) == 10
    assert len(real_gee) == 14
    assert len(labels) == 34


def test_page_has_no_region_or_coordinate_widgets():
    at = _fresh_page()
    number_labels = [n.label for n in at.number_input]
    assert number_labels == ["Upstream drainage area (km²)", "Local relief (m)"]
    assert len(at.selectbox) == 1


def test_default_run_shows_dbi_verdicts():
    at = _fresh_page()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("ran end to end" in s.value for s in at.success)
    assert len(at.dataframe) >= 1


def test_custom_terrain_values_change_the_verdict():
    at = _fresh_page()
    _select_example(at, "image_825")
    at.number_input[0].set_value(0.05).run()
    at.number_input[1].set_value(5000.0).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("ran end to end" in s.value for s in at.success)


def test_zero_detection_example_shows_info_not_crash():
    at = _fresh_page()
    _select_example(at, "train_COMP_0_1")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("No landslide scars detected" in i.value for i in at.info)


def test_upload_path_runs_without_exception():
    at = _fresh_page()
    at.radio[0].set_value("Upload your own").run()
    img_bytes = (REPO_ROOT / "datasets" / "landslide_seg" / "test" / "images" / "image_825.png").read_bytes()
    at.get("file_uploader")[0].upload("my_tile.png", img_bytes, "image/png").run()
    assert not at.exception, at.exception
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("ran end to end" in s.value for s in at.success)


def _all_example_labels():
    at = _fresh_page()
    return list(at.selectbox[0].options)


@pytest.mark.parametrize("label", _all_example_labels())
def test_every_curated_example_runs_without_exception(label):
    at = _fresh_page()
    idx = at.selectbox[0].options.index(label)
    at.selectbox[0].set_value(at.selectbox[0].options[idx]).run()
    assert not at.exception, f"{label}: {at.exception}"
    at.button[0].click().run()
    assert not at.exception, f"{label}: {at.exception}"

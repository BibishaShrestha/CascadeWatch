"""Permanent regression coverage for app/pages/3_Historical_Validation.py."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_PATH = str(REPO_ROOT / "app" / "Home.py")
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fresh_page() -> AppTest:
    at = AppTest.from_file(HOME_PATH, default_timeout=90)
    at.run()
    at.switch_page("pages/3_Historical_Validation.py")
    at.run()
    assert not at.exception, at.exception
    return at


def test_page_loads_without_exception():
    _fresh_page()


def test_shows_both_jure_and_control_cases():
    at = _fresh_page()
    expander_labels = [e.label for e in at.expander]
    assert any(label.startswith("Jure") for label in expander_labels)
    assert any(label.startswith("Control") for label in expander_labels)

"""Run the shared offline smoke checks, including the rendered Streamlit app."""

from scripts.smoke_test import run_checks


def test_foundation_smoke():
    result = run_checks("configs/base.yaml")
    assert result["imports"] == "passed"
    assert result["device"] in {"cpu", "cuda"}
    assert result["streamlit_app"] == "passed (AppTest)"

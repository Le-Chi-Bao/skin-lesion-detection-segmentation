"""Check honest empty-state messaging and all checkpoint states."""

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.utils.config import PROJECT_ROOT, load_config


def test_app_renders_upload_and_result_sections():
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Skin Lesion Detection and Segmentation"
    assert len(app.get("file_uploader")) == 1
    assert any("Detection" in section.value for section in app.subheader)
    assert any("Segmentation" in section.value for section in app.subheader)
    assert sum("chưa được huấn luyện" in message.value for message in app.info) == 2
    assert len(app.get("image")) + len(app.get("imgs")) == 0


@pytest.mark.parametrize("exists", [False, True])
def test_checkpoint_presence_never_enables_inference(tmp_path, monkeypatch, exists):
    path = tmp_path / "not_a_real_model.pth"
    if exists:
        path.write_bytes(b"test file only; must never be loaded")
    config = replace(load_config(), detector_checkpoint=path, segmenter_checkpoint=path)
    monkeypatch.setattr("src.utils.config.load_config", lambda: config)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    if exists:
        assert sum("chưa nạp trọng số" in message.value for message in app.info) == 2
    else:
        assert len(app.warning) == 2
    assert len(app.get("image")) + len(app.get("imgs")) == 0


def test_uploaded_image_is_displayed(monkeypatch):
    uploaded = BytesIO()
    Image.fromarray(np.full((4, 7, 3), 60, dtype=np.uint8)).save(uploaded, format="PNG")
    uploaded.seek(0)
    uploaded.name = "synthetic.png"
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    # AppTest uses "image" in newer releases, "imgs" in older 1.x versions.
    assert len(app.get("image")) + len(app.get("imgs")) == 1
    assert any("7 × 4" in message.value for message in app.caption)


def test_invalid_upload_shows_error(monkeypatch):
    uploaded = BytesIO(b"invalid png contents")
    uploaded.name = "broken.png"
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert len(app.error) == 1
    assert "Không thể đọc ảnh" in app.error[0].value

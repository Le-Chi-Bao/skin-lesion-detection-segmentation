import asyncio
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from urllib.error import URLError
from urllib.request import urlopen

import gradio as gr
import pytest
from gradio.state_holder import SessionState as GradioSessionState
from PIL import Image

from app.config import PROJECT_ROOT, AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.temporary_files import TemporaryFiles
from app.cv import segformer
from app.cv.availability import ModelStatus, RunStatus
from app.cv.checkpoint import CHECKPOINT_ID, SHA256
from app.cv.schemas import CVResult
from app.ui.shell import build_shell


def app_callbacks(config, monkeypatch):
    files = TemporaryFiles(config.runtime_dir, config.session_ttl_seconds)
    files.cache_dir.mkdir(parents=True)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(files.cache_dir))
    model = segformer.SegFormer(config)
    demo = build_shell(config, files, model)
    deps = {dep["api_name"]: dep for dep in demo.config["dependencies"]}
    state_id = deps["upload"]["outputs"][0]

    async def call(event, session, *args):
        return (await demo.process_api(deps[event]["id"], list(args), state=session))["data"]

    return files, model, demo, state_id, call


def upload_payload(files, width):
    path = files.cache_dir / f"{width}.png"
    with Image.new("RGB", (width, 12)) as image:
        image.save(path)
    return gr.FileData(path=str(path)).model_dump()


def assert_unchanged_updates(updates):
    for update in updates:
        assert update["__type__"] == "update"
        assert set(update) <= {"__type__", "props"}
        assert set(update.get("props", {})) <= {"__type__"}


@pytest.fixture
def callback_app(tmp_path, monkeypatch):
    monkeypatch.setattr(segformer, "_load_verified_model", lambda config: (object(), "cpu"))
    return app_callbacks(AppConfig(runtime_dir=tmp_path, segformer_checkpoint=tmp_path / "unit-test.pt"), monkeypatch)


@pytest.mark.parametrize("edit", ["upload", "reset"])
def test_stale_inference_cannot_replace_edit_even_with_duplicate_request(callback_app, monkeypatch, edit):
    files, model, demo, state_id, call = callback_app
    session = GradioSessionState(demo)
    entered, release = Event(), Event()
    calls = []

    def predict(image):
        calls.append(image.size)
        entered.set()
        assert release.wait(10)
        return segformer.SegmentationPrediction(Image.new("L", image.size), 1)

    monkeypatch.setattr(model, "predict", predict)

    async def race():
        await call("upload", session, upload_payload(files, 25), None, 1)
        current = session[state_id]
        image_path = files.root / current.session_id.hex / current.image.temporary_file
        await call("begin_analysis", session, None, 1)
        working = asyncio.create_task(call("run_analysis", session, None))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert current.processing_status is RunStatus.RUNNING
            assert_unchanged_updates(await call("begin_analysis", session, None, 1))
            await call("run_analysis", session, None)
            if edit == "upload":
                await call("upload", session, upload_payload(files, 31), None, 2)
            else:
                await call("reset", session, None, 2)
            assert image_path.is_file()
            assert current.result is None and current.active_run is None
        finally:
            release.set()
            await working
        assert_unchanged_updates(await call("show_analysis", session, None, 2))
        assert calls == [(25, 12)]
        assert current.result is None
        assert not image_path.exists()
        retained = list(files.root.glob("*/*.png"))
        if edit == "upload":
            assert current.image.metadata.width == 31
            assert [path.name for path in retained] == [current.image.temporary_file]
        else:
            assert current.image is None and not retained

    asyncio.run(race())


def test_completed_failure_empty_mask_and_session_isolation(callback_app, monkeypatch):
    files, model, demo, state_id, call = callback_app
    first, second = GradioSessionState(demo), GradioSessionState(demo)

    def predict(image):
        mask = Image.new("L", image.size)
        if image.width == 25:
            mask.putpixel((24, 11), 255)
        return segformer.SegmentationPrediction(mask, 12.5)

    def fail(image):
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)

    async def exercise():
        for session, width in ((first, 25), (second, 31)):
            await call("upload", session, upload_payload(files, width), None, 1)
            started = await call("begin_analysis", session, None, 1)
            assert "Đang phân tích" in started[2]
            await call("run_analysis", session, None)
            assert_unchanged_updates(await call("show_analysis", session, None, 2))
            shown = await call("show_analysis", session, None, 1)
            assert "data:image/png;base64," in shown[1]
            assert CHECKPOINT_ID in shown[2]

        current = first[state_id]
        result = current.result
        assert result.metadata.status is result.segmentation.status is RunStatus.COMPLETED
        assert result.metadata.inference_duration_ms == 12.5
        assert result.segmentation.threshold == 0.7
        assert result.segmentation.model.checkpoint_sha256 == SHA256
        assert result.detection.status is RunStatus.NOT_AVAILABLE
        assert result.localization.source == "derived_from_mask"
        assert tuple(result.localization.box.model_dump().values()) == (24, 11, 25, 12)
        assert CVResult.model_validate_json(result.model_dump_json()) == result
        reference = result.segmentation.mask.reference
        with pytest.raises(AppError) as error, files.open_mask(second[state_id].session_id, reference):
            pytest.fail("A second session could read the first session's mask")
        assert error.value.code is ErrorCode.MISSING_FILE

        empty = second[state_id].result
        assert empty.localization is None and empty.metadata.warning_codes == ("empty_mask",)
        assert empty.segmentation.warnings
        await call("upload", second, upload_payload(files, 32), None, 2)
        assert second[state_id].result is None
        with pytest.raises(AppError), files.open_mask(second[state_id].session_id, empty.segmentation.mask.reference):
            pytest.fail("Replacing an image retained its previous mask")

        monkeypatch.setattr(model, "predict", fail)
        await call("begin_analysis", first, None, 1)
        await call("run_analysis", first, None)
        failed = await call("show_analysis", first, None, 1)
        assert "sai kích thước" in failed[2] and "data:image/png" not in failed[1]
        assert current.result.metadata.error_code is ErrorCode.INVALID_MODEL_OUTPUT
        assert current.result.segmentation.status is RunStatus.FAILED and current.result.segmentation.mask is None
        with pytest.raises(AppError), files.open_mask(current.session_id, reference):
            pytest.fail("Rerunning retained the previous mask")

        image = current.image
        with files.operation(current.session_id):
            files.discard(current.session_id, image.temporary_file)
        await call("begin_analysis", first, None, 1)
        await call("run_analysis", first, None)
        assert current.result.metadata.error_code is ErrorCode.MISSING_FILE
        await call("reset", first, None, 2)
        assert current.result is None and current.image is None

    monkeypatch.setattr(model, "predict", predict)
    asyncio.run(exercise())


def test_incompatible_checkpoint_keeps_upload_available(tmp_path, monkeypatch):
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"invalid")
    config = AppConfig(runtime_dir=tmp_path, segformer_checkpoint=invalid)
    files, model, demo, state_id, call = app_callbacks(config, monkeypatch)
    assert model.availability.status is ModelStatus.FAILED
    assert model.error.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
    session = GradioSessionState(demo)
    output = asyncio.run(call("upload", session, upload_payload(files, 24), None, 1))
    assert session[state_id].image.metadata.width == 24
    assert output[6]["interactive"] is False and "SHA-256" in output[8]


def test_mask_write_failure_clears_the_run_without_a_preview(callback_app, monkeypatch):
    files, model, demo, state_id, call = callback_app
    session = GradioSessionState(demo)

    def predict(image):
        return segformer.SegmentationPrediction(Image.new("L", image.size), 1)

    def disk_error(session_id, writer):
        raise AppError(ErrorCode.TEMPORARY_FILE)

    monkeypatch.setattr(model, "predict", predict)
    monkeypatch.setattr(files, "write_mask", disk_error)

    async def exercise():
        await call("upload", session, upload_payload(files, 25), None, 1)
        await call("begin_analysis", session, None, 1)
        await call("run_analysis", session, None)
        shown = await call("show_analysis", session, None, 1)
        assert "tệp tạm" in shown[2] and "data:image/png" not in shown[1]
        current = session[state_id]
        assert current.active_run is None and current.result.metadata.error_code is ErrorCode.TEMPORARY_FILE
        assert current.result.segmentation.mask is None
        assert [path.name for path in files.root.glob("*/*.png")] == [current.image.temporary_file]

    asyncio.run(exercise())


@pytest.fixture(scope="module")
def published_sample():
    checkpoint = os.environ.get("WP2_SMOKE_CHECKPOINT")
    image = os.environ.get("WP2_SMOKE_IMAGE")
    if not checkpoint and not image:
        pytest.skip("Đặt WP2_SMOKE_CHECKPOINT và WP2_SMOKE_IMAGE để kiểm tra checkpoint thật")
    assert checkpoint and image, "Cần cả WP2_SMOKE_CHECKPOINT và WP2_SMOKE_IMAGE"
    return Path(checkpoint).resolve(strict=True), Path(image).resolve(strict=True), os.getenv("WP2_SMOKE_DEVICE", "cpu")


def test_real_checkpoint_app_callbacks(published_sample, tmp_path, monkeypatch):
    import torch

    checkpoint, sample, device = published_sample
    files, model, demo, state_id, call = app_callbacks(
        AppConfig(runtime_dir=tmp_path, segformer_checkpoint=checkpoint, segformer_device=device), monkeypatch,
    )
    assert model.availability.status is ModelStatus.READY, str(model.error)
    session = GradioSessionState(demo)
    uploaded = files.cache_dir / sample.name
    uploaded.write_bytes(sample.read_bytes())
    observed = []

    def observe_logits(module, inputs, logits):
        observed.append((tuple(logits.shape), torch.isfinite(logits).all().item(), torch.is_inference_mode_enabled()))

    async def exercise():
        await call("upload", session, gr.FileData(path=str(uploaded)).model_dump(), None, 1)
        await call("begin_analysis", session, None, 1)
        await call("run_analysis", session, None)
        displayed = await call("show_analysis", session, None, 1)
        assert "data:image/png;base64," in displayed[1]
        current = session[state_id]
        result = current.result
        assert result.metadata.status is result.segmentation.status is RunStatus.COMPLETED
        assert result.detection.status is RunStatus.NOT_AVAILABLE
        assert result.segmentation.model.checkpoint_sha256 == SHA256
        assert result.segmentation.model.checkpoint_id == CHECKPOINT_ID
        assert result.segmentation.threshold == 0.7
        assert CVResult.model_validate_json(result.model_dump_json()) == result
        reference = result.segmentation.mask
        with files.open_mask(current.session_id, reference.reference) as stream, Image.open(stream) as mask:
            assert mask.size == (current.image.metadata.width, current.image.metadata.height)
            assert mask.size == (reference.width, reference.height)
            histogram = mask.histogram()
            assert histogram[0] + histogram[255] == mask.width * mask.height
            assert histogram[255] > 0, "Ảnh mẫu phải cho thấy vùng tiền cảnh để kiểm tra box thật"
            box = tuple(result.localization.box.model_dump().values())
            assert result.localization.source == "derived_from_mask" and box == mask.getbbox()
            data = displayed[1].split("data:image/png;base64,", 1)[1].split('"', 1)[0]
            with Image.open(io.BytesIO(base64.b64decode(data))) as preview:
                assert preview.tobytes() == mask.tobytes()
        print(json.dumps({
            "sha256": SHA256, "device": result.segmentation.model.device,
            "inference_ms": result.metadata.inference_duration_ms, "logits_shape": observed[0][0],
            "mask_size": [reference.width, reference.height], "foreground_pixels": histogram[255], "box": box,
        }))
        await call("reset", session, None, 2)
        assert current.result is None and not list(files.root.glob("*/*.png"))

    with model._model.register_forward_hook(observe_logits):
        asyncio.run(exercise())
    assert observed == [((1, 1, 512, 512), True, True)]


@pytest.fixture
def real_browser(published_sample, tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    checkpoint, sample, device = published_sample
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("LESION_")}
    env.update(
        LESION_PORT=str(port), LESION_RUNTIME_DIR=str(tmp_path), PYTHONUTF8="1",
        LESION_SEGFORMER_CHECKPOINT=str(checkpoint), LESION_SEGFORMER_DEVICE=device,
    )
    url = f"http://127.0.0.1:{port}"
    with playwright_api.sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            pytest.skip("Cần Chromium từ playwright install chromium")
        with (tmp_path / "app.log").open("w+", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "app.main"], cwd=PROJECT_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        with urlopen(f"{url}/config", timeout=2) as response:
                            assert response.status == 200
                            break
                    except (URLError, TimeoutError):
                        time.sleep(0.2)
                else:
                    log.seek(0)
                    pytest.fail(f"Gradio không khởi động\n{log.read()}")

                browser = playwright.chromium.launch()
                try:
                    yield browser, url, sample
                finally:
                    browser.close()
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)


def test_real_checkpoint_browser(real_browser):
    from playwright.sync_api import expect

    browser, url, sample = real_browser
    page = browser.new_page()
    assert page.goto(url).status == 200
    page.get_by_role("radio", name="Computer Vision").check()
    analyze = page.get_by_role("button", name="Phân tích")
    expect(analyze).to_be_disabled()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Tải ảnh lên").click()
    chooser.value.set_files(sample)
    expect(analyze).to_be_enabled()
    analyze.click()
    completed = page.get_by_text("Phân đoạn hoàn tất", exact=False)
    expect(completed).to_be_visible(timeout=120_000)
    expect(page.get_by_text(CHECKPOINT_ID, exact=True)).to_be_visible()
    page.get_by_role("tab", name="Mask").click()
    mask = page.get_by_role("img", name="Mặt nạ tổn thương nhị phân")
    expect(mask).to_be_visible()
    original = page.locator("#lesion-preview img")
    assert mask.evaluate("(img) => [img.naturalWidth, img.naturalHeight]") == original.evaluate(
        "(img) => [img.naturalWidth, img.naturalHeight]",
    )

    other = browser.new_page()
    other.goto(url)
    other.get_by_role("radio", name="Computer Vision").check()
    other.get_by_role("tab", name="Mask").click()
    expect(other.get_by_role("img", name="Mặt nạ tổn thương nhị phân")).to_have_count(0)

    page.get_by_role("button", name="Đặt lại").click()
    expect(analyze).to_be_disabled()
    expect(completed).to_have_count(0)
    page.get_by_role("tab", name="Mask").click()
    expect(mask).to_have_count(0)

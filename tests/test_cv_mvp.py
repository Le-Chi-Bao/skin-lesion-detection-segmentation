import asyncio
import io
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import gradio as gr
import pytest
from gradio.state_holder import SessionState as GradioSessionState
from PIL import Image

from app.config import AppConfig, PROJECT_ROOT
from app.core.errors import AppError, ErrorCode
from app.core.image_validation import ImageMetadata
from app.core.session import SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.adapters import DetectionOutput, SegmentationPrediction
from app.cv.analysis import execute_analysis
from app.cv.availability import ModelAvailability, ModelStatus, RunStatus
from app.cv.orchestrator import CVOrchestrator, create_orchestrator
from app.cv.registry import ModelRegistry, UnavailableAdapter
from app.cv.schemas import BoundingBox, CVResult, DetectionPrediction, ModelIdentity
from app.cv.visualization import BOX_COLOR, DETECTION_COLOR
from app.ui.cv_page import render_result_images
from app.ui.png_export import ExportRequest, PNGExports
from app.ui.shell import build_shell


def adapter(module, state):
    if state in {"off", "bad_config"}:
        return UnavailableAdapter(module, AppError(ErrorCode.INVALID_MODEL_CONFIG) if state == "bad_config" else None)

    def predict(image):
        if state == "fail":
            raise AppError(ErrorCode.INFERENCE_FAILED)
        if module == "segmentation":
            mask = Image.new("L", image.size)
            mask.paste(255, (2, 3, 10, 12))
            return SegmentationPrediction(mask, 3.0)
        return DetectionOutput((DetectionPrediction(box=BoundingBox(x1=14, y1=15, x2=25, y2=27), score=0.8),), 4.0)

    return SimpleNamespace(
        load=lambda: None, predict=predict, error=None,
        identity=ModelIdentity(model_id=module, checkpoint_id="unit-v1", checkpoint_sha256="a" * 64, device="cpu"),
        availability=ModelAvailability(module, ModelStatus.READY, "unit-v1"),
    )


def session(files, owner="A" * 43):
    current = SessionState(browser_owner=owner)
    current.browser_proof = "1" * 64
    current.accept_image_action(1)
    with files.operation(current.session_id), Image.new("RGB", (40, 30), (80, 90, 100)) as image:
        source = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
    current.replace_image(ImageMetadata(40, 30, "PNG", 100, "a" * 64), source)
    current.begin_run()
    return current


@pytest.mark.parametrize("seg,det,status", [
    ("ok", "ok", RunStatus.COMPLETED), ("ok", "fail", RunStatus.PARTIAL),
    ("fail", "ok", RunStatus.PARTIAL), ("ok", "off", RunStatus.COMPLETED),
    ("off", "ok", RunStatus.COMPLETED), ("off", "off", RunStatus.NOT_AVAILABLE),
    ("fail", "fail", RunStatus.FAILED), ("ok", "bad_config", RunStatus.PARTIAL),
    ("bad_config", "ok", RunStatus.PARTIAL),
])
def test_independent_modules_and_aggregate(tmp_path, seg, det, status):
    files = TemporaryFiles(tmp_path, 30)
    current = session(files)
    execute_analysis(current, files, CVOrchestrator(adapter("segmentation", seg), adapter("detection", det)))
    result = current.result
    assert result.metadata.status is status and current.active_run is None
    assert bool(result.segmentation.mask) == (seg == "ok")
    assert bool(result.detection.boxes) == (det == "ok")
    assert bool(result.localization) == (seg == "ok")
    assert CVResult.model_validate_json(result.model_dump_json()) == result
    if seg == det == "ok":
        assert result.metadata.inference_duration_ms == 7
        assert result.localization.box != result.detection.boxes[0].box
    if "fail" in (seg, det):
        failed = result.segmentation if seg == "fail" else result.detection
        assert failed.error_code is ErrorCode.INFERENCE_FAILED and failed.inference_duration_ms >= 0


def test_invalid_detector_output_preserves_successful_mask(tmp_path):
    files = TemporaryFiles(tmp_path, 30)
    current = session(files)
    detector = adapter("detection", "ok")
    detector.predict = lambda image: DetectionOutput((DetectionPrediction(box=BoundingBox(x1=0, y1=0, x2=41, y2=10), score=0.9),), 1)
    execute_analysis(current, files, CVOrchestrator(adapter("segmentation", "ok"), detector))
    assert current.result.metadata.status is RunStatus.PARTIAL
    assert current.result.segmentation.mask is not None
    assert current.result.detection.error_code is ErrorCode.INVALID_MODEL_OUTPUT
    assert not current.result.detection.boxes


def test_mask_storage_failure_preserves_successful_detection(tmp_path, monkeypatch):
    files = TemporaryFiles(tmp_path, 30)
    current = session(files)

    def full_disk(*args):
        raise AppError(ErrorCode.TEMPORARY_FILE)

    monkeypatch.setattr(files, "write_mask", full_disk)
    execute_analysis(current, files, CVOrchestrator(adapter("segmentation", "ok"), adapter("detection", "ok")))
    assert current.result.metadata.status is RunStatus.PARTIAL
    assert current.result.segmentation.mask is None
    assert current.result.segmentation.error_code is ErrorCode.TEMPORARY_FILE
    assert current.result.detection.boxes


@pytest.mark.parametrize("seg,det", [("ok", "ok"), ("off", "ok"), ("ok", "fail")])
def test_combined_and_detection_only_exports_are_authorized_full_resolution(tmp_path, seg, det):
    files = TemporaryFiles(tmp_path, 30)
    current = session(files)
    execute_analysis(current, files, CVOrchestrator(adapter("segmentation", seg), adapter("detection", det)))
    exports = PNGExports(files)
    exports.issue(current, current.result)
    requested = ExportRequest(session=current.session_id, run=current.result.run_id, action=1, proof=current.browser_proof, kind="localization", opacity=40)
    assert exports.response("B" * 43, requested).status_code == 404
    response = exports.response(current.browser_owner, requested)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    prefix = "cv" if det == "ok" else "segformer"
    filename = f"{prefix}-localization-{current.result.run_id.hex[:12]}.png"
    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
    from app.ui.cv_page import export_panel
    assert f'data-filename="{filename}"' in export_panel(current, current.result, 1)
    with Image.open(io.BytesIO(response.body)) as image:
        assert image.size == (40, 30)
        if det == "ok":
            assert image.getpixel((24, 26)) == DETECTION_COLOR
        if seg == "ok":
            assert image.getpixel((9, 11)) == BOX_COLOR
    overlay = exports.response(current.browser_owner, requested.model_copy(update={"kind": "overlay"}))
    assert overlay.status_code == (200 if seg == "ok" else 404)
    views = render_result_images(files, current, current.image.temporary_file, current.result)
    assert "data:image/png" in views[2]
    assert ("lesion-layer" in views[1]) == (seg == "ok")
    current.reset()
    assert exports.response(current.browser_owner, requested).status_code == 404


@pytest.mark.parametrize("edit", ["reset", "upload"])
def test_late_detector_never_restores_old_mask_or_boxes(tmp_path, edit):
    from concurrent.futures import ThreadPoolExecutor

    files = TemporaryFiles(tmp_path, 30)
    current = session(files)
    detector = adapter("detection", "ok")
    predict = detector.predict
    entered, release = Event(), Event()

    def delayed(image):
        entered.set()
        assert release.wait(10)
        return predict(image)

    detector.predict = delayed
    with ThreadPoolExecutor(1) as workers:
        task = workers.submit(execute_analysis, current, files, CVOrchestrator(adapter("segmentation", "ok"), detector))
        try:
            assert entered.wait(5)
            assert len(list(files.root.glob("*/*.png"))) == 2
            with current.lock, files.operation(current.session_id):
                source = current.image.temporary_file
                if edit == "reset":
                    current.reset()
                else:
                    current.replace_image(current.image.metadata, source)
                if edit == "reset":
                    files.discard(current.session_id, source)
        finally:
            release.set()
        task.result(10)
    assert current.result is None and current.active_run is None
    assert len(list(files.root.glob("*/*.png"))) == (1 if edit == "upload" else 0)


def registry_data():
    return json.loads((PROJECT_ROOT / "config/models.json").read_text())


def write_registry(tmp_path, data):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_selection_and_rollback_without_loading_inactive_models(tmp_path):
    data = registry_data()
    previous = data["models"][1]
    future = dict(previous, version="E2-final", active=False, checkpoint_path="does-not-exist.pt", sha256="b" * 64)
    data["models"].append(future)
    path = write_registry(tmp_path, data)
    old_registry = ModelRegistry(path)
    assert old_registry.selected("detection").version == "E1-best-completed"
    previous["active"], future["active"] = False, True
    write_registry(tmp_path, data)
    assert ModelRegistry(path).selected("detection").version == "E2-final"
    assert old_registry.selected("detection").version == "E1-best-completed"
    previous["active"], future["active"] = True, False
    write_registry(tmp_path, data)
    assert ModelRegistry(path).selected("detection").version == "E1-best-completed"


def test_startup_instantiates_only_active_detector(tmp_path, monkeypatch):
    from app.cv import detection

    data = registry_data()
    data["models"][0]["active"] = False
    data["models"].append(dict(data["models"][1], version="E3-final", active=False, checkpoint_path="missing-E3.pt"))
    loaded = []

    def construct(entry, *args):
        loaded.append(entry.version)
        return adapter("detection", "ok")

    monkeypatch.setattr(detection, "FasterRCNN", construct)
    models = create_orchestrator(AppConfig(model_registry=write_registry(tmp_path, data)))
    assert loaded == ["E1-best-completed"]
    assert models.segmentation.availability.status is ModelStatus.NOT_AVAILABLE


@pytest.mark.parametrize("bad", ["hash", "duplicate_active", "type"])
def test_bad_detector_configuration_does_not_poison_segmentation(tmp_path, bad):
    data = registry_data()
    if bad == "hash":
        data["models"][1]["sha256"] = "wrong"
    elif bad == "duplicate_active":
        data["models"][2]["active"] = True
    else:
        data["models"][1]["active"] = "yes"
    registry = ModelRegistry(write_registry(tmp_path, data))
    assert registry.selected("segmentation").model_id == "segformer-mitb2"
    with pytest.raises(AppError):
        registry.selected("detection")
    assert registry.create("detection", AppConfig()).availability.status is ModelStatus.FAILED


@pytest.mark.parametrize("bad", [[], {"schema_version": 2, "models": []}, {"schema_version": 1, "models": [{"module": []}]}])
def test_malformed_registry_reports_readiness_instead_of_crashing(tmp_path, bad):
    registry = ModelRegistry(write_registry(tmp_path, bad))
    for module in ("segmentation", "detection"):
        with pytest.raises(AppError):
            registry.selected(module)


@pytest.mark.parametrize("seg_state,det_state", [("off", "ok"), ("off", "fail"), ("fail", "fail")])
def test_detection_only_gradio_uses_same_callbacks_and_clears_outputs(tmp_path, monkeypatch, seg_state, det_state):
    files = TemporaryFiles(tmp_path, 30)
    files.cache_dir.mkdir(parents=True)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(files.cache_dir))
    models = CVOrchestrator(adapter("segmentation", seg_state), adapter("detection", det_state))
    demo = build_shell(AppConfig(runtime_dir=tmp_path), files, orchestrator=models)
    deps = {dep["api_name"]: dep for dep in demo.config["dependencies"]}
    state_id = deps["upload"]["outputs"][0]
    current = GradioSessionState(demo)
    path = files.cache_dir / "synthetic.png"
    with Image.new("RGB", (40, 30)) as image:
        image.save(path)

    async def call(name, *args):
        return (await demo.process_api(deps[name]["id"], list(args), state=current))["data"]

    async def exercise():
        uploaded = await call("upload", gr.FileData(path=str(path)).model_dump(), None, 1)
        assert uploaded[6]["interactive"] is True
        await call("begin_analysis", None, 1)
        await call("run_analysis", None)
        shown = await call("show_analysis", None, 1)
        if det_state == "ok":
            assert "data:image/png" in shown[3] and "data:image" not in shown[1]
            assert shown[-1]["selected"] == "localization"
        else:
            assert "data:image" not in shown[3] and "data:image" not in shown[1]
            assert "Faster R-CNN: Suy luận Faster R-CNN thất bại" in shown[2]
            assert ("SegFormer" in shown[2]) == (seg_state == "fail")
        assert shown[5]["interactive"] is False
        assert "detection" in shown[6]
        reset = await call("reset", None, 2)
        assert current[state_id].result is None
        assert "data:image" not in reset[9]

    asyncio.run(exercise())


@pytest.fixture(scope="module")
def real_models():
    path = os.getenv("CV_SMOKE_REGISTRY")
    if not path:
        pytest.skip("Đặt CV_SMOKE_REGISTRY và WP2_SMOKE_IMAGE để chạy hai checkpoint thật")
    models = create_orchestrator(AppConfig(model_registry=Path(path), segformer_device="cpu", detection_device="cpu"))
    assert models.segmentation.availability.status is ModelStatus.READY, str(models.segmentation.error)
    assert models.detection.availability.status is ModelStatus.READY, str(models.detection.error)
    return models


def test_real_models_coexist_and_export(real_models, tmp_path):
    from app.core.image_validation import validate_image

    sample = Path(os.environ["WP2_SMOKE_IMAGE"])
    config = AppConfig(runtime_dir=tmp_path)
    files = TemporaryFiles(tmp_path, 30)
    current = SessionState(browser_owner="A" * 43)
    current.browser_proof = "1" * 64
    current.accept_image_action(1)
    image, metadata = validate_image(sample, config)
    with image, files.operation(current.session_id):
        source = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
    current.replace_image(metadata, source)
    current.begin_run()
    execute_analysis(current, files, real_models)
    result = current.result
    assert result.metadata.status is RunStatus.COMPLETED
    assert result.segmentation.mask and result.segmentation.threshold == 0.7
    assert result.detection.status is RunStatus.COMPLETED and result.detection.boxes
    assert result.detection.model.artifact_format == "isic-fasterrcnn-inference-v1"
    assert result.localization.source == "derived_from_mask"
    assert result.detection.inference_duration_ms > 0 and result.segmentation.inference_duration_ms > 0
    exports = PNGExports(files)
    exports.issue(current, result)
    request = ExportRequest(session=current.session_id, run=result.run_id, action=1, proof=current.browser_proof, kind="localization", opacity=40)
    response = exports.response(current.browser_owner, request)
    assert response.status_code == 200
    with Image.open(io.BytesIO(response.body)) as png:
        assert png.size == (metadata.width, metadata.height)
    print(json.dumps({
        "segmentation_ms": result.segmentation.inference_duration_ms, "detection_ms": result.detection.inference_duration_ms,
        "detection_sha256": result.detection.model.checkpoint_sha256,
        "boxes": [p.model_dump() for p in result.detection.boxes], "size": [metadata.width, metadata.height],
    }))


def test_real_pilot_artifact_when_available():
    from app.cv.detection import FasterRCNN, PILOT_SHA256
    from app.cv.registry import RegistryEntry

    data = registry_data()
    raw = next(row for row in data["models"] if row["research_pilot"])
    path = PROJECT_ROOT / raw["checkpoint_path"]
    if not path.is_file():
        pytest.skip("BLOCKED: thiếu E1_step07200_PILOT_UNFINISHED.pt đúng SHA-256 yêu cầu")
    detector = FasterRCNN(RegistryEntry.model_validate(raw), "cpu", pilot_mode=True)
    detector.load()
    assert detector.availability.status is ModelStatus.READY, str(detector.error)
    assert detector.identity.checkpoint_sha256 == PILOT_SHA256 and detector.identity.research_pilot
    sample = os.environ.get("WP2_SMOKE_IMAGE")
    assert sample, "Cần WP2_SMOKE_IMAGE để kiểm chứng pilot thật"
    with Image.open(sample) as image:
        output = detector.predict(image.convert("RGB"))
    assert output.duration_ms > 0


@pytest.mark.parametrize("entrypoint", ["module", "root"])
def test_real_dual_model_browser_upload_export_reset(tmp_path, entrypoint):
    import socket
    import subprocess
    import sys
    import time
    from urllib.error import URLError
    from urllib.parse import quote
    from urllib.request import urlopen

    registry = os.environ.get("CV_SMOKE_REGISTRY")
    sample = os.environ.get("WP2_SMOKE_IMAGE")
    if not registry or not sample:
        pytest.skip("Đặt CV_SMOKE_REGISTRY và WP2_SMOKE_IMAGE để kiểm tra trình duyệt thật")
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            pytest.skip("Chưa cài Chromium")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {key: value for key, value in os.environ.items() if not key.startswith("LESION_")}
        env.update(
            LESION_RUNTIME_DIR=str(tmp_path), LESION_PORT=str(port), PYTHONUTF8="1",
        )
        command = [sys.executable, "-u", "main.py"]
        if entrypoint == "module":
            env.update(
                LESION_MODEL_REGISTRY=str(Path(registry).resolve()),
                LESION_SEGFORMER_DEVICE="cpu", LESION_DETECTION_DEVICE="cpu",
            )
            command = [sys.executable, "-u", "-m", "app.main"]
        url = f"http://127.0.0.1:{port}"
        with (tmp_path / "server.log").open("w+", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urlopen(url + "/config", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        time.sleep(0.2)
                else:
                    log.seek(0)
                    pytest.fail(log.read())
                log.seek(0)
                startup = log.read()
                expected_registry = Path(registry).resolve() if entrypoint == "module" else PROJECT_ROOT / "config/models.json"
                assert f"Registry: {expected_registry}" in startup
                assert "SegFormer: READY | version=checkpoint-v1 | device=" in startup
                assert "Faster R-CNN: READY | version=E1-best-completed | device=" in startup
                assert url in startup
                browser = playwright.chromium.launch()
                try:
                    first = browser.new_context(accept_downloads=True)
                    second = browser.new_context(accept_downloads=True)
                    page, other = first.new_page(), second.new_page()
                    expect = playwright_api.expect
                    assert page.goto(url).status == other.goto(url).status == 200
                    for tab in (page, other):
                        tab.get_by_role("radio", name="Computer Vision").check()
                    with page.expect_file_chooser() as choose:
                        page.get_by_role("button", name="Tải ảnh lên").click()
                    choose.value.set_files(sample)
                    analyze = page.get_by_role("button", name="Phân tích")
                    expect(analyze).to_be_enabled()
                    analyze.click()
                    expect(page.locator("#segmentation-status")).to_contain_text("Phân đoạn hoàn tất", timeout=120_000)
                    expect(page.locator("#lesion-overlay .lesion-layer")).to_be_visible()
                    slider = page.locator("#overlay-opacity input[type=range]")
                    slider.fill("65")
                    expect(page.locator("#lesion-overlay .lesion-layer")).to_have_css("opacity", "0.65")
                    page.get_by_role("tab", name="Localization").click()
                    expect(page.locator("#lesion-localization img")).to_be_visible()
                    expect(page.locator("#lesion-localization")).to_contain_text("Faster R-CNN: detection độc lập")
                    page.get_by_role("button", name="Chi tiết mô hình", exact=False).click()
                    expect(page.locator("#lesion-model-details")).to_contain_text("E1-best-completed")
                    expect(page.locator("#lesion-model-details")).to_contain_text("isic-fasterrcnn-inference-v1")
                    expect(page.locator("#lesion-model-details")).to_contain_text("Số detection")
                    page.screenshot(path=str(tmp_path / "dual-model-localization.png"), full_page=True)
                    requests = []
                    for label in ("Tải Overlay PNG", "Tải Localization PNG"):
                        with page.expect_download() as download, page.expect_request("**/lesion-export") as request:
                            page.get_by_role("button", name=label).click()
                        requests.append(request.value.post_data_json)
                        expected_prefix = "cv" if label == "Tải Localization PNG" else "segformer"
                        kind = request.value.post_data_json["kind"]
                        run = request.value.post_data_json["run"]
                        assert download.value.suggested_filename == f"{expected_prefix}-{kind}-{run[:12]}.png"
                        with Image.open(download.value.path()) as exported, Image.open(sample) as source:
                            assert exported.size == source.size and exported.format == "PNG"
                    assert second.request.post(url + "/lesion-export", data=requests[1]).status == 404
                    other.get_by_role("tab", name="Localization").click()
                    expect(other.locator("#lesion-localization img")).to_have_count(0)
                    for row in registry_data()["models"]:
                        assert first.request.get(url + "/gradio_api/file=" + quote(str(PROJECT_ROOT / row["checkpoint_path"]), safe="")).status == 403
                    # Replacing the image revokes both exports immediately after upload.
                    with page.expect_file_chooser() as choose:
                        page.get_by_role("button", name="Tải ảnh lên").click()
                    choose.value.set_files(sample)
                    expect(page.get_by_role("button", name="Tải Overlay PNG")).to_be_disabled()
                    assert first.request.post(url + "/lesion-export", data=requests[1]).status == 404
                    analyze.click()
                    expect(page.locator("#segmentation-status")).to_contain_text("Phân đoạn hoàn tất", timeout=120_000)
                    page.get_by_role("button", name="Đặt lại").click()
                    expect(analyze).to_be_disabled()
                    expect(slider).to_be_disabled()
                    for label in ("Original", "Mask", "Overlay", "Localization"):
                        page.get_by_role("tab", name=label).click()
                        expect(page.get_by_role("tabpanel").locator("img")).to_have_count(0)
                    assert first.request.post(url + "/lesion-export", data=requests[0]).status == 404
                    page.screenshot(path=str(tmp_path / "dual-model-reset.png"))
                finally:
                    browser.close()
            finally:
                process.terminate()
                process.wait(timeout=10)

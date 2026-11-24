import asyncio
import base64
import io
from threading import Event
from urllib.parse import quote

import pytest
from gradio.state_holder import SessionState as GradioSessionState
from PIL import Image, ImageChops, ImageOps
from test_wp2_application import (
    app_callbacks, assert_unchanged_updates, callback_app, published_sample, real_browser, upload_payload,
)

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.cv.segformer import SegmentationPrediction
from app.cv.visualization import BOX_COLOR, DEFAULT_OPACITY, OVERLAY_COLOR
from app.ui import cv_page


@pytest.mark.parametrize("configured", [False, True])
def test_upload_without_an_available_checkpoint(tmp_path, monkeypatch, configured):
    path = tmp_path / "missing.pt" if configured else None
    files, _, demo, _, call = app_callbacks(AppConfig(runtime_dir=tmp_path, segformer_checkpoint=path), monkeypatch)
    shown = asyncio.run(call("upload", GradioSessionState(demo), upload_payload(files, 25), None, 1))
    assert "data:image/jpeg" in shown[4]
    assert shown[6]["interactive"] is False
    assert "Chưa có checkpoint SegFormer" in shown[8]
    assert "data:image" not in shown[10]
    assert shown[11]["interactive"] is False
    assert shown[14].count("disabled") == 2


def test_result_details_and_visualization_are_cleared(callback_app, monkeypatch):
    files, model, demo, state_id, call = callback_app
    session = GradioSessionState(demo)

    def predict(image):
        if image.width == 25:
            return SegmentationPrediction(Image.new("L", image.size, 255), 17.65)
        if image.width == 26:
            return SegmentationPrediction(Image.new("L", image.size), 20)
        raise AppError(ErrorCode.INFERENCE_FAILED)

    monkeypatch.setattr(model, "predict", predict)

    async def analyze(action):
        await call("begin_analysis", session, None, action)
        await call("run_analysis", session, None)
        return await call("show_analysis", session, None, action)

    async def exercise():
        await call("upload", session, upload_payload(files, 25), None, 1)
        shown = await analyze(1)
        assert shown[-1]["selected"] == "overlay"
        assert "data:image/png" in shown[1] and "data:image/png" in shown[3]
        assert 'class="lesion-layer"' in shown[4]
        assert shown[5]["value"] == DEFAULT_OPACITY and shown[5]["interactive"] is True
        assert "25 × 12 px" in shown[6] and "17.65 ms" in shown[6] and "CPU" in shown[6]
        cleared = await call("upload", session, upload_payload(files, 26), None, 2)
        assert cleared[5]["selected"] == "original"
        assert all("data:image" not in cleared[index] for index in (7, 9, 10))
        assert cleared[14].count("disabled") == 2 and session[state_id].result is None
        empty = await analyze(2)
        assert "Không có pixel tiền cảnh" in empty[7] and "không dùng để kết luận" in empty[7]
        assert "Mặt nạ rỗng" in empty[3] and "data:image" not in empty[3]
        assert 'class="lesion-layer"' in empty[4]
        await call("upload", session, upload_payload(files, 27), None, 3)
        failed = await analyze(3)
        assert 'role="alert"' in failed[2]
        assert "thất bại" in failed[2] and "Phân tích thất bại" in failed[6]
        assert all("data:image" not in failed[index] for index in (1, 3, 4))
        assert failed[8].count("disabled") == 2
        for action in (4, 5):
            reset = await call("reset", session, None, action)
            assert all("data:image" not in reset[index] for index in (1, 4, 7, 9, 10))
            assert "Chưa có lượt phân tích" in reset[12]
        assert not list(files.root.glob("*/*.png"))

    asyncio.run(exercise())


@pytest.mark.parametrize("edit", ["reset", "upload"])
def test_delayed_view_render_cannot_restore_results(callback_app, monkeypatch, edit):
    files, model, demo, state_id, call = callback_app
    session = GradioSessionState(demo)
    entered, release = Event(), Event()
    render = cv_page.render_result_images

    def delayed(*args):
        images = render(*args)
        entered.set()
        assert release.wait(10)
        return images

    monkeypatch.setattr(cv_page, "render_result_images", delayed)
    monkeypatch.setattr(model, "predict", lambda image: SegmentationPrediction(Image.new("L", image.size), 5))

    async def exercise():
        await call("upload", session, upload_payload(files, 25), None, 1)
        await call("begin_analysis", session, None, 1)
        await call("run_analysis", session, None)
        pending = asyncio.create_task(call("show_analysis", session, None, 1))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if edit == "upload":
                await call("upload", session, upload_payload(files, 31), None, 2)
            else:
                await call("reset", session, None, 2)
        finally:
            release.set()
        assert_unchanged_updates(await pending)
        assert session[state_id].result is None

    asyncio.run(exercise())


def inline_image(locator):
    return Image.open(io.BytesIO(base64.b64decode(locator.get_attribute("src").split(",", 1)[1])))


def download_png(page, button, kind):
    with page.expect_request("**/lesion-export") as sent, page.expect_download() as saved:
        button.click()
    download = saved.value
    data = sent.value.post_data_json
    assert download.suggested_filename == f"segformer-{kind}-{data['run'][:12]}.png"
    with Image.open(download.path()) as png:
        assert png.format == "PNG" and not png.info
        return png.copy(), data


def check_browser_session(page, url, upload, normalized):
    from playwright.sync_api import expect

    assert page.goto(url).status == 200
    page.get_by_role("radio", name="Computer Vision").check()
    analyze = page.get_by_role("button", name="Phân tích")
    overlay_download = page.get_by_role("button", name="Tải Overlay PNG")
    box_download = page.get_by_role("button", name="Tải Localization PNG")
    expect(overlay_download).to_be_disabled()
    expect(box_download).to_be_disabled()
    with page.expect_file_chooser() as choose:
        page.get_by_role("button", name="Tải ảnh lên").click()
    choose.value.set_files(upload)
    expect(analyze).to_be_enabled()
    original_view = page.locator("#lesion-original img")
    expect(original_view).to_be_visible()
    original_url = original_view.get_attribute("src")
    analyze.click()
    overlay_tab = page.get_by_role("tab", name="Overlay")
    expect(overlay_tab).to_have_attribute("aria-selected", "true", timeout=120_000)
    overlay_base = page.locator("#lesion-overlay .lesion-stack img").first
    expect(overlay_base).to_have_attribute("src", original_url)
    layer = page.locator("#lesion-overlay .lesion-layer")
    expect(layer).to_be_visible()
    expect(layer).to_have_css("opacity", "0.4")
    expect(overlay_download).to_be_enabled()
    expect(box_download).to_be_enabled()

    model_requests = []
    page.on("request", lambda req: model_requests.append(req.url) if "/queue/join" in req.url else None)
    slider = page.locator("#overlay-opacity input[type=range]")
    viewer = page.locator("#lesion-overlay .lesion-stack")
    default_pixels = viewer.screenshot(animations="disabled")
    slider.press("Home")
    expect(layer).to_have_css("opacity", "0")
    zero_pixels = viewer.screenshot(animations="disabled")
    slider.press("End")
    expect(layer).to_have_css("opacity", "1")
    full_pixels = viewer.screenshot(animations="disabled")
    assert len({default_pixels, zero_pixels, full_pixels}) == 3
    assert not model_requests
    field = page.locator("#overlay-opacity input[type=number]")
    field.fill("55")
    field.press("Tab")
    expect(slider).to_have_value("55")
    expect(layer).to_have_css("opacity", "0.55")
    slider.fill("65")
    expect(layer).to_have_css("opacity", "0.65")

    page.get_by_role("tab", name="Mask").click()
    mask_view = page.locator("#lesion-mask img")
    expect(mask_view).to_be_visible()
    page.get_by_role("tab", name="Localization").click()
    localized = page.locator("#lesion-localization img")
    expect(localized).to_be_visible()
    expect(page.locator("#lesion-localization").get_by_text("Bounding box suy ra từ mask", exact=True)).to_be_visible()
    page.get_by_role("button", name="Chi tiết mô hình", exact=False).click()
    details = page.locator("#lesion-model-details")
    for text in ("SHA-256", "Ngưỡng phân đoạn", "Thời gian suy luận", "Nguồn định vị"):
        expect(details.get_by_text(text, exact=True)).to_be_visible()
    expect(details.get_by_text("0.7", exact=True)).to_be_visible()

    overlay, request = download_png(page, overlay_download, "overlay")
    placement, localization_request = download_png(page, box_download, "localization")
    assert request["opacity"] == 65 and request["run"] == localization_request["run"]
    with overlay, placement, inline_image(mask_view) as mask, inline_image(localized) as displayed_box:
        assert mask.size == displayed_box.size == placement.size == overlay.size == normalized.size
        assert mask.mode == "L" and set(mask.get_flattened_data()) == {0, 255}
        box = mask.getbbox()
        x1, y1, x2, y2 = box
        expect(page.locator("#lesion-localization code")).to_have_text(f"[{x1}, {y1}, {x2}, {y2}]")
        expected_alpha = round(255 * 65 / 100)
        expected_pixels = [
            tuple((old * (255 - expected_alpha) + teal * expected_alpha + 127) // 255
                  for old, teal in zip(pixel, OVERLAY_COLOR, strict=True)) if is_foreground else pixel
            for pixel, is_foreground in zip(normalized.get_flattened_data(), mask.get_flattened_data(), strict=True)
        ]
        assert list(overlay.get_flattened_data()) == expected_pixels
        assert placement.getpixel((x1, y1)) == placement.getpixel((x2 - 1, y2 - 1)) == BOX_COLOR
        assert displayed_box.tobytes() == placement.tobytes()
        with ImageChops.difference(placement, normalized) as differences:
            assert differences.getbbox() == box
    assert not model_requests
    return request


def test_wp3_real_browser_visualization_and_downloads(real_browser, published_sample, tmp_path):
    from playwright.sync_api import expect

    browser, url, sample = real_browser
    first = browser.new_context(accept_downloads=True)
    second = browser.new_context(accept_downloads=True)
    try:
        page_a, page_b = first.new_page(), second.new_page()
        with Image.open(sample) as raw, ImageOps.exif_transpose(raw).convert("RGB") as image:
            request_a = check_browser_session(page_a, url, sample, image)
            with image.transpose(Image.Transpose.ROTATE_90) as rotated, io.BytesIO() as buffer:
                rotated.save(buffer, format="PNG")
                request_b = check_browser_session(page_b, url, {
                    "name": "rotated-sample.png", "mimeType": "image/png", "buffer": buffer.getvalue(),
                }, rotated)

        assert request_a["session"] != request_b["session"] and request_a["run"] != request_b["run"]
        assert first.request.post(f"{url}/lesion-export", data=request_b).status == 404
        assert second.request.post(f"{url}/lesion-export", data=request_a).status == 404
        assert first.request.post(f"{url}/lesion-export", data={**request_a, "proof": request_b["proof"]}).status == 404
        assert first.request.post(f"{url}/lesion-export", data={**request_a, "path": str(sample)}).status == 404
        assert first.request.post(f"{url}/lesion-export", data={**request_a, "kind": str(sample)}).status == 404
        assert first.request.post(f"{url}/lesion-export", data=request_a, headers={"Origin": "https://example.invalid"}).status == 404
        assert first.request.get(f"{url}/lesion-export").status == 405
        private_files = list((tmp_path / "sessions").rglob("*.png"))
        assert len(private_files) == 4
        for path in (*private_files, published_sample[0]):
            for prefix in ("/gradio_api/file=", "/file="):
                assert first.request.get(url + prefix + quote(path.as_posix(), safe="")).status == 403

        with page_a.expect_file_chooser() as choose:
            page_a.get_by_role("button", name="Tải ảnh lên").click()
        choose.value.set_files(sample)
        expect(page_a.get_by_role("tab", name="Original")).to_have_attribute("aria-selected", "true")
        expect(page_a.get_by_role("button", name="Tải Overlay PNG")).to_be_disabled()
        assert first.request.post(f"{url}/lesion-export", data=request_a).status == 404
        expect(page_b.get_by_role("button", name="Tải Overlay PNG")).to_be_enabled()

        for page, context, request in ((page_a, first, request_a), (page_b, second, request_b)):
            slider = page.locator("#overlay-opacity input[type=range]")
            page.get_by_role("button", name="Đặt lại").click()
            expect(slider).to_be_disabled()
            expect(slider).to_have_value("40")
            slider.evaluate("input => input.dispatchEvent(new Event('input', {bubbles: true}))")
            expect(page.get_by_role("button", name="Tải Overlay PNG")).to_be_disabled()
            expect(page.get_by_role("button", name="Tải Localization PNG")).to_be_disabled()
            expect(page.get_by_role("button", name="Phân tích")).to_be_disabled()
            for tab in ("Original", "Mask", "Overlay", "Localization"):
                page.get_by_role("tab", name=tab).click()
                expect(page.get_by_role("tabpanel").locator("img")).to_have_count(0)
            expect(page.locator("#lesion-model-details")).to_contain_text("Chưa có lượt phân tích")
            assert context.request.post(f"{url}/lesion-export", data=request).status == 404
        assert not list((tmp_path / "sessions").rglob("*.png"))

        page_a.set_viewport_size({"width": 390, "height": 844})
        assert page_a.locator("#cv-viewer").bounding_box()["y"] > page_a.locator("#cv-controls").bounding_box()["y"]
        assert page_a.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        first.close()
        second.close()

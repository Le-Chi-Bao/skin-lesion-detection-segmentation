import asyncio
import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import gradio as gr
import pytest
from gradio.state_holder import SessionState as GradioSessionState
from PIL import Image

from app.config import PROJECT_ROOT, AppConfig
from app.core.temporary_files import TemporaryFiles
from app.main import launch_options
from app.ui.shell import build_shell


def test_blocks_expose_only_wp1_modules_and_tabs(tmp_path):
    config = AppConfig(runtime_dir=tmp_path)
    demo = build_shell(config, TemporaryFiles(tmp_path, config.session_ttl_seconds))
    assert isinstance(demo, gr.Blocks)
    assert demo.delete_cache == (60, 3600)
    components = demo.get_config_file()["components"]
    markdown = "\n".join(component["props"]["value"] for component in components if component["type"] == "markdown")
    for name in ("Tổng quan", "Computer Vision", "Medical RAG", "Clinical Cases", "Consultation Reports"):
        assert name in markdown
    assert "không tự động chẩn đoán" in markdown and "Chưa tích hợp" in markdown
    assert [tab["props"]["label"] for tab in components if tab["type"] == "tabitem"] == [
        "Original", "Mask", "Overlay", "Localization",
    ]
    analyze = next(component for component in components if component["props"].get("elem_id") == "analyze-image")
    assert not analyze["props"]["interactive"]
    assert all(target[0] != analyze["id"] for dep in demo.config["dependencies"] for target in dep["targets"])


def test_launch_options_are_local_and_do_not_allow_static_paths():
    config = AppConfig()
    options = launch_options(config)
    assert options["server_name"] == "127.0.0.1" and options["server_port"] == 7860
    assert options["share"] is False and options["show_error"] is False
    assert options["allowed_paths"] == [] and options["max_file_size"] == config.max_upload_bytes
    assert str(config.model_store_dir) in options["blocked_paths"]
    assert str(config.runtime_dir / "sessions") in options["blocked_paths"]


@pytest.mark.parametrize("latest_edit", ["reset", "upload"])
def test_gradio_ignores_upload_queued_after_a_newer_edit(tmp_path, monkeypatch, latest_edit):
    config = AppConfig(runtime_dir=tmp_path)
    files = TemporaryFiles(tmp_path, config.session_ttl_seconds)
    files.cache_dir.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(files.cache_dir))

    def fixture(name, width):
        path = files.cache_dir / name
        with Image.new("RGB", (width, 12)) as image:
            image.save(path)
        return gr.FileData(path=str(path)).model_dump()

    first = fixture("first.png", 20)
    newest = fixture("newest.png", 23)
    demo = build_shell(config, files)
    dependencies = {event["api_name"]: event for event in demo.config["dependencies"]}
    upload = dependencies["upload"]
    assert upload["trigger_mode"] == "multiple"
    session = GradioSessionState(demo)
    state_id = upload["outputs"][0]

    async def deliver_out_of_order():
        await demo.process_api(upload["id"], [first, None, 1], state=session)
        assert session[state_id].image.metadata.width == 20
        if latest_edit == "reset":
            await demo.process_api(dependencies["reset"]["id"], [None, 3], state=session)
            assert session[state_id].image is None
        else:
            await demo.process_api(upload["id"], [newest, None, 3], state=session)
            assert session[state_id].image.metadata.width == 23
        current = session[state_id].image
        revision = session[state_id].image_revision
        stored_files = set(files.root.glob("*/*.png"))

        stale = await demo.process_api(upload["id"], [first, None, 2], state=session)
        assert len(stale["data"]) == len(upload["outputs"])
        for update in stale["data"][1:]:
            assert update["__type__"] == "update"
            assert set(update) <= {"__type__", "props"}
            assert set(update.get("props", {})) <= {"__type__"}
        assert session[state_id].image is current
        assert session[state_id].image_revision == revision
        assert set(files.root.glob("*/*.png")) == stored_files

    asyncio.run(deliver_out_of_order())


@pytest.fixture(scope="module")
def local_server(tmp_path_factory):
    runtime_dir = tmp_path_factory.mktemp("local-server")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("LESION_")}
    env.update(LESION_PORT=str(port), LESION_RUNTIME_DIR=str(runtime_dir), PYTHONUTF8="1")
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "app.main"], cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"{url}/config", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.2)
        else:
            process.terminate()
            output, _ = process.communicate(timeout=10)
            pytest.fail(f"Gradio did not start at {url}\n{output}")
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)


def test_cli_serves_over_http_and_disables_direct_file_delivery(local_server, tmp_path):
    private_file = tmp_path / "private.txt"
    private_file.write_text("test fixture", encoding="utf-8")
    with pytest.raises(HTTPError) as error:
        urlopen(f"{local_server}/gradio_api/file={quote(private_file.as_posix(), safe='')}", timeout=2)
    assert error.value.code == 403


def test_browser_ignores_an_old_multipart_upload(local_server):
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            pytest.skip("Install Chromium with python -m playwright install chromium")
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            assert page.goto(local_server).status == 200
            page.get_by_role("radio", name="Computer Vision").check()
            playwright_api.expect(page.get_by_role("heading", name="Xem ảnh")).to_be_visible()
            playwright_api.expect(page.locator("#image-action-index")).to_be_hidden()
            playwright_api.expect(page.locator("#oversize-image")).to_be_hidden()
            held = []
            page.route("**/gradio_api/upload", lambda route: held.append(route))

            for expected, width in enumerate((21, 32), start=1):
                with Image.new("RGB", (width, 14)) as image, io.BytesIO() as stream:
                    image.save(stream, format="PNG")
                    with page.expect_file_chooser() as chooser:
                        page.get_by_role("button", name="Tải ảnh lên").click()
                    chooser.value.set_files({"name": "fixture.png", "mimeType": "image/png", "buffer": stream.getvalue()})
                for _ in range(100):
                    if len(held) == expected:
                        break
                    page.wait_for_timeout(25)
                assert len(held) == expected

            with page.expect_request(lambda req: "/gradio_api/queue/join" in req.url) as old_event:
                held[0].continue_()
            assert old_event.value.post_data_json["data"][0] is None
            held[1].continue_()
            playwright_api.expect(page.get_by_text("32 × 14 px", exact=True)).to_be_visible()
            playwright_api.expect(page.get_by_text("21 × 14 px", exact=True)).to_have_count(0)
        finally:
            browser.close()

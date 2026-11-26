"""Browser integration with fixed SYNTHETIC DEMO data and independent browser sessions."""

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from app.config import PROJECT_ROOT


def test_browser_medical_rag_demo_and_cv_navigation(tmp_path):
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            pytest.skip("Chromium is not installed")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LESION_", "RAG_"))}
        env.update(LESION_PORT=str(port), LESION_RUNTIME_DIR=str(tmp_path / "runtime"),
                   RAG_SYNTHETIC_DEMO="1", PYTHONUTF8="1", GRADIO_ANALYTICS_ENABLED="False")
        log = tmp_path / "server.txt"
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen([sys.executable, "-u", "main.py"], cwd=PROJECT_ROOT, env=env,
                                       stdout=output, stderr=subprocess.STDOUT)
            url = f"http://127.0.0.1:{port}"
            browser = None
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        with urlopen(url + "/config", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        time.sleep(0.2)
                else:
                    pytest.fail("RAG browser server did not start: " + log.read_text(encoding="utf-8"))
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                assert page.goto(url).status == 200
                page.get_by_role("radio", name="Medical RAG", exact=True).check()
                page.get_by_role("textbox", name="Câu hỏi", exact=True).fill("tổn thương da")
                page.get_by_role("button", name="Search", exact=True).click()
                api.expect(page.get_by_role("textbox", name="Trạng thái và thời gian tìm kiếm")).to_have_value(
                    re.compile(r"BM25 · SYNTHETIC DEMO.*OK"), timeout=30000)
                api.expect(page.get_by_text("skin lesion color border size observation.", exact=False)).to_be_visible()
                page.screenshot(path=str(tmp_path / "rag-browser.png"), full_page=True)
                # Returning to CV preserves its existing viewer and tab layout.
                page.get_by_role("radio", name="Computer Vision", exact=True).check()
                api.expect(page.get_by_role("heading", name="Xem ảnh", exact=True)).to_be_visible()
                assert page.get_by_role("tab", name="Original", exact=True).is_visible()
                # A new browser context has no question/result history from the first session.
                context = browser.new_context()
                other = context.new_page()
                other.goto(url)
                other.get_by_role("radio", name="Medical RAG", exact=True).check()
                api.expect(other.get_by_role("textbox", name="Câu hỏi", exact=True)).to_have_value("")
                assert other.locator("article").count() == 0
                context.close()
            finally:
                if browser:
                    browser.close()
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=15)

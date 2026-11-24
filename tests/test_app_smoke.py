import os
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import gradio as gr
import pytest

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


def test_cli_serves_over_http_and_disables_direct_file_delivery(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("LESION_")}
    env.update(LESION_PORT=str(port), LESION_RUNTIME_DIR=str(tmp_path), PYTHONUTF8="1")
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
        private_file = tmp_path / "private.txt"
        private_file.write_text("test fixture", encoding="utf-8")
        with pytest.raises(HTTPError) as error:
            urlopen(f"{url}/gradio_api/file={quote(private_file.as_posix(), safe='')}", timeout=2)
        assert error.value.code == 403
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)

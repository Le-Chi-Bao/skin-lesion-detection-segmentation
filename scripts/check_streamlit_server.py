"""Start a temporary local Streamlit server, check HTTP, then stop it."""

import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from src.utils.config import PROJECT_ROOT


def check_server(timeout: float = 45.0) -> None:
    """Verify server startup and frontend HTML; AppTest checks Python UI logic."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app/streamlit_app.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    # Do not send localhost requests through a machine's configured HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + timeout
            last_error = "Server did not start"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    base_url = f"http://127.0.0.1:{port}"
                    with opener.open(f"{base_url}/_stcore/health", timeout=2) as health:
                        if health.status != 200 or health.read().strip() != b"ok":
                            raise RuntimeError("Streamlit health check failed")
                    with opener.open(base_url, timeout=2) as page:
                        if page.status != 200 or b"<html" not in page.read().lower():
                            raise RuntimeError("Streamlit frontend did not serve HTML")
                    print("Streamlit HTTP smoke: health=200, frontend=200; passed")
                    return
                except (URLError, TimeoutError, OSError) as error:
                    last_error = str(error)
                    time.sleep(0.25)
            log.seek(0)
            details = log.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{last_error}\nStreamlit logs:\n{details}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    check_server()

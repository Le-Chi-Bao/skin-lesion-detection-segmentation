import logging
import os

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.logging_config import configure_logging, log_error
from app.core.temporary_files import TemporaryFiles, checked_path
from app.ui.cv_page import upload_browser_setup
from app.ui.shell import build_shell
from app.ui.theme import CSS, clinical_theme

logger = logging.getLogger(__name__)


class FileTransportPolicy(BaseHTTPMiddleware):
    def __init__(self, app, limit: int):
        super().__init__(app)
        self.limit = limit

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path.startswith(("/gradio_api/file=", "/file=")):
            return PlainTextResponse(str(AppError(ErrorCode.UNSAFE_PATH)), 403)
        response = await call_next(request)
        if request.method == "POST" and path == "/gradio_api/upload" and response.status_code == 413:
            return PlainTextResponse(str(AppError(ErrorCode.UPLOAD_TOO_LARGE, limit=self.limit)), 413)
        return response


def launch_options(config: AppConfig) -> dict:
    return {
        "server_name": config.host,
        "server_port": config.port,
        "share": False,
        "allowed_paths": [],
        "blocked_paths": [str(config.runtime_dir / "sessions"), str(config.model_store_dir)],
        "max_file_size": config.max_upload_bytes,
        "show_error": False,
        "run_history": False,
        "enable_monitoring": False,
        "mcp_server": False,
        "ssr_mode": False,
        "footer_links": [],
        "theme": clinical_theme(),
        "css": CSS,
        "js": upload_browser_setup(config.max_upload_bytes),
        "app_kwargs": {"middleware": [Middleware(FileTransportPolicy, limit=config.max_upload_mib)]},
    }


def main() -> int:
    try:
        config = AppConfig.from_env()
        configure_logging(config.log_level)
        files = TemporaryFiles(config.runtime_dir, config.session_ttl_seconds)
        with files.maintain(config.cleanup_interval):
            os.environ["GRADIO_TEMP_DIR"] = str(checked_path(files.cache_dir))
            demo = build_shell(config, files)
            try:
                demo.launch(**launch_options(config))
            finally:
                demo.close()
    except (AppError, OSError, ValueError, RuntimeError) as exc:
        log_error(logger, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

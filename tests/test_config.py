import pytest

from app.config import MIB, PROJECT_ROOT, AppConfig
from app.core.errors import AppError, ErrorCode


def test_defaults_and_relative_paths():
    config = AppConfig.from_env({})
    assert config.host == "127.0.0.1" and config.port == 7860
    assert config.runtime_dir == PROJECT_ROOT / "runtime"
    assert config.model_store_dir == PROJECT_ROOT / "model_store"
    assert config.research_mode and config.session_ttl_seconds == 3600
    assert config.max_upload_bytes == 10 * MIB
    override = AppConfig.from_env({"LESION_RUNTIME_DIR": "runtime/demo", "LESION_LOG_LEVEL": "debug"})
    assert override.runtime_dir == PROJECT_ROOT / "runtime" / "demo"
    assert override.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "70000"), ("PORT", "abc"), ("MAX_UPLOAD_MIB", "0"), ("MAX_IMAGE_PIXELS", "-1"),
        ("MAX_IMAGE_EDGE", "0"), ("SESSION_TTL_SECONDS", "0"), ("LOG_LEVEL", "TRACE"),
        ("HOST", "https://example.invalid"), ("APP_NAME", " "), ("RESEARCH_MODE", "false"),
        ("RUNTIME_DIR", "docs"), ("RUNTIME_DIR", ""), ("MODEL_STORE_DIR", "runtime/models"),
    ],
)
def test_invalid_configuration(name, value):
    with pytest.raises(AppError) as error:
        AppConfig.from_env({f"LESION_{name}": value})
    assert error.value.code is ErrorCode.INVALID_CONFIG
    assert error.value.field == name

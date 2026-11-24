import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import AppError, ErrorCode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str = "Hệ thống hỗ trợ phân tích tổn thương da"
    host: str = "127.0.0.1"
    port: int = 7860
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    model_store_dir: Path = PROJECT_ROOT / "model_store"
    segformer_checkpoint: Path | None = None
    segformer_device: str = "auto"
    max_upload_mib: int = 10
    max_image_pixels: int = 16_000_000
    max_image_edge: int = 8192
    session_ttl_seconds: int = 3600
    log_level: str = "INFO"
    research_mode: bool = True

    def __post_init__(self) -> None:
        if not self.app_name.strip():
            raise AppError(ErrorCode.INVALID_CONFIG, field="APP_NAME")
        if self.host != "localhost":
            try:
                ipaddress.ip_address(self.host)
            except ValueError as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, field="HOST") from exc
        for field, (minimum, maximum) in {
            "port": (1, 65535),
            "max_upload_mib": (1, 64),
            "max_image_pixels": (1, 36_000_000),
            "max_image_edge": (1, 16_384),
            "session_ttl_seconds": (1, 604_800),
        }.items():
            value = getattr(self, field)
            if type(value) is not int or not minimum <= value <= maximum:
                raise AppError(ErrorCode.INVALID_CONFIG, field=field.upper())
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise AppError(ErrorCode.INVALID_CONFIG, field="LOG_LEVEL")
        if self.research_mode is not True:
            raise AppError(ErrorCode.INVALID_CONFIG, field="RESEARCH_MODE")
        if self.segformer_device not in {"auto", "cpu", "cuda"}:
            raise AppError(ErrorCode.INVALID_CONFIG, field="SEGFORMER_DEVICE")
        if self.segformer_checkpoint is not None:
            try:
                path = self.segformer_checkpoint.expanduser()
                path = (PROJECT_ROOT / path).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, field="SEGFORMER_CHECKPOINT") from exc
            object.__setattr__(self, "segformer_checkpoint", path)

        for field in ("runtime_dir", "model_store_dir"):
            try:
                path = getattr(self, field).expanduser().resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, field=field.upper()) from exc
            object.__setattr__(self, field, path)
        for field in ("runtime_dir", "model_store_dir"):
            path = getattr(self, field)
            ignored = PROJECT_ROOT / field.removesuffix("_dir")
            if PROJECT_ROOT.is_relative_to(path) or (
                path.is_relative_to(PROJECT_ROOT) and not path.is_relative_to(ignored)
            ):
                raise AppError(ErrorCode.INVALID_CONFIG, field=field.upper())
        if self.runtime_dir.is_relative_to(self.model_store_dir) or self.model_store_dir.is_relative_to(
            self.runtime_dir
        ):
            raise AppError(ErrorCode.INVALID_CONFIG, field="RUNTIME_DIR")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mib * MIB

    @property
    def cleanup_interval(self) -> int:
        return min(60, self.session_ttl_seconds)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppConfig":
        env = os.environ if env is None else env
        values = {}
        for name in cls.__dataclass_fields__:
            key = f"LESION_{name.upper()}"
            if key not in env:
                continue
            raw = env[key].strip()
            try:
                if name in {"port", "max_upload_mib", "max_image_pixels", "max_image_edge", "session_ttl_seconds"}:
                    values[name] = int(raw)
                elif name.endswith("_dir") or name == "segformer_checkpoint":
                    if not raw:
                        raise ValueError
                    path = Path(raw)
                    values[name] = path if path.is_absolute() else PROJECT_ROOT / path
                elif name == "research_mode":
                    values[name] = raw.lower() in {"true", "1"}
                elif name == "log_level":
                    values[name] = raw.upper()
                elif name == "segformer_device":
                    values[name] = raw.lower()
                else:
                    values[name] = raw
            except ValueError as exc:
                raise AppError(ErrorCode.INVALID_CONFIG, field=name.upper()) from exc
        return cls(**values)

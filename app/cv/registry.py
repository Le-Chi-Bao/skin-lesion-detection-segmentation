"""Versioned local artifact selection. Invalid modules fail independently."""

import json
import logging
from dataclasses import replace
from pathlib import Path


from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.cv.availability import ModelAvailability, ModelStatus
from app.cv.artifacts import RegistryEntry

logger = logging.getLogger(__name__)


class UnavailableAdapter:
    identity = None

    def __init__(self, module: str, error: AppError | None = None, reason: str = ""):
        self.error = error
        self.reason = reason
        self.availability = ModelAvailability(
            module, ModelStatus.FAILED if error else ModelStatus.NOT_AVAILABLE,
        )

    def load(self) -> None:
        pass

    def predict(self, image):
        raise self.error or AppError(ErrorCode.MODEL_UNAVAILABLE)


class ModelRegistry:
    def __init__(self, path: Path):
        self.entries: tuple[RegistryEntry, ...] = ()
        self.errors: dict[str, AppError] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) != {"schema_version", "models"} or type(data["schema_version"]) is not int or data["schema_version"] != 1 or not isinstance(data["models"], list):
                raise ValueError("Unsupported registry")
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Không thể đọc registry %s: %s: %s", path, type(exc).__name__, exc)
            self.errors = {key: AppError(ErrorCode.INVALID_MODEL_CONFIG) for key in ("segmentation", "detection")}
            return
        entries = []
        for raw in data["models"]:
            module = raw.get("module") if isinstance(raw, dict) else None
            module = module if isinstance(module, str) else None
            try:
                entry = RegistryEntry.model_validate(raw)
                if any((old.module, old.model_id, old.version) == (entry.module, entry.model_id, entry.version) for old in entries):
                    raise ValueError("Duplicate model version")
                entries.append(entry)
            except ValueError as exc:
                logger.error("Entry registry không hợp lệ: registry=%s module=%s | %s", path, module, exc)
                for key in ((module,) if module in {"segmentation", "detection"} else ("segmentation", "detection")):
                    self.errors[key] = AppError(ErrorCode.INVALID_MODEL_CONFIG)
        self.entries = tuple(entries)
        for module in ("segmentation", "detection"):
            if sum(entry.active and entry.module == module for entry in entries) > 1:
                logger.error("Registry %s: module=%s có nhiều hơn một model active.", path, module)
                self.errors[module] = AppError(ErrorCode.INVALID_MODEL_CONFIG)

    def selected(self, module: str) -> RegistryEntry | None:
        if module in self.errors:
            raise self.errors[module]
        return next((entry for entry in self.entries if entry.module == module and entry.active), None)

    def create(self, module: str, config: AppConfig):
        try:
            entry = self.selected(module)
            if entry is None:
                return UnavailableAdapter(module, reason="Không có mô hình active trong registry.")
            if module == "detection":
                from app.cv.detection import FasterRCNN
                return FasterRCNN(entry, config.detection_device, config.detection_pilot_mode)
            from app.cv.checkpoint import SHA256 as SEG_SHA
            from app.cv.segformer import SegFormer, THRESHOLD
            if (entry.sha256, entry.architecture, entry.artifact_format, entry.threshold, entry.preprocessing, entry.research_pilot) != (
                SEG_SHA, "segformer_mit_b2", "segformer-gate2-state-dict", THRESHOLD, "segformer-gate2-512-v1", False,
            ):
                raise AppError(ErrorCode.INVALID_MODEL_CONFIG)
            adapter = SegFormer(replace(config, segformer_checkpoint=entry.path))
            adapter.load()
            if adapter.identity:
                adapter.identity = adapter.identity.model_copy(update={
                    "model_version": entry.version, "architecture": entry.architecture,
                    "artifact_format": entry.artifact_format,
                })
            return adapter
        except (AppError, OSError, ValueError) as exc:
            return UnavailableAdapter(module, exc if isinstance(exc, AppError) else AppError(ErrorCode.INVALID_MODEL_CONFIG))

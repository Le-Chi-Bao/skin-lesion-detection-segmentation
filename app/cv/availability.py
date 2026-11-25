from dataclasses import dataclass, field
from enum import StrEnum

from app.core.errors import AppError, ErrorCode


class ModelStatus(StrEnum):
    NOT_AVAILABLE = "not_available"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class RunStatus(StrEnum):
    NOT_AVAILABLE = "not_available"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelAvailability:
    model_id: str
    status: ModelStatus = ModelStatus.NOT_AVAILABLE
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.model_id.strip()
            or not isinstance(self.status, ModelStatus)
            or (self.checkpoint_id is not None and not self.checkpoint_id.strip())
            or (self.status is ModelStatus.READY and self.checkpoint_id is None)
        ):
            raise AppError(ErrorCode.INVALID_MODEL_CONFIG)

    def require_ready(self) -> None:
        if self.status is not ModelStatus.READY:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE)

    @property
    def label(self) -> str:
        return {
            ModelStatus.NOT_AVAILABLE: "Không khả dụng",
            ModelStatus.LOADING: "Đang tải mô hình",
            ModelStatus.READY: "Sẵn sàng",
            ModelStatus.FAILED: "Không thể nạp mô hình",
        }[self.status]


@dataclass(frozen=True, slots=True)
class CVAvailability:
    segmentation: ModelAvailability = field(default_factory=lambda: ModelAvailability("SegFormer MiT-B2"))
    detection: ModelAvailability = field(default_factory=lambda: ModelAvailability("Faster R-CNN ResNet50-FPN V2"))

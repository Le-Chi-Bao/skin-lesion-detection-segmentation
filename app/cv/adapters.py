"""Small application-facing contracts; adapters own all tensor operations."""

from typing import Protocol
from dataclasses import dataclass

from PIL import Image

from app.core.errors import AppError
from app.cv.availability import ModelAvailability
from app.cv.schemas import ModelIdentity, DetectionPrediction


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    mask: Image.Image
    duration_ms: float
    threshold: float = 0.7


@dataclass(frozen=True, slots=True)
class DetectionOutput:
    boxes: tuple[DetectionPrediction, ...]
    duration_ms: float


class ModelAdapter(Protocol):
    identity: ModelIdentity | None
    availability: ModelAvailability
    error: AppError | None

    def load(self) -> None: ...


class SegmentationAdapter(ModelAdapter, Protocol):
    def predict(self, image: Image.Image) -> SegmentationPrediction: ...


class DetectionAdapter(ModelAdapter, Protocol):
    def predict(self, image: Image.Image) -> DetectionOutput: ...

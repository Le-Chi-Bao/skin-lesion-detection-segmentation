"""Local inference for the verified SegFormer MiT-B2 checkpoint."""

import logging
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

from PIL import Image

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.cv.availability import ModelAvailability, ModelStatus
from app.cv.checkpoint import CHECKPOINT_ID, SHA256, verified_checkpoint
from app.cv.schemas import BoundingBox, ModelIdentity

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
MODEL_ID = "SegFormer MiT-B2"
INPUT_SIZE = 512
LOGITS_SHAPE = (1, 1, INPUT_SIZE, INPUT_SIZE)
THRESHOLD = 0.7
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(image: Image.Image) -> "torch.Tensor":
    import cv2
    import numpy as np
    import torch

    resized = cv2.resize(np.asarray(image), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = resized.astype(np.float32) / 255.0
    normal = (rgb - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).contiguous()


def binary_mask(logits: "torch.Tensor", original_size: tuple[int, int]) -> Image.Image:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.shape != LOGITS_SHAPE or not logits.is_floating_point():
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    logits = logits.float().cpu()
    if not torch.isfinite(logits).all().item():
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    binary = (logits[0, 0].sigmoid() >= THRESHOLD).to(torch.uint8).mul_(255)
    with Image.fromarray(binary.numpy()) as image:
        return image.resize(original_size, Image.Resampling.NEAREST)


def mask_box(mask: Image.Image) -> BoundingBox | None:
    edges = mask.getbbox()
    if edges is None:
        return None
    x1, y1, x2, y2 = edges
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def model_state(checkpoint: object) -> dict[str, "torch.Tensor"]:
    import torch

    if not isinstance(checkpoint, Mapping):
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="state_dict")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="state_dict")
    wrapped = [key.startswith("model.") for key in state]
    if all(wrapped):
        return {key.removeprefix("model."): value for key, value in state.items()}
    if any(wrapped):
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="mixed_wrapper_prefix")
    return dict(state)


def _load_verified_model(config: AppConfig) -> tuple["torch.nn.Module", str]:
    if config.segformer_checkpoint is None:
        raise AppError(ErrorCode.CHECKPOINT_MISSING)
    with verified_checkpoint(config.segformer_checkpoint) as stream:
        try:
            import cv2  # noqa: F401
            import segmentation_models_pytorch as smp
            import torch
        except Exception as exc:
            raise AppError(ErrorCode.MODEL_DEPENDENCIES) from exc
        try:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise AppError(ErrorCode.CHECKPOINT_WEIGHTS_ONLY) from exc
        except OSError as exc:
            raise AppError(ErrorCode.CHECKPOINT_READ) from exc
        except (MemoryError, RuntimeError) as exc:
            raise _runtime_error(torch, exc, ErrorCode.CHECKPOINT_INCOMPATIBLE) from exc
        except (EOFError, ValueError, TypeError) as exc:
            raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE) from exc

    state = model_state(checkpoint)
    try:
        model = smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=1)
    except (MemoryError, RuntimeError) as exc:
        raise _runtime_error(torch, exc, ErrorCode.MODEL_DEVICE) from exc
    expected = model.state_dict()
    if expected.keys() != state.keys():
        missing = len(expected.keys() - state.keys())
        unexpected = len(state.keys() - expected.keys())
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field=f"missing={missing} unexpected={unexpected}")
    try:
        model.load_state_dict(state, strict=True)
    except (MemoryError, RuntimeError) as exc:
        raise _runtime_error(torch, exc, ErrorCode.CHECKPOINT_INCOMPATIBLE) from exc
    try:
        device = config.segformer_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise AppError(ErrorCode.MODEL_DEVICE)
        return model.eval().requires_grad_(False).to(device), device
    except (MemoryError, RuntimeError, OSError) as exc:
        raise _runtime_error(torch, exc, ErrorCode.MODEL_DEVICE) from exc


def _runtime_error(torch_module, exc: Exception, default: ErrorCode) -> AppError:
    message = str(exc).lower()
    if isinstance(exc, (torch_module.OutOfMemoryError, MemoryError)) or any(
        text in message for text in ("out of memory", "not enough memory", "can't allocate memory")
    ):
        return AppError(ErrorCode.MODEL_OUT_OF_MEMORY)
    return AppError(default)


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    mask: Image.Image
    duration_ms: float


class SegFormer:
    def __init__(self, config: AppConfig):
        self._config = config
        self._load_lock = Lock()
        self._model = None
        self.identity: ModelIdentity | None = None
        self.availability = ModelAvailability(MODEL_ID)
        self.error: AppError | None = None

    def load(self) -> None:
        with self._load_lock:
            if self.availability.status is not ModelStatus.NOT_AVAILABLE or self.error is not None:
                return
            if self._config.segformer_checkpoint is not None:
                self.availability = ModelAvailability(MODEL_ID, ModelStatus.LOADING, CHECKPOINT_ID)
            try:
                self._model, device = _load_verified_model(self._config)
            except Exception as exc:
                code = exc.code if isinstance(exc, AppError) else ErrorCode.MODEL_DEPENDENCIES
                self.error = AppError(code)
                missing = self.error.code is ErrorCode.CHECKPOINT_MISSING
                if not missing:
                    log_error(logger, exc)
                status = ModelStatus.NOT_AVAILABLE if missing else ModelStatus.FAILED
                self.availability = ModelAvailability(MODEL_ID, status)
                return
            self.identity = ModelIdentity(
                model_id=MODEL_ID, checkpoint_id=CHECKPOINT_ID, checkpoint_sha256=SHA256, device=device,
            )
            self.availability = ModelAvailability(MODEL_ID, ModelStatus.READY, CHECKPOINT_ID)

    def predict(self, image: Image.Image) -> SegmentationPrediction:
        self.availability.require_ready()
        import torch

        try:
            with torch.inference_mode():
                tensor = preprocess(image).to(self.identity.device)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                start = perf_counter()
                logits = self._model(tensor)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                duration_ms = (perf_counter() - start) * 1000
                mask = binary_mask(logits, image.size)
        except (MemoryError, RuntimeError, OSError, ValueError) as exc:
            raise _runtime_error(torch, exc, ErrorCode.INFERENCE_FAILED) from exc
        return SegmentationPrediction(mask, duration_ms)

"""Strict local Faster R-CNN inference exports, with an explicit pilot boundary."""

import logging
import math
from threading import Lock
from time import perf_counter

from PIL import Image

from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.cv.adapters import DetectionOutput
from app.cv.availability import ModelAvailability, ModelStatus
from app.cv.artifacts import RegistryEntry, verify_stream
from app.cv.schemas import BoundingBox, DetectionPrediction, ModelIdentity

logger = logging.getLogger(__name__)
ARCHITECTURE = "fasterrcnn_resnet50_fpn_v2"
OFFICIAL_FORMAT = "isic-fasterrcnn-inference-v1"
PILOT_FORMAT = "isic-fasterrcnn-pilot-v1"
PILOT_SHA256 = "327c2281b69a9591939c1abe883cb4337fecc9983fdf8746d26c7b5a32011141"
PREPROCESSING = {
    "input": "RGB float32 CHW", "range": [0, 1], "resize": "torchvision GeneralizedRCNNTransform",
    "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
    "box_format": "xyxy, exclusive right/bottom, in source-image pixels",
    "box_score_thresh": 0.0, "nms_threshold": 0.5, "detections_per_image": 100,
}


def out_of_memory(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or any(
        phrase in str(exc).lower() for phrase in ("out of memory", "not enough memory", "can't allocate memory" )
    )


def validate_artifact(artifact, entry: RegistryEntry, pilot_mode: bool) -> None:
    if not isinstance(artifact, dict) or artifact.get("format") != entry.artifact_format:
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="format")
    if entry.artifact_format == PILOT_FORMAT:
        if not pilot_mode or not entry.research_pilot or entry.sha256 != PILOT_SHA256:
            raise AppError(ErrorCode.INVALID_MODEL_CONFIG, field="pilot_mode")
    elif entry.artifact_format != OFFICIAL_FORMAT or entry.research_pilot:
        raise AppError(ErrorCode.INVALID_MODEL_CONFIG, field="format")
    cfg = artifact.get("model_config", {})
    if not isinstance(cfg, dict) or set(cfg) != {"model", "min_size", "max_size", "trainable_backbone_layers"}:
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="model_config")
    if (
        entry.architecture != ARCHITECTURE or cfg["model"] != ARCHITECTURE
        or type(cfg["min_size"]) is not int or type(cfg["max_size"]) is not int
        or (cfg["min_size"], cfg["max_size"]) != (entry.min_size, entry.max_size)
        or cfg["min_size"] > cfg["max_size"]
        or type(cfg["trainable_backbone_layers"]) is not int or not 0 <= cfg["trainable_backbone_layers"] <= 5
        or artifact.get("labels") not in ({0: "__background__", 1: "skin_lesion"}, {0: "background", 1: "skin_lesion"})
        or artifact.get("preprocessing") != PREPROCESSING
        or entry.preprocessing != "torchvision-rgb-float32-v1"
        or isinstance(artifact.get("operating_score_threshold"), bool)
        or artifact.get("operating_score_threshold") != entry.threshold
        or not isinstance(artifact.get("model"), dict) or not artifact["model"]
    ):
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="metadata")


def predictions(output, size: tuple[int, int], threshold: float) -> tuple[DetectionPrediction, ...]:
    import torch

    if not isinstance(output, dict) or not {"boxes", "scores", "labels"} <= output.keys():
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    if not all(isinstance(output[key], torch.Tensor) for key in ("boxes", "scores", "labels")):
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    boxes, scores, labels = (output[key].detach().cpu() for key in ("boxes", "scores", "labels"))
    if scores.ndim != 1:
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    count = len(scores)
    if (
        boxes.shape != (count, 4) or scores.shape != (count,) or labels.shape != (count,)
        or not boxes.is_floating_point() or not scores.is_floating_point() or labels.dtype != torch.int64
        or not all(torch.isfinite(value).all().item() for value in (boxes, scores))
    ):
        raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
    width, height = size
    result = []
    for coords, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True):
        x1, y1, x2, y2 = coords
        if label != 1 or not 0 <= score <= 1 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
        if score >= threshold:
            result.append(DetectionPrediction(box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), score=score))
    return tuple(result)


class FasterRCNN:
    def __init__(self, entry: RegistryEntry, device: str = "auto", pilot_mode: bool = False):
        self.entry = entry
        self.device = device
        self.pilot_mode = pilot_mode
        self.identity = None
        self.error = None
        self.availability = ModelAvailability(entry.model_id)
        self._model = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self.availability.status is not ModelStatus.NOT_AVAILABLE or self.error:
                return
            self.availability = ModelAvailability(self.entry.model_id, ModelStatus.LOADING)
            try:
                import torch
                from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

                with self.entry.path.open("rb") as stream:
                    verify_stream(stream, self.entry.sha256)
                    artifact = torch.load(stream, map_location="cpu", weights_only=True)
                validate_artifact(artifact, self.entry, self.pilot_mode)
                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
                    raise AppError(ErrorCode.MODEL_DEVICE)
                model = fasterrcnn_resnet50_fpn_v2(
                    weights=None, weights_backbone=None, min_size=self.entry.min_size, max_size=self.entry.max_size,
                    box_score_thresh=0.0, box_nms_thresh=0.5, box_detections_per_img=100,
                )
                features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes=2)
                model.load_state_dict(artifact["model"], strict=True)
                self._model = model.eval().requires_grad_(False).to(device)
                self.identity = ModelIdentity(
                    model_id=self.entry.model_id, model_version=self.entry.version, checkpoint_id=self.entry.version,
                    checkpoint_sha256=self.entry.sha256, device=device, architecture=self.entry.architecture,
                    artifact_format=self.entry.artifact_format, research_pilot=self.entry.research_pilot,
                )
                self.availability = ModelAvailability(self.entry.model_id, ModelStatus.READY, self.entry.version)
            except Exception as exc:
                self._model = None
                if isinstance(exc, AppError):
                    self.error = exc
                elif isinstance(exc, FileNotFoundError):
                    self.error = AppError(ErrorCode.CHECKPOINT_MISSING)
                elif isinstance(exc, (ImportError, ModuleNotFoundError)):
                    self.error = AppError(ErrorCode.MODEL_DEPENDENCIES)
                elif out_of_memory(exc):
                    self.error = AppError(ErrorCode.MODEL_OUT_OF_MEMORY)
                else:
                    self.error = AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE)
                self.availability = ModelAvailability(self.entry.model_id, ModelStatus.FAILED)
                log_error(logger, self.error)

    def predict(self, image: Image.Image) -> DetectionOutput:
        self.availability.require_ready()
        import numpy as np
        import torch

        try:
            with self._lock, torch.inference_mode(), image.convert("RGB") as rgb:
                tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous().to(self.identity.device)
                cuda = self.identity.device == "cuda"
                if cuda:
                    torch.cuda.synchronize()
                start = perf_counter()
                output = self._model([tensor])
                if cuda:
                    torch.cuda.synchronize()
                duration = (perf_counter() - start) * 1000
                if not isinstance(output, list) or len(output) != 1 or not math.isfinite(duration):
                    raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
                return DetectionOutput(predictions(output[0], image.size, self.entry.threshold), duration)
        except AppError:
            raise
        except Exception as exc:
            code = ErrorCode.MODEL_OUT_OF_MEMORY if isinstance(exc, torch.OutOfMemoryError) or out_of_memory(exc) else ErrorCode.INFERENCE_FAILED
            raise AppError(code) from exc

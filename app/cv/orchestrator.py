"""Independent sequential modules coordinated behind stable application contracts."""

import logging
from threading import Lock
from time import perf_counter

from PIL import Image

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.core.session import RunToken, SessionImage
from app.core.temporary_files import TemporaryFiles
from app.cv.adapters import DetectionAdapter, SegmentationAdapter
from app.cv.availability import CVAvailability, ModelStatus, RunStatus
from app.cv.registry import ModelRegistry, UnavailableAdapter
from app.cv.schemas import BoundingBox, CVResult, DerivedLocalization, DetectionResult, MaskReference, ResultMetadata, SegmentationResult

logger = logging.getLogger(__name__)


class CVOrchestrator:
    def __init__(self, segmentation: SegmentationAdapter, detection: DetectionAdapter | None = None):
        self.segmentation = segmentation
        self.detection = detection or UnavailableAdapter(
            "Faster R-CNN ResNet50-FPN V2", reason="Detection chưa được cấu hình qua registry.",
        )
        self._inference_lock = Lock()

    @property
    def availability(self) -> CVAvailability:
        return CVAvailability(self.segmentation.availability, self.detection.availability)

    def analyze(self, image: Image.Image, session_image: SessionImage, token: RunToken, files: TemporaryFiles) -> CVResult:
        segmentation = SegmentationResult(image_width=image.width, image_height=image.height)
        detection = DetectionResult()
        localization = None
        reference = None
        assembled = False
        try:
            with self._inference_lock:
                for module, adapter in (("segmentation", self.segmentation), ("detection", self.detection)):
                    dimensions = {"image_width": image.width, "image_height": image.height} if module == "segmentation" else {}
                    result_type = SegmentationResult if module == "segmentation" else DetectionResult
                    if adapter.availability.status is not ModelStatus.READY:
                        if adapter.error:
                            failed = result_type(**dimensions, status=RunStatus.FAILED, error_code=adapter.error.code)
                            if module == "segmentation":
                                segmentation = failed
                            else:
                                detection = failed
                        continue
                    start = perf_counter()
                    try:
                        output = adapter.predict(image)
                        if module == "segmentation":
                            with output.mask as mask:
                                if mask.size != image.size or mask.mode != "L":
                                    raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
                                colors = mask.getcolors(2)
                                if colors is None or any(value not in (0, 255) for _, value in colors):
                                    raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
                                edges = mask.getbbox()
                                box = BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), edges, strict=True))) if edges else None
                                reference = files.write_mask(token.session_id, lambda stream: mask.save(stream, format="PNG"))
                                segmentation = SegmentationResult(
                                    **dimensions, status=RunStatus.COMPLETED, model=adapter.identity,
                                    mask=MaskReference(reference=reference, width=mask.width, height=mask.height),
                                    threshold=output.threshold, inference_duration_ms=output.duration_ms,
                                    warnings=("Không có pixel tổn thương đạt ngưỡng phân đoạn.",) if box is None else (),
                                )
                                localization = DerivedLocalization(box=box) if box else None
                        else:
                            if any(p.box.x2 > image.width or p.box.y2 > image.height for p in output.boxes):
                                raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
                            detection = DetectionResult(
                                status=RunStatus.COMPLETED, model=adapter.identity, boxes=output.boxes,
                                inference_duration_ms=output.duration_ms,
                                warnings=(f"{adapter.identity.model_version}: pilot chưa hoàn tất; không phải mô hình được chọn cuối cùng.",) if adapter.identity.research_pilot else (),
                            )
                    except Exception as exc:
                        error = exc if isinstance(exc, AppError) else AppError(
                            ErrorCode.MODEL_OUT_OF_MEMORY if isinstance(exc, MemoryError) else ErrorCode.INFERENCE_FAILED,
                        )
                        log_error(logger, error, token.session_id)
                        failed = result_type(
                            **dimensions, status=RunStatus.FAILED, model=adapter.identity, error_code=error.code,
                            inference_duration_ms=(perf_counter() - start) * 1000,
                        )
                        if module == "segmentation":
                            segmentation = failed
                            localization = None
                            if reference:
                                files.discard_mask(token.session_id, reference)
                                reference = None
                        else:
                            detection = failed
            statuses = {segmentation.status, detection.status}
            if RunStatus.COMPLETED in statuses:
                status = RunStatus.PARTIAL if RunStatus.FAILED in statuses else RunStatus.COMPLETED
            else:
                status = RunStatus.FAILED if RunStatus.FAILED in statuses else RunStatus.NOT_AVAILABLE
            durations = [r.inference_duration_ms for r in (segmentation, detection) if r.inference_duration_ms is not None]
            result = CVResult(
                **session_image.identity.model_dump(), run_id=token.run_id, segmentation=segmentation,
                detection=detection, localization=localization, metadata=ResultMetadata(
                    status=status, inference_duration_ms=sum(durations) if durations else None,
                    error_code=(segmentation.error_code or detection.error_code) if status is RunStatus.FAILED else None,
                    warning_codes=("empty_mask",) if segmentation.mask and localization is None else (),
                ),
            )
            assembled = True
            return result
        finally:
            if reference and not assembled:
                files.discard_mask(token.session_id, reference)


def create_orchestrator(config: AppConfig, segmentation: SegmentationAdapter | None = None) -> CVOrchestrator:
    from app.cv.segformer import SegFormer
    logger.info("Registry: %s", config.model_registry or "không có; chế độ SegFormer cũ")
    registry = ModelRegistry(config.model_registry) if config.model_registry else None
    segmenter = segmentation if segmentation is not None else (
        registry.create("segmentation", config) if registry else SegFormer(config)
    )
    detector = registry.create("detection", config) if registry else None
    segmenter.load()
    if detector:
        detector.load()
    models = CVOrchestrator(segmenter, detector)
    for module, label, adapter, device in (
        ("segmentation", "SegFormer", models.segmentation, config.segformer_device),
        ("detection", "Faster R-CNN", models.detection, config.detection_device),
    ):
        entry = None
        if registry:
            try:
                entry = registry.selected(module)
            except AppError:
                pass  # The adapter already carries this module's registry error.
        identity = adapter.identity
        checkpoint = config.segformer_checkpoint if registry is None and module == "segmentation" else None
        if entry:
            try:
                checkpoint = entry.path
            except (OSError, ValueError, RuntimeError):
                checkpoint = repr(entry.checkpoint_path)
        status = adapter.availability.status
        reason = str(adapter.error).replace("SegFormer", label) if adapter.error else getattr(adapter, "reason", "")
        message = "%s: %s | version=%s | device=%s | checkpoint=%s | %s"
        logger.log(
            logging.INFO if status is ModelStatus.READY else logging.WARNING,
            message, label, "UNAVAILABLE" if status is ModelStatus.NOT_AVAILABLE else status.value.upper(),
            (identity.model_version or identity.checkpoint_id) if identity else (entry.version if entry else "không có"),
            identity.device if identity else f"chưa khởi tạo (yêu cầu {device})",
            checkpoint or "chưa cấu hình", reason or "đã nạp và xác minh checkpoint",
        )
    return models

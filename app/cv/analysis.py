"""Publish only the current RunToken; stale results never revive old images."""

import logging
from types import SimpleNamespace
from PIL import Image
from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.core.session import SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.adapters import SegmentationAdapter
from app.cv.availability import RunStatus, ModelAvailability, ModelStatus
from app.cv.orchestrator import CVOrchestrator
from app.cv.schemas import CVResult, ResultMetadata, SegmentationResult

logger = logging.getLogger(__name__)


def discard_result(files: TemporaryFiles, current: SessionState, previous: CVResult | None) -> None:
    if previous and previous.segmentation.mask:
        files.discard_mask(current.session_id, previous.segmentation.mask.reference)


def execute_analysis(current: SessionState, files: TemporaryFiles, models: CVOrchestrator | SegmentationAdapter) -> None:
    with current.lock:
        job = current.claim_run()
    if job is None:
        return
    token, session_image = job
    # Preserve WP2's direct predict/identity call contract for existing callers.
    if not isinstance(models, CVOrchestrator) and not hasattr(models, "availability"):
        models = SimpleNamespace(
            predict=models.predict, identity=models.identity, error=None,
            availability=ModelAvailability(models.identity.model_id, ModelStatus.READY, models.identity.checkpoint_id or models.identity.checkpoint_sha256),
        )
    orchestrator = models if isinstance(models, CVOrchestrator) else CVOrchestrator(models)
    try:
        with files.operation(token.session_id):
            with files.open_image(token.session_id, session_image.temporary_file) as stream, Image.open(stream) as image:
                image.load()
                result = orchestrator.analyze(image, session_image, token, files)
            with current.lock:
                accepted = current.accept_result(token, result)
            if not accepted:
                discard_result(files, current, result)
    except Exception as exc:
        error = exc if isinstance(exc, AppError) else AppError(ErrorCode.TEMPORARY_FILE)
        log_error(logger, error, token.session_id)
        result = CVResult(
            **session_image.identity.model_dump(), run_id=token.run_id,
            segmentation=SegmentationResult(
                image_width=session_image.metadata.width, image_height=session_image.metadata.height,
                status=RunStatus.FAILED, error_code=error.code,
            ), metadata=ResultMetadata(status=RunStatus.FAILED, error_code=error.code),
        )
        with current.lock:
            current.accept_result(token, result)

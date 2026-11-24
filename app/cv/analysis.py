"""Run a session's SegFormer request and publish only the current token's result."""

import logging

from PIL import Image

from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error
from app.core.session import RunToken, SessionImage, SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.availability import RunStatus
from app.cv.schemas import CVResult, DerivedLocalization, MaskReference, ModelIdentity, ResultMetadata, SegmentationResult
from app.cv.segformer import THRESHOLD, SegFormer, mask_box

logger = logging.getLogger(__name__)
EMPTY_MASK = "Không có pixel tổn thương đạt ngưỡng phân đoạn."


def discard_result(files: TemporaryFiles, current: SessionState, previous: CVResult | None) -> None:
    """Mark the previous mask for removal within an active file operation."""
    if previous and previous.segmentation.mask:
        files.discard_mask(current.session_id, previous.segmentation.mask.reference)


def _result(
    image: SessionImage,
    token: RunToken,
    model: ModelIdentity | None,
    metadata: ResultMetadata,
    *,
    mask: MaskReference | None = None,
    warnings: tuple[str, ...] = (),
    localization: DerivedLocalization | None = None,
) -> CVResult:
    identity = image.identity.model_dump()
    return CVResult(
        **identity, run_id=token.run_id, metadata=metadata,
        segmentation=SegmentationResult(
            image_width=identity["image_width"], image_height=identity["image_height"],
            status=metadata.status, model=model, threshold=THRESHOLD,
            mask=mask, warnings=warnings,
        ),
        localization=localization,
    )


def execute_analysis(current: SessionState, files: TemporaryFiles, segmenter: SegFormer) -> None:
    with current.lock:
        job = current.claim_run()
    if job is None:
        return
    token, session_image = job
    try:
        with files.operation(token.session_id):
            with files.open_image(token.session_id, session_image.temporary_file) as stream:
                try:
                    image = Image.open(stream)
                except (OSError, ValueError) as exc:
                    raise AppError(ErrorCode.TEMPORARY_FILE) from exc
                with image:
                    try:
                        image.load()
                    except (OSError, ValueError) as exc:
                        raise AppError(ErrorCode.TEMPORARY_FILE) from exc
                    prediction = segmenter.predict(image)

            with prediction.mask as mask:
                if mask.size != (session_image.metadata.width, session_image.metadata.height):
                    raise AppError(ErrorCode.INVALID_MODEL_OUTPUT)
                box = mask_box(mask)
                reference = files.write_mask(token.session_id, lambda stream: mask.save(stream, format="PNG"))
                published = False
                try:
                    result = _result(
                        session_image, token, segmenter.identity,
                        ResultMetadata(
                            status=RunStatus.COMPLETED, inference_duration_ms=prediction.duration_ms,
                            warning_codes=("empty_mask",) if box is None else (),
                        ),
                        mask=MaskReference(reference=reference, width=mask.width, height=mask.height),
                        warnings=(EMPTY_MASK,) if box is None else (),
                        localization=DerivedLocalization(box=box) if box is not None else None,
                    )
                    with current.lock:
                        published = current.accept_result(token, result)
                finally:
                    if not published:
                        files.discard_mask(token.session_id, reference)
    except Exception as exc:
        error = exc if isinstance(exc, AppError) else AppError(ErrorCode.INFERENCE_FAILED)
        log_error(logger, exc, token.session_id)
        result = _result(
            session_image, token, segmenter.identity, ResultMetadata(status=RunStatus.FAILED, error_code=error.code),
        )
        with current.lock:
            current.accept_result(token, result)

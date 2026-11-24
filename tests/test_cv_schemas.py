from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.errors import AppError, ErrorCode
from app.cv.availability import CVAvailability, ModelAvailability, ModelStatus, RunStatus
from app.cv.schemas import (
    BoundingBox,
    CVResult,
    DerivedLocalization,
    DetectionPrediction,
    DetectionResult,
    MaskReference,
    ModelIdentity,
    ResultMetadata,
    SegmentationResult,
)


def result(**overrides):
    fields = {
        "image_id": uuid4(), "run_id": uuid4(), "image_width": 60, "image_height": 40,
        "image_hash": "a" * 64, "segmentation": SegmentationResult(image_width=60, image_height=40),
    }
    return CVResult(**(fields | overrides))


def test_available_models_and_run_status_are_distinct():
    models = CVAvailability()
    assert models.segmentation.status is models.detection.status is ModelStatus.NOT_AVAILABLE
    with pytest.raises(AppError) as error:
        models.segmentation.require_ready()
    assert error.value.code is ErrorCode.MODEL_UNAVAILABLE
    with pytest.raises(AppError) as error:
        ModelAvailability("test", ModelStatus.READY)
    assert error.value.code is ErrorCode.INVALID_MODEL_CONFIG


def test_valid_unavailable_result_and_json_round_trip():
    current = result()
    assert current.metadata.status is RunStatus.NOT_AVAILABLE
    assert current.segmentation.mask is None and not current.detection.boxes
    assert CVResult.model_validate_json(current.model_dump_json()) == current


def test_valid_completed_contract_with_source_and_model_metadata():
    model = ModelIdentity(model_id="test", checkpoint_sha256="a" * 64, device="cpu")
    mask = MaskReference(reference="mask_123", width=60, height=40)
    segmentation = SegmentationResult(
        image_width=60, image_height=40, status=RunStatus.COMPLETED, model=model, mask=mask, threshold=0.5,
    )
    box = BoundingBox(x1=0, y1=0.5, x2=60, y2=40)
    detection = DetectionResult(
        status=RunStatus.COMPLETED, model=model, boxes=(DetectionPrediction(box=box, score=0.9),),
    )
    current = result(
        segmentation=segmentation, detection=detection, localization=DerivedLocalization(box=box),
        metadata=ResultMetadata(status=RunStatus.COMPLETED, inference_duration_ms=25, warning_codes=("test",)),
    )
    assert current.localization.source == "derived_from_mask"
    assert current.detection.coordinate_convention == "xyxy_exclusive"
    assert CVResult.model_validate_json(current.model_dump_json()) == current


@pytest.mark.parametrize("coords", [(-1, 0, 1, 1), (2, 0, 2, 1), (0, 2, 1, 1), (0, 0, float("inf"), 1)])
def test_invalid_bounding_box(coords):
    with pytest.raises(ValidationError):
        BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), coords, strict=True)))


def test_out_of_image_box_and_missing_mask():
    box = BoundingBox(x1=0, y1=0, x2=61, y2=40)
    with pytest.raises(ValidationError, match="outside"):
        result(
            detection=DetectionResult(
                status=RunStatus.COMPLETED, model=ModelIdentity(model_id="test", checkpoint_id="test-v1"),
                boxes=(DetectionPrediction(box=box, score=1),),
            ),
            metadata=ResultMetadata(status=RunStatus.COMPLETED),
        )
    with pytest.raises(ValidationError, match="requires a mask"):
        result(localization=DerivedLocalization(box=box))
    with pytest.raises(ValidationError, match="Mask dimensions"):
        SegmentationResult(
            image_width=60, image_height=40, status=RunStatus.COMPLETED,
            model=ModelIdentity(model_id="test", checkpoint_id="test-v1"),
            mask=MaskReference(reference="mask_123", width=40, height=60),
        )


def test_invalid_metadata_and_references():
    with pytest.raises(ValidationError):
        ModelIdentity(model_id="test")
    with pytest.raises(ValidationError):
        DetectionResult(status=RunStatus.COMPLETED)
    with pytest.raises(ValidationError):
        DetectionResult(model=ModelIdentity(model_id="test", checkpoint_id="test-v1"))
    with pytest.raises(ValidationError):
        ResultMetadata(status=RunStatus.FAILED)
    with pytest.raises(ValidationError):
        ResultMetadata(status=RunStatus.COMPLETED, inference_duration_ms=-1)
    with pytest.raises(ValidationError):
        result(metadata=ResultMetadata(status=RunStatus.COMPLETED))
    with pytest.raises(ValidationError):
        MaskReference(reference="../mask.png", width=60, height=40)

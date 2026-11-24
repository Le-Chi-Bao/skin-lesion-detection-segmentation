from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.core.errors import ErrorCode
from app.cv.availability import RunStatus

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Dimension = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImageIdentity(Contract):
    image_id: UUID
    image_width: Dimension
    image_height: Dimension
    image_hash: SHA256


class BoundingBox(Contract):
    """Pixel edges in the EXIF-oriented original, with exclusive right and bottom edges."""

    x1: Nonnegative
    y1: Nonnegative
    x2: Nonnegative
    y2: Nonnegative

    @model_validator(mode="after")
    def check_area(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self


class ModelIdentity(Contract):
    model_id: Text
    checkpoint_id: Text | None = None
    checkpoint_sha256: SHA256 | None = None
    device: Text | None = None

    @model_validator(mode="after")
    def check_checkpoint(self) -> "ModelIdentity":
        if self.checkpoint_id is None and self.checkpoint_sha256 is None:
            raise ValueError("Model metadata requires a checkpoint ID or SHA-256")
        return self


class MaskReference(Contract):
    """An opaque storage ID. The mask producer supplies its actual dimensions."""

    reference: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
    width: Dimension
    height: Dimension


class ModelResult(Contract):
    status: RunStatus = RunStatus.NOT_AVAILABLE
    model: ModelIdentity | None = None
    warnings: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def check_model_status(self) -> "ModelResult":
        if self.status is RunStatus.NOT_AVAILABLE and self.model is not None:
            raise ValueError("An unavailable model cannot supply inference metadata")
        if self.status in {RunStatus.RUNNING, RunStatus.COMPLETED} and self.model is None:
            raise ValueError("An active or completed model requires inference metadata")
        return self


class SegmentationResult(ModelResult):
    image_width: Dimension
    image_height: Dimension
    mask: MaskReference | None = None
    threshold: Score | None = None

    @model_validator(mode="after")
    def check_mask(self) -> "SegmentationResult":
        if (self.mask is not None) != (self.status is RunStatus.COMPLETED):
            raise ValueError("Only a completed segmentation may supply a mask")
        if self.mask and (self.mask.width, self.mask.height) != (self.image_width, self.image_height):
            raise ValueError("Mask dimensions must match the original image")
        if self.status is RunStatus.NOT_AVAILABLE and self.threshold is not None:
            raise ValueError("An unavailable model cannot supply an inference threshold")
        return self


class DetectionPrediction(Contract):
    box: BoundingBox
    score: Score  # Model detection score.


class DetectionResult(ModelResult):
    boxes: tuple[DetectionPrediction, ...] = ()
    coordinate_convention: Literal["xyxy_exclusive"] = "xyxy_exclusive"

    @model_validator(mode="after")
    def check_predictions(self) -> "DetectionResult":
        if self.boxes and self.status is not RunStatus.COMPLETED:
            raise ValueError("Only a completed detection may supply predictions")
        return self


class DerivedLocalization(Contract):
    box: BoundingBox
    source: Literal["derived_from_mask"] = "derived_from_mask"


class ResultMetadata(Contract):
    status: RunStatus = RunStatus.NOT_AVAILABLE
    inference_duration_ms: Nonnegative | None = None
    warning_codes: tuple[Text, ...] = ()
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def check_status(self) -> "ResultMetadata":
        if (self.error_code is not None) != (self.status is RunStatus.FAILED):
            raise ValueError("A failed run requires a UI error code")
        if self.inference_duration_ms is not None and self.status not in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError("Inference duration requires a finished run")
        return self


class CVResult(ImageIdentity):
    run_id: UUID
    segmentation: SegmentationResult
    detection: DetectionResult = Field(default_factory=DetectionResult)
    localization: DerivedLocalization | None = None
    metadata: ResultMetadata = Field(default_factory=ResultMetadata)

    @model_validator(mode="after")
    def check_outputs(self) -> "CVResult":
        if (self.segmentation.image_width, self.segmentation.image_height) != (self.image_width, self.image_height):
            raise ValueError("Segmentation dimensions must match the original image")
        if self.localization is not None and self.segmentation.mask is None:
            raise ValueError("Localization derived from a mask requires a mask")
        boxes = [prediction.box for prediction in self.detection.boxes]
        if self.localization is not None:
            boxes.append(self.localization.box)
        if any(box.x2 > self.image_width or box.y2 > self.image_height for box in boxes):
            raise ValueError("Bounding box lies outside the original image")
        statuses = {self.segmentation.status, self.detection.status}
        for status in (RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.NOT_AVAILABLE):
            if status in statuses:
                if self.metadata.status is not status:
                    raise ValueError("Run status does not match the model results")
                break
        return self

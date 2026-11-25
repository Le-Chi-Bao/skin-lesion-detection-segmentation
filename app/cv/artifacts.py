"""Immutable artifact metadata and checksum verification, without model loading."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.config import PROJECT_ROOT
from app.core.errors import AppError, ErrorCode
from app.cv.schemas import Contract, SHA256, Text, Score


class RegistryEntry(Contract):
    module: Literal["segmentation", "detection"]
    model_id: Text
    version: Text
    checkpoint_path: Text
    sha256: SHA256
    architecture: Text
    artifact_format: Text
    research_pilot: bool = False
    active: bool = False
    threshold: Score
    min_size: int = Field(default=800, gt=0, le=8192)
    max_size: int = Field(default=1333, gt=0, le=8192)
    preprocessing: Text

    @property
    def path(self) -> Path:
        return (PROJECT_ROOT / self.checkpoint_path).resolve()



def verify_stream(stream, digest: str) -> None:
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="sha256")
    stream.seek(0)

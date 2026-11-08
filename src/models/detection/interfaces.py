"""Inference contract; the Faster R-CNN adapter will be added in a later phase."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.data.geometry import BoundingBox


@dataclass(frozen=True)
class Detection:
    """A single lesion box in ORIGINAL image coordinates and its confidence."""

    box: BoundingBox
    score: float


class Detector(Protocol):
    """Accept RGB uint8, return original-coordinate detections, possibly empty."""

    def predict(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Detect lesions; implementations own resize/normalization internally."""
        ...

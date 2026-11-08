"""Spatial preparation only: no detector, segmenter, or predictions here."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.data.geometry import (
    BoundingBox,
    CropMetadata,
    Size,
    crop_roi,
    resize_image,
)


@dataclass(frozen=True)
class PreparedROI:
    image: NDArray[np.uint8]
    metadata: CropMetadata


def prepare_roi(
    image: NDArray[np.uint8],
    box: BoundingBox,
    size: Size,
    padding: float = 0.0,
) -> PreparedROI:
    """Crop in ORIGINAL coordinates, then resize while retaining exact metadata."""
    cropped, metadata = crop_roi(image, box, padding)
    return PreparedROI(resize_image(cropped, size), metadata)

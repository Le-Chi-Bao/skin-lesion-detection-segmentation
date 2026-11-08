"""Pixel geometry: sizes are (height, width); boxes use half-open xyxy edges."""

from dataclasses import dataclass
from math import ceil, floor, isfinite
from numbers import Integral

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

Size = tuple[int, int]


def _size(size: Size) -> Size:
    if len(size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in size
    ):
        raise ValueError("size must be (height, width), both positive integers")
    return int(size[0]), int(size[1])


def _rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be RGB uint8 with shape (height, width, 3)")
    _size(image.shape[:2])
    return image


def as_binary_mask(mask: ArrayLike) -> NDArray[np.bool_]:
    """Validate a 2D binary mask encoded as bool, 0/1 or 0/255.

    Probability maps must be explicitly thresholded by the caller first.
    """
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind not in "buif":
        raise ValueError("mask must be a 2D binary numeric array")
    _size(array.shape)
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError("mask must contain only binary values (0, 1 or 255)")
    return array.astype(bool)


@dataclass(frozen=True)
class BoundingBox:
    """Continuous xyxy edges: x_max/y_max are exclusive, never width-1."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in self.xyxy):
            raise ValueError("Bounding box coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding box must have positive width and height")

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class CropMetadata:
    """Exact integer crop window in the original image, before ROI resize."""

    original_size: Size
    window: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        height, width = _size(self.original_size)
        if len(self.window) != 4 or any(
            isinstance(v, bool) or not isinstance(v, Integral) for v in self.window
        ):
            raise ValueError("Crop window must contain four integers")
        x1, y1, x2, y2 = self.window
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Crop window must lie inside the original image")

    @property
    def crop_size(self) -> Size:
        x1, y1, x2, y2 = self.window
        return y2 - y1, x2 - x1


def resize_image(image: NDArray[np.uint8], size: Size) -> NDArray[np.uint8]:
    """Resize RGB with area interpolation when shrinking, linear otherwise."""
    image = _rgb(image)
    height, width = _size(size)
    shrinking = height <= image.shape[0] and width <= image.shape[1]
    method = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=method)


def resize_mask(mask: ArrayLike, size: Size) -> NDArray[np.bool_]:
    """Nearest-neighbor resize preserves binary labels, not small structures."""
    binary = as_binary_mask(mask)
    height, width = _size(size)
    resized = cv2.resize(
        binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def mask_to_bbox(mask: ArrayLike) -> BoundingBox | None:
    """Return the tight union box of foreground pixels, or None if empty."""
    binary = as_binary_mask(mask)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return None
    return BoundingBox(
        int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
    )


def scale_bbox(box: BoundingBox, from_size: Size, to_size: Size) -> BoundingBox:
    """Map box edges between full-image sizes; retain fractional precision.

    For a detector run on a resized image: from_size is detector image size
    and to_size is original image size. Padding/crop happens after mapping.
    """
    from_h, from_w = _size(from_size)
    to_h, to_w = _size(to_size)
    return BoundingBox(
        box.x_min * to_w / from_w,
        box.y_min * to_h / from_h,
        box.x_max * to_w / from_w,
        box.y_max * to_h / from_h,
    )


def crop_roi(
    image: NDArray[np.uint8], box: BoundingBox, padding: float = 0.0
) -> tuple[NDArray[np.uint8], CropMetadata]:
    """Pad by a fraction of each box dimension, round outward, clip to image.

    The box must overlap the image before padding. Always use the returned
    metadata for restoration, never the unpadded or unclipped input box.
    """
    image = _rgb(image)
    if not isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and non-negative")
    height, width = image.shape[:2]
    if box.x_max <= 0 or box.y_max <= 0 or box.x_min >= width or box.y_min >= height:
        raise ValueError("Bounding box does not overlap the image")
    pad_x = (box.x_max - box.x_min) * padding
    pad_y = (box.y_max - box.y_min) * padding
    x1 = max(0, floor(box.x_min - pad_x))
    y1 = max(0, floor(box.y_min - pad_y))
    x2 = min(width, ceil(box.x_max + pad_x))
    y2 = min(height, ceil(box.y_max + pad_y))
    metadata = CropMetadata((height, width), (x1, y1, x2, y2))
    return image[y1:y2, x1:x2].copy(), metadata


def crop_mask(mask: ArrayLike, metadata: CropMetadata) -> NDArray[np.bool_]:
    """Apply an image's exact crop window to its ground-truth mask."""
    binary = as_binary_mask(mask)
    if binary.shape != metadata.original_size:
        raise ValueError("Mask dimensions must match the original image")
    x1, y1, x2, y2 = metadata.window
    return binary[y1:y2, x1:x2].copy()


def restore_roi_mask(
    roi_mask: ArrayLike, metadata: CropMetadata
) -> NDArray[np.bool_]:
    """Resize binary ROI labels back to the crop, paste on an empty original.

    A size-changing resize can lose boundaries; exact inversion is not
    guaranteed. This function restores coordinates, not lost information.
    """
    mask = resize_mask(roi_mask, metadata.crop_size)
    restored = np.zeros(metadata.original_size, dtype=bool)
    x1, y1, x2, y2 = metadata.window
    restored[y1:y2, x1:x2] = mask
    return restored

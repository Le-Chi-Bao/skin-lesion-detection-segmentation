"""RGB uint8 images and binary boolean masks in stored pixel orientation."""

from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageSource = str | Path | BinaryIO


def read_rgb(source: ImageSource) -> NDArray[np.uint8]:
    """Read RGB (H, W, 3); do not apply EXIF rotation independently of masks."""
    with Image.open(source) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def read_binary_mask(source: ImageSource) -> NDArray[np.bool_]:
    """Read an ISIC-style grayscale mask; every nonzero pixel is foreground.

    Ground-truth masks should be lossless PNGs; this is not a probability
    thresholding function. Outputs always contain only False and True.
    """
    with Image.open(source) as mask:
        return np.array(mask.convert("L")) > 0

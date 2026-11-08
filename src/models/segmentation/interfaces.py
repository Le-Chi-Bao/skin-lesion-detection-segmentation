"""One contract for a future U-Net on ROI or full-image inputs."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Segmenter(Protocol):
    """Accept RGB uint8, return a binary mask at the supplied image's size."""

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Implementations own normalization, thresholding and output resizing."""
        ...

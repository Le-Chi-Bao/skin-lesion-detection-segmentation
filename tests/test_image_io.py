"""Lossless synthetic image/mask I/O; no dataset files required."""

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from src.data.image_io import read_binary_mask, read_rgb


def test_read_rgb_preserves_channels(tmp_path):
    path = tmp_path / "ảnh.png"
    pixels = np.zeros((3, 5, 3), dtype=np.uint8)
    pixels[:] = [213, 42, 17]
    Image.fromarray(pixels).save(path)
    np.testing.assert_array_equal(read_rgb(path), pixels)


def test_read_grayscale_as_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((4, 6), 90, dtype=np.uint8)).save(path)
    result = read_rgb(path)
    assert result.shape == (4, 6, 3)
    assert result.dtype == np.uint8
    assert (result == 90).all()


@pytest.mark.parametrize("foreground", [1, 255])
def test_read_binary_mask(tmp_path, foreground):
    path = tmp_path / "mask.png"
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = foreground
    Image.fromarray(mask).save(path)
    result = read_binary_mask(path)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, mask > 0)


def test_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_rgb(tmp_path / "missing.png")


def test_corrupt_image(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        read_rgb(path)

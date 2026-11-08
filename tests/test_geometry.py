"""Coordinate contracts including edge pixels, padding and non-square resize."""

import numpy as np
import pytest

from src.data.geometry import (
    BoundingBox,
    CropMetadata,
    as_binary_mask,
    crop_mask,
    crop_roi,
    mask_to_bbox,
    resize_image,
    resize_mask,
    restore_roi_mask,
    scale_bbox,
)
from src.pipeline.spatial import prepare_roi


def test_bbox_exclusive_maxima_and_disjoint_foreground():
    mask = np.zeros((8, 11), dtype=bool)
    mask[2:5, 3:7] = True
    mask[7, 10] = True
    assert mask_to_bbox(mask) == BoundingBox(3, 2, 11, 8)


def test_empty_mask_has_no_box():
    assert mask_to_bbox(np.zeros((5, 7), dtype=bool)) is None


@pytest.mark.parametrize("row,col", [(0, 0), (3, 6)])
def test_single_pixel_box_round_trip(row, col):
    mask = np.zeros((4, 7), dtype=bool)
    mask[row, col] = True
    box = mask_to_bbox(mask)
    assert box == BoundingBox(col, row, col + 1, row + 1)
    _, metadata = crop_roi(np.zeros((4, 7, 3), dtype=np.uint8), box)
    np.testing.assert_array_equal(restore_roi_mask([[1]], metadata), mask)


def test_full_image_mask_includes_last_row_and_column():
    mask = np.ones((4, 7), dtype=bool)
    assert mask_to_bbox(mask) == BoundingBox(0, 0, 7, 4)


def test_padding_is_fractional_per_axis_and_clipped():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    cropped, metadata = crop_roi(image, BoundingBox(0, 1, 4, 5), padding=0.5)
    assert metadata.window == (0, 0, 6, 7)
    assert cropped.shape == (7, 6, 3)
    assert not np.shares_memory(cropped, image)


def test_float_coordinates_round_outward():
    _, meta = crop_roi(
        np.zeros((10, 20, 3), dtype=np.uint8), BoundingBox(2.2, 1.8, 8.1, 7.2)
    )
    assert meta.window == (2, 1, 9, 8)


def test_scale_bbox_handles_non_square_detector_resize():
    box = BoundingBox(10, 5, 40, 25)
    original = scale_bbox(box, (50, 100), (200, 300))
    assert original == BoundingBox(30, 20, 120, 100)
    assert scale_bbox(original, (200, 300), (50, 100)) == box


def test_rgb_resize_uses_height_width_order_and_preserves_channels():
    image = np.full((8, 12, 3), [12, 70, 255], dtype=np.uint8)
    for size in [(3, 5), (15, 20)]:
        result = resize_image(image, size)
        assert result.shape == (*size, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[0, 0], image[0, 0])


def test_mask_resize_is_nearest_neighbor_and_binary():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    expected = np.repeat(np.repeat(mask > 0, 2, axis=0), 3, axis=1)
    np.testing.assert_array_equal(resize_mask(mask, (4, 6)), expected)
    assert resize_mask(mask, (4, 6)).dtype == np.bool_


def test_roi_restore_resized_prediction_to_exact_window():
    image = np.zeros((10, 16, 3), dtype=np.uint8)
    roi = prepare_roi(image, BoundingBox(4, 2, 12, 8), (3, 4))
    assert roi.image.shape == (3, 4, 3)
    restored = restore_roi_mask(np.ones((3, 4), dtype=bool), roi.metadata)
    expected = np.zeros((10, 16), dtype=bool)
    expected[2:8, 4:12] = True
    np.testing.assert_array_equal(restored, expected)


@pytest.mark.parametrize("padding", [0, 0.2, 1.0])
def test_mask_bbox_crop_restore_round_trip(padding):
    image = np.zeros((13, 21, 3), dtype=np.uint8)
    mask = np.zeros((13, 21), dtype=bool)
    mask[0:8, 5:17] = True
    mask[3:5, 9:11] = False
    _, metadata = crop_roi(image, mask_to_bbox(mask), padding)
    restored = restore_roi_mask(crop_mask(mask, metadata), metadata)
    np.testing.assert_array_equal(restored, mask)


@pytest.mark.parametrize("padding", [-0.1, float("nan"), float("inf")])
def test_invalid_padding(padding):
    with pytest.raises(ValueError, match="padding"):
        crop_roi(np.zeros((4, 7, 3), dtype=np.uint8), BoundingBox(1, 1, 3, 3), padding)


@pytest.mark.parametrize("box", [(2, 2, 2, 3), (3, 1, 1, 2), (0, 0, 1, float("nan"))])
def test_invalid_box(box):
    with pytest.raises(ValueError):
        BoundingBox(*box)


def test_outside_box_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        crop_roi(np.zeros((4, 7, 3), dtype=np.uint8), BoundingBox(7, 1, 9, 3))


@pytest.mark.parametrize("mask", [np.zeros((2, 2, 3)), [[0.2, 0.8]], [[0, np.nan]]])
def test_non_binary_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        as_binary_mask(mask)


@pytest.mark.parametrize("size", [(0, 2), (2, -1), (2.5, 3), (True, 3)])
def test_invalid_resize_size(size):
    with pytest.raises(ValueError):
        resize_mask([[1]], size)


def test_crop_mask_requires_matching_image_shape():
    with pytest.raises(ValueError, match="dimensions"):
        crop_mask(np.zeros((3, 4)), CropMetadata((5, 7), (1, 1, 4, 3)))


@pytest.mark.parametrize("window", [(0, 0, 8, 4), (-1, 0, 2, 3), (1, 1, 1, 3)])
def test_invalid_crop_metadata(window):
    with pytest.raises(ValueError):
        CropMetadata((4, 7), window)

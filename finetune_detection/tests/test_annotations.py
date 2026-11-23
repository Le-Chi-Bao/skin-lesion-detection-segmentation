from itertools import product

import numpy as np
import pytest
import torch
from PIL import Image

from src.annotations import mask_to_box, validate_box
from src.transforms import transform_geometry


def test_mask_to_box_uses_exclusive_edges():
    mask = np.zeros((5, 8), np.uint8)
    mask[1, 2] = 1
    mask[4, 7] = 255
    assert mask_to_box(mask) == [2, 1, 8, 5]
    assert validate_box([2, 1, 8, 5], width=8, height=5) == 24
    assert mask_to_box(np.full((3, 1), 255, np.uint8)) == [0, 0, 1, 3]


@pytest.mark.parametrize("mask", [np.zeros((2, 2)), np.full((2, 2), 100), np.ones((2, 2, 3))])
def test_rejects_invalid_masks(mask):
    with pytest.raises(ValueError):
        mask_to_box(mask)


@pytest.mark.parametrize("horizontal,vertical,turns", list(product((False, True), (False, True), range(4))))
def test_transform_matches_pixels(horizontal, vertical, turns):
    mask = np.zeros((5, 8), np.uint8)
    mask[1:3, 4:8] = 255
    original = torch.tensor([mask_to_box(mask)], dtype=torch.float32)
    image, boxes = transform_geometry(Image.fromarray(mask), original, horizontal, vertical, turns)
    assert boxes.tolist() == [mask_to_box(image)]
    assert original.tolist() == [[4, 1, 8, 3]]

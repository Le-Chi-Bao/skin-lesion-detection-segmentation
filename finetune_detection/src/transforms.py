"""Joint image and xyxy box transforms on original-resolution images."""

import random

import torch
from torchvision.transforms import functional as F


def transform_geometry(image, boxes, horizontal=False, vertical=False, turns=0):
    boxes = boxes.clone()
    width, height = image.size
    if horizontal:
        image = F.hflip(image)
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    if vertical:
        image = F.vflip(image)
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
    turns %= 4
    if turns:
        # PIL rotates exactly in 90-degree increments without interpolation.
        from PIL import Image

        rotations = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}
        image = image.transpose(rotations[turns])
        x1, y1, x2, y2 = boxes.unbind(1)
        if turns == 1:
            boxes = torch.stack((y1, width - x2, y2, width - x1), dim=1)
        elif turns == 2:
            boxes = torch.stack((width - x2, height - y2, width - x1, height - y1), dim=1)
        else:
            boxes = torch.stack((height - y2, x1, height - y1, x2), dim=1)
    return image, boxes


class DetectionTransform:
    def __init__(self, options=None):
        self.options = options or {}

    def __call__(self, image, target):
        image, target["boxes"] = transform_geometry(
            image, target["boxes"],
            horizontal=self.options.get("horizontal_flip", False) and random.random() < 0.5,
            vertical=self.options.get("vertical_flip", False) and random.random() < 0.5,
            turns=random.randrange(4) if self.options.get("rotate_90") else 0,
        )
        if self.options.get("brightness_contrast") and random.random() < 0.5:
            image = F.adjust_brightness(image, random.uniform(0.90, 1.10))
            image = F.adjust_contrast(image, random.uniform(0.90, 1.10))
        return F.pil_to_tensor(image).to(torch.float32).div_(255), target

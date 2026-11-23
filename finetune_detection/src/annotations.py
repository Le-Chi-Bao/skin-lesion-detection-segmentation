"""Mask annotations with exclusive right and bottom coordinates."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_box(box, width, height):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid box {box} for image {width}x{height}")
    return (x2 - x1) * (y2 - y1)


def mask_to_box(mask):
    pixels = np.asarray(mask)
    if pixels.ndim != 2 or not np.isin(pixels, [0, 1, 255]).all():
        raise ValueError("Expected a single-channel binary mask with values 0, 1 or 255")
    foreground = pixels != 0
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if not rows.size:
        raise ValueError("Empty lesion mask")
    box = [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]
    validate_box(box, pixels.shape[1], pixels.shape[0])
    return box


def index_files(directory, mask=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing extracted dataset directory: {directory}")
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if mask:
            if path.suffix.lower() != ".png" or not path.stem.endswith("_segmentation"):
                continue
            image_id = path.stem.removesuffix("_segmentation")
        else:
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            image_id = path.stem
        if image_id in files:
            raise ValueError(f"Duplicate ID {image_id}: {files[image_id]} and {path}")
        files[image_id] = path
    return files


def build_annotations(root, images, masks):
    root = Path(root)
    images, masks = index_files(images), index_files(masks, mask=True)
    records, invalid = [], []
    for index, image_id in enumerate(sorted(images.keys() | masks.keys()), 1):
        image_path, mask_path = images.get(image_id), masks.get(image_id)
        try:
            if image_path is None or mask_path is None:
                raise ValueError("Missing image or ground-truth mask")
            if not image_id.startswith("ISIC_") or not image_id[5:].isdigit():
                raise ValueError("Unexpected ISIC image ID")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                width, height = image.size
                image.load()
                if mask.size != image.size:
                    raise ValueError(f"Mask dimensions {mask.size} differ from image {image.size}")
                box = mask_to_box(mask)
            records.append({
                "id": image_id, "numeric_id": index, "width": width, "height": height,
                "box": box, "area": validate_box(box, width, height), "label": 1,
                "image": image_path.relative_to(root).as_posix(),
                "mask": mask_path.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(mask_path),
            })
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            invalid.append({"id": image_id, "error": str(error)})
    return records, invalid, len(images), len(masks)

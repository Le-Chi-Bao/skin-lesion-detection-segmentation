from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import prepare_data
from src.annotations import build_annotations
from src.dataset import LesionDataset, detection_collate, load_manifest, split_records


def encoded(array, format):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format=format)
    return stream.getvalue()


def archives_fixture(directory, count=5, invalid=False):
    directory.mkdir()
    images, masks, _ = prepare_data.ARCHIVES["training"]
    with ZipFile(directory / images, "w") as inputs, ZipFile(directory / masks, "w") as targets:
        for index in range(count):
            name = f"ISIC_{index:07d}"
            photo = np.full((6, 8, 3), index * 20, np.uint8)
            shape = (4, 8) if invalid and index == 0 else (6, 8)
            mask = np.zeros(shape, np.uint8)
            mask[1:3, 2:6] = 255
            inputs.writestr(f"images/{name}.jpg", encoded(photo, "JPEG"))
            targets.writestr(f"masks/{name}_segmentation.png", encoded(mask, "PNG"))


def test_prepare_archives_pair_and_persist(tmp_path, monkeypatch):
    source = tmp_path / "drive"
    archives_fixture(source)
    images, masks, _ = prepare_data.ARCHIVES["training"]
    monkeypatch.setitem(prepare_data.ARCHIVES, "training", (images, masks, 5))
    root = tmp_path / "runtime"
    path = prepare_data.prepare_dataset(root, source)
    manifest = load_manifest(path, verify_files=True)
    assert manifest["image_count"] == len(manifest["records"]) == 5
    assert len(manifest["splits"]["train"]) == 4 and len(manifest["splits"]["validation"]) == 1
    assert not set(manifest["splits"]["train"]) & set(manifest["splits"]["validation"])
    assert all((row["box"], row["area"]) == ([2, 1, 6, 3], 8) for row in manifest["records"])
    destination = source / "manifest.json"
    prepare_data.persist_manifest(path, destination)
    assert load_manifest(destination)["sha256"] == manifest["sha256"]
    assert prepare_data.prepare_dataset(root, source) == path
    assert load_manifest(path)["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match="type"):
        load_manifest(path, expected="test")
    dataset = LesionDataset(path, manifest, "train")
    images, targets = detection_collate([dataset[0], dataset[1]])
    assert images[0].dtype == torch.float32 and tuple(images[0].shape) == (3, 6, 8)
    assert 0 <= images[0].min() <= images[0].max() <= 1
    assert targets[0]["boxes"].shape == (1, 4) and targets[0]["boxes"].dtype == torch.float32
    assert targets[0]["labels"].tolist() == [1] and targets[0]["labels"].dtype == torch.int64
    (root / manifest["records"][0]["image"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path, verify_files=True)


def test_invalid_report_and_explicit_exclusion(tmp_path, monkeypatch):
    source = tmp_path / "drive"
    archives_fixture(source, invalid=True)
    images, masks, _ = prepare_data.ARCHIVES["training"]
    monkeypatch.setitem(prepare_data.ARCHIVES, "training", (images, masks, 5))
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="1 invalid samples"):
        prepare_data.prepare_dataset(runtime, source)
    assert "Mask dimensions" in (runtime / "invalid_samples.json").read_text()
    manifest = load_manifest(prepare_data.prepare_dataset(runtime, source, invalid_policy="exclude"))
    assert len(manifest["records"]) == 4
    assert manifest["invalid"][0]["id"] == "ISIC_0000000"


def test_missing_pairs_and_duplicate_ids_are_reported(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    photo = encoded(np.zeros((6, 8, 3), np.uint8), "JPEG")
    (images / "ISIC_0000001.jpg").write_bytes(photo)
    records, errors, image_count, mask_count = build_annotations(tmp_path, images, masks)
    assert (image_count, mask_count) == (1, 0)
    assert not records and "Missing" in errors[0]["error"]
    (images / "ISIC_0000001.jpeg").write_bytes(photo)
    with pytest.raises(ValueError, match="Duplicate ID"):
        build_annotations(tmp_path, images, masks)


def test_split_is_reproducible_with_duplicate_content_and_groups():
    rows = [{"id": str(i), "image_sha256": str(i)} for i in range(25)]
    rows[2]["image_sha256"] = rows[1]["image_sha256"]
    groups = {"0": "lesion-A", "1": "lesion-A"}
    split = split_records(rows, groups, seed=42)
    assert split == split_records(list(reversed(rows)), groups, seed=42)
    assert split != split_records(rows, groups, seed=43)
    assert 4 <= len(split["validation"]) <= 6
    assert all(set(("0", "1", "2")).issubset(ids) or not set(("0", "1", "2")).intersection(ids) for ids in split.values())

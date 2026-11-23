"""Portable ISIC manifests, reproducible splits and Torchvision targets."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .annotations import digest_json, sha256_file, validate_box


def read_groups(path):
    groups = {}
    if path is None:
        return groups
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"image", "lesion_id"}.issubset(rows.fieldnames or []):
            raise ValueError("Grouping CSV must contain image and lesion_id columns")
        for row in rows:
            image_id, group = row["image"].strip(), row["lesion_id"].strip()
            if group:
                if image_id in groups and groups[image_id] != group:
                    raise ValueError(f"Conflicting groups for {image_id}")
                groups[image_id] = group
    return groups


def split_records(records, groups=None, seed=42):
    groups = groups or {}
    parents = {row["id"]: row["id"] for row in records}
    if len(parents) != len(records) or len(parents) < 2:
        raise ValueError("Splitting requires at least two unique image IDs")

    def root(image_id):
        while image_id != parents[image_id]:
            parents[image_id] = parents[parents[image_id]]
            image_id = parents[image_id]
        return image_id

    links = {}
    for row in sorted(records, key=lambda row: row["id"]):
        image_id = row["id"]
        keys = [("sha256", row["image_sha256"])]
        if image_id in groups:
            keys.append(("lesion", groups[image_id]))
        for key in keys:
            previous = links.setdefault(key, image_id)
            parents[root(image_id)] = root(previous)
    components = defaultdict(list)
    for image_id in sorted(parents):
        components[root(image_id)].append(image_id)
    blocks = list(components.values())
    if len(blocks) < 2:
        raise ValueError("Need at least two independent groups to make a split")
    random.Random(seed).shuffle(blocks)
    target = max(1, round(len(records) * 0.2))
    validation = []
    for block in blocks:
        combined = len(validation) + len(block)
        if combined < len(records) and abs(combined - target) < abs(len(validation) - target):
            validation.extend(block)
    if not validation:
        validation = min(blocks, key=len)
    return {"train": sorted(parents.keys() - set(validation)), "validation": sorted(validation)}


def load_manifest(path, expected="training", verify_files=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}. Run scripts.prepare_data first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed = manifest.pop("sha256", None)
    if manifest.get("version") != 1 or manifest.get("dataset") != expected or digest_json(manifest) != claimed:
        raise ValueError(f"Incorrect manifest type, version or checksum: {path}")
    valid = {row["id"]: row for row in manifest["records"]}
    expected_keys = {"train", "validation"} if expected == "training" else {"test"}
    assigned = [image_id for split in manifest["splits"].values() for image_id in split]
    if (set(manifest["splits"]) != expected_keys or not all(manifest["splits"].values())
            or len(assigned) != len(set(assigned)) or set(assigned) != valid.keys()
            or len(valid) != len(manifest["records"])
            or len({row["numeric_id"] for row in valid.values()}) != len(valid)):
        raise ValueError("Manifest has empty splits, duplicate IDs or mismatched membership")
    seen = {}
    for split, ids in manifest["splits"].items():
        for image_id in ids:
            row = valid[image_id]
            keys = [("image", row["image_sha256"])]
            if row.get("lesion_group"):
                keys.append(("lesion", row["lesion_group"]))
            if any(seen.setdefault(key, split) != split for key in keys):
                raise ValueError("Identical image content or lesion group crosses splits")
    for row in valid.values():
        if row["label"] != 1 or row["area"] != validate_box(row["box"], row["width"], row["height"]):
            raise ValueError(f"Invalid target for {row['id']}")
        for kind in ("image", "mask"):
            relative = Path(row[kind])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid manifest path: {relative}")
            if verify_files and sha256_file(path.parent / relative) != row[f"{kind}_sha256"]:
                raise ValueError(f"Dataset content changed: {relative}")
    manifest["sha256"] = claimed
    return manifest


class LesionDataset:
    def __init__(self, manifest_path, manifest, split, augmentation=None):
        from .transforms import DetectionTransform

        self.root = Path(manifest_path).parent
        records = {row["id"]: row for row in manifest["records"]}
        self.records = [records[image_id] for image_id in manifest["splits"][split]]
        self.transform = DetectionTransform(augmentation)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from PIL import Image

        row = self.records[index]
        path = self.root / row["image"]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except OSError as error:
            raise RuntimeError(f"Cannot read image {path}; rerun scripts.verify_data") from error
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions differ from manifest: {path}")
        target = {
            "boxes": torch.tensor([row["box"]], dtype=torch.float32),
            "labels": torch.tensor([row["label"]], dtype=torch.int64),
            "image_id": torch.tensor(row["numeric_id"], dtype=torch.int64),
            "area": torch.tensor([row["area"]], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return self.transform(image, target)


def detection_collate(samples):
    return tuple(zip(*samples))

"""Typed YAML configuration with shallow experiment files and project paths."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    dataset_root: Path
    images_dir: Path
    masks_dir: Path
    splits_dir: Path
    checkpoint_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    paths: ProjectPaths
    image_size: tuple[int, int]
    roi_padding: float
    batch_size: int
    seed: int
    device: str
    model_name: str
    input_mode: str
    detector_checkpoint: Path | None
    segmenter_checkpoint: Path | None


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path, active: frozenset[Path]) -> dict[str, Any]:
    if path in active:
        raise ValueError(f"Circular configuration inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    parent = values.pop("extends", None)
    if parent is None:
        return values
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("extends must be a non-empty YAML path")
    base = _read_yaml((path.parent / parent).resolve(), active | {path})
    return _merge(base, values)


def _resolve_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be non-empty strings")
    path = Path(value).expanduser()
    return (root / path).resolve()


def load_config(
    config_path: str | Path = "configs/base.yaml",
    *,
    project_root: str | Path | None = None,
) -> ProjectConfig:
    """Load YAML without creating folders or requiring a dataset/checkpoint.

    Config paths and configured data paths are relative to project_root.
    An extends path is relative to its containing YAML file.
    """
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    values = _read_yaml((root / config_path).resolve(), frozenset())
    expected = set(ProjectConfig.__dataclass_fields__)
    if set(values) != expected:
        raise ValueError(
            f"Invalid config keys; missing={expected - set(values)}, "
            f"unknown={set(values) - expected}"
        )
    paths = values["paths"]
    if not isinstance(paths, dict) or set(paths) != set(
        ProjectPaths.__dataclass_fields__
    ):
        raise ValueError("paths must contain exactly the ProjectPaths fields")
    size = values["image_size"]
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(item) is not int or item <= 0 for item in size)
    ):
        raise ValueError("image_size must be [height, width], both positive integers")
    for key in ("batch_size", "seed"):
        if type(values[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if values["batch_size"] <= 0 or not 0 <= values["seed"] < 2**32:
        raise ValueError("batch_size must be positive; seed must be in [0, 2**32)")
    padding = values["roi_padding"]
    if type(padding) not in (int, float) or not 0 <= padding < float("inf"):
        raise ValueError("roi_padding must be finite and non-negative")
    allowed = {
        "device": ("auto", "cpu", "cuda"),
        "model_name": ("two_stage", "faster_rcnn", "unet"),
        "input_mode": ("roi", "full_image"),
    }
    for key, choices in allowed.items():
        if values[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    if (
        not isinstance(values["project_name"], str)
        or not values["project_name"].strip()
    ):
        raise ValueError("project_name must be a non-empty string")
    values["paths"] = ProjectPaths(
        **{key: _resolve_path(value, root) for key, value in paths.items()}
    )
    for key in ("detector_checkpoint", "segmenter_checkpoint"):
        if values[key] is not None:
            values[key] = _resolve_path(values[key], root)
    values["image_size"] = tuple(size)
    values["roi_padding"] = float(padding)
    return ProjectConfig(**values)

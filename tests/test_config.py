"""Configuration inheritance, validation and working-directory independence."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import PROJECT_ROOT, load_config


@pytest.mark.parametrize(
    "filename,model,mode,size",
    [
        ("base.yaml", "two_stage", "roi", (256, 256)),
        ("faster_rcnn.yaml", "faster_rcnn", "full_image", (512, 512)),
        ("unet_roi_256.yaml", "unet", "roi", (256, 256)),
        ("unet_roi_512.yaml", "unet", "roi", (512, 512)),
        ("unet_full_image_512.yaml", "unet", "full_image", (512, 512)),
    ],
)
def test_all_experiment_configs(filename, model, mode, size):
    config = load_config(Path("configs") / filename)
    assert config.model_name == model
    assert config.input_mode == mode
    assert config.image_size == size
    assert config.paths.images_dir.is_absolute()
    assert config.paths.splits_dir == PROJECT_ROOT / "data/splits"
    assert config.detector_checkpoint is None
    assert config.segmenter_checkpoint is None


def test_config_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.paths.checkpoint_dir == PROJECT_ROOT / "checkpoints"


def test_root_override_and_checkpoint_path(tmp_path):
    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text("utf-8"))
    base["detector_checkpoint"] = "checkpoints/detector.pth"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    config = load_config("local.yaml", project_root=tmp_path)
    assert config.paths.images_dir.is_relative_to(tmp_path)
    assert config.detector_checkpoint == tmp_path / "checkpoints/detector.pth"
    assert not config.detector_checkpoint.exists()  # loading never creates data


@pytest.mark.parametrize(
    "key,value",
    [
        ("image_size", [0, 256]),
        ("image_size", [256]),
        ("image_size", [True, 256]),
        ("batch_size", 0),
        ("batch_size", True),
        ("seed", -1),
        ("seed", 2**32),
        ("device", "tpu"),
        ("roi_padding", -0.5),
        ("roi_padding", float("nan")),
        ("model_name", "unknown"),
        ("input_mode", "unknown"),
        ("project_name", ""),
        ("detector_checkpoint", ""),
        ("typo", 123),
    ],
)
def test_invalid_config_values(tmp_path, key, value):
    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text("utf-8"))
    base[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_circular_inheritance_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")

"""Startup selection and failure isolation; synthetic loaders are unit tests only."""

import asyncio
import json
import logging
import os
import runpy
from pathlib import Path

import gradio as gr
import pytest
from gradio.state_holder import SessionState as GradioSessionState
from PIL import Image

from app.config import PROJECT_ROOT, AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.temporary_files import TemporaryFiles
from app.cv.availability import ModelAvailability, ModelStatus
from app.cv.orchestrator import create_orchestrator
from app.cv.registry import ModelRegistry
from app.cv.schemas import ModelIdentity
from app.ui.shell import build_shell


@pytest.fixture
def registry_data():
    # Preserve the historical E1 selection scenarios independently of deployment.
    return json.loads((Path(__file__).parent / "fixtures/cv_registry_e1.json").read_text(encoding="utf-8-sig"))


def write_registry(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def project_env(root, **values):
    return {"LESION_RUNTIME_DIR": str(root / "runtime"),
            "LESION_MODEL_STORE_DIR": str(root / "model_store"), **values}


def test_default_registry_is_project_relative_even_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.startswith("LESION_"):
            monkeypatch.delenv(key)
    config = AppConfig.from_env()
    assert config.model_registry == PROJECT_ROOT / "config/models.json"
    assert config.segformer_device == config.detection_device == "auto"
    assert not config.detection_pilot_mode


@pytest.mark.parametrize("relative", [False, True])
def test_registry_environment_override(tmp_path, monkeypatch, relative):
    from app import config as config_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    chosen = Path("custom/models.json") if relative else tmp_path / "custom/models.json"
    monkeypatch.setenv("LESION_MODEL_REGISTRY", str(chosen))
    for key, value in project_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    assert AppConfig.from_env().model_registry == tmp_path / "custom/models.json"


def test_empty_override_is_an_error():
    with pytest.raises(AppError) as error:
        AppConfig.from_env({"LESION_MODEL_REGISTRY": " "})
    assert error.value.code is ErrorCode.INVALID_CONFIG
    assert error.value.field == "MODEL_REGISTRY"


def test_missing_default_registry_preserves_explicit_legacy_checkpoint(tmp_path, monkeypatch, caplog):
    from app import config as config_module
    from app.cv import segformer

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    calls = []

    def load(config):
        calls.append(config.segformer_checkpoint)
        return object(), "cpu"

    monkeypatch.setattr(segformer, "_load_verified_model", load)
    config = AppConfig.from_env(project_env(tmp_path, LESION_SEGFORMER_CHECKPOINT="legacy.pt"))
    assert config.model_registry is None
    models = create_orchestrator(config)
    assert calls == [tmp_path / "legacy.pt"]
    assert models.segmentation.availability.status is ModelStatus.READY
    assert models.detection.availability.status is ModelStatus.NOT_AVAILABLE
    assert str(tmp_path / "config/models.json") in caplog.text
    assert "chế độ SegFormer cũ" in caplog.text


@pytest.mark.parametrize("content", [None, "{invalid", '{"schema_version": 2, "models": []}'])
@pytest.mark.parametrize("explicit_override", [False, True])
def test_selected_bad_registry_never_falls_back(tmp_path, monkeypatch, caplog, content, explicit_override):
    from app import config as config_module
    from app.cv import segformer

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "config/models.json"
    if content is not None:
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
    if not explicit_override and content is None:
        # The default was selected while present, then disappeared before loading.
        write_registry(path, {"schema_version": 1, "models": []})
    env = project_env(tmp_path)
    if explicit_override:
        env["LESION_MODEL_REGISTRY"] = str(path)
    config = AppConfig.from_env(env)
    if content is None:
        path.unlink(missing_ok=True)
    monkeypatch.setattr(segformer, "SegFormer", lambda *a: pytest.fail("No legacy fallback allowed"))
    models = create_orchestrator(config)
    for adapter in (models.segmentation, models.detection):
        assert adapter.availability.status is ModelStatus.FAILED
        assert adapter.error.code is ErrorCode.INVALID_MODEL_CONFIG
    assert str(path) in caplog.text
    assert "Không thể đọc registry" in caplog.text
    assert "SegFormer: FAILED" in caplog.text and "Faster R-CNN: FAILED" in caplog.text


def test_missing_checkpoints_report_selected_versions_and_paths(tmp_path, registry_data, caplog):
    for entry in registry_data["models"]:
        entry["checkpoint_path"] = str(tmp_path / f"{entry['version']}.pt")
    path = write_registry(tmp_path / "models.json", registry_data)
    models = create_orchestrator(AppConfig.from_env({"LESION_MODEL_REGISTRY": str(path)}))
    for adapter in (models.segmentation, models.detection):
        assert adapter.availability.status is not ModelStatus.READY
        assert adapter.error.code is ErrorCode.CHECKPOINT_MISSING
        assert adapter.identity is None
    assert "SegFormer: UNAVAILABLE | version=checkpoint-v1" in caplog.text
    assert "Faster R-CNN: FAILED | version=E1-best-completed" in caplog.text
    for entry in registry_data["models"][:2]:
        assert entry["checkpoint_path"] in caplog.text
    assert "Chưa có checkpoint Faster R-CNN" in caplog.text


def test_invalid_checkpoint_path_does_not_crash_diagnostics(tmp_path, registry_data, caplog):
    registry_data["models"][0]["checkpoint_path"] = "invalid\0.pt"
    registry_data["models"][1]["active"] = False
    path = write_registry(tmp_path / "models.json", registry_data)
    models = create_orchestrator(AppConfig(model_registry=path))
    assert models.segmentation.availability.status is ModelStatus.FAILED
    assert models.segmentation.error.code is ErrorCode.INVALID_MODEL_CONFIG
    assert "invalid\\x00.pt" in caplog.text
    assert models.detection.availability.status is ModelStatus.NOT_AVAILABLE


@pytest.mark.parametrize("failed", [None, "segmentation", "detection"])
def test_active_models_load_independently_with_actual_devices(tmp_path, registry_data, monkeypatch, caplog, failed):
    from app.cv import detection, segformer

    loaded = []

    def load_seg(config):
        loaded.append(config.segformer_checkpoint)
        if failed == "segmentation":
            raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE)
        return object(), "cpu"

    def load_det(adapter):
        loaded.append(adapter.entry.path)
        if failed == "detection":
            adapter.error = AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE)
            adapter.availability = ModelAvailability(adapter.entry.model_id, ModelStatus.FAILED)
            return
        adapter.identity = ModelIdentity(
            model_id=adapter.entry.model_id, model_version=adapter.entry.version,
            checkpoint_id=adapter.entry.version, checkpoint_sha256=adapter.entry.sha256, device="cpu",
        )
        adapter.availability = ModelAvailability(adapter.entry.model_id, ModelStatus.READY, adapter.entry.version)

    monkeypatch.setattr(segformer, "_load_verified_model", load_seg)
    monkeypatch.setattr(detection.FasterRCNN, "load", load_det)
    caplog.set_level(logging.INFO)
    path = write_registry(tmp_path / "models.json", registry_data)
    models = create_orchestrator(AppConfig.from_env({"LESION_MODEL_REGISTRY": str(path)}))
    assert loaded == [PROJECT_ROOT / entry["checkpoint_path"] for entry in registry_data["models"][:2]]
    for module in ("segmentation", "detection"):
        adapter = getattr(models, module)
        assert adapter.availability.status is (ModelStatus.FAILED if module == failed else ModelStatus.READY)
        if module != failed:
            assert adapter.identity.device == "cpu"
    assert "Registry:" in caplog.text and "device=cpu" in caplog.text
    assert "version=checkpoint-v1" in caplog.text and "version=E1-best-completed" in caplog.text
    assert "pilot" not in caplog.text


def test_no_active_models_disables_analyze_after_upload(tmp_path, registry_data, monkeypatch):
    for entry in registry_data["models"]:
        entry["active"] = False
    path = write_registry(tmp_path / "models.json", registry_data)
    config = AppConfig.from_env({"LESION_MODEL_REGISTRY": str(path), "LESION_RUNTIME_DIR": str(tmp_path / "runtime")})
    files = TemporaryFiles(config.runtime_dir, 30)
    files.cache_dir.mkdir(parents=True)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(files.cache_dir))
    demo = build_shell(config, files)
    components = demo.config["components"]
    analyze = next(c for c in components if c["props"].get("elem_id") == "analyze-image")
    assert not analyze["props"]["interactive"]
    assert all(t[0] != analyze["id"] for d in demo.config["dependencies"] for t in d["targets"])
    upload = next(d for d in demo.config["dependencies"] if d["api_name"] == "upload")
    image_path = files.cache_dir / "test.png"
    with Image.new("RGB", (20, 12)) as image:
        image.save(image_path)
    session = GradioSessionState(demo)
    shown = asyncio.run(demo.process_api(
        upload["id"], [gr.FileData(path=str(image_path)).model_dump(), None, 1], state=session,
    ))["data"]
    assert shown[6]["interactive"] is False
    assert "Không có mô hình active trong registry" in shown[8]
    assert "Không có mô hình sẵn sàng" in shown[8]
    assert "Chưa tích hợp" not in shown[8]


def test_e3_can_replace_e1_only_when_selected(tmp_path, registry_data):
    from app.cv.detection import FasterRCNN, validate_artifact
    from test_detection import artifact

    future = dict(registry_data["models"][1], version="E3-verified", active=False,
                  checkpoint_path="model_store/detection/E3-inference.pt", sha256="b" * 64)
    registry_data["models"].append(future)
    path = write_registry(tmp_path / "models.json", registry_data)
    assert ModelRegistry(path).selected("detection").version == "E1-best-completed"
    registry_data["models"][1]["active"] = False
    future["active"] = True
    write_registry(path, registry_data)
    registry = ModelRegistry(path)
    selected = registry.selected("detection")
    assert selected.version == "E3-verified" and selected.sha256 == "b" * 64
    validate_artifact(artifact(), selected, False)
    adapter = registry.create("detection", AppConfig())
    assert isinstance(adapter, FasterRCNN) and adapter.entry is selected
    assert adapter.device == "auto" and not adapter.pilot_mode
    assert adapter.availability.status is ModelStatus.NOT_AVAILABLE


def test_root_entrypoint_only_calls_app_main(monkeypatch):
    from app import main as app_main

    calls = []
    monkeypatch.setattr(app_main, "main", lambda: calls.append(True) or 7)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
    assert calls == [True] and exit_info.value.code == 7

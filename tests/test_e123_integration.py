"""WP-CV4 provenance, activation and opt-in actual artifact verification."""

import copy
import json
import os

import pytest

from app.config import PROJECT_ROOT
from app.cv.registry import ModelRegistry
from app.cv.verify_detection import EXPECTED, validate_selection, verify_checkpoint


def selection_artifact():
    return {"selection": {
        "experiment": "E2", "run_id": "a2301ed5919a4b9089dc3cc73af65e70",
        "best_epoch": 6, "manifest_sha256": "b" * 64,
        "best_map_50_95": 0.7021721840671518, "selection_metric": "validation map_50_95",
    }}


@pytest.mark.parametrize("key,value", [
    ("experiment", "E1"), ("run_id", "c" * 32), ("best_epoch", 9), ("best_epoch", True),
    ("manifest_sha256", "not-a-hash"), ("manifest_sha256", None),
    ("selection_metric", "official_test map_50_95"), ("best_map_50_95", float("nan")),
    ("best_map_50_95", float("inf")), ("best_map_50_95", True), ("best_map_50_95", 1.1),
])
def test_rejects_wrong_selection_provenance(key, value):
    artifact = selection_artifact()
    validate_selection(artifact, "E2", "a2301ed5919a4b9089dc3cc73af65e70", 6)
    artifact["selection"][key] = value
    with pytest.raises(ValueError):
        validate_selection(artifact, "E2", "a2301ed5919a4b9089dc3cc73af65e70", 6)


def test_deployment_registry_and_lossless_e1_rollback(tmp_path):
    data = json.loads((PROJECT_ROOT / "config/models.json").read_text(encoding="utf-8"))
    registry = ModelRegistry(PROJECT_ROOT / "config/models.json")
    assert not registry.errors
    assert registry.selected("segmentation").sha256 == "d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87"
    assert registry.selected("segmentation").threshold == 0.7
    assert registry.selected("detection").version == "E2-a2301ed5919a4b9089dc3cc73af65e70"
    assert registry.selected("detection").sha256 == "9c083aadba4b47cf4b5d5ff794cd23856977ffa65dc35bf5217cfa1e27b4a78c"
    originals = copy.deepcopy(data)
    for experiment, (minimum, maximum, _) in EXPECTED.items():
        matches = [entry for entry in registry.entries if entry.version.startswith(experiment + "-") and not entry.research_pilot]
        assert len(matches) == 1
        assert (matches[0].min_size, matches[0].max_size, matches[0].threshold) == (minimum, maximum, 0.5)
    for experiment in ("E1", "E2"):
        for row in data["models"]:
            if row["module"] == "detection":
                row["active"] = row["version"].startswith(experiment + "-") and not row["research_pilot"]
        path = tmp_path / "models.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        selected = ModelRegistry(path).selected("detection")
        assert selected.version.startswith(experiment + "-")
        assert sum(row["active"] for row in data["models"] if row["module"] == "detection") == 1
        for before, after in zip(originals["models"], data["models"], strict=True):
            assert {k: v for k, v in before.items() if k != "active"} == {k: v for k, v in after.items() if k != "active"}
    assert data == originals


def test_verifier_rejects_hash_before_deserialization(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from app.cv.artifacts import RegistryEntry

    path = tmp_path / "E2-a2301ed5919a4b9089dc3cc73af65e70.pt"
    path.write_bytes(b"untrusted checkpoint")
    raw = ModelRegistry(PROJECT_ROOT / "config/models.json").selected("detection").model_dump()
    raw.update(checkpoint_path=str(path), sha256="0" * 64)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Unverified deserialization"))
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(RegistryEntry.model_validate(raw), "E2")
    assert path.read_bytes() == b"untrusted checkpoint"


def test_verifier_detects_source_change_even_on_load_failure(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from app.cv.artifacts import RegistryEntry
    from app.cv.verify_detection import sha256_file

    path = tmp_path / "E2-a2301ed5919a4b9089dc3cc73af65e70.pt"
    path.write_bytes(b"initial bytes")
    raw = ModelRegistry(PROJECT_ROOT / "config/models.json").selected("detection").model_dump()
    raw.update(checkpoint_path=str(path), sha256=sha256_file(path))

    def concurrent_change(source, *, weights_only, map_location):
        assert weights_only is True and map_location == "cpu"
        path.write_bytes(b"changed by an external writer")
        raise ValueError("Simulated deserialization failure")

    monkeypatch.setattr(torch, "load", concurrent_change)
    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        verify_checkpoint(RegistryEntry.model_validate(raw), "E2")


@pytest.mark.parametrize("experiment", EXPECTED)
def test_real_e123_independent_cpu_inference(experiment):
    if os.getenv("WP_CV4_REAL") != "1":
        pytest.skip("Set WP_CV4_REAL=1 for actual E1/E2/E3 CPU verification")
    registry = ModelRegistry(PROJECT_ROOT / "config/models.json")
    entry = next(entry for entry in registry.entries if entry.version.startswith(experiment + "-") and not entry.research_pilot)
    report = verify_checkpoint(entry, experiment, PROJECT_ROOT / "model_store/wp2-verification/dermoscopy.jpg")
    assert report["strict_state_dict"] and report["inference"]["status"] == "PASS"
    assert report["inference"]["raw_predictions"] > 0
    assert report["inference"]["finite_in_bounds_threshold_checked"]
    print(json.dumps(report))


def test_real_registry_e2_to_e1_rollback(tmp_path):
    import gc
    from PIL import Image
    from app.config import AppConfig
    from app.cv.availability import ModelStatus

    if os.getenv("WP_CV4_REAL") != "1":
        pytest.skip("Set WP_CV4_REAL=1 for actual registry rollback inference")
    original = (PROJECT_ROOT / "config/models.json").read_bytes()
    data = json.loads(original)
    path = tmp_path / "rollback.json"
    evidence = []
    for experiment in ("E2", "E1"):
        for row in data["models"]:
            if row["module"] == "detection":
                row["active"] = row["version"].startswith(experiment + "-") and not row["research_pilot"]
        path.write_text(json.dumps(data), encoding="utf-8")
        registry = ModelRegistry(path)
        selected = registry.selected("detection")
        detector = registry.create("detection", AppConfig(detection_device="cpu"))
        detector.load()
        assert detector.availability.status is ModelStatus.READY, str(detector.error)
        assert detector.identity.model_version == selected.version
        assert detector.identity.checkpoint_sha256 == selected.sha256
        with Image.open(PROJECT_ROOT / "model_store/wp2-verification/dermoscopy.jpg") as source:
            output = detector.predict(source)
        assert len(output.boxes) == 1
        evidence.append({"version": selected.version, "sha256": selected.sha256,
                         "boxes": [box.model_dump(mode="json") for box in output.boxes]})
        del detector
        gc.collect()
    (tmp_path / "rollback-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    assert (PROJECT_ROOT / "config/models.json").read_bytes() == original
    assert ModelRegistry(PROJECT_ROOT / "config/models.json").selected("detection").version.startswith("E2-")

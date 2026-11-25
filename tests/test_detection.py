"""Unit contracts use synthetic outputs; real inference is tested separately."""

import copy
import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.errors import AppError, ErrorCode
from app.cv.availability import ModelStatus
from app.cv.detection import FasterRCNN, OFFICIAL_FORMAT, PILOT_FORMAT, PREPROCESSING, predictions, validate_artifact
from app.cv.registry import RegistryEntry


def entry(**overrides):
    return RegistryEntry(**({
        "module": "detection", "model_id": "fasterrcnn", "version": "unit-v1", "checkpoint_path": "missing.pt",
        "sha256": "a" * 64, "architecture": "fasterrcnn_resnet50_fpn_v2", "artifact_format": OFFICIAL_FORMAT,
        "threshold": 0.5, "preprocessing": "torchvision-rgb-float32-v1",
    } | overrides))


def artifact():
    return {
        "format": OFFICIAL_FORMAT, "labels": {0: "__background__", 1: "skin_lesion"},
        "model_config": {"model": "fasterrcnn_resnet50_fpn_v2", "min_size": 800, "max_size": 1333, "trainable_backbone_layers": 3},
        "preprocessing": copy.deepcopy(PREPROCESSING), "operating_score_threshold": 0.5, "model": {"unit": 1},
    }


def test_official_and_pilot_boundaries():
    validate_artifact(artifact(), entry(), False)
    pilot = artifact() | {"format": PILOT_FORMAT}
    from app.cv.detection import PILOT_SHA256
    pilot_entry = entry(artifact_format=PILOT_FORMAT, research_pilot=True, sha256=PILOT_SHA256)
    with pytest.raises(AppError):
        validate_artifact(pilot, pilot_entry, False)
    validate_artifact(pilot, pilot_entry, True)
    with pytest.raises(AppError):
        validate_artifact(pilot, pilot_entry.model_copy(update={"research_pilot": False}), True)
    with pytest.raises(AppError):
        validate_artifact(artifact(), entry(research_pilot=True), True)


@pytest.mark.parametrize("key,value", [
    ("format", "training-v1"), ("labels", {0: "background", 1: "cancer"}), ("model", {}),
    ("preprocessing", {}), ("operating_score_threshold", 0.4), ("model_config", {}),
])
def test_rejects_incompatible_metadata(key, value):
    with pytest.raises(AppError) as error:
        validate_artifact(artifact() | {key: value}, entry(), False)
    assert error.value.code in {ErrorCode.CHECKPOINT_INCOMPATIBLE, ErrorCode.INVALID_MODEL_CONFIG}


def output(box=(0., 1., 20., 10.), score=0.8, label=1):
    torch = pytest.importorskip("torch")
    return {"boxes": torch.tensor([box], dtype=torch.float32), "scores": torch.tensor([score]), "labels": torch.tensor([label])}


def test_original_coordinate_and_inclusive_threshold_contract():
    assert len(predictions(output(score=0.5), (20, 10), 0.5)) == 1
    assert predictions(output(score=0.499), (20, 10), 0.5) == ()
    box = predictions(output(), (20, 10), 0.5)[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == (0., 1., 20., 10.)


@pytest.mark.parametrize("box,score,label", [
    ((-1., 0., 2., 2.), 0.8, 1), ((0., 0., 21., 2.), 0.8, 1), ((0., 0., 2., 11.), 0.8, 1),
    ((2., 0., 2., 2.), 0.8, 1), ((0., 0., float("nan"), 2.), 0.8, 1),
    ((0., 0., 2., 2.), float("inf"), 1), ((0., 0., 2., 2.), 1.1, 1),
    ((0., 0., 2., 2.), -0.1, 1), ((0., 0., 2., 2.), 0.8, 2),
])
def test_rejects_all_invalid_predictions_including_below_threshold(box, score, label):
    with pytest.raises(AppError) as error:
        predictions(output(box, score, label), (20, 10), 0.9)
    assert error.value.code is ErrorCode.INVALID_MODEL_OUTPUT


def test_loader_checks_hash_before_safe_deserialization(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "bad.pt"
    path.write_bytes(b"untrusted")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Must verify checksum first"))
    model = FasterRCNN(entry(checkpoint_path=str(path)))
    model.load()
    assert model.availability.status is ModelStatus.FAILED
    assert model.error.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
    assert model.identity is None


def test_loader_safe_strict_no_download_and_float_rgb(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torchvision.models.detection as tv
    import torchvision.models.detection.faster_rcnn as frc
    path = tmp_path / "synthetic.pt"
    torch.save(artifact(), path)
    calls = {}

    class Model:
        roi_heads = SimpleNamespace(box_predictor=SimpleNamespace(cls_score=SimpleNamespace(in_features=4)))

        def load_state_dict(self, state, *, strict):
            calls["strict"] = strict
            assert state == {"unit": 1}

        def eval(self):
            calls["eval"] = True
            return self

        def requires_grad_(self, value):
            return self

        def to(self, device):
            return self

        def __call__(self, tensors):
            tensor = tensors[0]
            assert torch.is_inference_mode_enabled()
            assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 10, 20)
            assert 0 <= tensor.min() <= tensor.max() <= 1
            return [output()]

    def build(**kwargs):
        calls["build"] = kwargs
        return Model()

    original_load = torch.load

    def safe_load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(tv, "fasterrcnn_resnet50_fpn_v2", build)
    monkeypatch.setattr(frc, "FastRCNNPredictor", lambda features, num_classes: object())
    model = FasterRCNN(entry(checkpoint_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()), "cpu")
    model.load()
    assert model.availability.status is ModelStatus.READY
    assert calls["strict"] is calls["eval"] is True
    assert calls["build"]["weights"] is calls["build"]["weights_backbone"] is None
    with Image.new("RGBA", (20, 10)) as image:
        prediction = model.predict(image)
    assert len(prediction.boxes) == 1 and prediction.duration_ms >= 0

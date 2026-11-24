import hashlib
import io
from threading import Event, Thread

import pytest
from PIL import Image

from app.config import PROJECT_ROOT, AppConfig
from app.core.errors import AppError, ErrorCode
from app.cv import checkpoint, segformer
from app.cv.availability import ModelStatus


def test_configuration_and_unavailable_checkpoint(tmp_path):
    config = AppConfig.from_env({
        "LESION_SEGFORMER_CHECKPOINT": "model_store/custom.pt", "LESION_SEGFORMER_DEVICE": "CPU",
    })
    assert config.segformer_checkpoint == PROJECT_ROOT / "model_store/custom.pt"
    assert config.segformer_device == "cpu"
    with pytest.raises(AppError):
        AppConfig.from_env({"LESION_SEGFORMER_DEVICE": "mps"})
    assert checkpoint.checkpoint_path(AppConfig()) == PROJECT_ROOT / "model_store/segformer/real_only/best.pt"
    for path in (None, tmp_path / "missing.pt"):
        model = segformer.SegFormer(AppConfig(segformer_checkpoint=path))
        model.load()
        assert model.availability.status is ModelStatus.NOT_AVAILABLE
        assert model.error.code is ErrorCode.CHECKPOINT_MISSING
        with Image.new("RGB", (10, 10)) as image, pytest.raises(AppError):
            model.predict(image)


@pytest.mark.parametrize("failure", [None, ErrorCode.MODEL_DEVICE, ErrorCode.CHECKPOINT_WEIGHTS_ONLY])
def test_loading_transitions_and_runs_once(tmp_path, monkeypatch, failure):
    model = segformer.SegFormer(AppConfig(segformer_checkpoint=tmp_path / "test.pt"))
    entered, release = Event(), Event()
    calls = []

    def load(config):
        calls.append(config)
        entered.set()
        assert release.wait(5)
        if failure:
            raise AppError(failure)
        return object(), "cpu"

    monkeypatch.setattr(segformer, "_load_verified_model", load)
    worker = Thread(target=model.load)
    worker.start()
    try:
        assert entered.wait(5)
        assert model.availability.status is ModelStatus.LOADING
    finally:
        release.set()
        worker.join(5)
    model.load()
    assert len(calls) == 1
    if failure:
        assert model.availability.status is ModelStatus.FAILED
        assert model.error.code is failure
        assert model.identity is None
    else:
        assert model.availability.status is ModelStatus.READY
        assert model.identity.checkpoint_id == checkpoint.CHECKPOINT_ID
        assert model.identity.checkpoint_sha256 == checkpoint.SHA256


def test_acquisition_checks_size_hash_and_reuses_existing_file(tmp_path, monkeypatch):
    contents = b"test fixture"
    path = tmp_path / "real_only/best.pt"
    monkeypatch.setattr(checkpoint, "SIZE_BYTES", len(contents))
    monkeypatch.setattr(checkpoint, "SHA256", hashlib.sha256(contents).hexdigest())

    def serve_wrong(*args, **kwargs):
        return io.BytesIO(b"X" * len(contents))

    monkeypatch.setattr(checkpoint, "urlopen", serve_wrong)
    with pytest.raises(AppError) as error:
        checkpoint.acquire(path)
    assert error.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
    assert error.value.field == "sha256"
    assert not path.exists() and not list(path.parent.glob("*.part"))

    calls = []

    def serve(url, **kwargs):
        calls.append(url)
        return io.BytesIO(contents)

    monkeypatch.setattr(checkpoint, "urlopen", serve)
    checkpoint.acquire(path)
    checkpoint.acquire(path)
    assert calls == [checkpoint.DOWNLOAD_URL]
    with checkpoint.verified_checkpoint(path) as stream:
        assert stream.read() == contents
    path.write_bytes(b"short")
    with pytest.raises(AppError) as error, checkpoint.verified_checkpoint(path):
        pytest.fail("A truncated checkpoint passed verification")
    assert error.value.field == "size_bytes"
    assert calls == [checkpoint.DOWNLOAD_URL]


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_preprocessing_matches_rgb_area_resize_and_imagenet(torch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    with Image.new("RGB", (730, 620), color=(255, 31, 0)) as image:
        image.paste((20, 63, 209), (111, 70, 482, 396))
        expected = cv2.resize(np.asarray(image), (512, 512), interpolation=cv2.INTER_AREA)
        tensor = segformer.preprocess(image)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32 and tensor.is_contiguous()
    expected = (expected / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    np.testing.assert_allclose(tensor[0].permute(1, 2, 0), expected, atol=1e-6)


@pytest.mark.parametrize("bad", ["shape", "dtype", "NaN", "Inf", "not_tensor"])
def test_invalid_logits(torch, bad):
    logits = torch.zeros(segformer.LOGITS_SHAPE)
    if bad == "shape":
        logits = logits[:, 0]
    elif bad == "dtype":
        logits = logits.int()
    elif bad == "not_tensor":
        logits = {"logits": logits}
    else:
        logits[0, 0, 0, 0] = float(bad)
    with pytest.raises(AppError) as error:
        segformer.binary_mask(logits, (512, 512))
    assert error.value.code is ErrorCode.INVALID_MODEL_OUTPUT


def test_threshold_empty_full_and_exclusive_one_pixel_box(torch):
    logits = torch.full(segformer.LOGITS_SHAPE, torch.logit(torch.tensor(0.699)).item())
    with segformer.binary_mask(logits, (35, 90)) as empty:
        assert empty.getextrema() == (0, 0)
        assert segformer.mask_box(empty) is None
    cutoff = torch.logit(torch.tensor(0.7))
    logits[0, 0, 511, 511] = cutoff
    with segformer.binary_mask(logits, (512, 512)) as mask:
        assert mask.histogram()[255] == 1
        assert tuple(segformer.mask_box(mask).model_dump().values()) == (511, 511, 512, 512)
    logits.fill_(cutoff)
    with segformer.binary_mask(logits, (35, 90)) as full:
        assert full.getextrema() == (255, 255)
        assert tuple(segformer.mask_box(full).model_dump().values()) == (0, 0, 35, 90)


def test_rectangular_resize_is_binary_and_box_uses_native_coordinates(torch):
    logits = torch.full(segformer.LOGITS_SHAPE, -1.0)
    logits[0, 0, :256, 256:] = 1
    with segformer.binary_mask(logits, (280, 120)) as mask:
        assert mask.size == (280, 120)
        assert mask.histogram()[0] + mask.histogram()[255] == 280 * 120
        assert tuple(segformer.mask_box(mask).model_dump().values()) == (140, 0, 280, 60)


def test_explicit_checkpoint_wrapper_reconciliation(torch):
    value = torch.tensor(1.0)
    assert segformer.model_state({"encoder.weight": value}) == {"encoder.weight": value}
    assert segformer.model_state({"model": {"model.encoder.weight": value}, "epoch": 21}) == {"encoder.weight": value}
    for bad in ([], {"model": {}}, {"model.bad": value, "bare": value}, {"bad": "not-a-tensor"}):
        with pytest.raises(AppError) as error:
            segformer.model_state(bad)
        assert error.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_oom_is_reported_without_raw_exception(torch):
    error = segformer._runtime_error(torch, torch.OutOfMemoryError("private details"), ErrorCode.INFERENCE_FAILED)
    assert error.code is ErrorCode.MODEL_OUT_OF_MEMORY
    assert "private details" not in str(error)

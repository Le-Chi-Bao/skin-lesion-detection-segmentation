import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.image_validation import ImageMetadata
from app.core.session import SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.analysis import execute_analysis
from app.cv.schemas import ModelIdentity
from app.cv.segformer import SegmentationPrediction
from app.cv.visualization import BOX_COLOR, OVERLAY_COLOR
from app.ui import png_export

OWNER = "A" * 43
PROOF = "1" * 64


@pytest.fixture
def exports(tmp_path):
    return png_export.PNGExports(TemporaryFiles(tmp_path, 30))


def analyzed(exports, *, size=(31, 22), color=(120, 84, 66), owner=OWNER, proof=PROOF, empty=False):
    files = exports.files
    current = SessionState(browser_owner=owner)
    current.browser_proof = proof
    current.accept_image_action(1)
    with files.operation(current.session_id), Image.new("RGB", size, color) as image:
        source = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
    current.replace_image(ImageMetadata(*size, "PNG", 300, "0" * 64), source)

    def predict(image):
        mask = Image.new("L", image.size)
        if not empty:
            mask.paste(255, (2, 3, image.width, image.height))
        return SegmentationPrediction(mask, 12.5)

    current.begin_run()
    execute_analysis(current, files, SimpleNamespace(predict=predict, identity=ModelIdentity(
        model_id="SegFormer MiT-B2", checkpoint_id="fixture", device="cpu",
    )))
    exports.issue(current, current.result)
    request = png_export.ExportRequest(
        session=current.session_id, run=current.result.run_id, action=1, proof=proof, kind="overlay", opacity=100,
    )
    return current, request


def test_export_png_is_private_full_resolution_and_on_demand(exports):
    size = (1673, 23)
    first, requested = analyzed(exports, size=size)
    second, other = analyzed(exports, size=(23, 219), color=(8, 16, 32), owner="B" * 43, proof="2" * 64)
    paths = set(exports.files.root.glob("*/*"))
    assert exports.response(second.browser_owner, requested).status_code == 404
    assert exports.response(first.browser_owner, other).status_code == 404
    assert exports.response(first.browser_owner, requested.model_copy(update={"proof": other.proof})).status_code == 404
    assert exports.response(first.browser_owner, requested.model_copy(update={"run": other.run})).status_code == 404
    assert exports.response(first.browser_owner, requested.model_copy(update={"action": 2})).status_code == 404
    overlay = exports.response(first.browser_owner, requested)
    assert overlay.status_code == 200 and overlay.headers["cache-control"] == "no-store"
    assert overlay.headers["content-type"] == "image/png"
    assert overlay.headers["content-disposition"] == f'attachment; filename="segformer-overlay-{first.result.run_id.hex[:12]}.png"'
    with Image.open(io.BytesIO(overlay.body)) as image:
        assert image.size == size and image.format == "PNG"
        assert image.getpixel((1, 3)) == (120, 84, 66)
        assert image.getpixel((2, 3)) == OVERLAY_COLOR
        assert image.getpixel((1672, 22)) == OVERLAY_COLOR
    localization = exports.response(first.browser_owner, requested.model_copy(update={"kind": "localization"}))
    with Image.open(io.BytesIO(localization.body)) as image:
        assert image.size == size
        assert image.getpixel((2, 3)) == BOX_COLOR
        assert image.getpixel((1672, 22)) == BOX_COLOR
        assert image.getpixel((40, 13)) == (120, 84, 66)
    assert set(exports.files.root.glob("*/*")) == paths


def test_empty_and_expired_result_exports(exports, monkeypatch):
    current, requested = analyzed(exports, empty=True)
    assert exports.response(OWNER, requested.model_copy(update={"kind": "localization"})).status_code == 404
    overlay = exports.response(OWNER, requested)
    with Image.open(io.BytesIO(overlay.body)) as image:
        assert image.getextrema() == ((120, 120), (84, 84), (66, 66))
    assert current.result.localization is None
    now = png_export.monotonic()
    monkeypatch.setattr(png_export, "monotonic", lambda: now + 30)
    assert exports.response(OWNER, requested).status_code == 404


@pytest.mark.parametrize("operation", ["reset", "upload", "analyze"])
def test_delayed_png_is_rejected_after_result_changes(exports, monkeypatch, operation):
    current, requested = analyzed(exports)
    entered, release = Event(), Event()
    source = current.image.temporary_file
    mask = current.result.segmentation.mask.reference
    encode = png_export.encode_png

    def delayed(image, buffer):
        entered.set()
        assert release.wait(10)
        encode(image, buffer)

    monkeypatch.setattr(png_export, "encode_png", delayed)
    with ThreadPoolExecutor(max_workers=1) as workers:
        export = workers.submit(exports.response, OWNER, requested)
        try:
            assert entered.wait(5)
            assert exports.response(OWNER, requested).status_code == 409
            with current.lock, exports.files.operation(current.session_id):
                exports.revoke(current.session_id)
                if operation == "analyze":
                    current.begin_run()
                else:
                    current.accept_image_action(2)
                    if operation == "upload":
                        replacement = exports.files.write(current.session_id, lambda f: f.write(b"fixture"))
                        current.replace_image(current.image.metadata, replacement)
                    else:
                        current.reset()
                        current.reset()
                    exports.files.discard(current.session_id, source)
                exports.files.discard_mask(current.session_id, mask)
            assert (exports.files.root / current.session_id.hex / f"{mask}.png").is_file()
            assert exports.response(OWNER, requested).status_code == 404
        finally:
            release.set()
        response = export.result(timeout=10)
    assert response.status_code == 404 and not response.body.startswith(b"\x89PNG")
    assert not (exports.files.root / current.session_id.hex / f"{mask}.png").exists()

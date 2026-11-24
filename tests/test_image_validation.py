from dataclasses import replace
from hashlib import sha256

import pytest
from PIL import Image

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.image_validation import uploaded_path, validate_image


@pytest.mark.parametrize(("fmt", "mode"), [("JPEG", "L"), ("PNG", "RGBA")])
def test_supported_image_uses_bytes_instead_of_suffix(tmp_path, fmt, mode):
    path = tmp_path / "misleading.gif"
    Image.new(mode, (13, 6)).save(path, fmt)
    image, details = validate_image(path, AppConfig())
    with image:
        assert image.mode == "RGB" and image.size == (13, 6)
    assert (details.format, details.width, details.height) == (fmt, 13, 6)
    content = path.read_bytes()
    assert details.sha256 == sha256(content).hexdigest()
    assert details.size_bytes == len(content)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (None, ErrorCode.MISSING_FILE), (b"", ErrorCode.INVALID_IMAGE),
        (b"not an image", ErrorCode.UNSUPPORTED_FORMAT),
        (b"\x89PNG\r\n\x1a\ncorrupt", ErrorCode.INVALID_IMAGE),
        (b"\xff\xd8\xffcorrupt", ErrorCode.INVALID_IMAGE),
    ],
)
def test_rejects_invalid_or_corrupt_files(tmp_path, content, code):
    path = tmp_path / "photo.png"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(AppError) as error:
        validate_image(path, AppConfig())
    assert error.value.code is code


def test_unsupported_and_animated_formats(tmp_path):
    path = tmp_path / "photo.png"
    with Image.new("RGB", (16, 12)) as frame:
        frame.save(path, "GIF")
        with pytest.raises(AppError) as error:
            validate_image(path, AppConfig())
        assert error.value.code is ErrorCode.UNSUPPORTED_FORMAT
        frame.save(path, "PNG", save_all=True, append_images=[Image.new("RGB", (16, 12), "red")])
    with pytest.raises(AppError) as error:
        validate_image(path, AppConfig())
    assert error.value.code is ErrorCode.INVALID_IMAGE


@pytest.mark.parametrize(("image_format", "missing_bytes"), [("JPEG", 16), ("PNG", 1)])
def test_rejects_truncated_image(tmp_path, image_format, missing_bytes):
    path = tmp_path / "photo.bin"
    Image.new("RGB", (25, 16)).save(path, image_format)
    path.write_bytes(path.read_bytes()[:-missing_bytes])
    with pytest.raises(AppError) as error:
        validate_image(path, AppConfig())
    assert error.value.code is ErrorCode.INVALID_IMAGE


def test_file_size_limit(tmp_path):
    config = AppConfig(max_upload_mib=1)
    path = tmp_path / "oversized.jpg"
    with path.open("wb") as stream:
        stream.truncate(config.max_upload_bytes + 1)
    with pytest.raises(AppError) as error:
        validate_image(path, config)
    assert error.value.code is ErrorCode.UPLOAD_TOO_LARGE
    assert "1 MiB" in str(error.value)


@pytest.mark.parametrize("override", [{"max_image_edge": 22}, {"max_image_pixels": 120}])
def test_dimension_and_pixel_limits(tmp_path, override):
    path = tmp_path / "large.png"
    Image.new("RGB", (23, 12)).save(path)
    with pytest.raises(AppError) as error:
        validate_image(path, replace(AppConfig(), **override))
    assert error.value.code is ErrorCode.IMAGE_TOO_LARGE


def test_exif_orientation_and_metadata_are_consistent(tmp_path):
    path = tmp_path / "orientation.jpg"
    image = Image.new("RGB", (45, 21), "blue")
    image.paste("red", (0, 0, 20, 21))
    image.getexif()[274] = 6
    image.save(path, exif=image.getexif())
    rgb, details = validate_image(path, AppConfig())
    with rgb:
        assert rgb.size == (21, 45) == (details.width, details.height)
        assert rgb.getpixel((10, 5))[0] > 230 and rgb.getpixel((10, 40))[2] > 230
        assert not rgb.getexif()


@pytest.mark.parametrize("location", ["outside.png", "https://example.invalid/photo.png", "../photo.png"])
def test_uploaded_paths_must_be_inside_the_cache(tmp_path, location):
    with pytest.raises(AppError) as error:
        uploaded_path(location, tmp_path / "cache")
    assert error.value.code is ErrorCode.UNSAFE_PATH
    absolute = tmp_path / "photo.png"
    with pytest.raises(AppError) as error:
        uploaded_path(str(absolute), tmp_path / "cache")
    assert error.value.code is ErrorCode.UNSAFE_PATH

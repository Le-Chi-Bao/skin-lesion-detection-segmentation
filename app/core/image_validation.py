import hashlib
import io
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    format: Literal["JPEG", "PNG"]
    size_bytes: int
    sha256: str


def uploaded_path(value: str | None, cache_dir: Path) -> Path:
    if not value:
        raise AppError(ErrorCode.MISSING_FILE)
    try:
        path = Path(value)
        resolved = path.resolve()
        if not path.is_absolute() or path != resolved or not resolved.is_relative_to(cache_dir):
            raise AppError(ErrorCode.UNSAFE_PATH)
        return resolved
    except (OSError, RuntimeError, ValueError) as exc:
        raise AppError(ErrorCode.UNSAFE_PATH) from exc


def validate_image(path: Path, config: AppConfig) -> tuple[Image.Image, ImageMetadata]:
    try:
        details = path.stat()
        if not stat.S_ISREG(details.st_mode):
            raise AppError(ErrorCode.UNSAFE_PATH)
        if details.st_size > config.max_upload_bytes:
            raise AppError(ErrorCode.UPLOAD_TOO_LARGE, limit=config.max_upload_mib)
        with path.open("rb") as stream:
            content = stream.read(config.max_upload_bytes + 1)
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.MISSING_FILE) from exc
    except OSError as exc:
        raise AppError(ErrorCode.TEMPORARY_FILE) from exc
    if len(content) > config.max_upload_bytes:
        raise AppError(ErrorCode.UPLOAD_TOO_LARGE, limit=config.max_upload_mib)
    if not content:
        raise AppError(ErrorCode.INVALID_IMAGE)
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        # Pillow's PNG verifier stops before checking the final chunk's CRC.
        if not content.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
            raise AppError(ErrorCode.INVALID_IMAGE)
        image_format = "PNG"
    elif content.startswith(b"\xff\xd8\xff"):
        image_format = "JPEG"
    else:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT)

    limits = {"edge": config.max_image_edge, "pixels": config.max_image_pixels}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content), formats=(image_format,)) as source:
                width, height = source.size
                if max(width, height) > config.max_image_edge or width * height > config.max_image_pixels:
                    raise AppError(ErrorCode.IMAGE_TOO_LARGE, **limits)
                if getattr(source, "n_frames", 1) != 1:
                    raise AppError(ErrorCode.INVALID_IMAGE)
                source.verify()
            with Image.open(io.BytesIO(content), formats=(image_format,)) as source:
                ImageOps.exif_transpose(source, in_place=True)
                image = source.convert("RGB")
                image.info.clear()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise AppError(ErrorCode.IMAGE_TOO_LARGE, **limits) from exc
    except (OSError, SyntaxError, ValueError, TypeError, OverflowError) as exc:
        raise AppError(ErrorCode.INVALID_IMAGE) from exc
    metadata = ImageMetadata(image.width, image.height, image_format, len(content), hashlib.sha256(content).hexdigest())
    return image, metadata

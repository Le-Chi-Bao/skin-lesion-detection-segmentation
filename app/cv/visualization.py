"""Render SegFormer results in the normalized image's pixel coordinates."""

from collections.abc import Iterator
from contextlib import contextmanager
from math import ceil, floor
from typing import BinaryIO
from uuid import UUID

from PIL import Image, ImageDraw

from app.core.temporary_files import TemporaryFiles
from app.cv.schemas import BoundingBox, CVResult

DISPLAY_EDGE = 1600
DEFAULT_OPACITY = 40
OVERLAY_COLOR = (20, 184, 166)
BOX_COLOR = (255, 207, 73)
BOX_OUTLINE = (23, 50, 77)
DETECTION_COLOR = (255, 96, 96)


@contextmanager
def open_source(files: TemporaryFiles, session_id: UUID, source_file: str, result: CVResult) -> Iterator[Image.Image]:
    with files.operation(session_id), files.open_image(session_id, source_file) as stream, Image.open(stream, formats=("PNG",)) as source:
        if source.mode != "RGB" or source.size != (result.image_width, result.image_height):
            raise ValueError("Source dimensions do not match the result")
        yield source


def draw_combined_localization(original: Image.Image, result: CVResult) -> Image.Image | None:
    if result.localization is None and not result.detection.boxes:
        return None
    size = result.image_width, result.image_height
    output = draw_localization(original, result.localization.box, source_size=size) if result.localization else original.copy()
    output.info.clear()
    draw = ImageDraw.Draw(output)
    stroke = max(1, min(12, round(max(output.size) / 450)))
    if result.localization:
        box = result.localization.box
        # Put mask and detector labels on opposite sides of overlapping boxes.
        position = (floor(box.x1 * output.width / size[0]), min(max(0, output.height - 12), ceil(box.y2 * output.height / size[1]) + 2))
        draw.text(position, "SegFormer: mask box", fill=BOX_COLOR, stroke_width=1, stroke_fill=BOX_OUTLINE)
    for prediction in result.detection.boxes:
        box = prediction.box
        coords = (
            floor(box.x1 * output.width / size[0]), floor(box.y1 * output.height / size[1]),
            min(output.width - 1, ceil(box.x2 * output.width / size[0]) - 1),
            min(output.height - 1, ceil(box.y2 * output.height / size[1]) - 1),
        )
        draw.rectangle(coords, outline=DETECTION_COLOR, width=stroke)
        draw.text((coords[0], max(0, coords[1] - 12)), f"Faster R-CNN: {prediction.score:.3f}", fill=DETECTION_COLOR, stroke_width=1, stroke_fill=BOX_OUTLINE)
    return output


def display_size(size: tuple[int, int]) -> tuple[int, int]:
    scale = min(1, DISPLAY_EDGE / max(size))
    return tuple(max(1, round(edge * scale)) for edge in size)


def validate_alignment(original: Image.Image, mask: Image.Image, size: tuple[int, int]) -> None:
    if original.mode != "RGB" or mask.mode != "L" or original.size != size or mask.size != size:
        raise ValueError("Expected aligned RGB source and grayscale binary mask")
    # The producer stores exactly 0 and 255. Reject a damaged mask before drawing it.
    colors = mask.getcolors(2)
    if colors is None or any(value not in (0, 255) for _, value in colors):
        raise ValueError("Expected a binary mask with values 0 and 255")


@contextmanager
def open_aligned_images(
    files: TemporaryFiles, session_id: UUID, source_file: str, result: CVResult,
) -> Iterator[tuple[Image.Image, Image.Image]]:
    reference = result.segmentation.mask
    if reference is None:
        raise ValueError("A completed segmentation is required")
    with (
        files.operation(session_id),
        files.open_image(session_id, source_file) as source_stream,
        files.open_mask(session_id, reference.reference) as mask_stream,
        Image.open(source_stream, formats=("PNG",)) as source,
        Image.open(mask_stream, formats=("PNG",)) as binary,
    ):
        validate_alignment(source, binary, (result.image_width, result.image_height))
        yield source, binary


def compose_overlay(original: Image.Image, mask: Image.Image, opacity: int) -> Image.Image:
    validate_alignment(original, mask, original.size)
    if type(opacity) is not int or not 0 <= opacity <= 100:
        raise ValueError("Opacity must be an integer from 0 to 100")
    output = original.copy()
    output.info.clear()
    if opacity:
        alpha = round(255 * opacity / 100)
        with mask.point([0] * 255 + [alpha]) as coverage:
            output.paste(OVERLAY_COLOR, (0, 0, *output.size), coverage)
    return output


def overlay_layer(mask: Image.Image) -> Image.Image:
    """Make a two-color PNG layer. The browser varies its opacity without a request."""
    layer = mask.point([0] * 255 + [1])
    layer.putpalette([0, 0, 0, *OVERLAY_COLOR])
    layer.info["transparency"] = 0
    return layer


def draw_localization(
    original: Image.Image, box: BoundingBox | None, *, source_size: tuple[int, int] | None = None,
) -> Image.Image | None:
    if box is None:
        return None
    width, height = source_size or original.size
    if original.mode != "RGB" or box.x2 > width or box.y2 > height:
        raise ValueError("Bounding box lies outside the RGB source")
    sx, sy = original.width / width, original.height / height
    # Pillow includes both end pixels. Round outward when projecting onto the preview.
    rectangle = (
        floor(box.x1 * sx), floor(box.y1 * sy),
        min(original.width - 1, ceil(box.x2 * sx) - 1),
        min(original.height - 1, ceil(box.y2 * sy) - 1),
    )
    output = original.copy()
    output.info.clear()
    draw = ImageDraw.Draw(output)
    stroke = max(1, min(12, round(max(output.size) / 450)))
    # Degenerate boxes need a line or a point. Wider Pillow strokes can extend a
    # one-pixel rectangle outside its coordinates, so cap the stroke on small boxes.
    left, top, right, bottom = rectangle
    if left == right or top == bottom:
        draw.line(rectangle, fill=BOX_COLOR)
    else:
        inner_limit = (min(right - left, bottom - top) + 1) // 2
        if stroke > 1 and inner_limit > 1:
            draw.rectangle(rectangle, outline=BOX_OUTLINE, width=min(stroke + 1, inner_limit))
        draw.rectangle(rectangle, outline=BOX_COLOR, width=min(stroke, max(1, inner_limit - 1)))
    return output


def encode_png(image: Image.Image, stream: BinaryIO) -> None:
    image.save(stream, format="PNG", compress_level=3)

"""Render SegFormer results in the normalized image's pixel coordinates."""

from math import ceil, floor
from typing import BinaryIO

from PIL import Image, ImageDraw

from app.cv.schemas import BoundingBox

DISPLAY_EDGE = 1600
DEFAULT_OPACITY = 40
OVERLAY_COLOR = (20, 184, 166)
BOX_COLOR = (255, 207, 73)
BOX_OUTLINE = (23, 50, 77)


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

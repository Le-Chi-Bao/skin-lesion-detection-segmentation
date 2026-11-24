import io

import pytest
from PIL import Image

from app.cv.schemas import BoundingBox
from app.cv.visualization import (
    BOX_COLOR, OVERLAY_COLOR, compose_overlay, display_size, draw_localization, encode_png,
    overlay_layer, validate_alignment,
)


@pytest.mark.parametrize("size", [(5, 3), (3, 5), (3, 3)])
@pytest.mark.parametrize(("opacity", "foreground"), [(0, (120, 84, 66)), (40, (80, 124, 106)), (100, OVERLAY_COLOR)])
def test_overlay_pixels_and_input_immutability(size, opacity, foreground):
    color = (120, 84, 66)
    with Image.new("RGB", size, color) as image, Image.new("L", size) as mask:
        point = (image.width - 1, image.height - 1)
        mask.putpixel(point, 255)
        before = image.tobytes(), mask.tobytes()
        with compose_overlay(image, mask, opacity) as overlay:
            assert overlay.size == image.size
            assert overlay.getpixel(point) == foreground
            assert overlay.getpixel((0, 0)) == color
        assert (image.tobytes(), mask.tobytes()) == before


@pytest.mark.parametrize("value", [0, 255])
def test_empty_and_full_foreground(value):
    with Image.new("RGB", (4, 7)) as image, Image.new("L", image.size, value) as mask:
        with compose_overlay(image, mask, 100) as overlay:
            pixel = OVERLAY_COLOR if value else (0, 0, 0)
            assert list(overlay.get_flattened_data()) == [pixel] * 28
        assert draw_localization(image, None) is None


@pytest.mark.parametrize("edges", [(0, 0, 6, 4), (5, 3, 6, 4), (1, 0, 2, 4), (1, 1, 5, 2), (1, 1, 5, 3)])
def test_exclusive_box_matches_exact_border_pixels(edges):
    x1, y1, x2, y2 = edges
    box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    with Image.new("RGB", (6, 4)) as image, draw_localization(image, box) as localized:
        for y in range(image.height):
            for x in range(image.width):
                boundary = x1 <= x < x2 and y1 <= y < y2 and (x in (x1, x2 - 1) or y in (y1, y2 - 1))
                assert localized.getpixel((x, y)) == (BOX_COLOR if boundary else (0, 0, 0))
        assert image.getbbox() is None


def test_preview_box_reaches_last_pixel_and_sizes_are_bounded():
    assert display_size((3001, 2000)) == (1600, 1066)
    assert display_size((2000, 3001)) == (1066, 1600)
    assert display_size((10000, 1)) == (1600, 1)
    box = BoundingBox(x1=2999, y1=2199, x2=3000, y2=2200)
    with Image.new("RGB", (16, 12)) as image, draw_localization(image, box, source_size=(3000, 2200)) as localized:
        assert localized.getpixel((15, 11)) == BOX_COLOR


@pytest.mark.parametrize(("size", "mode", "value"), [((2, 3), "L", 0), ((3, 2), "RGB", 0), ((3, 2), "L", 20)])
def test_rejects_misaligned_or_nonbinary_mask(size, mode, value):
    with Image.new("RGB", (3, 2)) as image, Image.new(mode, size, value) as mask:
        with pytest.raises(ValueError):
            validate_alignment(image, mask, image.size)


@pytest.mark.parametrize("opacity", [-1, 101, True, 40.5])
def test_rejects_invalid_opacity(opacity):
    with Image.new("RGB", (3, 2)) as image, Image.new("L", image.size) as mask:
        with pytest.raises(ValueError):
            compose_overlay(image, mask, opacity)


def test_png_round_trip_and_transparent_overlay_layer():
    with Image.new("L", (2, 3)) as mask, io.BytesIO() as data:
        mask.putpixel((0, 0), 255)
        with overlay_layer(mask) as layer:
            encode_png(layer, data)
        data.seek(0)
        with Image.open(data) as decoded, decoded.convert("RGBA") as rgba:
            assert decoded.format == "PNG" and decoded.size == mask.size
            assert rgba.getpixel((0, 0)) == (*OVERLAY_COLOR, 255)
            assert rgba.getpixel((1, 2))[3] == 0

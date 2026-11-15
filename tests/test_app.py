"""Kiểm thử cơ bản bằng ảnh hình học tự tạo, không dùng dữ liệu bệnh nhân."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from app import preview_image


class PreviewImageTests(unittest.TestCase):
    def test_no_image(self):
        image, message = preview_image(None)
        self.assertIsNone(image)
        self.assertIn("Vui lòng chọn ảnh", message)

    def test_jpg_and_png_preserve_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Image.new("RGB", (96, 64), "navy")
            ImageDraw.Draw(sample).rectangle((16, 16, 48, 48), fill="orange")
            for extension in ("png", "jpg"):
                with self.subTest(extension=extension):
                    path = Path(directory) / f"sample.{extension}"
                    sample.save(path)
                    result, message = preview_image(str(path))
                    self.assertIsNotNone(result)
                    with Image.open(path) as original:
                        self.assertEqual(result.size, original.size)
                        self.assertEqual(result.mode, original.mode)
                        self.assertEqual(result.tobytes(), original.tobytes())
                    self.assertIn("Chưa phân tích tổn thương", message)

    def test_invalid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            path.write_text("This is not an image.", encoding="utf-8")
            image, message = preview_image(str(path))
            self.assertIsNone(image)
            self.assertIn("Không đọc được ảnh", message)

    def test_unsupported_format(self):
        image, message = preview_image("sample.gif")
        self.assertIsNone(image)
        self.assertIn("Chỉ hỗ trợ ảnh JPG hoặc PNG", message)


if __name__ == "__main__":
    unittest.main()

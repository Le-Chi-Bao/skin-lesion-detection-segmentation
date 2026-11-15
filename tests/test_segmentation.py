"""Kiểm thử pipeline và lỗi CLI; không cần tải checkpoint hoặc chạy mô hình."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from teledermoscopy.vision.segmentation import preprocess_rgb, restore_probabilities, select_device


ROOT = Path(__file__).resolve().parents[1]


class SegmentationTests(unittest.TestCase):
    def test_rgb_normalization_and_tensor_layout(self):
        red = np.full((8, 12, 3), [255, 0, 0], dtype=np.uint8)
        tensor = preprocess_rgb(red)
        self.assertEqual(tuple(tensor.shape), (1, 3, 512, 512))
        np.testing.assert_allclose(
            tensor[0, :, 0, 0].numpy(), [(1 - .485) / .229, -.456 / .224, -.406 / .225], rtol=1e-6,
        )

    def test_resize_probabilities_before_fixed_threshold(self):
        # Điểm giữa nội suy đúng 0.5 phải nằm trong mask theo >=, không dùng >.
        probability = np.array([[.25, .75], [.25, .75]], dtype=np.float32)
        restored, mask = restore_probabilities(probability, (3, 2))
        np.testing.assert_allclose(restored, [[.25, .5, .75], [.25, .5, .75]])
        np.testing.assert_array_equal(mask, [[0, 255, 255], [0, 255, 255]])

    def test_explicit_cuda_does_not_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "không tự chuyển sang CPU"):
                select_device("cuda")
            self.assertEqual(str(select_device("auto")), "cpu")

    def run_cli(self, image, checkpoint, output):
        return subprocess.run(
            [sys.executable, "-m", "teledermoscopy.segment", "--image", str(image),
             "--checkpoint", str(checkpoint), "--output-dir", str(output), "--device", "cpu"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
        )

    def test_cli_rejects_invalid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / "invalid.png"
            image.write_text("This is not an image.", encoding="utf-8")
            result = self.run_cli(image, ROOT / "checkpoints/segformer-gate2/real_only/best.pt", folder / "output")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Ảnh không hợp lệ", result.stderr)
            self.assertFalse((folder / "output").exists())

    def test_cli_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / "valid.png"
            Image.new("RGB", (16, 12), "navy").save(image)
            result = self.run_cli(image, folder / "missing.pt", folder / "output")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Không tìm thấy checkpoint", result.stderr)
            self.assertFalse((folder / "output").exists())


if __name__ == "__main__":
    unittest.main()

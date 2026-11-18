"""Kiểm thử trạng thái/cache bằng mock, tách biệt lượt inference thật trên UI."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from app import build_demo, preview_and_reset, reset_results
from teledermoscopy.gradio_segmentation import SegmentationController


class GradioSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.image = self.folder / "sample.png"
        Image.new("RGB", (16, 12), "navy").save(self.image)
        self.checkpoint = self.folder / "unused.pt"
        # Chỉ là path fixture để kiểm thử controller với mock, không được torch.load.
        self.checkpoint.touch()
        self.state = {"revision": 0}
        self.controller = SegmentationController(self.checkpoint)

    def assert_cleared(self, response):
        self.assertEqual(response[:4], (None, None, None, ""))

    def test_open_app_does_not_load_model(self):
        with patch("teledermoscopy.gradio_segmentation.SegmentationPredictor") as constructor:
            build_demo(controller=self.controller)
            self.assertEqual(self.controller.model_load_count, 0)
            constructor.assert_not_called()

    def test_no_image_clears_results(self):
        response = list(self.controller.segment(None, self.state))[-1]
        self.assert_cleared(response)
        self.assertIn("Vui lòng chọn ảnh", response[-1])
        self.assertEqual(self.controller.model_load_count, 0)

    def test_invalid_image_clears_results(self):
        invalid = self.folder / "invalid.png"
        invalid.write_text("Not an image", encoding="utf-8")
        response = list(self.controller.segment(str(invalid), self.state))[-1]
        self.assert_cleared(response)
        self.assertIn("Ảnh không hợp lệ", response[-1])

    def test_missing_checkpoint_clears_results(self):
        controller = SegmentationController(self.folder / "missing.pt")
        response = list(controller.segment(str(self.image), self.state))[-1]
        self.assert_cleared(response)
        self.assertIn("Không tìm thấy checkpoint local", response[-1])

    def test_cuda_oom_clears_results_without_cpu_fallback(self):
        with patch("teledermoscopy.gradio_segmentation.SegmentationPredictor") as constructor:
            constructor.return_value.predict.side_effect = torch.cuda.OutOfMemoryError("unit test")
            response = list(self.controller.segment(str(self.image), self.state))[-1]
            self.assert_cleared(response)
            self.assertIn("CUDA hết VRAM", response[-1])
            self.assertIn("không tự chuyển sang CPU", response[-1])
            constructor.assert_called_once_with(self.checkpoint, device="auto")

    def test_missing_checkpoint_is_checked_even_with_cached_predictor(self):
        cached = Mock()
        self.controller._predictor = cached
        self.controller.checkpoint_path = self.folder / "missing.pt"
        response = list(self.controller.segment(str(self.image), self.state))[-1]
        self.assert_cleared(response)
        self.assertIn("Không tìm thấy checkpoint local", response[-1])
        cached.predict.assert_not_called()

    def test_predictor_is_reused(self):
        # Kết quả mock chỉ phục vụ test adapter, không ghi file hoặc báo là inference thật.
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        result = SimpleNamespace(original_rgb=rgb, mask=np.zeros((12, 16), dtype=np.uint8),
                                 overlay=Mock(return_value=rgb), device="cpu")
        with patch("teledermoscopy.gradio_segmentation.SegmentationPredictor") as constructor:
            constructor.return_value.predict.return_value = result
            for _ in range(2):
                responses = list(self.controller.segment(str(self.image), self.state))
                self.assert_cleared(responses[0])
                self.assertEqual(responses[-1][3], "cpu")
            constructor.assert_called_once_with(self.checkpoint, device="auto")
            self.assertEqual(constructor.return_value.predict.call_count, 2)
            self.assertEqual(self.controller.model_load_count, 1)

    def test_changed_image_discards_in_flight_result(self):
        def changed_while_running(_):
            self.state["revision"] += 1
            return None  # Không được đọc kết quả này sau khi revision đã thay đổi.
        with patch("teledermoscopy.gradio_segmentation.SegmentationPredictor") as constructor:
            constructor.return_value.predict.side_effect = changed_while_running
            responses = list(self.controller.segment(str(self.image), self.state))
            self.assertEqual(len(responses), 1)
            self.assert_cleared(responses[0])

    def test_change_clear_and_preview_invalidate_old_results(self):
        for path in (str(self.image), None):
            self.assert_cleared(reset_results(path, self.state))
        response = preview_and_reset(str(self.image), self.state)
        self.assertEqual(response[0].size, (16, 12))
        self.assertEqual(response[1:4], (None, None, ""))
        self.assertEqual(self.state["revision"], 3)


if __name__ == "__main__":
    unittest.main()

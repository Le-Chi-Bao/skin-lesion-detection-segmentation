"""Nối giao diện với predictor 2B; không chứa tiền/hậu xử lý ảnh."""

from pathlib import Path
from threading import Lock

from PIL import Image
import torch

from teledermoscopy.vision.segmentation import SegmentationPredictor, read_rgb_image


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/segformer-gate2/real_only/best.pt"


class SegmentationController:
    """Cache theo tiến trình app; khóa dùng chung tránh inference GPU đồng thời."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self._predictor = None
        self._lock = Lock()
        self.model_load_count = 0

    def segment(self, image_path: str | None, image_state: dict):
        """Generator xóa kết quả trước khi chạy; không trả kết quả của ảnh đã đổi."""
        revision = image_state["revision"]
        yield None, None, None, "", "Đang chuẩn bị phân vùng thử nghiệm…"
        try:
            if not image_path:
                raise ValueError("Vui lòng chọn ảnh JPG hoặc PNG trước khi nhấn Phân vùng.")
            if Path(image_path).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Chỉ hỗ trợ ảnh JPG hoặc PNG.")
            rgb = read_rgb_image(image_path)
            with self._lock:
                if revision != image_state["revision"]:
                    return
                # Kiểm tra cả khi đã cache để không che giấu checkpoint bị mất.
                if not self.checkpoint_path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy checkpoint local: {self.checkpoint_path}")
                if self._predictor is None:
                    self._predictor = SegmentationPredictor(self.checkpoint_path, device="auto")
                    self.model_load_count += 1
                result = self._predictor.predict(rgb)
            if revision != image_state["revision"]:
                return
            device = result.device
            if device.startswith("cuda"):
                device += f" — {torch.cuda.get_device_name(torch.device(result.device))}"
            yield (
                Image.fromarray(result.original_rgb), Image.fromarray(result.mask),
                Image.fromarray(result.overlay()), device,
                "Đã hoàn tất phân vùng thử nghiệm (ngưỡng 0.5). Đây không phải chẩn đoán.",
            )
        except Exception as error:
            if revision != image_state["revision"]:
                return
            if isinstance(error, torch.cuda.OutOfMemoryError):
                message = "CUDA hết VRAM khi phân vùng. Kết quả đã được xóa; không tự chuyển sang CPU."
            else:
                message = f"Phân vùng thất bại: {error}"
                if isinstance(error, RuntimeError):
                    message += " Không tự chuyển sang CPU."
            yield None, None, None, "", message

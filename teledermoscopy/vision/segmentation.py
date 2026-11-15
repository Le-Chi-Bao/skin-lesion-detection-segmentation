"""SegFormer inference theo cấu hình đã xác minh ở Phase 2A.1.

Nguồn: ishaannk/skin-lesion-seg, revision
d3d3195f13b02366cbc1332cbb57d7ca9234b7a7 (registry, dataset, preprocess, evaluate).
Không tải tài nguyên hoặc thay đổi thiết bị khi inference thất bại.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from torch import nn


IMAGE_SIZE = 512
THRESHOLD = 0.5
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


class SMPWrapper(nn.Module):
    """Giữ namespace model.* của checkpoint gốc."""

    def __init__(self, model: nn.Module, name: str = "segformer_mitb2"):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, x, provenance=None, **kw):
        return {"logits": self.model(x)}


def select_device(requested: str) -> torch.device:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Thiết bị phải là auto, cuda hoặc cpu.")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested == "cuda":
        raise RuntimeError("CUDA không khả dụng trong môi trường hiện tại; không tự chuyển sang CPU.")
    return torch.device("cpu")


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Đọc như upstream bằng OpenCV; hỗ trợ đường dẫn Unicode trên Windows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    try:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Ảnh không hợp lệ hoặc không đọc được: {path}") from error
    if bgr is None:
        raise ValueError(f"Ảnh không hợp lệ hoặc không đọc được: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Đầu vào phải là mảng ảnh RGB uint8 kích thước H×W×3.")
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)


def restore_probabilities(probabilities: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize xác suất về (width, height) trước khi threshold, không resize mask."""
    restored = cv2.resize(probabilities, size, interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(restored).all():
        raise RuntimeError("Mô hình trả bản đồ xác suất có NaN hoặc Inf.")
    mask = (restored >= THRESHOLD).astype(np.uint8) * 255
    return restored, mask


@dataclass
class SegmentationResult:
    original_rgb: np.ndarray
    probabilities: np.ndarray
    mask: np.ndarray
    device: str
    elapsed_seconds: float

    def overlay(self) -> np.ndarray:
        output = self.original_rgb.copy()
        foreground = self.mask == 255
        # Chỉ tô vùng mask; ngoài mask giữ nguyên pixel gốc.
        output[foreground] = np.rint(
            0.6 * output[foreground].astype(np.float32) + 0.4 * np.array([255, 0, 0])
        ).astype(np.uint8)
        return output


class SegmentationPredictor:
    """Nạp model một lần; predict nhận RGB ndarray để có thể tái sử dụng ở 2C."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {self.checkpoint_path}")
        self.device = select_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        cfg = checkpoint["cfg"]
        if (cfg["model"] != "segformer_mitb2" or cfg["model_overrides"] != {}
                or cfg["data"]["img_size"] != IMAGE_SIZE):
            raise ValueError("Cấu hình checkpoint không khớp SegFormer MiT-B2 512×512 đã xác minh.")
        self.model = SMPWrapper(smp.Segformer(
            encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=1,
        ))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        # Nếu .to(cuda) thất bại, truyền lỗi ra CLI; tuyệt đối không fallback.
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> SegmentationResult:
        started = perf_counter()
        inputs = preprocess_rgb(rgb).to(self.device)
        logits = self.model(inputs)["logits"]
        probabilities = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        height, width = rgb.shape[:2]
        probabilities, mask = restore_probabilities(probabilities, (width, height))
        return SegmentationResult(rgb.copy(), probabilities, mask, str(self.device), perf_counter() - started)


def save_result(result: SegmentationResult, image_path: str | Path,
                checkpoint_path: str | Path, output_dir: str | Path) -> dict:
    """Lưu kết quả thật, mở lại PNG và trả metadata của lần chạy."""
    image_path, checkpoint_path, output_dir = Path(image_path), Path(checkpoint_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    summary_path = output_dir / f"{image_path.stem}_result.json"
    Image.fromarray(result.mask).save(mask_path)
    Image.fromarray(result.overlay()).save(overlay_path)
    height, width = result.original_rgb.shape[:2]
    with Image.open(mask_path) as saved_mask, Image.open(overlay_path) as saved_overlay:
        saved_mask.load()
        saved_overlay.load()
        if (saved_mask.format != "PNG" or saved_mask.mode != "L" or saved_mask.size != (width, height)
                or saved_overlay.format != "PNG" or saved_overlay.mode != "RGB"
                or saved_overlay.size != (width, height)):
            raise RuntimeError("File mask/overlay ghi ra không đúng định dạng hoặc kích thước ảnh gốc.")
        mask_values = np.unique(np.asarray(saved_mask)).tolist()
        if not set(mask_values) <= {0, 255}:
            raise RuntimeError(f"Mask lưu ra không nhị phân: {mask_values}")
        mask_pixels = int(np.count_nonzero(np.asarray(saved_mask)))
    ratio = mask_pixels / (width * height)
    summary = {
        "image": str(image_path.resolve()), "checkpoint": str(checkpoint_path.resolve()),
        "device": result.device, "gpu_name": (
            torch.cuda.get_device_name(torch.device(result.device)) if result.device.startswith("cuda") else None),
        "precision": "float32", "threshold": THRESHOLD, "input_size_wh": [width, height],
        "model_input_size_wh": [IMAGE_SIZE, IMAGE_SIZE], "mask_size_wh": [width, height],
        "overlay_size_wh": [width, height], "mask_unique_values": mask_values,
        "mask_pixels": mask_pixels, "total_pixels": width * height, "mask_fraction": ratio,
        "mask_state": "empty" if ratio == 0 else "full" if ratio == 1 else "non_degenerate",
        "probability_min": float(result.probabilities.min()),
        "probability_max": float(result.probabilities.max()),
        "probability_mean": float(result.probabilities.mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "cuda_peak_allocated_mib": (torch.cuda.max_memory_allocated(torch.device(result.device)) / 1024**2
                                    if result.device.startswith("cuda") else None),
        "mask": str(mask_path.resolve()), "overlay": str(overlay_path.resolve()),
        "outputs_reopened_and_verified": True, "purpose": "smoke test only; no accuracy evaluation",
    }
    for name, path in (("image", image_path), ("checkpoint", checkpoint_path)):
        with path.open("rb") as stream:
            summary[f"{name}_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

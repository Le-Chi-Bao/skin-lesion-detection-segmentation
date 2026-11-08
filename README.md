# Skin Lesion Detection and Segmentation

Nền tảng đồ án Computer Vision với **ISIC 2018 Task 1**, Python 3.11,
PyTorch/Torchvision và Streamlit. Hiện chỉ triển khai **Phase 1 — Project
Foundation**: utility, cấu hình, giao diện xem ảnh và kiểm thử synthetic.
Chưa huấn luyện, chưa xây dựng Faster R-CNN/U-Net, chưa suy luận AI.

## 1. Kiến trúc hệ thống

Luồng dự kiến cho các phase tiếp theo:

```text
Ảnh RGB gốc → preprocessing của detector → Faster R-CNN
   → box quy về tọa độ ảnh gốc → crop ROI có padding → resize ROI → U-Net
   → binary ROI mask → resize về kích thước crop → paste lên ảnh gốc
   → hiển thị bounding box / mask / overlay trong Streamlit

Đối chứng: ảnh RGB gốc → resize full image → U-Net → mask ảnh gốc
```

Phase 1 mới thực hiện các phép biến đổi không gian và phần xem ảnh đầu vào.
`Detector` và `Segmenter` là `Protocol` định nghĩa giao diện, không phải mô hình
giả hay implementation có thể dự đoán. Chi tiết trong
[docs/system_architecture.md](docs/system_architecture.md).

## 2. Cấu trúc thư mục

```text
.
├── configs/
│   ├── base.yaml
│   ├── faster_rcnn.yaml
│   ├── unet_roi_256.yaml
│   ├── unet_roi_512.yaml
│   └── unet_full_image_512.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── splits/.gitkeep
├── src/
│   ├── __init__.py
│   ├── data/{__init__,image_io,geometry}.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detection/{__init__,interfaces}.py
│   │   └── segmentation/{__init__,interfaces}.py
│   ├── pipeline/{__init__,spatial}.py
│   ├── utils/{__init__,config,device,seed}.py
│   └── evaluation/__init__.py
├── app/{__init__,streamlit_app}.py
├── scripts/{__init__,smoke_test,check_streamlit_server}.py
├── notebooks/
│   ├── README.md
│   └── 01_foundation_smoke.ipynb
├── tests/
│   ├── test_image_io.py
│   ├── test_geometry.py
│   ├── test_config.py
│   ├── test_device.py
│   ├── test_app.py
│   └── test_smoke.py
├── checkpoints/.gitkeep
├── outputs/.gitkeep
├── docs/
│   ├── system_architecture.md
│   ├── file_manifest.md
│   └── verification.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Các tệp PDF và `.drawio` có sẵn được giữ nguyên. Dùng package `src` theo cấu trúc
bạn yêu cầu, không thêm tầng repository/service, model registry hay framework
cấu hình lớn. `opencv-python-headless` cung cấp resize mà không cần GUI hệ điều
hành. `evaluation` được dành chỗ bằng package có tài liệu; chưa có metrics.

## 3. Cài đặt môi trường Windows

Cài **Python 3.11 64-bit** và chọn interpreter đó trong VS Code. Chạy tại thư mục
gốc project; dùng trực tiếp executable trong `.venv` để không cần đổi PowerShell
ExecutionPolicy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Baseline CPU: cài cặp phiên bản tương thích trước khi cài project.
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Với GPU NVIDIA trên Windows, chọn cặp torch/torchvision và CUDA wheel phù hợp
từ [hướng dẫn chính thức PyTorch](https://pytorch.org/get-started/locally/) hoặc
[các cặp phiên bản đã phát hành](https://pytorch.org/get-started/previous-versions/).
Không cài CPU wheel trước nếu muốn dùng GPU. Project không yêu cầu torchaudio.
Các khoảng phiên bản trong `pyproject.toml` giúp dùng lại PyTorch có sẵn trên
Colab; phiên bản kiểm thử thực tế được ghi trong `docs/verification.md`.

Trong workspace hiện tại, máy ban đầu chưa có Python hệ thống khả dụng. Một
runtime Python 3.11 cục bộ và dependency đã được chuẩn bị tại `.tools/python311/`
để kiểm chứng. Có thể chạy ngay tại root mà không cài lại:

```powershell
.\.tools\python311\python.exe -m pytest -q
.\.tools\python311\python.exe -m scripts.smoke_test
.\.tools\python311\python.exe -m streamlit run app/streamlit_app.py
```

`.tools/` được Git ignore, chỉ dùng cục bộ; khi chia sẻ project hoặc phát triển
lâu dài, dùng Python 3.11 và môi trường `.venv` theo hướng dẫn phía trên.

Trên Linux/Colab, dùng `python -m pip install -r requirements.txt` tại project
root; hướng dẫn notebook ở mục 8. Không có lệnh nào tự tải ISIC hay trọng số.

## 4. Cấu hình

```python
from src.utils.config import load_config
from src.utils.device import get_device

config = load_config("configs/unet_roi_512.yaml")
device = get_device(config.device)
print(config.image_size, config.paths.images_dir, device)
```

| Tệp | Mô hình dự kiến | Chế độ | Kích thước H × W | Batch size |
| --- | --- | --- | --- | --- |
| `base.yaml` | Hai giai đoạn | ROI | 256 × 256 | 4 |
| `faster_rcnn.yaml` | Faster R-CNN | Full image | 512 × 512 | 2 |
| `unet_roi_256.yaml` | U-Net | ROI | 256 × 256 | 4 |
| `unet_roi_512.yaml` | U-Net | ROI | 512 × 512 | 2 |
| `unet_full_image_512.yaml` | U-Net đối chứng | Full image | 512 × 512 | 2 |

Đây là cấu hình dự phòng, chưa đảm bảo batch size tối ưu cho T4. Các tệp thí
nghiệm `extends: base.yaml`, chỉ ghi đè phần khác biệt. Đường dẫn dữ liệu,
checkpoint và output tương đối được resolve từ project root theo vị trí
`src/utils/config.py`, độc lập current working directory. Có thể dùng
`load_config(..., project_root=...)` cho test hoặc bản sao project; đường dẫn
tuyệt đối do người dùng cung cấp trong cấu hình cũng được chấp nhận.

`device: auto` chọn CUDA nếu `torch.cuda.is_available()`, còn lại CPU.
Yêu cầu `cuda` tường minh khi CUDA không khả dụng sẽ báo lỗi rõ ràng.
`set_seed` seed Python/NumPy/PyTorch nhưng không đảm bảo deterministic tuyệt đối
giữa các GPU hoặc phiên bản thư viện. Loader không tạo thư mục hay nạp checkpoint.

`image_size` của Faster R-CNN là kích thước dự kiến, chưa áp dụng transform cho
detector; phase xây mô hình cần thống nhất với transform nội bộ của Torchvision.
`checkpoint_dir`/`output_dir` là nơi lưu kết quả sau này. Hai trường
`detector_checkpoint`/`segmenter_checkpoint` là đường dẫn tệp tùy chọn, mặc định
`null`; đặt tên thư mục không tự chọn checkpoint.

## 5. Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_test
.\.venv\Scripts\python.exe -m scripts.smoke_test --config configs/unet_roi_512.yaml
.\.venv\Scripts\python.exe -m scripts.check_streamlit_server
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Smoke test kiểm tra import cả Torchvision, load cấu hình, chọn thiết bị và cấp
phát tensor, round-trip tọa độ synthetic, cùng giao diện Streamlit qua `AppTest`.
`--skip-app` chỉ dành cho chẩn đoán, không tương đương smoke test đầy đủ.
`AppTest` thực thi app trong test harness, không phải trình duyệt hay server HTTP.
`scripts.check_streamlit_server` khởi động server headless tạm thời trên localhost,
kiểm tra health và HTML qua HTTP, rồi dừng đúng process mà script đã tạo.
Tham khảo [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing).
Kết quả và giới hạn kiểm chứng xem [docs/verification.md](docs/verification.md).

## 6. Chạy Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Mở URL được Streamlit in ra, thường là `http://localhost:8501`. App dùng
`configs/base.yaml`, nhận JPG/JPEG/PNG, đổi ảnh sang RGB và hiển thị ảnh đầu vào.
Hai vùng Detection và Segmentation luôn giải thích trạng thái chưa triển khai
suy luận. Tệp checkpoint có tồn tại cũng không được load hoặc trình bày như
một mô hình đã xác minh. Không có nút tạo dự đoán hay kết quả giả.

## 7. Chuẩn bị dataset thủ công

Phase 1 không cần dataset. Khi chuẩn bị Phase 2, tự tải và giải nén **ISIC 2018
Task 1 training input** và **training ground truth**, theo điều kiện sử dụng
của [ISIC Challenge](https://challenge.isic-archive.com/data/#2018).
Tên thư mục mặc định:

```text
data/raw/ISIC2018/
├── ISIC2018_Task1-2_Training_Input/
│   └── ISIC_0000000.jpg
└── ISIC2018_Task1_Training_GroundTruth/
    └── ISIC_0000000_segmentation.png
```

Tên trên minh họa quy tắc ghép cặp; chưa có dữ liệu đi kèm. Nếu thư mục giải nén
khác, cập nhật các path trong YAML. Ảnh và mask cùng stem phải có kích thước và
pixel orientation tương ứng. `read_rgb` không tự xoay EXIF để tránh xoay riêng
ảnh mà không xoay mask. Mask đọc từ PNG grayscale, pixel khác 0 là foreground;
utility hình học chỉ nhận bool hoặc giá trị 0/1/255. Xác suất đầu ra phải được
threshold tường minh trước khi dùng utility mask.

Phase 2 mới kiểm tra cặp ảnh/mask, tạo split và data loader. Cần chia split trước
khi augmentation và xây ROI; mọi thí nghiệm dùng chung split. Chưa sinh bounding
box annotation, chưa tạo train/validation/test split ở Phase 1.
Dataset, checkpoint, output, local config và secrets được bỏ qua bởi `.gitignore`.

## 8. Google Colab

Mở `notebooks/01_foundation_smoke.ipynb`, đưa mã nguồn lên Colab hoặc Drive và
nhập project root qua ô nhập; không có đường dẫn Drive cá nhân hardcode.
Notebook chỉ cài project và chạy test, không tải dữ liệu hoặc trọng số.

Để huấn luyện ở phase sau, chọn runtime T4. Giữ cặp torch/torchvision tương thích
có sẵn trên Colab nếu import được; không cài CPU wheel vào GPU runtime.
Python của runtime phải >= 3.11. Sử dụng các lệnh sau sau khi đã chuyển tới root:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
subprocess.run([sys.executable, "-m", "scripts.smoke_test"], check=True)
```

## 9. Quy ước tọa độ và ví dụ utility

```python
from src.data.geometry import crop_mask, mask_to_bbox, restore_roi_mask
from src.data.image_io import read_binary_mask, read_rgb
from src.pipeline.spatial import prepare_roi

# Chỉ chạy ví dụ này sau khi đã tự chuẩn bị ảnh và ground truth tương ứng.
image = read_rgb(config.paths.images_dir / "ISIC_0000000.jpg")
mask = read_binary_mask(config.paths.masks_dir / "ISIC_0000000_segmentation.png")
box = mask_to_bbox(mask)
if box is not None:
    roi = prepare_roi(image, box, config.image_size, config.roi_padding)
    gt_roi = crop_mask(mask, roi.metadata)
    restored_gt = restore_roi_mask(gt_roi, roi.metadata)
```

Ví dụ dùng **ground truth** để kiểm tra phép biến đổi, không phải kết quả detector.
Mọi size là `(height, width)`, box là `(x_min, y_min, x_max, y_max)` theo khoảng
nửa mở `[x_min, x_max) × [y_min, y_max)`. Padding là phần tỉ lệ chiều rộng/cao
của box trên mỗi cạnh. Crop làm tròn ra ngoài và clip theo biên ảnh. Luôn khôi
phục bằng `CropMetadata` thực tế; không dùng lại box chưa padding.
Resize RGB dùng area khi thu nhỏ, linear khi phóng to; mask dùng nearest neighbor.
Resize xuống có thể làm mất chi tiết nên không hứa khôi phục mask chính xác sau
mọi lần resize. `scale_bbox` chuyển box từ ảnh detector đã resize về ảnh gốc.

## 10. Roadmap

1. **Phase 1 — hoàn thiện foundation:** utility, config, app và synthetic tests.
2. **Phase 2 — Data Pipeline:** inventory ảnh/mask, kiểm tra cặp và kích thước,
   split tái lập, PyTorch Dataset/collate, annotation bbox từ ground truth,
   augmentation đồng bộ, dataset ROI và full-image dùng cùng split.
3. **Phase 3 — Models & training:** triển khai Faster R-CNN/U-Net và training
   loop trên T4; chỉ tải pretrained weights khi được yêu cầu.
4. **Phase 4 — Evaluation & experiments:** IoU/Dice/detection metrics, so sánh
   ROI 256/512 với full-image 512, phân biệt ROI ground truth và ROI detector
   để tránh kết quả quá lạc quan.
5. **Phase 5 — Inference & app:** nạp checkpoint đã xác minh, chính sách chọn
   detection/no-detection, pipeline end-to-end và visualization thật.

Không tự chuyển sang Phase 2 trong lần triển khai này.

# Hỗ trợ phân tích ảnh soi da và chuẩn bị hội chẩn

Dự án hỗ trợ phân tích ảnh soi da và chuẩn bị hội chẩn. Gradio cho phép tải ảnh,
xem trước và chạy **Phân vùng thử nghiệm** bằng SegFormer MiT-B2 khi bấm nút.
CLI Phase 2B vẫn chạy độc lập; UI Phase 2C dùng lại cùng predictor và ngưỡng 0.5.
Đây là smoke test kỹ thuật, không phải chẩn đoán và chưa đánh giá độ chính xác.
Chưa có detection, classification, RAG, cơ sở dữ liệu hoặc báo cáo AI.

Xem [tài liệu và kết quả kiểm chứng Phase 2C](docs/phase-2c.md).

## Cài đặt trên Windows PowerShell

Yêu cầu Python 3.12. Nếu đã có môi trường Phase 2B, dùng lại `.venv` và tài nguyên
local, bỏ qua bước cài đặt. Khi dựng môi trường Windows CUDA 12.6 mới, mở
PowerShell ngay tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

Không cần kích hoạt môi trường ảo hoặc thay đổi Execution Policy.
Gradio tự cài các thư viện phụ thuộc; dự án không xây dựng dịch vụ FastAPI riêng.
`requirements-vision.txt` gồm dependency giao diện và vision đã kiểm chứng.
Lệnh cài dependency không tải checkpoint hoặc ảnh mẫu.

## Chạy giao diện

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở http://127.0.0.1:7860. Chọn ảnh JPG/PNG rồi nhấn **Xem ảnh** để xem trước,
hoặc **Phân vùng** để nhận ảnh gốc, mask, overlay và thiết bị thực tế.
Mô hình chỉ được nạp ở lần phân vùng đầu và dùng lại cho các lượt sau.
Checkpoint mặc định là `checkpoints/segformer-gate2/real_only/best.pt` local;
không tải weights khi mở app. Chế độ `auto` dùng CUDA khi khả dụng, nếu không dùng
CPU ngay từ đầu; lỗi inference không tự chuyển thiết bị.

Đổi/xóa ảnh, xem trước lại hoặc lượt chạy thất bại sẽ xóa mask, overlay và thiết bị
cũ. Muốn chọn ảnh khác trong ô upload đang có file, bấm nút **Clear** (dấu ×)
rồi chọn ảnh mới. Thông báo lỗi bằng tiếng Việt hiển thị tại **Thông báo**.
Ứng dụng chỉ lắng nghe trên localhost và tắt chia sẻ công khai (`share=False`).
Nhấn `Ctrl+C` trong PowerShell để dừng.

## Phase 2B — CLI segmentation độc lập

Tái sử dụng `.venv`, checkpoint và ảnh đã xác minh ở
[`outputs/phase2a1/report.md`](outputs/phase2a1/report.md). Môi trường hiện tại:
Python 3.12.5, torch 2.11.0+cu126, torchvision 0.26.0+cu126, SMP 0.5.0,
timm 1.0.31, NumPy 2.5.4, Pillow 13.0.0 và OpenCV headless 4.13.0.92.
Phase 2B chỉ cài thêm OpenCV còn thiếu, không cài lại bộ torch/SMP.

**Chạy lại ngay trong môi trường đã chuẩn bị**, không cần tải lại tài nguyên:

```powershell
.\.venv\Scripts\python.exe -m teledermoscopy.segment `
  --image data\samples\ISIC_0012169.jpg `
  --checkpoint checkpoints\segformer-gate2\real_only\best.pt `
  --output-dir outputs\phase2b `
  --device cuda
```

Mỗi lần nhận một ảnh. `--device auto` chọn CUDA khi khả dụng, nếu không chọn CPU.
`--device cuda` trả mã lỗi 1 khi CUDA thất bại, **không tự chuyển CPU**. Để thử CPU
riêng, đổi thành `--device cpu --output-dir outputs\phase2b\cpu`.
Thiết bị thực tế được in ra terminal và lưu trong JSON.

Pipeline cố định theo upstream: đọc RGB, resize 512×512 bằng `INTER_AREA`, chuẩn hóa
ImageNet `(pixel/255 - mean)/std`, mean `(0.485, 0.456, 0.406)`, std
`(0.229, 0.224, 0.225)`. Nạp `SMPWrapper` bọc SegFormer `mit_b2`, 3 kênh vào,
1 kênh ra, `encoder_weights=None`, `torch.load(weights_only=True, map_location="cpu")`
và `load_state_dict(strict=True)`. Dùng `eval()` và `inference_mode()` với float32.
Áp dụng sigmoid, resize **xác suất** về kích thước ảnh gốc bằng `INTER_LINEAR`,
sau đó threshold cố định **>= 0.5**. Không có tham số chỉnh ngưỡng hoặc hậu xử lý mask.

Đầu ra trong `outputs/phase2b/`:

- `ISIC_0012169_mask.png`: PNG một kênh, chỉ có giá trị 0/255.
- `ISIC_0012169_overlay.png`: PNG RGB tô đỏ vùng mask, cùng kích thước ảnh gốc.
- `ISIC_0012169_result.json`: thiết bị, kích thước, tỷ lệ mask, ngưỡng và SHA-256 đầu vào.

CLI đọc lại hai PNG để kiểm tra định dạng, kích thước và mask nhị phân. Mask rỗng
hoặc phủ kín được ghi nhận và cảnh báo, không thay thế bằng kết quả khác. Chạy lại
cùng tên ảnh trong cùng output-dir sẽ ghi đè các file kết quả tương ứng.

`SegmentationPredictor` trong `teledermoscopy/vision/segmentation.py` nạp mô hình
một lần; phương thức `predict(rgb)` nhận mảng RGB uint8 H×W×3 để 2C có thể dùng lại.
Không phụ thuộc Gradio và không tự tải weights hoặc dữ liệu.

Ảnh `ISIC_0012169` có giấy phép CC-0, ghi công Anonymous qua ISIC Archive;
metadata và provenance nằm bên cạnh ảnh trong `data/samples/`.
[Trang metadata gốc](https://api.isic-archive.com/images/ISIC_0012169/).
Chưa độc lập xác minh split chính thức; **chỉ dùng smoke test, không tính Dice/IoU
hay kết luận độ chính xác hoặc chẩn đoán**.

Chỉ khi dựng một `.venv` mới trên Windows CUDA 12.6, cài bộ dependency đã ghim:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

Lệnh cài dependency không tải checkpoint hoặc ảnh mẫu. Các file này giữ local
trong `checkpoints/` và `data/`, được `.gitignore` bỏ qua.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Để chạy toàn bộ kiểm thử cần bộ dependency vision bên trên. Kiểm thử tự sinh ảnh
hình học tạm thời; kiểm tra luồng xem ảnh, RGB/chuẩn hóa, thứ tự resize xác suất
trước threshold, quy tắc chọn thiết bị, CLI với ảnh không hợp lệ và checkpoint
bị thiếu. Kiểm thử lỗi yêu cầu exit code 1 và không sinh output giả; không tải
weights hoặc chạy mô hình. Lệnh CLI Phase 2B bên trên là smoke test inference thật.
Các test Phase 2C bổ sung kiểm tra cache, xóa trạng thái cũ, lỗi thiếu checkpoint,
hết VRAM giả lập và bỏ kết quả của ảnh đã thay đổi. Phần thao tác UI và inference
thật được ghi riêng trong [docs/phase-2c.md](docs/phase-2c.md).

Kiểm tra trực tiếp trên trình duyệt: nhấn **Xem ảnh** khi chưa chọn file để xem
thông báo; sau đó tải lần lượt một JPG và một PNG bất kỳ không chứa dữ liệu bệnh
nhân, nhấn **Xem ảnh** và đối chiếu ảnh trong vùng hiển thị với ảnh gốc.

## Cấu trúc hiện tại

```text
.
├── app.py
├── teledermoscopy/
│   ├── __init__.py
│   ├── gradio_segmentation.py
│   ├── segment.py
│   └── vision/
│       ├── __init__.py
│       └── segmentation.py
├── tests/
│   ├── test_app.py
│   ├── test_gradio_segmentation.py
│   └── test_segmentation.py
├── docs/
│   ├── Define_System.md
│   ├── System_Design.md
│   └── phase-2c.md
├── requirements.txt
├── requirements-vision.txt
├── .gitignore
└── README.md
```

`app.py` phụ trách giao diện và xem trước. `teledermoscopy/gradio_segmentation.py`
nối nút bấm với predictor, quản lý cache và trạng thái kết quả. Logic inference
dùng chung với CLI nằm trong `teledermoscopy/vision/`.

Cấu trúc đích trong `teledermoscopy/` gồm `vision/` (Detection, Segmentation),
`ingestion/` (nạp tài liệu), `rag/` (tra cứu tài liệu) và `reports/` (báo cáo nháp).
Hiện mới triển khai segmentation trong `vision/`; chưa tạo các module còn lại.

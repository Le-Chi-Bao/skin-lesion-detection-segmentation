# Báo cáo Phase 2B — 2026-11-16 (Asia/Bangkok)

Đầu vào: `outputs/phase2a1/report.md`. Tái sử dụng đúng `.venv`, checkpoint
và ảnh đã tải. SHA-256 của cả hai tài nguyên vẫn khớp 2A.1; không tải lại.

| Kiểm tra | Kết quả thực tế |
|---|---|
| Inference CUDA | PASS, CLI exit code 0; `cuda:0`, NVIDIA GeForce RTX 3050 Laptop GPU |
| Inference CPU | NOT RUN: CUDA thành công, không cần chạy CPU dự phòng |
| Mở lại mask và overlay | PASS, cả hai PNG có kích thước 3008×2000 |
| Mask nhị phân | PASS, mode L, chỉ có giá trị 0 và 255 |
| Kiểm thử tự động | PASS, 9 tests, gồm 2 trường hợp lỗi CLI |
| Dependency | `pip check`: No broken requirements found |

## Code và dependency

- `teledermoscopy/vision/__init__.py`: package xử lý ảnh.
- `teledermoscopy/vision/segmentation.py`: model wrapper, chọn thiết bị, đọc RGB,
  tiền xử lý, predictor dùng lại được, lưu và kiểm tra kết quả.
- `teledermoscopy/segment.py`: CLI độc lập với Gradio, một ảnh mỗi lần chạy.
- `tests/test_segmentation.py`: 5 kiểm thử mới.
- `requirements-vision.txt`: dependency vision đã ghim cho môi trường Windows CUDA hiện tại.
- `README.md`: lệnh Windows, pipeline, đầu ra và phạm vi smoke test.

Python 3.12.5. Bộ thư viện hiện hữu được giữ nguyên: torch 2.11.0+cu126,
torchvision 0.26.0+cu126, SMP 0.5.0, timm 1.0.31, NumPy 2.5.4, Pillow 13.0.0.
Chỉ cài thêm thư viện còn thiếu `opencv-python-headless==4.13.0.92` vào `.venv`.

## Pipeline đã chạy

SMPWrapper bọc `smp.Segformer(encoder_name="mit_b2", encoder_weights=None,
in_channels=3, classes=1)`. Nạp trên CPU bằng `weights_only=True`, lấy
`checkpoint["model"]`, `load_state_dict(strict=True)`. Không bỏ qua/đổi tên keys,
không tải encoder weights. Kiểm tra cấu hình checkpoint là `segformer_mitb2`,
không có model overrides và working resolution 512×512.

Ảnh được đọc thành RGB bằng OpenCV (imdecode hỗ trợ đường dẫn Unicode trên Windows),
resize 512×512 bằng INTER_AREA, float32 và chuẩn hóa `(pixel/255 - mean)/std`.
Mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
Mô hình ở eval(), chạy trong inference_mode(), dùng float32, không TTA.
Sigmoid logits, resize bản đồ xác suất bằng INTER_LINEAR về 3008×2000, rồi
threshold cố định **>= 0.5**. Không hậu xử lý hoặc điều chỉnh ngưỡng.

`auto` chọn CUDA khi khả dụng, nếu không chọn CPU. Khi đã chọn CUDA mà gặp lỗi,
CLI trả lỗi và không tự chuyển CPU. Thiết bị thực tế được in trước inference.

## Kết quả smoke test thật

- Ảnh: `data/samples/ISIC_0012169.jpg`, 3008×2000.
- Checkpoint: `checkpoints/segformer-gate2/real_only/best.pt`.
- Thiết bị: `cuda:0`, RTX 3050 Laptop, CUDA build 12.6.
- Inference và resize: khoảng 0.743 giây trong lần chạy này, không gồm load model/lưu file.
- Peak bộ nhớ tensor do PyTorch cấp phát: khoảng 291.68 MiB, không phải tổng VRAM tiến trình.
- Pixel mask: **801770 / 6016000 = 13.327293882978725%**.
- Mask không rỗng và không phủ kín; không sửa mask để thay đổi hình dạng.
- Xác suất native: min 1.77699305936585e-07, max 0.9997411370277405.

Đã mở lại hai PNG bằng Pillow và kiểm tra độc lập với metadata do CLI ghi.
Kết quả kiểm tra lưu trong `outputs/phase2b/verification.json`.

## Output local

- `outputs/phase2b/ISIC_0012169_mask.png`: PNG L nhị phân, 3008×2000.
- `outputs/phase2b/ISIC_0012169_overlay.png`: PNG RGB, 3008×2000, tô đỏ mask với alpha 0.4.
- `outputs/phase2b/ISIC_0012169_result.json`: thiết bị, thống kê, kích thước, SHA-256 đầu vào.
- `outputs/phase2b/verification.json`: kết quả đọc lại PNG, checksum output và phiên bản.

SHA-256:

```text
checkpoint: d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87
image:      45ec44071e288dfe84b889189968ea3a343933d05451219ca4e345c54f1a1653
mask:       1276e69204efe607b13083589bd416f2b15fb455f7842b9ce3d172889b1a0d4e
overlay:    b4614c12a85e54da742a6c3d0c9b6885c2614a4435034960c3d0c20b688599ac
```

## Kiểm thử

`python -m unittest discover -s tests -v`: 9/9 tests đạt.

- CLI ảnh không hợp lệ: trả exit code 1, stderr có thông báo ảnh không hợp lệ;
  không tạo thư mục output hay mask giả.
- CLI checkpoint bị thiếu, ảnh hợp lệ: trả exit code 1, stderr báo không tìm thấy
  checkpoint; không tạo thư mục output hay mask giả.
- Kiểm thử CUDA không khả dụng (mock trạng thái thiết bị): `--device cuda` phải
  lỗi; `auto` chọn CPU. Đây không phải một lần inference CPU.
- Kiểm tra RGB/chuẩn hóa/layout tensor và resize xác suất trước threshold >=0.5.
- Bốn test giao diện Bước 1 vẫn đạt.

## Lệnh đã chạy trong PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python-headless==4.13.0.92 -c requirements.txt
.\.venv\Scripts\python.exe -m teledermoscopy.segment --image data\samples\ISIC_0012169.jpg --checkpoint checkpoints\segformer-gate2\real_only\best.pt --output-dir outputs\phase2b --device cuda
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
git diff --check
git diff --exit-code -- app.py requirements.txt
```

Đã chạy thêm kiểm tra độc lập bằng Pillow/NumPy để mở lại PNG, xác nhận kích thước,
giá trị mask, số pixel và SHA-256; không chạy lại mô hình trong bước kiểm tra này.

Không sửa app.py, Gradio, crawler/RAG hoặc requirements.txt gốc. Không train,
không push. Outputs, checkpoint và ảnh giữ local. Đây chỉ là smoke test;
không tính Dice/IoU, không kết luận độ chính xác hoặc chẩn đoán. Ảnh mẫu có
CC-0, attribution Anonymous qua ISIC Archive; split chính thức chưa được
đối chiếu độc lập như đã ghi trong báo cáo 2A.1.

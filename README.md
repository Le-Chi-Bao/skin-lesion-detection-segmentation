# Hệ thống hỗ trợ phân tích tổn thương da

Ứng dụng nghiên cứu dành cho nhân viên y tế đã được đào tạo. Mục tiêu của đồ án là hỗ trợ quan sát tổn thương trên ảnh soi da, tra cứu kiến thức da liễu và chuẩn bị hội chẩn. **WP3 bổ sung trình xem phân đoạn SegFormer MiT-B2, điều chỉnh độ mờ và xuất ảnh PNG.** Nguyên mẫu này chưa được thẩm định để sử dụng lâm sàng và không tự động chẩn đoán bệnh da.

| Mô-đun | Trạng thái WP3 |
| --- | --- |
| Tổng quan | Thông tin ứng dụng và trạng thái mô-đun |
| Computer Vision | Tải JPEG/PNG, phân đoạn, xem ảnh gốc, mask, lớp phủ và vùng định vị. Xuất PNG lớp phủ và định vị ở kích thước gốc. |
| Medical RAG, Clinical Cases, Consultation Reports | Chưa triển khai |

Nút **Phân tích** khả dụng khi checkpoint SegFormer đã được xác minh, nạp thành công và phiên có ảnh hợp lệ. Faster R-CNN ResNet50-FPN V2 hiển thị **Chưa tích hợp**. Khi hoàn tất, trình xem tự mở **Overlay** với độ mờ 40%. **Original** hiển thị ảnh đã chuẩn hóa hướng EXIF, **Mask** hiển thị mặt nạ nhị phân và **Localization** vẽ bounding box suy ra từ mặt nạ. Phần **Chi tiết mô hình** ghi nguồn checkpoint, thiết bị, ngưỡng và thời gian suy luận. [Hướng dẫn và kiểm chứng WP3](docs/WP3_CLINICAL_VISUALIZATION.md)

## Cài đặt và chạy cục bộ

Dùng Python 3.12. Các lệnh dưới đây chạy tại thư mục gốc của kho mã.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m app.main
```

Trên Linux hoặc macOS, thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`. Khi đã kích hoạt môi trường ảo, chạy `python -m app.main`. Mở `http://127.0.0.1:7860`. Gradio chạy với `share=False`. Cài đặt cơ bản hỗ trợ tải và xem ảnh khi chưa cấu hình SegFormer.

### Bật phân đoạn SegFormer

Ví dụ cài đặt CPU cho Windows hoặc Linux. Nếu môi trường đã có bộ PyTorch/Torchvision đang hoạt động, kiểm tra phiên bản trước và dùng môi trường ảo riêng nếu cần một cặp phiên bản khác. Hướng dẫn CUDA và macOS có trong [tài liệu WP2](docs/WP2_SEGFORMER_INTEGRATION.md).

```powershell
.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-cv.txt
.venv\Scripts\python.exe -m app.cv.checkpoint
$env:LESION_SEGFORMER_CHECKPOINT = 'model_store/segformer/real_only/best.pt'
$env:LESION_SEGFORMER_DEVICE = 'cpu'
.venv\Scripts\python.exe -m app.main
```

Lệnh tải xác minh SHA-256 và dùng lại tệp hợp lệ nếu đã có. Để dùng bản sao cục bộ của checkpoint, trỏ `LESION_SEGFORMER_CHECKPOINT` đến tệp đó và chạy `python -m app.cv.checkpoint --verify-only`. Máy chủ chỉ đọc checkpoint cục bộ khi khởi động. Trong **Computer Vision**, tải ảnh soi da có giấy phép phù hợp rồi chọn **Phân tích**. Sau khi hoàn tất, chọn độ mờ và bấm **Tải Overlay PNG**. Khi có box, dùng **Tải Localization PNG** để tải ảnh định vị.

## Cấu hình

Các biến môi trường sau ghi đè cấu hình mặc định. Đường dẫn tương đối được tính từ thư mục dự án.

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LESION_APP_NAME` | Hệ thống hỗ trợ phân tích tổn thương da | Tên hiển thị |
| `LESION_HOST` / `LESION_PORT` | `127.0.0.1` / `7860` | Địa chỉ IP hoặc `localhost` và cổng lắng nghe |
| `LESION_RUNTIME_DIR` | `runtime/` | Dữ liệu tạm. Dùng thư mục dành riêng cho một tiến trình ứng dụng |
| `LESION_MODEL_STORE_DIR` | `model_store/` | Nơi CLI lưu checkpoint |
| `LESION_SEGFORMER_CHECKPOINT` | Chưa cấu hình | Đường dẫn đến bản sao đã xác minh của `real_only/best.pt` |
| `LESION_SEGFORMER_DEVICE` | `auto` | Chọn CUDA nếu PyTorch nhận diện được CUDA, CPU nếu không. Có thể chọn cố định `cpu` hoặc `cuda` |
| `LESION_MAX_UPLOAD_MIB` | `10` | Dung lượng tệp, từ 1 đến 64 MiB |
| `LESION_MAX_IMAGE_PIXELS` | `16000000` | Tổng số pixel, tối đa 36 triệu |
| `LESION_MAX_IMAGE_EDGE` | `8192` | Số pixel mỗi cạnh, tối đa 16.384 |
| `LESION_SESSION_TTL_SECONDS` | `3600` | Thời gian giữ tệp tạm của ứng dụng sau lần sử dụng cuối, từ 1 giây đến 7 ngày |
| `LESION_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` hoặc `CRITICAL` |
| `LESION_RESEARCH_MODE` | `true` | WP1 yêu cầu `true` hoặc `1` |

`runtime/` và `model_store/` nằm trong `.gitignore`. Nếu đặt lại vị trí trong kho mã, giữ chúng dưới hai thư mục tương ứng. Hai vị trí phải tách biệt. Cấu hình host ngoài loopback làm cổng lắng nghe có thể truy cập từ mạng, tùy cấu hình máy.

## Phiên và dữ liệu tạm

Mỗi phiên Gradio giữ một ảnh hiện tại và kết quả phân đoạn tương ứng. Ảnh được chuẩn hóa hướng EXIF và màu RGB. Khi thay ảnh hoặc đặt lại, ứng dụng hủy hiệu lực của lượt phân tích và yêu cầu xóa ảnh cùng mask trước đó. Dữ liệu của ứng dụng nằm trong `runtime/sessions/`. Một lượt phân tích đang đọc hoặc ghi có thể giữ tệp đến khi thao tác hoàn tất. Kết quả của lượt đã hết hiệu lực bị loại bỏ.

Ứng dụng mã hóa PNG xuất trong bộ nhớ khi người dùng yêu cầu. Máy chủ kiểm tra phiên trình duyệt, ảnh và lượt phân tích hiện tại trước khi trả PNG. Quyền xuất hết hạn theo TTL của phiên.

Gradio quản lý riêng tệp tải lên trong `runtime/gradio-cache/` theo tuổi tệp. Đặt lại ảnh không xóa ngay mọi bản sao thuộc cache của Gradio. Bộ nhớ trình duyệt, tệp còn lại sau khi tiến trình dừng đột ngột và chi tiết vòng đời được mô tả trong tài liệu nền tảng. [Kiến trúc và giới hạn WP1](docs/CV_MVP_FOUNDATION.md)

## Kiểm thử và tích hợp tiếp theo

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q tests --basetemp runtime/wp3-regression
```

Bộ kiểm thử thông thường không tải checkpoint. Để chạy đủ ca đơn vị CV, cài thêm `requirements-cv.txt`. Thiết lập checkpoint thật, mẫu có giấy phép, lệnh kiểm thử Chromium và kết quả kiểm chứng được ghi trong [tài liệu WP3](docs/WP3_CLINICAL_VISUALIZATION.md) và [tài liệu WP2](docs/WP2_SEGFORMER_INTEGRATION.md).

Dùng thư mục riêng cho `--basetemp` vì pytest sẽ dọn thư mục đó khi chạy lại.

Để pytest chạy các ca trình duyệt, cài Playwright và Chromium trong môi trường kiểm thử.

```powershell
.venv\Scripts\python.exe -m pip install playwright==1.65.0
.venv\Scripts\python.exe -m playwright install chromium
```

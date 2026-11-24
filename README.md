# Hệ thống hỗ trợ phân tích tổn thương da

Ứng dụng nghiên cứu dành cho nhân viên y tế đã được đào tạo. Mục tiêu của đồ án là hỗ trợ quan sát tổn thương trên ảnh soi da, tra cứu kiến thức da liễu và chuẩn bị hội chẩn. **WP1 hiện hỗ trợ tải và xem ảnh.** Nguyên mẫu này chưa được thẩm định để sử dụng lâm sàng và không tự động chẩn đoán bệnh da.

| Mô-đun | Trạng thái WP1 |
| --- | --- |
| Tổng quan | Thông tin ứng dụng và trạng thái mô-đun |
| Computer Vision | Tải JPEG/PNG, kiểm tra ảnh, xem ảnh gốc và đặt lại phiên |
| Medical RAG, Clinical Cases, Consultation Reports | Chưa triển khai |

SegFormer MiT-B2 và Faster R-CNN ResNet50-FPN V2 đều hiển thị **Chưa tích hợp**. Nút **Phân tích** bị vô hiệu hóa. Các thẻ Mask, Overlay và Localization hiển thị trạng thái chưa có kết quả. Ứng dụng khởi động mà không đọc hoặc tải checkpoint.

## Cài đặt và chạy cục bộ

Dùng Python 3.12. Các lệnh dưới đây chạy tại thư mục gốc của kho mã.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m app.main
```

Trên Linux hoặc macOS, thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`. Khi đã kích hoạt môi trường ảo, chạy `python -m app.main`. Mở `http://127.0.0.1:7860`. Gradio chạy với `share=False`. Dùng ảnh tổng hợp cục bộ hoặc ảnh ISIC minh họa có giấy phép phù hợp.

## Cấu hình

Các biến môi trường sau ghi đè cấu hình mặc định. Đường dẫn tương đối được tính từ thư mục dự án.

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LESION_APP_NAME` | Hệ thống hỗ trợ phân tích tổn thương da | Tên hiển thị |
| `LESION_HOST` / `LESION_PORT` | `127.0.0.1` / `7860` | Địa chỉ IP hoặc `localhost` và cổng lắng nghe |
| `LESION_RUNTIME_DIR` | `runtime/` | Dữ liệu tạm. Dùng thư mục dành riêng cho một tiến trình ứng dụng |
| `LESION_MODEL_STORE_DIR` | `model_store/` | Vị trí dành cho checkpoint khi tích hợp mô hình |
| `LESION_MAX_UPLOAD_MIB` | `10` | Dung lượng tệp, từ 1 đến 64 MiB |
| `LESION_MAX_IMAGE_PIXELS` | `16000000` | Tổng số pixel, tối đa 36 triệu |
| `LESION_MAX_IMAGE_EDGE` | `8192` | Số pixel mỗi cạnh, tối đa 16.384 |
| `LESION_SESSION_TTL_SECONDS` | `3600` | Thời gian giữ tệp tạm của ứng dụng sau lần sử dụng cuối, từ 1 giây đến 7 ngày |
| `LESION_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` hoặc `CRITICAL` |
| `LESION_RESEARCH_MODE` | `true` | WP1 yêu cầu `true` hoặc `1` |

`runtime/` và `model_store/` nằm trong `.gitignore`. Nếu đặt lại vị trí trong kho mã, giữ chúng dưới hai thư mục tương ứng. Hai vị trí phải tách biệt. Cấu hình host ngoài loopback làm cổng lắng nghe có thể truy cập từ mạng, tùy cấu hình máy.

## Phiên và dữ liệu tạm

Mỗi phiên Gradio giữ một ảnh hiện tại. Ảnh được chuẩn hóa hướng EXIF và màu RGB. Khi thay ảnh hoặc đặt lại, ứng dụng xóa trạng thái cũ và yêu cầu xóa tệp chuẩn hóa của ảnh trước. Dữ liệu của ứng dụng nằm trong `runtime/sessions/`. Tác vụ nền dọn phiên hết hạn, và thao tác đang đọc hoặc ghi tệp có thể giữ tệp đến khi hoàn tất.

Gradio quản lý riêng tệp tải lên trong `runtime/gradio-cache/` theo tuổi tệp. Đặt lại ảnh không xóa ngay mọi bản sao thuộc cache của Gradio. Bộ nhớ trình duyệt, tệp còn lại sau khi tiến trình dừng đột ngột và chi tiết vòng đời được mô tả trong tài liệu nền tảng. [Kiến trúc và giới hạn WP1](docs/CV_MVP_FOUNDATION.md)

## Kiểm thử và tích hợp tiếp theo

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q tests
```

Bộ kiểm thử dùng ảnh tự sinh và khởi động một tiến trình Gradio cục bộ để kiểm tra HTTP. WP2 tích hợp SegFormer qua hợp đồng trạng thái mô hình và `CVResult`. Faster R-CNN sẽ được tích hợp sau khi có checkpoint suy luận phù hợp. WP3 dành cho giao diện kết quả đầy đủ, điều khiển độ mờ và xuất PNG.

Nếu thư mục tạm mặc định của pytest bị từ chối truy cập trên Windows, thêm `--basetemp runtime/wp1-pytest`. Dùng một thư mục dành riêng cho kiểm thử vì pytest sẽ dọn thư mục này khi chạy lại.

Để pytest chạy thêm ca tải ảnh đồng thời trên trình duyệt, cài Playwright và Chromium trong môi trường kiểm thử.

```powershell
.venv\Scripts\python.exe -m pip install playwright==1.65.0
.venv\Scripts\python.exe -m playwright install chromium
```

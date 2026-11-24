# WP3 · Trình xem ảnh soi da và xuất PNG

WP3 bổ sung cách quan sát kết quả SegFormer MiT-B2 trong mô-đun **Computer Vision**. Đây là nguyên mẫu nghiên cứu. Mask và bounding box minh họa đầu ra mô hình. Việc thẩm định hiệu năng trên quần thể lâm sàng nằm ngoài phần kiểm chứng kỹ thuật này.

## Sử dụng trình xem

1. Mở **Computer Vision** và tải một tệp JPEG hoặc PNG. Trên màn hình rộng, cột trái chứa ảnh xem trước, nút **Phân tích**, nút **Đặt lại** và trạng thái mô hình. Trên màn hình hẹp, trình xem nằm dưới vùng tải ảnh.
2. Chọn **Phân tích**. Khi lượt chạy hoàn tất, ứng dụng chọn **Overlay**. Dùng thanh trượt hoặc ô số để chỉnh độ mờ từ 0% đến 100%. Mỗi lượt xem kết quả mới bắt đầu ở 40%. Tải ảnh khác hoặc đặt lại đưa giao diện về **Original** và độ mờ 40%.
3. Mở **Chi tiết mô hình** để xem tên mô hình, định danh checkpoint, SHA-256, thiết bị, ngưỡng, kích thước ảnh, thời gian forward và nguồn định vị nếu có. Thông báo đang chạy, hoàn tất và lỗi xuất hiện cạnh nút phân tích. Cảnh báo về mask xuất hiện dưới trình xem.
4. Chọn **Tải Overlay PNG** hoặc **Tải Localization PNG**. Overlay dùng độ mờ tại thời điểm nhấn nút. Khi mask rỗng, chỉ có thể tải Overlay. Cảnh báo giải thích rằng không có pixel tiền cảnh đạt ngưỡng và kết quả này không dùng để kết luận về bệnh lý.

| Thẻ | Nội dung |
| --- | --- |
| **Original** | Ảnh RGB sau khi áp dụng hướng EXIF, cùng nguồn hiển thị với ảnh xem trước. |
| **Mask** | Mask nhị phân SegFormer có giá trị 0 và 255. Khi thu nhỏ mặt nạ, trình xem dùng nearest-neighbor. |
| **Overlay** | Lớp màu xanh ngọc RGB `(20, 184, 166)` theo vùng tiền cảnh. Trình duyệt thay đổi độ mờ lớp này ngay trên ảnh gốc đã thu nhỏ. |
| **Localization** | Bounding box suy ra từ mask, vẽ trực tiếp trên ảnh. Tọa độ hiển thị theo pixel ảnh gốc. |

Các thẻ dùng cùng hướng và hệ tọa độ. Cạnh dài của ảnh xem trước tối đa 1.600 px. Các phép biến đổi hình học của trình xem chỉ áp dụng cho ảnh hiển thị. Khi chiếu box xuống ảnh xem trước, ứng dụng làm tròn các cạnh ra ngoài để giữ cả box một pixel. Cách biểu diễn `[x1, y1, x2, y2]` dùng cạnh phải và dưới loại trừ. Pillow vẽ đến pixel `x2−1` và `y2−1` ở kích thước gốc đối với tọa độ nguyên của mask.

## PNG, phiên và vòng đời dữ liệu

- Ảnh PNG dùng đúng kích thước của ảnh RGB đã chuẩn hóa EXIF. Điều chỉnh độ mờ và chuyển thẻ chỉ dùng dữ liệu hiển thị trong trình duyệt. Khi xuất Overlay, Pillow tạo `a = round(255 × độ mờ / 100)` và trộn từng kênh theo `(pixel gốc × (255 − a) + màu lớp phủ × a) / 255`, làm tròn về số nguyên gần nhất. Pixel nền giữ nguyên. Tại 0%, toàn bộ pixel trùng với ảnh RGB chuẩn hóa. Tại 100%, tiền cảnh dùng màu lớp phủ.
- Nút tải gửi `POST /lesion-export` với định danh phiên, lượt chạy, phiên bản thao tác ảnh, loại PNG và độ mờ. Máy chủ kiểm tra cookie `HttpOnly` cùng mã ngẫu nhiên riêng cho trang, kết quả hiện tại của `SessionState`, quyền xuất và thời hạn. Yêu cầu chỉ chấp nhận hai loại PNG cố định. Đường dẫn ảnh và tham chiếu mask lấy từ trạng thái phía máy chủ. Quyền xuất tồn tại tối đa bằng TTL tính từ lúc hiển thị kết quả.
- PNG được mã hóa trong bộ nhớ khi có thao tác tải xuống. Trình duyệt tạo Blob tạm thời để tải tệp có tên dạng `segformer-overlay-<mã-lượt>.png` hoặc `segformer-localization-<mã-lượt>.png`. Phản hồi có `Cache-Control: no-store`. `FileTransportPolicy` giữ chặn các tuyến tải tệp trực tiếp, kể cả nguồn, mask và trọng số mô hình.
- Khi đặt lại, tải ảnh khác hoặc bắt đầu lượt chạy mới, quyền xuất cũ bị thu hồi. Máy chủ kiểm tra lại kết quả sau mã hóa. Trình duyệt hủy yêu cầu đang chờ khi người dùng đổi ảnh, đặt lại hoặc phân tích lại. `RunToken` và phiên bản ảnh giữ đầu ra suy luận cũ khỏi phiên mới. Bộ quản lý tệp hoãn dọn nguồn và mask cho đến khi thao tác đang đọc hoặc ghi hoàn tất. Cache tải lên của Gradio có vòng đời riêng. [Vòng đời tệp tạm của WP1](CV_MVP_FOUNDATION.md#phiên-thứ-tự-cập-nhật-và-vòng-đời-tệp)

## Kiểm chứng

Kiểm thử dùng Python 3.12, Gradio 6.32.0, Playwright và Chromium. Đặt biến môi trường sau để chạy với checkpoint đã xác minh và ảnh mẫu có giấy phép. [Thiết lập checkpoint và ảnh mẫu WP2](WP2_SEGFORMER_INTEGRATION.md)

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:WP2_SMOKE_CHECKPOINT = 'model_store/segformer/real_only/best.pt'
$env:WP2_SMOKE_IMAGE = 'model_store/wp2-verification/dermoscopy.jpg'
$env:WP2_SMOKE_DEVICE = 'cpu'
.venv\Scripts\python.exe -m pytest -q -s tests --basetemp runtime/wp3-regression

$env:WP2_SMOKE_DEVICE = 'cuda'
python -m pytest -q -s tests/test_wp2_application.py tests/test_wp3_application.py -k 'real_checkpoint or wp3_real_browser' --basetemp runtime/wp3-real-cuda
```

Lệnh CUDA cần chạy trong môi trường PyTorch CUDA tương thích. Chromium tải thật cả hai PNG trong hai phiên độc lập, với ảnh mẫu gốc và bản xoay 90°. Bài kiểm thử kiểm tra toàn bộ pixel Overlay, tọa độ đường bao, chuyển thẻ, đổi độ mờ bằng bàn phím, trạng thái sau thay ảnh hoặc đặt lại và truy cập tệp trực tiếp bị chặn.

| Nhóm | Đạt | Lỗi | Bỏ qua | Bằng chứng |
| --- | ---: | ---: | ---: | --- |
| Hồi quy đầy đủ trên CPU | 112 | 0 | 0 | `runtime/wp3-regression-cpu.log` |
| WP1/WP2 trong hồi quy CPU | 76 | 0 | 0 | Các tệp kiểm thử có sẵn trước WP3, gồm ca Chromium tải ảnh đến sai thứ tự. |
| Đơn vị hiển thị | 25 | 0 | 0 | `tests/test_visualization.py` |
| Trạng thái UI và hiển thị đến muộn | 5 | 0 | 0 | `tests/test_wp3_application.py`, ngoài ca Chromium dùng mô hình thật. |
| Quyền xuất PNG, hết hạn và xử lý đồng thời | 5 | 0 | 0 | `tests/test_png_export.py`. Ca Chromium WP3 cũng kiểm tra HTTP 403, truy cập chéo phiên, tham số đường dẫn và phiên bản cũ. |
| Bảo mật chạy riêng | 7 | 0 | 0 | `python -m pytest -q tests/test_png_export.py tests/test_app_smoke.py::test_launch_options_are_local_and_do_not_allow_static_paths tests/test_app_smoke.py::test_cli_serves_over_http_and_disables_direct_file_delivery --basetemp runtime/wp3-security` |
| Checkpoint thật trên CUDA | 3 | 0 | 0 | `runtime/wp3-real-cuda.log`. Bộ lọc loại 10 ca ngoài phạm vi. |

CPU gồm ba ca Chromium, trong đó hai ca chạy SegFormer thật. CUDA gồm hai ca Chromium với SegFormer thật và một ca kiểm tra callback trực tiếp. Cả CPU và CUDA trả logits `(1, 1, 512, 512)`, mask 330 × 244 px với 1.016 pixel tiền cảnh và box `[138, 89, 181, 124]` trên mẫu gốc. Thời gian forward của ca callback trực tiếp là 913,61 ms trên CPU và 793,86 ms trên NVIDIA GeForce RTX 3050 Laptop GPU. Mỗi giá trị chỉ là phép đo của một lượt chạy.

## Giới hạn và công việc tiếp theo

- Bản thu nhỏ của mask dùng lấy mẫu lân cận gần nhất. Vùng tiền cảnh rất nhỏ có thể biến mất trong bản xem trước. Tệp mask gốc và ảnh xuất dùng lưới ảnh gốc. PNG kích thước lớn cần thời gian mã hóa và bộ nhớ tương ứng.
- WP2 áp ngưỡng 0,7 cho dự đoán 512 × 512 trước khi resize mask. Quy trình đánh giá gốc của tác giả resize bản đồ xác suất trước khi áp ngưỡng. WP3 giữ nguyên thứ tự của WP2. Kiểm thử này xác nhận hành vi phần mềm và khả năng chạy checkpoint với ảnh mẫu. Việc đo Dice, IoU và hiệu năng trên dữ liệu đánh giá riêng cần một quy trình nghiên cứu riêng. [Chi tiết hậu xử lý WP2](WP2_SEGFORMER_INTEGRATION.md)
- Phần tích hợp và kiểm chứng Faster R-CNN thuộc công việc tiếp theo cần duyệt ở WP4. Brief WP3 chưa định nghĩa chi tiết WP5, vì vậy phạm vi WP5 cần được xác lập trong brief riêng. Medical RAG, Clinical Cases và Consultation Reports vẫn hiển thị trạng thái **Chưa triển khai**.

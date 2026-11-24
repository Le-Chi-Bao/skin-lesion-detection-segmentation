# Nền tảng CV MVP trong WP1

WP1 ưu tiên sản phẩm trước trong định hướng cân bằng 50/50 giữa sản phẩm và nghiên cứu. Chiến lược CV Hybrid Progressive đưa ứng dụng CV lên trước, trong một shell chính có thể mở rộng cho RAG, ca lâm sàng và báo cáo hội chẩn. Giao diện Balanced Clinical UI dùng theme Gradio sáng, điểm nhấn xanh navy và xanh teal, thanh điều hướng bên trái và màn hình Computer Vision hai cột. [Tài liệu WP2](WP2_SEGFORMER_INTEGRATION.md) mô tả phần suy luận SegFormer triển khai trên nền tảng này.

[Định nghĩa hệ thống](Define_System.md) và [thiết kế hệ thống](System_Design.md) trình bày lộ trình tổng thể gồm tra cứu, tổng hợp ca bệnh và tạo báo cáo. Lộ trình triển khai hiện tại ưu tiên shell và CV. Ba mô-đun còn lại xuất hiện ở trạng thái **Chưa triển khai**.

## Kiến trúc và luồng tải ảnh

| Thành phần | Trách nhiệm |
| --- | --- |
| `app.main`, `app.config` | Cấu hình, khởi động Gradio, định tuyến tệp và vòng đời tác vụ dọn tệp |
| `app.ui` | Điều hướng, Tổng quan, giao diện CV và thông báo bằng tiếng Việt |
| `app.core` | Kiểm tra nội dung ảnh, phiên, thư mục tạm và logging |
| `app.cv` | Hợp đồng trạng thái mô hình, trạng thái lượt chạy và kết quả CV độc lập với giao diện |

Gradio nhận từng tệp, có giới hạn dung lượng ngay tại điểm tải lên. Ứng dụng chỉ đọc đường dẫn đã kiểm tra trong cache tải lên. Bộ kiểm tra nhận dạng chữ ký JPEG/PNG, kiểm tra dung lượng, kích thước cạnh và số pixel trước khi giải mã đầy đủ. Tệp hỏng và PNG động bị từ chối. Ứng dụng áp dụng hướng EXIF, chuẩn hóa RGB, bỏ metadata khi ghi ảnh PNG chuẩn hóa và lưu SHA-256 của byte tệp đã nhận trong metadata phiên.

Ảnh xem trước và thẻ **Original** lấy từ cùng một bản hiển thị với cạnh dài tối đa 1.600 px. Giao diện giữ tỉ lệ ảnh. Pixel và kích thước dùng cho suy luận lấy từ PNG chuẩn hóa đầy đủ. **Phân tích** khả dụng khi có SegFormer sẵn sàng và ảnh hợp lệ. **Mask** hiển thị mask nhị phân đúng kích thước ảnh và **Localization** hiển thị tọa độ tính từ mask. **Overlay** dành cho WP3.

## Phiên, thứ tự cập nhật và vòng đời tệp

- Mỗi `gr.State` lưu một `SessionState` do máy chủ tạo khi phiên có thao tác đầu tiên. Phiên có UUID riêng, ID và phiên bản ảnh, metadata, tham chiếu tệp nội bộ và trạng thái lượt chạy. Trình duyệt đánh số tăng dần mỗi lần chọn ảnh hoặc đặt lại và gắn số đó vào tên tệp gửi lên. Nếu tệp hoàn tất sau một thao tác mới hơn, trình duyệt bỏ kết quả tải cũ. Các yêu cầu đã gửi đi qua cùng một hàng đợi Gradio để cập nhật tuần tự. Máy chủ chỉ áp dụng thao tác mới hơn thao tác đã xử lý gần nhất trong phiên, kể cả khi các yêu cầu đến hàng đợi sai thứ tự.
- Ảnh được ghi xong trước khi thay thế trạng thái. Mỗi lần ứng dụng áp dụng ảnh mới, báo lỗi xác thực ảnh hoặc đặt lại sẽ tăng `image_revision` và bỏ kết quả cùng lượt phân tích trước. Khi áp dụng kết quả, `SessionState` kiểm tra `session_id`, `image_revision`, `run_id`, định danh ảnh, SHA-256 và kích thước. Một lượt chỉ nhận một kết quả cuối cùng. WP2 chạy lượt suy luận ở hàng đợi riêng và chỉ gửi kết quả về giao diện qua hàng đợi cập nhật ảnh hiện có.
- `runtime/sessions/<session_uuid>/` chứa PNG chuẩn hóa và mask của phiên với tên ngẫu nhiên. Trình quản lý kiểm tra phạm vi đường dẫn và từ chối symlink hoặc junction chuyển hướng ra ngoài. Thao tác đang dùng tệp được giữ đến khi hoàn tất. Yêu cầu dọn ảnh, mask cũ hoặc đóng phiên được thực thi sau khi các thao tác trên thư mục phiên kết thúc. Khóa runtime dành thư mục cho một tiến trình ứng dụng.
- Tác vụ nền quét thư mục phiên khi khởi động và sau mỗi `min(60, TTL)` giây. TTL mặc định là 60 phút kể từ thao tác cuối trên tệp ứng dụng. Gradio gọi `gr.State.delete_callback` khi xóa trạng thái phiên sau khi ngắt kết nối hoặc thu hồi phiên. Một tab vẫn mở có thể thấy ảnh đã hiển thị sau khi tệp hết hạn. Đọc lại tệp đã hết hạn trả lỗi yêu cầu tải lại ảnh.
- Gradio giữ tệp tải lên riêng tại `runtime/gradio-cache/`. `Blocks(delete_cache=(chu_kỳ, TTL))` dùng chính sách tuổi tệp của Gradio cho các mục cache mà tiến trình theo dõi. Cache tải lên do Gradio chia sẻ ở cấp tiến trình, với các mục dữ liệu có thể được dùng lại khi nội dung trùng. Gradio có thể giữ mục cache sau đặt lại hoặc sau khi tiến trình dừng bất thường. Xem ảnh bằng dữ liệu nội tuyến trong phiên và chặn tuyến HTTP tải tệp trực tiếp. Cam kết chờ thao tác đang dùng tệp áp dụng cho tệp do WP1 quản lý. Bộ nhớ trình duyệt và cache Gradio có vòng đời riêng. [Vòng đời tài nguyên Gradio](https://www.gradio.app/guides/resource-cleanup)

## Hợp đồng mô hình và `CVResult`

`ModelAvailability` mô tả khả năng sử dụng mô hình qua `not_available`, `loading`, `ready` và `failed`. `RunStatus` mô tả lượt xử lý qua `not_available`, `running`, `completed` và `failed`. SegFormer chuyển sang `ready` sau khi nạp checkpoint đã xác minh trên thiết bị đã chọn. Faster R-CNN giữ trạng thái `not_available`.

`CVResult` định danh ảnh đã áp dụng EXIF, lượt chạy, kích thước và SHA-256. Kết quả phân đoạn lưu kích thước ảnh, tham chiếu mask có kích thước tương ứng, ngưỡng, cảnh báo và metadata mô hình. Kết quả phát hiện lưu box và điểm phát hiện do mô hình trả về. Điểm phát hiện không biểu thị xác suất ác tính. Box dùng tọa độ theo pixel, thứ tự `[x1, y1, x2, y2]`, với cạnh phải và cạnh dưới loại trừ. Cả box phát hiện và box `derived_from_mask` phải nằm trong ảnh. Box `derived_from_mask` yêu cầu mask hợp lệ theo hợp đồng. Ở WP2, bộ sinh mask chịu trách nhiệm khai báo kích thước thực và tính box từ pixel mask.

Hợp đồng lưu mã lỗi có thể ánh xạ sang thông báo người dùng, thời gian suy luận nếu có và cảnh báo. Metadata mô hình gồm định danh, checkpoint hoặc SHA-256 của checkpoint và thiết bị nếu có. Một lượt đã hoàn tất có thể chứa kết quả thành công của một mô hình cùng trạng thái lỗi của mô hình còn lại. Mô hình chưa có trả `not_available` và kết quả rỗng. Hợp đồng hỗ trợ JSON với tham chiếu mask dạng định danh nội bộ.

## Mốc tích hợp sau WP1

- **WP2 SegFormer MiT-B2** dùng `ModelAvailability`, ảnh RGB trong phiên, `RunToken` và `CVResult` với checkpoint `real_only/best.pt` đã ghim phiên bản. [Quy trình và bằng chứng kiểm thử](WP2_SEGFORMER_INTEGRATION.md)
- Faster R-CNN ResNet50-FPN V2 đang được fine-tune riêng trên Google Colab. Tích hợp ứng dụng cần checkpoint suy luận phù hợp.
- **WP3** bổ sung hiển thị kết quả đầy đủ, độ mờ và xuất PNG. Phần nghiên cứu CV tiếp tục đánh giá mô hình trên dữ liệu phù hợp. WP1 chỉ phục vụ nghiên cứu trên máy phát triển, không có xác thực hay kho lưu hồ sơ bệnh nhân.

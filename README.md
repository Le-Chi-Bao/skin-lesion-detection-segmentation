# Skin Lesion Detection, Segmentation & Clinical Consultation Support System

Đồ án tốt nghiệp xây dựng hệ thống hỗ trợ phân tích tổn thương da trên ảnh soi da,
tra cứu kiến thức y khoa và chuẩn bị hồ sơ hội chẩn. Hệ thống hướng tới hỗ trợ bác sĩ;
kết quả cần được kiểm tra chuyên môn, không thay thế chẩn đoán hoặc quyết định điều trị.

## Mục tiêu

- Hỗ trợ xác định vị trí và ranh giới tổn thương trên ảnh dermoscopy.
- Tra cứu tài liệu y khoa có nguồn trích dẫn và có thể kiểm tra lại.
- Kết nối kết quả phân tích ảnh, tài liệu tham khảo và thông tin ca khám.
- Chuẩn bị báo cáo nháp để bác sĩ rà soát trước hội chẩn.

## Bốn giai đoạn dự kiến

1. **Computer Vision (CV):** phát hiện và phân đoạn tổn thương da.
2. **RAG:** truy xuất kiến thức y khoa có trích dẫn.
3. **Integration:** tích hợp CV, RAG và báo cáo ca lâm sàng.
4. **Evaluation & Finalization:** đánh giá, hoàn thiện và tổng hợp kết quả đồ án.

## Trạng thái hiện tại

**Preparing for Phase 1 — Đang chuẩn bị triển khai lại Phase 1.**

Repository đã được dọn về nền tảng tài liệu để bắt đầu triển khai mới với kiến trúc
đơn giản, dễ bảo trì. Chưa triển khai Phase 1 mới; hiện không có mã ứng dụng,
module CV, RAG hoặc tích hợp hoạt động trong phiên bản baseline này.
Các triển khai trước được lưu trong lịch sử Git và nhánh backup.

Pipeline crawling và tiền xử lý tài liệu RAG được quản lý ở repository riêng.
Tài nguyên local được Git bỏ qua, như môi trường ảo, dữ liệu, checkpoint và output
thử nghiệm cũ, không thuộc baseline và chưa bị xóa.

## Tài liệu dự án

- [Define System — Định nghĩa hệ thống](docs/Define_System.md)
- [System Design — Thiết kế hệ thống](docs/System_Design.md)

Hai tài liệu được giữ nguyên để tham chiếu phạm vi và kiến trúc tổng thể; các mô tả
tiến độ cũ trong đó không xác nhận chức năng hiện có của baseline này.
Các liên kết sơ đồ docs/assets/ hiện thiếu file nguồn từ trước đợt dọn dẹp;
cần bổ sung lại tài sản gốc khi có sẵn.

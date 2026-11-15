# DEFINE SYSTEM
## Định nghĩa hệ thống hỗ trợ phân tích tổn thương da và hội chẩn

> **Phiên bản V3 — Từ tổng quan đến mức chi tiết vừa phải**  
> **Đối tượng đọc:** Thành viên nhóm đồ án, giảng viên hướng dẫn và người tham gia đánh giá.  
> **Mục tiêu tài liệu:** Trả lời ba câu hỏi: *Hệ thống giải quyết vấn đề gì? Có những chức năng nào? Ai sử dụng và chịu trách nhiệm?*

---

## 01. Giới thiệu hệ thống

Hệ thống được xây dựng để **hỗ trợ bác sĩ quan sát tổn thương da trên ảnh dermoscopy, tra cứu kiến thức da liễu và chuẩn bị hồ sơ hội chẩn**. Thay vì để ảnh chụp, ghi chú ca khám và tài liệu chuyên ngành nằm rời rạc, hệ thống đặt chúng trong một quy trình thống nhất.

Đây là **công cụ hỗ trợ chuyên môn**, không phải hệ thống tự chẩn đoán hay tự quyết định điều trị. Mọi kết quả cần được bác sĩ kiểm tra trước khi sử dụng.

### Giá trị mà hệ thống hướng tới

![Hành trình giá trị của hệ thống từ trái sang phải](assets/02_define_value_flow.png)

| Khó khăn thường gặp | Cách hệ thống hỗ trợ |
|---|---|
| Khó trình bày rõ vùng tổn thương trên ảnh | Hiển thị vị trí và ranh giới vùng tổn thương để bác sĩ quan sát |
| Tốn thời gian tìm tài liệu chuyên ngành | Tra cứu nội dung liên quan và dẫn về nguồn tài liệu |
| Thông tin ca khám nằm ở nhiều nơi | Đặt ảnh, bệnh sử, kết quả AI và tài liệu cùng một hồ sơ |
| Chuẩn bị hội chẩn thủ công | Tạo bản nháp có cấu trúc để bác sĩ rà soát |

## 02. Người sử dụng

**Bác sĩ đa khoa tuyến huyện đã được đào tạo và có dermatoscope** là người dùng chính: ghi nhận ca khám, xem kết quả phân tích ảnh, tra cứu tài liệu và chuẩn bị hội chẩn. **Bác sĩ da liễu** có thể xem hồ sơ được chia sẻ theo quy trình phù hợp và đưa ra nhận định chuyên môn. **Người phụ trách kho tài liệu** kiểm tra nguồn, độ đầy đủ và trạng thái được phép sử dụng.

Hệ thống **không hướng tới việc để người bệnh tự đưa ảnh lên và nhận kết luận bệnh**.

## 03. Những module chính

![Bốn module chính trình bày dạng box](assets/01_define_modules.png)

### Module 1 — Phân tích hình ảnh

Hỗ trợ bác sĩ nhìn thấy **vùng có tổn thương** trên ảnh và **ranh giới** của vùng đó. Hai kết quả cơ bản là *Detection* (xác định vị trí) và *Segmentation* (phân đoạn vùng tổn thương). Chúng **không đồng nghĩa với việc đã phân loại được bệnh**.

### Module 2 — Tra cứu kiến thức y khoa (RAG)

Tìm những phần tài liệu da liễu liên quan đến câu hỏi hoặc chủ đề bác sĩ đang quan tâm. Kết quả cần đi cùng **tên tài liệu, nguồn và vị trí trích dẫn** để có thể kiểm tra lại. Nếu thiếu nguồn đáng tin cậy, hệ thống phải nói rõ giới hạn.

### Module 3 — Kết nối thông tin ca bệnh

Tập hợp ba loại thông tin khác nhau: **bệnh sử/quan sát do bác sĩ ghi nhận**, **kết quả xử lý ảnh của AI** và **kiến thức được truy xuất từ tài liệu**. Các thông tin được trình bày riêng rẽ để tránh nhầm một dự đoán của máy với dữ kiện thực tế.

### Module 4 — Hỗ trợ báo cáo hội chẩn

Sắp xếp thông tin thành **bản nháp báo cáo** gồm tóm tắt ca khám, ảnh và vùng tổn thương, tài liệu tham khảo, thông tin còn thiếu và những điều cần bác sĩ xác minh. **Bác sĩ là người chỉnh sửa và xác nhận**.

## 04. Dữ liệu hệ thống sử dụng

| Nhóm dữ liệu | Dùng để làm gì? | Lưu ý |
|---|---|---|
| **Ảnh dermoscopy / dữ liệu ISIC** | Phát triển và đánh giá khả năng phát hiện, phân đoạn tổn thương | Không mặc nhiên có đầy đủ bệnh sử bệnh nhân |
| **Kho tài liệu da liễu** | Tìm kiếm và đối chiếu kiến thức y khoa | Cần kiểm tra nguồn, phiên bản và điều kiện sử dụng |
| **Thông tin ca khám** | Đặt ảnh và kiến thức trong bối cảnh cụ thể | Do bác sĩ cung cấp, cần được bảo vệ |

Theo báo cáo hoàn thành **Phase 2** mà nhóm đã cung cấp, dữ liệu văn bản hiện có **1.294 tài liệu hợp nhất**, trong đó tập thử nghiệm ISIC gồm **56 tài liệu nguồn, 90 tham chiếu và 493 đoạn văn (chunks)**. Các nhãn MEL, NV, BCC, AK, BKL, DF, VASC và SCC được dùng để **tra cứu theo chủ đề**, không phải kết quả chẩn đoán suy ra từ mặt nạ ảnh.

## 05. Các tình huống sử dụng tiêu biểu

| Tình huống | Kết quả mong đợi |
|---|---|
| Bác sĩ muốn quan sát rõ vùng tổn thương | Có ảnh gốc và vùng được AI đánh dấu để so sánh |
| Bác sĩ cần tham khảo đặc điểm một nhóm bệnh | Có tài liệu liên quan và nguồn để đọc lại |
| Hồ sơ chưa đủ thông tin | Có danh sách những mục cần kiểm tra thêm, không tự điền giả định |
| Bác sĩ cần chuẩn bị hội chẩn | Có bản nháp tổng hợp ảnh, ca khám và bằng chứng truy xuất |
| Chưa tìm thấy tài liệu đủ tin cậy | Hệ thống thể hiện rõ giới hạn, không đưa ra kết luận tùy đoán |

## 06. Ranh giới và nguyên tắc sử dụng

- **AI hỗ trợ quan sát và tìm tài liệu; bác sĩ chịu trách nhiệm quyết định chuyên môn.**
- Bounding box và segmentation mask **không phải nhãn bệnh**. Không tự kết luận melanoma, BCC hoặc SCC chỉ từ hai kết quả này.
- Chỉ dùng tài liệu **đã được duyệt phù hợp** cho câu trả lời phục vụ lâm sàng. Các nguồn chưa duyệt chỉ dùng để kiểm thử kỹ thuật trong môi trường tách biệt.
- Không tự kê đơn, không tự quyết định chuyển tuyến, không khẳng định tổn thương lành tính hay an toàn.
- Mọi nội dung tạo ra cần phân biệt rõ **dữ kiện đã ghi nhận**, **kết quả do AI tạo**, **tài liệu tham khảo** và **điểm chưa xác minh**.

## 07. Phạm vi phiên bản đầu tiên

**Tập trung phát triển:** phân tích ảnh dermoscopy (Detection + Segmentation), tìm tài liệu có trích dẫn, ghép thông tin ca khám và tạo bản nháp báo cáo hội chẩn. **Chưa đặt mục tiêu:** tự động chẩn đoán, kê đơn, thay thế hội chẩn hoặc sử dụng ảnh điện thoại thông thường ngoài phạm vi được kiểm chứng.

**Một câu để giới thiệu đồ án:**  
> *Hệ thống giúp bác sĩ nhìn rõ hơn vùng tổn thương trên ảnh, tìm nhanh kiến thức có nguồn và chuẩn bị hồ sơ hội chẩn dễ kiểm tra — còn kết luận chuyên môn vẫn do bác sĩ đưa ra.*

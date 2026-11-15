# SYSTEM DESIGN
## Thiết kế hệ thống hỗ trợ phân tích ảnh và tra cứu kiến thức da liễu

> **Phiên bản V3 — Kiến trúc mức tổng quan và chi tiết vừa phải**  
> **Mục tiêu tài liệu:** Cho thấy những module của hệ thống, cách chúng kết nối và thứ tự dữ liệu đi qua từng khối.  
> **Quy ước:** Tất cả sơ đồ được trình bày **từ trái sang phải**; hình ảnh là sơ đồ đã dựng sẵn, không cần công cụ vẽ hay mã lập trình để xem.

---

## 01. Kiến trúc tổng thể

![Sơ đồ kiến trúc hệ thống với các module dạng box](assets/03_architecture_modules.png)

Hệ thống có **hai nhánh xử lý chính**. Nhánh thứ nhất xem xét **ảnh dermoscopy** để xác định vị trí và ranh giới vùng tổn thương. Nhánh thứ hai tìm **tài liệu da liễu** dựa trên câu hỏi và thông tin ca bệnh. Hai kết quả gặp nhau tại phần **tổng hợp hồ sơ**, sau đó tạo **báo cáo nháp** để bác sĩ rà soát.

| Module | Vai trò | Kết quả đưa sang module tiếp theo |
|---|---|---|
| **1. Tiếp nhận ca khám** | Nhận ảnh, ghi chú và câu hỏi của bác sĩ | Thông tin ban đầu của ca khám |
| **2. Phân tích hình ảnh** | Xác định vị trí và phân đoạn vùng tổn thương | Vùng được đánh dấu, mặt nạ và ảnh trực quan |
| **3. Tra cứu RAG** | Tìm phần tài liệu liên quan, kèm trích dẫn | Những nguồn có thể đối chiếu |
| **4. Tổng hợp ca bệnh** | Kết nối thông tin ca khám với kết quả hai nhánh | Bản thông tin tổng hợp và những mục còn thiếu |
| **5. Tạo báo cáo** | Trình bày các kết quả theo một bố cục chung | Báo cáo nháp hỗ trợ hội chẩn |
| **6. Bác sĩ rà soát** | Kiểm tra, chỉnh sửa và xác nhận | Nhận định và quyết định chuyên môn |

## 02. Thiết kế luồng dữ liệu tài liệu

![Pipeline tài liệu từ trái sang phải](assets/04_data_pipeline.png)

Kho tài liệu đã được crawl từ các nguồn da liễu và tổ chức lại để có thể tìm kiếm. **Phase 2 đã chuẩn bị dữ liệu, chưa tự động tạo ra một hệ thống RAG hoàn chỉnh.**

- **Nguồn ban đầu:** tài liệu PDF/HTML, cùng thông tin xuất xứ và các chỉ mục liên quan.
- **Hợp nhất và làm sạch:** chuẩn hóa nội dung, giữ nguồn gốc và đánh dấu các tài liệu cần rà soát.
- **Tập thử nghiệm ISIC:** lựa chọn tài liệu liên quan đến các nhóm tổn thương trong phạm vi nghiên cứu.
- **Chia đoạn:** biến nội dung dài thành các đoạn ngắn hơn, vẫn truy ngược được về nguồn.
- **Bước tiếp theo:** xây khả năng tìm kiếm và đo xem có truy xuất đúng đoạn tài liệu hay không.

> **Trạng thái theo báo cáo Phase 2:** 1.294 tài liệu hợp nhất · 56 nguồn ISIC · 90 tham chiếu · 493 chunks thử nghiệm. Các nguồn **chưa được tự động phê duyệt về chuyên môn**.

## 03. Thiết kế module xử lý ảnh

**Đầu vào:** ảnh dermoscopy do bác sĩ cung cấp. **Đầu ra:** ảnh gốc, vùng tổn thương được đánh dấu và mặt nạ phân đoạn. Module này giúp bác sĩ quan sát, không tự kết luận một tổn thương thuộc nhóm bệnh nào.

Nếu ảnh không đạt điều kiện hoặc mô hình cho kết quả chưa đáng tin cậy, cần thông báo rõ. Những phép đo trên ảnh được hiểu là **đơn vị pixel**, không phải kích thước thực tế nếu chưa có hiệu chuẩn.

## 04. Thiết kế module RAG

![Luồng RAG theo các module dạng box](assets/05_rag_module.png)

RAG tìm kiến thức liên quan đến **câu hỏi của bác sĩ** và **bối cảnh ca khám**. Hệ thống ưu tiên kết quả có nguồn, tên tài liệu và vị trí trích dẫn. Trường hợp không đủ bằng chứng phải được thể hiện rõ, không cố đưa ra câu trả lời có vẻ chắc chắn.

**Ba lớp thông tin cần thể hiện trên giao diện:**

1. **Nội dung được tìm thấy:** đoạn tài liệu nào liên quan đến câu hỏi.
2. **Bằng chứng nguồn:** tài liệu nào, nguồn nào, trang/mục nào.
3. **Điều kiện sử dụng:** tài liệu đã được kiểm tra và cho phép phục vụ loại câu hỏi đó hay chưa.

Trong giai đoạn nghiên cứu, có thể thử nghiệm truy xuất với các chunk chưa duyệt trong môi trường tách biệt. **Chúng không được mặc định chuyển thành câu trả lời lâm sàng cho bệnh nhân.**

## 05. Luồng xử lý một ca khám

![Luồng một ca khám qua các khối từ trái sang phải](assets/06_clinical_workflow.png)

Bác sĩ cung cấp ảnh, bệnh sử và câu hỏi. Phần xử lý ảnh hỗ trợ quan sát vùng tổn thương; phần RAG hỗ trợ tìm tài liệu có liên quan. Thông tin được ghép thành hồ sơ rồi chuyển thành bản nháp báo cáo. Cuối cùng, **bác sĩ xem lại, chỉnh sửa và quyết định**.

## 06. Cấu trúc báo cáo hỗ trợ hội chẩn

![Các khối tạo báo cáo từ trái sang phải](assets/08_consultation_report.png)

Bản báo cáo nên có những phần dễ đọc dưới đây:

| Phần | Nội dung hiển thị |
|---|---|
| **Thông tin ca khám** | Thông tin do bác sĩ ghi nhận, không tự suy đoán dữ kiện còn thiếu |
| **Ảnh và phân tích** | Ảnh gốc, vùng phát hiện, mặt nạ phân đoạn và giới hạn của kết quả |
| **Kiến thức tham khảo** | Những đoạn tài liệu liên quan, trích dẫn và trạng thái kiểm duyệt |
| **Những điểm cần bổ sung** | Nội dung bác sĩ cần xác minh hoặc khai thác thêm |
| **Xác nhận của bác sĩ** | Phần xem xét, chỉnh sửa và xác nhận trước khi dùng trong hội chẩn |

**Nguyên tắc trình bày:** không gộp dữ kiện do bác sĩ ghi nhận, đầu ra AI và nội dung được tổng hợp từ tài liệu thành một kết luận duy nhất không thể kiểm tra nguồn.

## 07. Kiểm soát chất lượng và an toàn

![Sơ đồ kiểm soát tài liệu theo module](assets/07_source_governance.png)

**Kiểm soát nguồn:** một tài liệu đã tải về, đã làm sạch hoặc đã chia chunk **không có nghĩa** đã được duyệt y khoa. Cần có khả năng đánh dấu tài liệu nào chỉ dùng nghiên cứu, tài liệu nào đã được phép sử dụng cho những mục đích cụ thể.

**Kiểm soát đầu ra:** kết quả AI phải thể hiện giới hạn; báo cáo là bản nháp để bác sĩ xác nhận. Hệ thống không tự kê thuốc, xác định chẩn đoán hoặc quyết định chuyển tuyến.

**Bảo vệ ca khám:** thông tin bệnh nhân cần được giữ riêng với bộ dữ liệu nghiên cứu và chỉ chia sẻ cho người có quyền theo quy trình phù hợp.

## 08. Lộ trình triển khai tổng quát

| Nhóm công việc | Đã có / dự kiến | Mục tiêu dễ kiểm tra |
|---|---|---|
| Thu thập và tiền xử lý tài liệu | **Đã làm đến Phase 2, theo báo cáo Codex** | Kho tài liệu và các chunks có nguồn truy vết |
| Tìm kiếm kiến thức | **Dự kiến Phase 3** | Tìm được đoạn liên quan và hiển thị đúng nguồn |
| Đánh giá truy xuất | **Dự kiến** | Biết chất lượng tìm kiếm và các trường hợp thất bại |
| Kiểm tra và tích hợp CV | **Cần thực hiện/kiểm chứng** | Có ảnh, vùng phát hiện, mask được đánh giá độc lập |
| Tổng hợp và tạo báo cáo | **Dự kiến** | Có bản nháp rõ ràng để bác sĩ chỉnh sửa |
| Rà soát chuyên môn | **Điều kiện bắt buộc trước sử dụng lâm sàng** | Nguồn và quy trình được phê duyệt phù hợp |

## 09. Tóm tắt thiết kế

Thiết kế xoay quanh một nguyên tắc: **hệ thống hỗ trợ bác sĩ, không thay bác sĩ**. Ở một bên, Computer Vision giúp thể hiện vùng tổn thương; ở bên còn lại, RAG tìm kiến thức có nguồn. Phần tổng hợp đặt hai loại kết quả cạnh thông tin ca khám để chuẩn bị một **bản nháp hội chẩn có thể kiểm tra lại**.

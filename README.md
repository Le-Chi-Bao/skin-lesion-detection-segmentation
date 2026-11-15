# Hỗ trợ phân tích ảnh soi da và chuẩn bị hội chẩn

Dự án Python dùng Gradio. **Bước 1 chỉ là demo tải và xem ảnh gốc JPG/PNG,
chưa phân tích tổn thương.** Chưa có mô hình, RAG, cơ sở dữ liệu hoặc báo cáo AI.

## Cài đặt trên Windows PowerShell

Yêu cầu Python 3.12. Mở PowerShell ngay tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không cần kích hoạt môi trường ảo hoặc thay đổi Execution Policy.
Gradio tự cài các thư viện phụ thuộc; dự án không xây dựng dịch vụ FastAPI riêng.

## Chạy giao diện

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở http://127.0.0.1:7860. Chọn ảnh JPG/PNG rồi nhấn **Xem ảnh**.
Vùng **Ảnh gốc** hiển thị ảnh đã tải, không có dự đoán hoặc đánh dấu tổn thương.
Nếu chưa chọn ảnh, vùng **Thông báo** nhắc chọn ảnh bằng tiếng Việt.
Ứng dụng chỉ lắng nghe trên localhost và tắt chia sẻ công khai (`share=False`).
Nhấn `Ctrl+C` trong PowerShell để dừng.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm thử tự sinh ảnh hình học tạm thời, xác minh JPG/PNG giữ nguyên kích thước
và pixel khi đọc, trường hợp chưa chọn ảnh, file hỏng và định dạng không hỗ trợ.
Không dùng dữ liệu bệnh nhân.

Kiểm tra trực tiếp trên trình duyệt: nhấn **Xem ảnh** khi chưa chọn file để xem
thông báo; sau đó tải lần lượt một JPG và một PNG bất kỳ không chứa dữ liệu bệnh
nhân, nhấn **Xem ảnh** và đối chiếu ảnh trong vùng hiển thị với ảnh gốc.

## Cấu trúc hiện tại

```text
.
├── app.py
├── teledermoscopy/
│   └── __init__.py
├── tests/
│   └── test_app.py
├── docs/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

`app.py` chỉ phụ trách giao diện và đọc ảnh để hiển thị. `teledermoscopy/`
dành cho xử lý về sau. `docs/` dành cho Define System và System Design;
chưa viết lại hai tài liệu.

Cấu trúc đích trong `teledermoscopy/` sẽ gồm `vision/` (Detection, Segmentation),
`ingestion/` (nạp tài liệu), `rag/` (tra cứu tài liệu) và `reports/` (báo cáo nháp).
**Chưa tạo các module này trong Bước 1.**

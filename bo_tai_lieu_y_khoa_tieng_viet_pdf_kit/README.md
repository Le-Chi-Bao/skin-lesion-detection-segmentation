# Vietnamese Skin / Medical PDF Collection — Download kit

**Mục đích:** thu thập **PDF gốc**, chưa chuyển sang Markdown và chưa đưa vào RAG.

**Tình trạng thực tế của gói này:** có `sources.json`, `sources.csv`, `download_pdfs.ps1`; **không chứa PDF gốc** vì môi trường tạo gói không tải được binary từ website nhà xuất bản. Dữ liệu `downloaded` chỉ được xác nhận sau khi bạn chạy script trên máy mình.

## Dùng ngay trên Windows

1. Giải nén gói vào thư mục muốn lưu PDF.
2. Mở PowerShell ở thư mục vừa giải nén.
3. Chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_pdfs.ps1
```

4. Kiểm tra các PDF tại `pdf_originals/`, kết quả gồm SHA-256 ở `pdf_originals/download_report.csv`.

Script chỉ tải 8 bài có giấy phép Creative Commons được công bố tại trang bài; không tải các nguồn còn thiếu giấy phép rõ ràng. Nếu server từ chối, file lỗi sẽ không được giữ lại, báo cáo ghi `failed`. Không ghi đè bản gốc đã có.

## Giữ cấu trúc cho Markdown sau này

```text
vietnamese_medical_pdf_kit/
  sources.json               # siêu dữ liệu, URL, license, ID cố định
  sources.csv                # dùng để review / citation
  download_pdfs.ps1
  pdf_originals/             # file gốc sau khi script chạy
    *.pdf
    download_report.csv      # SHA-256 của từng file
```

Sau này có thể tạo `markdown/<id>.md` (không sửa / ghi đè PDF) đồng thời giữ `source_url`, `pdf_url`, `license`, `sha256`, `page_number` và chú thích nội dung hình/bảng để truy xuất nguồn.

## Bản quyền và chất lượng

- Creative Commons không đồng nghĩa mọi ảnh/tài liệu bên thứ ba trong PDF đều có cùng giấy phép; phải kiểm tra từng trường hợp.
- **CC BY 4.0** khác **CC BY-SA 4.0**: ghi rõ giấy phép đúng theo từng bài, bảo đảm ghi công và điều kiện ShareAlike khi áp dụng.
- Ba PDF melanoma thuộc nhóm `UNVERIFIED` để đọc/xem hoặc xin phép, không được tải tự động trong script.
- Việc tải PDF thành công chưa chứng minh nội dung chính xác, đầy đủ, có thể dùng làm RAG, hay bản PDF có thể chuyển Markdown mà không mất bảng, hình.
- Một số tài liệu tập trung vào chất lượng cuộc sống hoặc Y học cổ truyền, có tính bổ sung, không thay thế hướng dẫn chuyên khoa ung thư da.

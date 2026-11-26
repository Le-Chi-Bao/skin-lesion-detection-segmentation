# FINAL MVP v1.0

Ngày kiểm thử 26/11/2026. Bản này tích hợp CV và Medical RAG trong cùng Gradio shell, giữ chế độ nghiên cứu và không tự chẩn đoán.

## Trạng thái tính năng

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Faster R-CNN E2 | PASS | Registry active, checkpoint/hash đã xác minh, inference CPU |
| SegFormer MiT-B2 | PASS | Registry active, checkpoint/hash đã xác minh, inference CPU |
| Detection và segmentation trong một phiên | PASS | Upload, phân tích, localization, mask, overlay |
| Medical RAG BM25 | PASS | Local corpus và UI retrieval |
| Dense E5 Small / Hybrid RRF | PASS | ChromaDB hiện có 496 vectors, tái sử dụng theo content hash |
| Citation metadata | PASS | Có document ID, source, PDF path và số trang |
| Citation content verification | BLOCKED | Tám tài liệu vẫn `needs_human_review` |
| RAG Precision@5, Recall@5, MRR@10, nDCG@5 | N/A | 40 câu hỏi đều `pending_review`, chưa có ground truth hợp lệ |
| Cloud generation | BLOCKED | Không có quyền outbound được cấp, retrieval-only vẫn hoạt động |
| Report tab | PASS | Có trong Gradio shell |

## Kiểm thử thực tế

Lệnh đầy đủ:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Kết quả cuối cùng là `497 passed, 13 skipped, 0 failed`. Pytest dùng `runtime/pytest-tmp` qua `pytest.ini`, tránh lỗi quyền Windows trên thư mục temp hệ thống.

Kiểm thử trình duyệt Medical RAG và chuyển lại Computer Vision đạt. CV dùng ảnh mẫu local và checkpoint E2/SegFormer đã đăng ký. Các ca thiếu model, output lỗi, reset, upload và bảo vệ file được kiểm tra trong suite hiện có.

WP6E chạy bằng:

```powershell
.venv\Scripts\python.exe scripts\rag_evaluate.py --repeats 1
```

Kết quả đo CPU sau một warm-up là BM25 255.580 ms, Dense 321.241 ms và Hybrid 287.561 ms trung bình trên 40 câu hỏi. Đây là quan sát trên máy kiểm thử, không phải chỉ tiêu hiệu năng. MRR@10 truy xuất đủ 10 hit. PNG giữ metric thiếu nhãn ở `N/A`, không chuyển thành 0.

## Chạy ứng dụng

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-cv.txt
.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.venv\Scripts\python.exe main.py
```

Mở `http://127.0.0.1:7860`. Chọn Computer Vision để tải ảnh, xem Mask, Overlay và Localization. Chọn Medical RAG để tìm bằng BM25, Dense hoặc Hybrid. Chọn Report để xem trạng thái bàn giao. Chế độ local không gửi corpus lên cloud.

## Giới hạn

Ảnh, mask và bounding box chỉ hỗ trợ quan sát vùng tổn thương. Hệ thống không suy luận loại bệnh, mức độ bệnh hoặc chẩn đoán từ kết quả CV. Chất lượng Markdown và page alignment của corpus chưa được xác minh toàn văn bởi con người. Metric RAG chưa thể kết luận phương pháp nào tốt nhất. Latency CPU phụ thuộc máy và model loading. CUDA chưa được kiểm chứng trong mốc này.

## Future Work

- Hoàn tất human review cho tám tài liệu, xác nhận relevant chunk IDs và chạy lại WP6E.
- Bổ sung tập kiểm thử độc lập có đánh giá kép cho câu hỏi ngoài phạm vi.
- Đo latency/RAM trên cấu hình mục tiêu và ghi baseline cố định.
- Chỉ bật generation hoặc reranking sau khi có phê duyệt xử lý outbound và ngân sách vận hành.

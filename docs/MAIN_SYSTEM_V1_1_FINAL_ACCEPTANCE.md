# Main System v1.1 Acceptance

Ngày kiểm tra: 26/11/2026. Branch: `feat/main-system-v1.1`.

## Kiến trúc

Computer Vision và Medical RAG vẫn là hai module độc lập. CV dùng registry hiện tại với Faster R-CNN E2 và SegFormer MiT-B2. RAG nạp tám tài liệu local đã đồng ý indexing, chia đoạn có giữ ranh giới trang, embedding multilingual E5 Small 384 chiều trên CPU, ChromaDB, BM25, Dense và Hybrid RRF.

RAG có ba chế độ. Retrieve Only hiển thị nguồn và điểm retrieval. Evidence Summary trích nguyên văn các câu có giao từ khóa với câu hỏi, gắn ID E1... và citation từ chính chunk. Summary không gọi LLM và không thêm kết luận ngoài phần trích dẫn. Nếu không có câu nào phù hợp, summary abstain và không trình bày các hit không liên quan. Ask with LLM thực hiện local retrieval rồi báo trạng thái generation. Cấu hình corpus local hiện tại không cho phép outbound generation.

Kết quả dùng Gradio Markdown với `sanitize_html=True`. URL hiển thị chỉ được tạo link khi là HTTPS. Tiêu đề, page, chunk ID, score và trạng thái kiểm duyệt đặt gần từng trích đoạn. Accordion riêng hiển thị văn bản Markdown gốc dưới dạng code để nội dung không bị sửa khi kiểm tra.

## Kiểm kê Môi trường

| Thành phần | Trạng thái quan sát |
| --- | --- |
| Corpus local | 8 tài liệu Markdown và 8 hồ sơ đồng ý indexing |
| ChromaDB | Collection `medical-e5-1365be459f2425d53c18688117dc1538`, 496 vectors trước và sau kiểm thử |
| Embedding | E5 Small local, cấu hình 384 chiều, CPU; Transformers 5.24.1 và PyTorch 2.10.0+cpu |
| Ollama | Binary có trong PATH; `ollama list` không có model và API local chưa chạy |
| CUDA | Không khả dụng trong môi trường hiện tại |
| Detection E2 | `model_store/detection/E2-a2301ed5919a4b9089dc3cc73af65e70.pt`, SHA-256 khớp registry |
| SegFormer | `model_store/segformer/real_only/best.pt`, SHA-256 khớp registry |
| Quyền corpus | Chỉ local indexing; không cấp quyền gửi tài liệu tới OpenRouter hoặc nhà cung cấp khác |

Không tải model hoặc tài liệu y khoa mới. OpenRouter generation hiện có trong code vẫn chịu rights gate hiện hữu và không được dùng với corpus này.

## WP6E Retrieval

Lệnh đã chạy trên corpus và ChromaDB sẵn có:

```powershell
.venv\Scripts\python.exe scripts\rag_evaluate.py --repeats 1
```

Có 40 câu hỏi, 0 câu được xác minh và 40 câu `pending_review`. Vì chưa có ground truth do con người duyệt, Precision@5, Recall@5, MRR@10 và nDCG@5 đều `N/A` cho BM25, Dense và Hybrid. Không suy diễn điểm metric từ các nhãn pending.

| Chế độ | Latency trung bình CPU sau warm-up |
| --- | ---: |
| BM25 | 278.9 ms |
| Dense | 328.1 ms |
| Hybrid | 308.4 ms |

Runner kiểm tra 1.200 citation metadata và cả 1.200 đạt các trường metadata bắt buộc. `content_verified=0`; đây chỉ là kiểm tra metadata, không phải review tính đúng của nội dung hay vị trí trang. Kết quả chi tiết được ghi trong `runtime/rag/wp6e/results.json` và `per_query.csv`.

Live Hybrid smoke với câu hỏi về yếu tố liên quan viêm da cơ địa đã trả về citation `E5` trích đúng câu trong corpus về tiền sử gia đình dị ứng và rối loạn giấc ngủ. Nguồn là bài “Đặc điểm lâm sàng và một số yếu tố liên quan đến bệnh viêm da cơ địa ở sinh viên Trường Đại học Y Dược Thái Bình năm 2024”, PDF trang 1; response hiển thị chunk ID đầy đủ. Tài liệu vẫn `needs_human_review`; smoke này xác nhận đường đi và metadata, không phải xác minh kết luận y khoa.

## Kiểm thử

Baseline trước sửa đổi: `497 passed, 13 skipped` trong 128.29 giây. Regression trên tree cuối: `511 passed, 3 skipped` trong 250.79 giây, chạy với registry E2 và `WP_CV4_REAL=1`. Hai cảnh báo là API Pillow `Image.getdata` trong test CV, không làm test thất bại.

Các kiểm tra mới bao gồm summary chỉ trích đoạn tồn tại trong chunk, citation có document ID khớp, generation bị tắt vẫn giữ kết quả retrieval và summary abstain khi không có câu phù hợp. Browser Chromium với corpus synthetic và browser với corpus local đã kiểm tra. Luồng local thực hiện Hybrid retrieval, Summary, PDF page, chunk ID, trạng thái `needs_human_review`, mở văn bản gốc và câu hỏi ngoài phạm vi.

Sau hiệu chỉnh xếp hạng trích đoạn, toàn bộ RAG test group đạt `313 passed, 2 skipped` trong 89.46 giây. Browser local xác nhận câu về tiền sử dị ứng gia đình và rối loạn giấc ngủ được trích với citation, đồng thời kiểm tra bảng HTML trong chunk được render thành bảng.

| Kiểm tra cuối | Kết quả |
| --- | --- |
| Regression pytest đầy đủ trên Windows, CV registry E1/E2/E3 thật bật | PASS, 511 passed, 3 skipped |
| Browser Medical RAG synthetic | PASS, kiểm tra rendering và điều hướng sang Computer Vision |
| Browser local RAG | PASS, Hybrid, Evidence Summary, citations, original text và abstention |
| CV upload, inference, mask, overlay, localization, PNG export và session isolation qua Chromium | PASS trong `test_real_dual_model_browser_upload_export_reset`, với registry E2 hiện hành |
| Ground truth retrieval metrics | N/A, chưa được con người duyệt |
| Natural-language LLM generation trên corpus | BLOCKED theo quyền xử lý dữ liệu và chưa có local model |

Các skip cuối gồm thiếu checkpoint pilot E1 step 7200 (inactive), Windows symlink privilege unavailable và multilingual-E5-base chưa được xác minh cục bộ. Suite RAG có test adapter generation giả lập để kiểm tra schema, prompt-injection handling, trích dẫn giả và quyền outbound. Đây không phải lần chạy model LLM thật. Test doubles không được tính là browser hoặc model acceptance.

## Các Tệp Thay Đổi

- `app/rag/evidence_summary.py` thêm trích đoạn theo từ khóa có ID evidence.
- `app/rag/service.py` bổ sung chế độ summary và giữ hit đã truy xuất khi generation bị chặn.
- `app/ui/rag_page.py` đổi sang giao diện Markdown đã sanitize, ba mode, metadata nguồn, trạng thái và accordion văn bản gốc.
- `tests/test_rag_service.py` bổ sung test citation, generation blocked và abstention.
- `tests/test_rag_browser.py` cập nhật smoke test UI và thêm kiểm tra browser với corpus local khi tài nguyên có sẵn.
- `README.md` thêm hướng dẫn RAG, quyền dữ liệu và đánh giá.

## Chạy Demo Windows

```powershell
$env:RAG_CONFIG = (Resolve-Path 'config/rag.local.json').Path
.venv\Scripts\python.exe main.py
```

Mở `http://127.0.0.1:7860`, chọn **Medical RAG**, nhập câu hỏi và chọn Hybrid. Chọn Evidence Summary để xem trích đoạn có citation. Chọn Retrieve Only để duyệt kết quả truy xuất. Mở accordion để kiểm tra Markdown gốc. Cấu hình mặc định vẫn là retrieval-only. Xem thêm [README](../README.md), [demo runbook](DEMO_RUNBOOK.md) và [WP6E](WP6E_RAG_EVALUATION.md).

## Giới Hạn và Future Work

- Tám tài liệu đều đang `needs_human_review`. Cần người có chuyên môn kiểm chứng nội dung, trang và relevance trước khi tuyên bố evidence đã được xác thực.
- Evidence Summary là extractive, không phải câu trả lời tự nhiên tổng hợp. Các câu trích nguyên văn có thể thiếu ngữ cảnh nếu bản Markdown chuyển đổi không hoàn chỉnh.
- Ollama chưa có model local. Việc tải hoặc cài model cần được operator quyết định riêng; không tự tải trong acceptance này.
- Không bật cloud LLM hoặc reranker cho corpus local. Rights gate chỉ có thể thay đổi theo quyết định có thẩm quyền và phạm vi xử lý cụ thể.
- Cần human label WP6E rồi mới báo metric retrieval. Kết quả hiện tại chỉ cung cấp latency và consistency của metadata.
- CV browser acceptance chạy trong regression cuối với Faster R-CNN E2 và SegFormer registry hiện hành. Bản acceptance lịch sử ngày 25/11 dùng cấu hình tại thời điểm đó và được ghi riêng trong [DEMO_RELEASE_ACCEPTANCE](DEMO_RELEASE_ACCEPTANCE.md).
- CUDA, tải thực tế nhiều người dùng, quan trắc memory dài hạn và kiểm duyệt chất lượng Markdown toàn corpus chưa được xác minh ở mốc này.

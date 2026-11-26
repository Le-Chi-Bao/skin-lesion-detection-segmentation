# RAGAS Medical RAG Evaluation Report

Run time: 2026-11-26T11:31:03.940773+00:00

## 1. Mục tiêu đánh giá
Đo retrieval BM25, Dense multilingual E5 Small và Hybrid RRF trên cùng bộ 40 câu WP6E; kiểm tra citation metadata và Evidence Summary extractive. Không chạy LLM generation.

## 2. Dataset và tính hợp lệ của nhãn
WP6E có 40 câu. 0 verified, 40 pending_review. Không dùng nhãn pending_review làm ground truth. File review mẫu có 15 câu gồm câu được đánh dấu ngoài phạm vi để người đánh giá xác minh.

## 3. Phương pháp thực nghiệm
Corpus cố định ở 8 tài liệu và 496 chunks. Top-K=10, cùng embedding/chunking/corpus, 3 lượt đo sau một warm-up Dense không tính thời gian. Thiết bị CPU; model intfloat/multilingual-e5-small (384 chiều). RAGAS 0.4.4. E5 fingerprint 1365be459f2425d53c18688117dc1538a6d5ba93babac1c3de42c8fbdade2196. Trong lượt chỉ đọc này, hash trước-sau bằng nhau: True. So với hash pre-run đã lưu, baseline bit-for-bit được phục hồi: False.

## 4. Các metric và cách tính
Precision@5, Recall@5, MRR@10 và nDCG@5 dùng relevant_chunk_ids chỉ khi câu đã verified. Custom RAGAS Context Precision là average precision theo các relevant chunk ở vị trí xếp hạng; Context Recall là tỷ lệ relevant chunk IDs được truy xuất. Cả hai chỉ chạy khi có nhãn người duyệt. Citation validity kiểm tra document ID khớp, trang PDF dương, đường dẫn tồn tại, title và source locator. Evidence span validity kiểm tra trích đoạn nguyên văn có trong chunk được dẫn. Evidence coverage là proxy từ vựng, tính tỷ lệ token nội dung câu hỏi xuất hiện trong excerpt. Unsupported excerpt rate chỉ phát hiện chuỗi không khớp nguyên văn, không khẳng định tính đúng y khoa.

## 5. Kết quả đo thực tế
| Mode | P@5 | R@5 | MRR@10 | nDCG@5 | Latency mean (ms) | Citation hợp lệ |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | N/A | N/A | N/A | N/A | 300.64 | 1.0000 (400/400) |
| Dense | N/A | N/A | N/A | N/A | 305.17 | 1.0000 (400/400) |
| Hybrid | N/A | N/A | N/A | N/A | 307.06 | 1.0000 (400/400) |

Evidence Summary extractive, các chỉ số quan sát:
| Mode | Lexical coverage | Exact span | Citation consistency | Abstention trên 2 câu gắn cờ ngoài phạm vi* |
|---|---:|---:|---:|---:|
| BM25 | 0.8386 | 1.0000 | 1.0000 | 0.0000 |
| Dense | 0.8071 | 1.0000 | 1.0000 | 0.0000 |
| Hybrid | 0.8083 | 1.0000 | 1.0000 | 0.0000 |

## 6. So sánh ba phương pháp
Không xếp hạng chất lượng retrieval vì chưa có câu hỏi verified. Latency và citation metadata là số đo vận hành, không chứng minh relevance hoặc độ chính xác lâm sàng.
Hai câu được gắn cờ ngoài phạm vi trong dataset vẫn pending_review; abstention rate này chỉ là quan sát mô tả, không phải kết quả scope đã xác minh.

## 7. Phân tích lỗi
Có 0 citation metadata không hợp lệ trong 1200 hit. Có 0 excerpt không khớp nguyên văn nguồn trong 600 excerpt. Với 2 câu đang được gắn cờ ngoài phạm vi, 2 câu trả excerpt; tỷ lệ abstention quan sát là 0.00. Hai cờ scope vẫn pending_review. Kiểm tra này không đánh giá ngữ cảnh lâm sàng.

## 8. Giới hạn đánh giá tự động
40/40 nhãn hiện pending_review. Chưa có reference answer hoặc relevant chunk IDs được duyệt. Không chạy LLM judge; OpenRouter không được gọi trong lượt này. RAGAS Faithfulness, Answer Relevancy và unsupported claim rate được ghi N/A. Context Precision/Recall chỉ dùng custom RAGAS metric khi có chunk IDs đã duyệt. Evidence Summary là trích xuất theo từ khóa, không phải generation; lexical coverage và exact-span validation không thay thế review người. Lượt chạy đầu đã upsert lại Chroma và làm đổi hash do sai khác số thực ở bước chuẩn hóa. Event log cũ không giữ đủ 496 vector để xác nhận phục hồi bit-for-bit; hash baseline ban đầu không được tuyên bố là đã khôi phục. Các lượt evaluation sau dùng attach vector read-only.

## 9. Kết luận
Pipeline chạy được trên corpus local và đo latency/citation metadata. Chưa đủ bằng chứng để kết luận BM25, Dense hoặc Hybrid tốt hơn. Cần người đánh giá hoàn thành mẫu nhãn, sau đó chạy lại cùng pipeline.

## Tóm tắt cho slide bảo vệ
- Đã chạy 40 câu x 3 mode trên 496 chunks; citation metadata hợp lệ 1200/1200.
- Latency trung bình BM25/Dense/Hybrid lần lượt là 300.64/305.17/307.06 ms; retrieval quality metrics N/A do chưa có nhãn verified.

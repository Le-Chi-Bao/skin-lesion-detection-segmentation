# WP6A.2 — Real E5 & Hybrid Retrieval Acceptance

Ngày kiểm chứng: **2026-11-26 (Asia/Bangkok)**. Baseline:
`2d16b8d1823cdbc28d099072b8f0efd3c44f67a7`. Nhánh:
`feat/rag-real-e5-verification`.

**Trạng thái: PARTIAL / BLOCKED — chưa có pretrained E5 weights local và chưa nhận
được chấp thuận tải weights của người vận hành. Dense/Hybrid thật chưa được nghiệm thu.**

Đã gửi yêu cầu cho phép tải riêng `intfloat/multilingual-e5-base` từ Hugging Face
chính thức, khoảng 1,11 GB weights; không tải dataset. Không suy diễn yêu cầu thực thi
thành chấp thuận điều kiện tải mà người dùng đã đặt riêng. Không tải weights trong lần
kiểm chứng này. Đây là blocker về local assets/quyền tải, không phải kết luận mạng hỏng
hoặc thuật toán E5 sai. WP6B chưa đủ điều kiện bắt đầu dựa trên nghiệm thu real E5.

## Preflight và bảo toàn tài sản

- Đọc `docs/WP6A_1_VERIFICATION.md`, toàn bộ module `app/rag/`, CLI, Gradio RAG,
  `config/rag.example.json` và các test liên quan.
- Working tree ban đầu: notebook
  `finetune_detection/notebooks/Faster_RCNN_ISIC2018_Colab.ipynb` đã sửa;
  `finetune_detection/FasterRCNN_ISIC2018/` chưa được theo dõi. Giữ nguyên, không stage.
  File `.env` xuất hiện thêm trong lúc kiểm thử; không đọc nội dung, không sửa hoặc stage.
- Chụp SHA-256 và đối chiếu lại **84 files**: notebook, các `.pt` trong `model_store/`,
  các file trong thư mục Faster R-CNN chưa commit và hai JSON cấu hình. **Không đổi**.
- `config/models.json` giữ SHA-256
  `07ee693b8a01b043dc36a67f2c336092cde1b670454ff55951773334f5f35350`.
  Không thay checkpoint SegFormer/E1/E2/E3; E2 vẫn là detector active.
- Các Main System tại `127.0.0.1:7860` (PID 34584) và `:7861` (PID 27432) tiếp tục
  chạy. Không dừng hoặc dùng runtime của hai tiến trình này cho test server.
- Bằng chứng riêng ở `runtime/wp6a2-20261126-102120/` (gitignored).
  Không commit weights, runtime artifacts hay secrets. Không push/merge.

## Môi trường và model

Windows, Python **3.12.5**, interpreter `.venv/Scripts/python.exe` của repository.
CPU **Intel Core i5-12500H**, 12 cores / 16 logical processors, RAM khoảng 16 GB.
Không cài hoặc nâng cấp dependencies trong WP này. `pip check`: **PASS**,
`No broken requirements found`.

| Dependency | Phiên bản thực tế |
| --- | --- |
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| transformers | 5.24.1 |
| sentencepiece | 0.2.3 |
| numpy | 2.5.5 |
| huggingface-hub | 2.4.0 |
| safetensors | 0.8.0 |
| gradio / gradio-client | 6.32.0 / 2.7.2 |
| playwright | 1.65.0; Chromium có sẵn |
| pytest | 9.2.0 |
| medical-data-crawler | 0.1.0, pin commit trong `requirements-rag.txt` |

Model ID: **`intfloat/multilingual-e5-base`**. Cache mặc định:
`C:\Users\ADMIN\.cache\huggingface\hub`. Thư mục dự kiến
`models--intfloat--multilingual-e5-base` **không tồn tại** khi kiểm chứng.

Chỉ đọc metadata từ Hugging Face chính thức: revision hiện tại
`d128750597153bb5987e10b1c3493a34e5a4502a`, `model.safetensors` có
**1.112.201.288 bytes**; chưa tải file này. Recipe 768 chiều, prefix, masked mean
pooling và L2 normalization đối chiếu với
[model card chính thức](https://huggingface.co/intfloat/multilingual-e5-base).
Metadata/model card không phải bằng chứng đã load pretrained weights.

Tất cả lệnh kiểm chứng embedding/retrieval sử dụng `allow_download=false` và
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Không cần OpenRouter API key.

## Real E5 gate và retrieval

Gate chạy đúng test yêu cầu, với **`RAG_REQUIRE_REAL_E5=1`**:

```text
tests/test_rag_retrieval.py::test_real_e5_local_smoke
BLOCKED: real multilingual-e5-base NOT VERIFIED: embedding_model_unavailable_local
1 failed in 9.27s
exit 1
```

**Không SKIP, không PASS.** Không sử dụng FakeEmbedder để thay cho gate.
Do dừng ở model load, chưa xác minh bằng pretrained weights các điều kiện: query/passages
768 chiều, prefix thực tế, masked mean/L2, cosine, NaN/Inf, no-silent-truncation,
CPU stability hoặc thứ tự/score Dense/Hybrid. Unit tests hiện có kiểm tra các cơ chế
này bằng dữ liệu kiểm soát; kết quả đó không thay thế real-model acceptance.

Corpus duy nhất là ba tài liệu **SYNTHETIC DEMO** có sẵn, tự tạo từ repository,
đi qua importer, provenance/integrity verifier và `SyntheticRights`.

| Mode / CLI (`--demo --top-k 3`) | Kết quả thực tế |
| --- | --- |
| BM25 | **PASS**, exit 0, `OK`, synthetic=true, một hit cho query bên dưới. |
| Dense | **BLOCKED**, exit 2, `embedding_model_unavailable_local`, hits rỗng. |
| Hybrid | **BLOCKED**, exit 2, cùng lỗi, hits rỗng; không âm thầm fallback BM25. |

Query: `skin lesion color border size`. BM25 Top-1:

- Tiêu đề: `SYNTHETIC DEMO — Quan sát tổn thương da / Skin observation`.
- Score: **4.508681412363994** (BM25 raw score).
- Locator: `synthetic://wp6a/skin-observation`.
- Document: `doc:893d91dd76fec5b8956982756dc45e1434e1c26a5f807c3cb2e9f68b253fc11b`.
- Chunk: `chunk:e1dd8fb185d5d119070c04f465af0572b029bf2361cd65b45503163cb8eb9da5`.
- Citation giữ document ID, hashes, source locator, offsets, provenance và rights;
  Gradio đối chiếu cùng chunk/locator/score đã làm tròn 6 số thập phân với CLI.

Top-K là giới hạn trên; BM25 chỉ trả hit có lexical score > 0, nên query này không
bắt buộc có ba hit. Query `SYNTHETIC DEMO` khớp cả corpus, dùng để kiểm tra Top-K=2
trong trình duyệt. Không có bảng đối chiếu ranking/score Dense/Hybrid thật vì chưa có
weights. Không diễn giải synthetic results thành chất lượng truy xuất y khoa.

## JSON và rights gate

Không đổi `app/rag/config.py` hay `config/rag.example.json`, không thêm YAML hoặc
nguồn cấu hình runtime khác. Đã đọc luồng truyền các settings:

- `embedding` truyền nguyên typed config vào `LocalE5Embedder`; lazy-load, offline,
  device, prefixes, max_length và batch_size đi vào tokenizer/model đúng vị trí.
- BM25 `k1/b` đi vào `RAGService.bm25`; `rrf_k/candidates` đi vào Hybrid/RRF.
- Chunking đi vào `RecursiveChunker`; retrieval mode/Top-K mặc định đi vào CLI/UI.
- Các test cấu hình kiểm tra cả giá trị khác mặc định, binding paths và CLI override.
- **Chunking `max_tokens`/`overlap` đang đếm UTF-8 bytes (`utf8-bytes-v1`), không
  phải E5 tokenizer tokens.** `embedding.max_length` mới là tokenizer tokens gồm prefix
  và special tokens. Passage vượt giới hạn báo `passage_exceeds_model_tokens_rechunk_required`;
  kiểm chứng với tokenizer thật vẫn BLOCKED.

RAG regression đạt các ca ingestion bị từ chối, hash/provenance bị thay đổi,
registry/approval bị thu hồi, cập nhật citation rights và invalidation các indexes.
Service refresh quyền trước query và kiểm tra live rights lần nữa trước trả hits.
Không thay hay bỏ qua rights gate. Revocation trong lúc pretrained E5 chạy chưa
được thực thi; harness local đã chuẩn bị, chưa tính là PASS.

## Gradio và Computer Vision

Khởi động `main.py` riêng ở **`http://127.0.0.1:7862`**, với
`RAG_SYNTHETIC_DEMO=1`, CPU, HF offline. Lần kiểm tra hoàn tất dùng runtime riêng
`runtime/wp6a2-20261126-102120/gradio-runtime-blocked-102809/`, synthetic temp riêng;
đã dừng đúng test server khi xong.

- BM25: **PASS**, `OK`, SYNTHETIC DEMO; card có chunk ID/locator/score khớp CLI.
- Dense và Hybrid: đã thao tác cả hai trong Chromium; báo rõ
  `embedding_model_unavailable_local`, không trả card. **BLOCKED**, không ghi PASS.
- Top-K=2 với query synthetic chung: **PASS**.
- Chuyển Computer Vision → Medical RAG → Computer Vision → Medical RAG: **PASS**;
  viewer/Original tab CV hiện đúng, kết quả RAG giữ trong phiên.
- Browser context mới không có query hoặc cards của phiên trước: **PASS**.
- Startup của test server: SegFormer và Faster R-CNN **E2 đều READY**.
- Cấu hình registry gốc không thay đổi; 7860/7861 vẫn lắng nghe sau kiểm thử.

Hai lần thử harness UI ban đầu lỗi do giả định sai trang mặc định và yêu cầu BM25
có đủ ba hit. Đã sửa harness local theo hành vi thực tế, không sửa sản phẩm.

## Regression thực chạy

| Nhóm | Kết quả |
| --- | --- |
| Baseline toàn suite, loại real_e5 | **363 passed, 12 skipped, 1 deselected**, 80,12 s; exit 0. |
| Strict real E5 offline | **1 failed**, 9,27 s; exit 1, thiếu model. |
| RAG retrieval/config/ingestion/chunking/service/UI/browser + startup/config/CV schemas sau sửa test isolation | **217 passed, 1 skipped, 1 deselected**, 42,89 s; exit 0. |
| CV thật: `test_cv_mvp`, WP2, WP3, E123, bật đầy đủ local assets | **60 passed, 1 failed, 1 skipped**, 164,55 s. |
| Chạy lại đúng ca CV lỗi `test_real_dual_model_browser_upload_export_reset[module]` | **1 passed**, 35,59 s. |

Ca CV lỗi lần đầu timeout Playwright khi click tab Original; startup đã READY cả hai
model. Chạy lại riêng đạt, không sửa code CV hoặc nới timeout. Các ca CV còn lại gồm
SegFormer + E2 coexist/export/browser, độc lập E1/E2/E3 CPU inference và rollback E2→E1
trên bản sao registry đã PASS. Một warning Pillow `getdata()` deprecated, chưa ảnh hưởng kết quả.

12 skip baseline: 6 ca CV cần environment, 4 ca E123/rollback cần opt-in,
1 pilot E1 thiếu checkpoint và 1 Windows symlink privilege. Lần CV thật đã chạy thêm
10 ca opt-in; chỉ pilot E1 còn skip. RAG regression chỉ còn skip symlink privilege.
Không cộng số lượng các nhóm chồng lặp thành một tổng suite giả.

**Sửa duy nhất trong test harness:** `tests/test_rag_browser.py` vốn ghi screenshot
vào `runtime/wp6a-rag-browser.png` cố định. Đổi sang `tmp_path/rag-browser.png` để các
lần test sau không ghi đè artifact runtime chung. Browser regression chạy đạt và ảnh
đã được tạo bên trong basetemp riêng. Baseline ban đầu đã chạy đường dẫn screenshot
cũ; không có bản snapshot ảnh trước lần chạy để khẳng định artifact cũ được bảo toàn.
Không tìm thấy lỗi thực tế cần đổi production RAG/CV code.

## CPU performance và giới hạn

BM25 CLI một lần: **37,04 ms**. Mười lần gọi service trên corpus ba chunks, gồm
integrity/rights refresh: median **43,43 ms**, min **36,78 ms**, max **50,25 ms**.
Đây là phép đo nhỏ trên máy đang có hai Main System hoạt động, không phải benchmark.
Dense/Hybrid CLI mất 7.195,25 / 6.086,21 ms trước báo thiếu model; đây là thời gian
thất bại/import dependencies, **không phải E5 inference performance**. Chưa đo được
thời gian load model, embedding throughput hoặc latency Dense/Hybrid thành công.

## Khắc phục và tái kiểm chứng

1. Người vận hành chấp thuận tải pretrained weights; hoặc tự cung cấp cache Hugging
   Face đầy đủ của model chính thức. Không cần API key trả phí hoặc dữ liệu y khoa.
2. Sau khi có chấp thuận, chạy provisioning chọn đúng các file cần cho AutoModel và
   fast tokenizer; không tải PyTorch/ONNX/OpenVINO duplicates, dataset hay remote code:

```powershell
# CHỈ chạy sau khi người vận hành đã chấp thuận tải weights.
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('intfloat/multilingual-e5-base', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json']))"
```

`snapshot_download` tái sử dụng file cache đã có; không force download. Ghi lại snapshot
revision thực tải, hash weights và dung lượng. Không commit model/cache. Giữ JSON
`allow_download=false` và chạy lại offline:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:RAG_REQUIRE_REAL_E5 = '1'
$env:PYTHONUTF8 = '1'
$run = 'runtime/wp6a2-rerun-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.venv/Scripts/python.exe -m pytest tests/test_rag_retrieval.py::test_real_e5_local_smoke -q -ra --basetemp "$run/real-e5"
.venv/Scripts/python.exe -m app.rag --demo --mode Dense --top-k 3 --query 'skin lesion color border size'
.venv/Scripts/python.exe -m app.rag --demo --mode Hybrid --top-k 3 --query 'skin lesion color border size'
```

Sau gate phải chạy lại đối chiếu ranking/cosine/RRF, citation/identity/quyền, pooling,
no-truncation, cấu hình JSON khác mặc định với E5 thật, đo CPU và Gradio cả ba mode
trong runtime/port riêng. Harness local đã chuẩn bị tại `runtime/wp6a2-20261126-102120/acceptance.py`
và `gradio_acceptance.py`; chưa chạy đường đi E5 thành công. Chúng là artifact local,
không phải module cấu hình hay code runtime được commit.

Chỉ nâng trạng thái WP6A.2 sau khi Dense và Hybrid thực sự PASS với pretrained weights.
Cross-Encoder, OpenRouter, RAGAS và dữ liệu y khoa thật nằm ngoài WP này.

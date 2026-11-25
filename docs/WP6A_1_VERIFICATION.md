# WP6A.1 — Modular configuration & real E5 verification

Ngày kiểm chứng: **2026-11-26 (Asia/Bangkok)**. Base: `b8600b4a15bdbbab2a2f44ab2c6cdea52396e9d4`,
đã fetch và đối chiếu `origin/main` trước khi tạo nhánh `feat/rag-wp6a1`.

**Real E5: BLOCKED — chưa có snapshot/weights `intfloat/multilingual-e5-base` local.**
Đã cài và kiểm tra dependencies Dense. Không tải weights trong lần thực hiện này vì chưa
có cho phép rõ ràng của người vận hành. Không tuyên bố Dense/Hybrid real-model PASS từ
fake embeddings hoặc từ việc các unit test đạt.

## Kiến trúc và phạm vi

| Module | Trách nhiệm |
| --- | --- |
| `app/rag/config.py` | Frozen dataclasses, mặc định, validation, đọc JSON, resolve binding paths. Không import model/backend. |
| `app/rag/importer.py` | Ingestion qua Crawler verifier, schema, hash, provenance, original và store bindings. |
| `app/rag/rights.py` | Quyền dữ liệu hiện tại: reviewed registry, approvals theo record hash, expiry/revocation. |
| `app/rag/chunking.py` | Recursive splitting theo block/heading, budget, overlap, offset và citation. |
| `app/rag/retrieval.py` | BM25, lazy local E5, masked mean pooling, L2 normalization, cosine và RRF. |
| `app/rag/service.py` | Corpus/index theo phiên, refresh integrity/quyền trước query, invalidation và response. |
| `app/rag/runtime.py` | Nối typed settings với importer, rights, chunker và service. |
| `app/rag/models.py`, `synthetic.py` | Contracts chung và corpus SYNTHETIC DEMO tự viết cố định. |
| `app/rag/__main__.py`, `app/ui/rag_page.py` | CLI và Gradio: áp dụng retrieval defaults, cho phép chọn mode/Top-K. |

Giữ các ranh giới module sẵn có; chỉ thêm một module cấu hình, không tách retrieval thành
nhiều file. Không module RAG nào cần import pipeline CV. Không triển khai Cross-Encoder,
LLM, RAGAS hay WP6B.

`RAGService.search(query, mode="Hybrid", top_k=5) -> SearchResponse` giữ nguyên contract,
default và cấu trúc citation. Constructor chỉ bổ sung hai keyword tùy chọn `bm25_k1`,
`bm25_b`. CLI/UI đọc `retrieval.mode`, `retrieval.top_k` từ JSON rồi truyền vào search;
caller Python trực tiếp vẫn truyền mode/Top-K như trước. Runtime giữ typed snapshot tại
`service.settings`; đây không phải một nguồn cấu hình thứ hai.

## JSON configuration

Nguồn mẫu duy nhất: [`config/rag.example.json`](../config/rag.example.json). Không thêm YAML,
API key hoặc secret. Khóa `max_tokens`, `rrf_k`, `candidates` và năm binding paths giữ tên
cũ; các section mới có mặc định nên JSON WP6A cũ hợp lệ vẫn dùng được. Các import cũ
`app.rag.chunking.ChunkConfig` và `app.rag.retrieval.EmbeddingConfig` vẫn hoạt động.

```json
{
  "manifest": "../runtime/crawler-batches/REPLACE_WITH_BATCH_ID/manifest.json",
  "artifacts": "../runtime/crawler-artifacts",
  "provenance": "../runtime/crawler-provenance",
  "registry": "../runtime/reviewed-source-registry.json",
  "approvals": "../runtime/reviewed-indexing-approvals.json",
  "chunking": {
    "strategy": "recursive-blocks",
    "max_tokens": 384,
    "overlap": 48,
    "token_metric": "utf8-bytes-v1",
    "version": "recursive-blocks-1"
  },
  "bm25": {"k1": 1.5, "b": 0.75},
  "embedding": {
    "model": "intfloat/multilingual-e5-base",
    "allow_download": false,
    "device": "cpu",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "max_length": 512,
    "batch_size": 16
  },
  "retrieval": {"mode": "BM25", "top_k": 5},
  "rrf_k": 60,
  "candidates": 30
}
```

| Tham số | Mặc định | Validation/ý nghĩa |
| --- | --- | --- |
| `manifest`, `artifacts`, `provenance`, `registry`, `approvals` | Không có | Chuỗi path không rỗng; production cần đủ cả năm. Relative paths tính từ thư mục JSON; absolute paths và `~` được hỗ trợ. Integrity/quyền tiếp tục do importer/rights kiểm tra. |
| `chunking.strategy` | `recursive-blocks` | Chỉ hỗ trợ strategy hiện có. |
| `chunking.max_tokens` | `384` | Integer ≥ 4, **UTF-8 bytes**, giữ tên legacy. |
| `chunking.overlap` | `48` | Integer, 0 ≤ overlap < max_tokens, **UTF-8 bytes**. |
| `chunking.token_metric` | `utf8-bytes-v1` | Chỉ hỗ trợ metric hiện có. |
| `chunking.version` | `recursive-blocks-1` | Chỉ hỗ trợ version hiện có. |
| `bm25.k1` | `1.5` | Số hữu hạn > 0. |
| `bm25.b` | `0.75` | Số hữu hạn trong [0, 1]. |
| `embedding.model` | `intfloat/multilingual-e5-base` | Chuỗi model ID/local directory không rỗng, dùng cùng pooling recipe E5. |
| `embedding.allow_download` | `false` | JSON boolean; chỉ đặt `true` sau khi operator cho phép tải weights rõ ràng. |
| `embedding.device` | `cpu` | `cpu` hoặc `cuda`; CUDA cần thực sự khả dụng khi load. |
| `embedding.query_prefix` | `query: ` | Chuỗi, giữ dấu cách cuối; được phép rỗng cho model tương thích khác. |
| `embedding.passage_prefix` | `passage: ` | Chuỗi, giữ dấu cách cuối. |
| `embedding.batch_size` | `16` | Integer ≥ 1. |
| `embedding.max_length` | `512` | Integer trong [8, 512], **tokenizer tokens**, gồm prefix và special tokens. |
| `retrieval.mode` | `BM25` | `BM25`, `Dense`, `Hybrid`; default của CLI/UI. |
| `retrieval.top_k` | `5` | Integer trong [1, 100]; default của CLI/UI. |
| `rrf_k` | `60` | Số hữu hạn > 0; hằng số làm mượt fusion. |
| `candidates` | `30` | Integer ≥ 1; mỗi ranking lấy `max(top_k, candidates)`. |

**Chunk budget hiện tại được đo bằng UTF-8 bytes, không phải số từ hoặc tokenizer tokens.**
Ví dụ `len("da".encode("utf-8")) == 2`, `len("đỏ".encode("utf-8")) == 5`.
Overlap ưu tiên các fragment cấu trúc hoàn chỉnh nên có thể nhỏ hơn budget overlap hoặc
bằng 0 khi fragment không vừa. Dense kiểm tra độ dài bằng tokenizer thật; passage vượt
`embedding.max_length` báo `passage_exceeds_model_tokens_rechunk_required`, không truncate
âm thầm. Query được phép truncate theo max_length. Giảm chunk budget nếu cần.

Không thêm `strategy` vào payload băm/citation chunk vì strategy đã được biểu diễn bởi
`version`: giữ nguyên default chunk IDs/citations WP6A. Đã đối chiếu cả ba default chunks
của corpus tổng hợp với implementation tại `b8600b4`, bằng nhau hoàn toàn.

Validation từ chối object/section sai kiểu, unknown key, duplicate JSON key (kể cả nested),
NaN/Infinity, numeric string, boolean thay số và path rỗng/null. Không tự ép kiểu hay bỏ
qua khóa sai chính tả. Chỉ parse settings có thể bỏ cả năm paths; `create_service(config_path=...)`
luôn yêu cầu đủ paths trước ingestion. Demo và production config tiếp tục loại trừ nhau.

### Cách dùng

Sao chép mẫu thành JSON riêng và thay đủ binding paths bằng dữ liệu/quyền đã review.

```powershell
# CLI: --mode/--top-k có ưu tiên hơn retrieval defaults trong JSON.
.venv\Scripts\python.exe -m app.rag --config config/rag.local.json --query 'tổn thương da'
.venv\Scripts\python.exe -m app.rag --config config/rag.local.json --mode BM25 --top-k 3 --query 'skin'

# UI: đọc cùng JSON; mode và Top-K ban đầu lấy từ retrieval defaults.
Remove-Item Env:RAG_SYNTHETIC_DEMO -ErrorAction SilentlyContinue
$env:RAG_CONFIG = (Resolve-Path config/rag.local.json).Path
.venv\Scripts\python.exe main.py

# Demo không cần production config hay model E5 cho BM25.
.venv\Scripts\python.exe -m app.rag --demo --mode BM25 --query 'tổn thương da' --top-k 3
```

```python
from pathlib import Path
from app.rag.config import RAGConfig
from app.rag.runtime import create_service

path = Path("config/rag.local.json")
settings = RAGConfig.load(path)
service = create_service(config_path=path)
try:
    result = service.search("skin", settings.retrieval.mode, settings.retrieval.top_k)
finally:
    service.close()
```

UI chỉ cho chọn retrieval mode/Top-K, không cho sửa paths hoặc quyền. JSON lỗi không làm
trang CV mất khả dụng; thao tác search RAG tiếp tục báo lỗi cấu hình an toàn.

## RRF review

Mỗi ranking được loại duplicate theo `chunk_id` trước khi đếm rank, bắt đầu từ 1.
Mỗi chunk nhận tổng `1 / (k + rank)` từ các ranking có chứa nó. Không cộng duplicate
trong cùng ranking; có cộng đóng góp của cùng chunk ở hai ranking. Raw BM25/cosine score
không đi vào công thức. Tie sắp theo `chunk_id` để ổn định.

Đã bổ sung unequal-length rankings có duplicate và ranking rỗng với `k=1, 60, 80.5`,
cross-ranking contribution, Top-K và tie khi đảo thứ tự ranking. Các test xác nhận logic
fusion hiện tại đúng nên **không đổi thuật toán RRF**; constructor chỉ siết kiểu tham số.

## Kiểm chứng thực tế

Môi trường: Windows, Python 3.12, `.venv` của repository. Crawler được cài đúng commit
`1bcfb69ef8dc6991e206c5874b02779b6297204f` từ `requirements-rag.txt`.

| Dependency Dense sau khi cài requirements | Phiên bản |
| --- | --- |
| torch | 2.10.0+cpu |
| transformers | 5.24.1 |
| sentencepiece | 0.2.3 |
| numpy | 2.5.5 |
| pytest | 9.2.0 |

Các version này thỏa `requirements-rag-dense.txt`; không đổi requirements hoặc model
weights/configuration CV. `pip check`: **No broken requirements found**.

| Kiểm tra | Kết quả thực tế |
| --- | --- |
| Baseline trên main, `.venv`, basetemp riêng | **252 passed, 9 skipped**, 111.74 s. |
| Toàn bộ tests sau thay đổi | **349 passed, 9 skipped**, 85.05 s; exit 0. |
| RAG regression riêng (không gồm gate real E5) | **176 passed, 1 skipped, 181 deselected**, 41.80 s; exit 0. |
| CV regression với checkpoint thật sẵn có | **43 passed, 1 skipped**, 122.31 s; exit 0. |
| Strict real E5 smoke offline | **BLOCKED**, 1 failed, 4.34 s; exit 1, `embedding_model_unavailable_local`. |
| BM25 demo CLI | **OK**, exit 0, SYNTHETIC DEMO và citation đầy đủ. |
| Backward chunk compatibility so với `b8600b4` | Ba chunks mặc định có ID, nội dung và citation bằng nhau hoàn toàn. |
| Dependency consistency | `pip check` PASS, không có requirements bị hỏng. |

Full suite gồm các test hiện tại, RAG regression, Gradio isolation và browser Chromium.
Chín skip: sáu ca CV cần biến môi trường, một checkpoint pilot E1 vắng mặt, một quyền
symlink Windows không khả dụng, một real E5 thiếu local weights. Lần chạy CV riêng đặt
các biến môi trường và thực thi thêm sáu ca đó; chỉ còn ca pilot
`E1_step07200_PILOT_UNFINISHED.pt` thiếu đúng SHA-256 theo yêu cầu test.

Lần thăm dò đầu bằng `python` hệ thống và temp mặc định thất bại (2 failed, 166 errors),
gồm lỗi môi trường/temp Windows. Kết quả baseline và nghiệm thu phía trên dùng đúng
`.venv` của dự án và basetemp riêng trong `runtime/`, như hướng dẫn WP6A hiện có.
Không sửa test CV hoặc quyền thư mục người dùng để làm suite đạt.

Logs/artifacts local (được gitignore): `runtime/wp6a1-regression.txt`,
`runtime/wp6a1-rag-regression.txt`, `runtime/wp6a1-real-cv.txt`,
`runtime/wp6a1-real-e5.txt`, `runtime/wp6a1-e5-environment.txt`,
`runtime/wp6a1-dense-install.txt`, `runtime/wp6a1-bm25-cli.json`.

Real smoke test chỉ dùng `LocalE5Embedder` thật, corpus tổng hợp đã qua importer và
`RAGService`. Khi weights sẵn có, test kiểm tra document/query embeddings 768 chiều,
giá trị hữu hạn và unit norm, Dense bằng cosine tính độc lập, Hybrid bằng BM25 + Dense +
RRF tính độc lập, truy vấn Việt/Anh cho ba chủ đề, thứ tự/score ổn định khi lặp lại,
không duplicate, citation/hash/source/quyền và text đúng offsets. Batch size 2 kiểm tra
đường đi nhiều batch. Những assertion sau model load **chưa được thực thi trong môi
trường này** vì weights vắng mặt.

Kiểm tra trực tiếp `AutoTokenizer` và `AutoModel` với `local_files_only=True`,
`trust_remote_code=False` cùng `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` đều báo
`OSError` do không có cached files. Thư mục cache `models--intfloat--multilingual-e5-base`
không tồn tại. Đây là thiếu local assets; không suy ra model có lỗi hoặc mạng hỏng.

### Lệnh tái kiểm chứng

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-rag-dense.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest -q -ra --basetemp runtime/wp6a1-regression-rerun
.venv\Scripts\python.exe -m pytest tests -k rag -m 'not real_e5' -q -ra --basetemp runtime/wp6a1-rag-regression-rerun

# Gate nghiêm ngặt: thiếu dependencies/weights phải exit 1 và báo BLOCKED, không skip.
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:RAG_REQUIRE_REAL_E5 = '1'
.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval.py::test_real_e5_local_smoke -q -ra --basetemp runtime/wp6a1-real-e5-rerun
Remove-Item Env:RAG_REQUIRE_REAL_E5
```

CV smoke regression bằng các tài sản đã có, không tải hoặc thay weights:

```powershell
$env:CV_SMOKE_REGISTRY = (Resolve-Path config/models.json).Path
$env:WP2_SMOKE_CHECKPOINT = (Resolve-Path model_store/segformer/real_only/best.pt).Path
$env:WP2_SMOKE_IMAGE = (Resolve-Path model_store/wp2-verification/dermoscopy.jpg).Path
$env:WP2_SMOKE_DEVICE = 'cpu'
.venv\Scripts\python.exe -m pytest tests/test_cv_mvp.py tests/test_wp2_application.py tests/test_wp3_application.py -q -ra --basetemp runtime/wp6a1-real-cv-rerun
```

Trong full regression thông thường, E5 thiếu local assets được ghi **SKIPPED / BLOCKED**.
Gate `RAG_REQUIRE_REAL_E5=1` chuyển chính điều kiện đó thành FAIL rõ ràng để CI/operator
không nhầm suite xanh với real-model success. Một assertion sai sau khi load E5 luôn FAIL.

Chỉ sau khi operator cho phép tải weights, có thể tự chạy provisioning sau đây (lệnh
này **chưa được thực hiện** trong WP6A.1):

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -c "from app.rag.config import EmbeddingConfig; from app.rag.retrieval import LocalE5Embedder; LocalE5Embedder(EmbeddingConfig(allow_download=True)).embed_query('SYNTHETIC DEMO')"
# Sau provisioning, chạy lại gate offline phía trên.
```

Mặc định ứng dụng vẫn `allow_download=false`, `trust_remote_code=false`, tải model bằng
safetensors và lazy-load; BM25 không cần load torch/transformers/model. Corpus tổng hợp
chỉ kiểm tra cơ chế retrieval/citation, không phải benchmark hay bằng chứng chất lượng y khoa.

## Thay đổi code và nguyên nhân

- `config.py`: gom cấu hình có kiểu, validation và JSON parser vào một chỗ để runtime không
  unpack dict chưa kiểm tra vào backend. Không thêm source YAML.
- `chunking.py`, `retrieval.py`: dùng lại dataclasses từ config, giữ import cũ; bảo toàn
  chunk identity, thuật toán chunking, E5 pooling/cosine và RRF.
- `service.py`, `runtime.py`: nối BM25/config settings với service; không đổi search
  contract hoặc logic kiểm chứng lại quyền/corpus.
- CLI/UI: áp dụng retrieval defaults thực sự, cho phép override có chủ đích; Top-K UI
  hỗ trợ đủ giới hạn [1, 100] của service.
- `tests/test_rag_config.py`: backward compatibility, validation, binding paths, runtime,
  CLI override, UI defaults và không load model khi chỉ cấu hình/BM25.
- `tests/test_rag_retrieval.py`, `pytest.ini`: RRF edge cases và real E5 verification gate;
  fake embeddings chỉ còn là unit-test geometry, không chứng minh real-model success.

Không sửa pipeline CV, weights hay cấu hình E1/E3. Notebook CV đã có thay đổi trong working
tree trước tác vụ được giữ nguyên và không đưa vào commit WP6A.1. Không push/merge.

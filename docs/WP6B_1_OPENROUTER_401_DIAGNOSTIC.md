# WP6B.1 — OpenRouter Embedding HTTP 401 diagnostic

Ngày: **2026-11-26, Asia/Bangkok**. Baseline:
`f4f2e1d389c3bdd70a114ffd2a030d3c7b3b6140`.
Branch: `fix/openrouter-embedding-401`. Không push/merge.

**PARTIAL / ROOT CAUSE NOT CONFIRMED.** Safe authentication checks đạt; chưa có
chấp thuận mới cho POST embedding trong lượt chẩn đoán này. Không tuyên bố đã sửa
401, đã chứng minh lỗi account/provider hoặc WP6B đã VERIFIED.

## Lỗi ban đầu và authentication evidence

Operator báo live acceptance dừng ở request embedding đầu tiên với
`openrouter_unauthorized`, `requests=1`. Các kiểm tra GET `/key` trước đó được
operator báo HTTP 200 với cả `trust_env=True` và `False`; chưa có minimal POST độc lập.
Đó là evidence do operator cung cấp, không phải POST đã tái chạy trong lượt này.

Đã thực chạy an toàn:

```powershell
.venv\Scripts\python.exe -m app.rag.cloud.diagnostics --settings config/rag.synthetic.example.json
```

Report chỉ lưu whitelist booleans/status/config identifiers tại
`runtime/wp6b-401/safe-diagnostic.json`, không chứa account details, key, headers,
vectors hoặc raw response body. HTTPX **0.28.1**, cùng cấu hình production
`trust_env=False`, `follow_redirects=False`. Kết quả:

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| Project root dotenv tồn tại, startup loader được gọi | True | True | PASS |
| Key present sau startup | True | True; từ startup dotenv, không in key | PASS |
| Process environment có key trước startup | Chỉ quan sát | False trong tiến trình diagnostic | OBSERVED |
| Existing environment không bị override | True | True; `load_dotenv(..., override=False)` | PASS |
| Key có whitespace đầu/cuối | False | False | PASS |
| GET `/api/v1/key` | 200 | 200 | PASS — authentication endpoint only |
| GET `/api/v1/embeddings/models` | 200 | 200 | PASS — read-only catalog |
| `baai/bge-m3` trong catalog | True | True | PASS — không chứng minh quyền POST |
| Minimal POST `/embeddings` | 200 + valid vector | NOT RUN | BLOCKED — cần opt-in mới |
| Production POST `/embeddings` | 200 + valid vector | NOT RUN | BLOCKED — cần opt-in bổ sung và A PASS |
| Production Bearer header tại mock transport | Khớp loaded key | True; chỉ xuất boolean | PASS — mock wire audit |
| Base URL/method/JSON content type | Official HTTPS / POST / JSON | Đúng tại mock transport | PASS — mock wire audit |
| Provider routing schema | Hợp lệ | `{"allow_fallbacks": false}` | STATIC PASS; live chưa kiểm chứng |

**Tổng request thực của lượt này: 2 read-only GET, 0 paid/model POST.** Không gọi
LLM/rerank, không gửi corpus Crawler/ảnh bệnh nhân hoặc sửa tài nguyên bên ngoài.
Không đọc `.env` để trích nội dung vào báo cáo; chỉ dùng startup loader như ứng dụng.

## Static API contract và HTTP client audit

Đã đọc client, embeddings, live acceptance, runtime, config, service, synthetic JSON,
WP6B report và các test liên quan. Các module production giữ nguyên vì chưa có
evidence chỉ ra lỗi source gây HTTP 401.

| Hạng mục | Kết quả audit |
| --- | --- |
| URL và method | `POST https://openrouter.ai/api/v1/embeddings`, đúng reference. |
| Authentication | `Authorization: Bearer ` + environment key; HTTPX `json=` tạo Content-Type JSON. |
| Secret source | Chỉ `OPENROUTER_API_KEY`; startup load project root dotenv, override=False. Không có key trong JSON. |
| `input` | Adapter dùng list các chuỗi; minimal dùng một string. Cả hai kiểu được endpoint mô tả. |
| `model` | `baai/bge-m3` trong catalog thực; không đổi model. |
| `encoding_format` | `float` là giá trị được tài liệu endpoint hỗ trợ. |
| Provider | object `allow_fallbacks=false`; `only=[provider]` chỉ khi operator pin provider. Không tự bật fallback. |
| Optional parameters | `dimensions` và `input_type` chỉ xuất khi được cấu hình; synthetic JSON hiện không gửi. |
| Prefix | Cloud BGE-M3 hiện prefix rỗng; không chèn prefix E5. |
| Client/middleware | HTTPX client không gắn request hooks/custom middleware, proxy environment bị tắt, không redirect. Mock audit kiểm tra request sau khi HTTPX build. |
| Retry | Production có bounded retry cho timeout/429/5xx, **không retry 401**. Diagnostic gửi mỗi probe một lần, không đi qua vòng retry. |
| Quyền dữ liệu | Service kiểm tra outbound approval mỗi attempt. Diagnostic không tạo service/corpus; chỉ literal synthetic cố định. |

Nguồn chính thức:

- [Embedding contract](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)
  mô tả POST, Authorization, string/list input, float encoding, optional dimensions/input_type/provider.
- [Current key reference](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key)
  mô tả GET `/key`; endpoint này không chạy model embedding.
- [Embedding model catalog](https://openrouter.ai/docs/api/api-reference/embeddings/list-all-embeddings-models)
  mô tả GET `/embeddings/models`; đã kiểm tra catalog thật thay vì suy ra từ model name.
- [BGE-M3 model page](https://openrouter.ai/baai/bge-m3) có slug và ví dụ embedding;
  catalog listing không đồng nghĩa account/key có quyền inference.
- [Chat reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
  đã đối chiếu phạm vi endpoint; không gọi chat.
- Link [rerank reference](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank)
  trả 404 từ công cụ đọc docs trong lần này; đối chiếu thêm
  [official rerank SDK reference](https://openrouter.ai/docs/client-sdks/python/api-reference/rerank).
  Đây là trạng thái trang documentation, không phải evidence rằng API `/rerank` bị 404.

Không có evidence để đổi `trust_env`, bỏ provider object/Authorization, thêm headers
không bắt buộc, đổi URL/model hoặc vô hiệu hóa gate nhằm ép HTTP 200.

## Minimal và production payload

CLI lấy payload B từ **chính `OpenRouterEmbedder` hiện hành**, chạy một input synthetic
qua `MockTransport` rồi dừng tại transport, không sửa adapter hoặc gửi request thật.
Mock cũng kiểm tra method, URL, Bearer equality, Content-Type, client flags/hooks.
Request body/header không được đưa vào report. So sánh field-level hiện tại:

| Field | Test A minimal | Test B production |
| --- | --- | --- |
| `model` | `baai/bge-m3` | Cùng model |
| `input` | String literal `Synthetic embedding test` | List một phần tử, cùng literal |
| `encoding_format` | Omitted | `float` |
| `provider` | Omitted theo minimal contract | `{"allow_fallbacks": false}` |
| `dimensions` | Omitted | Omitted trong JSON đang audit |
| `input_type` | Omitted | Omitted trong JSON đang audit |

Đây là khác biệt hợp lệ theo static contract; chưa có live evidence cho thấy field
nào gây lỗi. Test A dùng routing mặc định của OpenRouter vì payload tối giản không
gửi `provider`; Test B giữ nguyên route cấu hình. Nếu operator đã pin provider ở JSON
khác, cần cân nhắc và chấp thuận rõ cả minimal route trước khi chạy. Diagnostic không
sửa policy/config production hoặc coi kết quả A là quyền bypass routing trong service.

## Root cause và evidence còn thiếu

**Chưa xác nhận root cause.** GET 200 chứng minh key được endpoint auth chấp nhận
trong tiến trình kiểm tra, không chứng minh quyền POST embeddings hoặc provider
inference. Static schema hợp lệ cũng chưa chứng minh request production sẽ thành công.

Các khả năng còn mở: account/key/organization restrictions theo endpoint/model,
provider routing/inference restrictions, sự cố upstream tạm thời hoặc sự khác biệt
runtime của lần POST trước. Chưa có bằng chứng đủ để chọn một khả năng. Không kết luận
key sai, proxy lỗi hoặc lỗi thuộc account chỉ từ HTTP 401.

Evidence tiếp theo cần là minimal POST được cấp phép. A thất bại thì dừng, giữ HTTP
status + normalized code; không tự thử model/provider/fields khác. Nếu A PASS, chỉ chạy
B khi có opt-in bổ sung. Nếu A PASS/B FAIL, cần xin thêm số request cụ thể trước khi
thử loại từng field có kiểm soát; hai POST đầu chưa thể chỉ ra field gây lỗi. Nếu cả
hai PASS, chỉ kết luận lỗi cũ chưa tái hiện, không tự nhận đã tìm/sửa root cause.

## Diagnostic CLI và opt-in

Mặc định safe mode có tối đa hai read-only GET và zero paid requests. Nếu key thiếu,
format không hợp lệ hoặc GET auth thất bại thì dừng sớm; không retry/redirect. Không
in account label/balances/IDs, provider body, exception text, vectors hay secret.
`safe_status=PASS`/exit 0 chỉ xác nhận các kiểm tra yêu cầu đã đạt, không phải DoD FIXED.

Chỉ operator chạy các lệnh sau khi cấp quyền riêng cho lần thử này:

```powershell
# Test A: cho phép tối đa MỘT paid POST, fixed synthetic string, zero retries.
.venv\Scripts\python.exe -m app.rag.cloud.diagnostics --settings config/rag.synthetic.example.json --live

# Cho phép cả A và B: tối đa HAI paid POST; B chỉ sau khi A PASS.
.venv\Scripts\python.exe -m app.rag.cloud.diagnostics --settings config/rag.synthetic.example.json --live --include-production
```

Hai lệnh là hai lựa chọn cho **một lượt mới**; không chạy nối tiếp để vô tình lặp A.
`--include-production` không có `--live` bị từ chối trước network. Flag cũ
`--budget-confirmed`, `cloud.synthetic_live_opt_in` hoặc feature flags không cấp phép
paid diagnostic. Mỗi lượt dùng key hiện hành và cùng HTTP client settings; không có
model substitution, request rerank/chat hoặc quyền đọc/gửi corpus thật.

Không có opt-in mới nhận được trong lúc thực hiện safe diagnostic thì giữ POST ở
NOT_RUN. Câu hỏi xin opt-in tách biệt đã nêu chính xác scope một hoặc hai request;
không xem thời gian chờ hoặc lựa chọn mặc định là đồng ý.

## Nếu cần operator kiểm tra account

Chỉ khi có thêm evidence POST: kiểm tra locally trên dashboard OpenRouter các hạn chế
key/organization cho endpoint, model và provider, key/account trạng thái active,
credit/spending limit và data policy. Không đổi key hoặc cấp rộng quyền chỉ để thử
mà chưa xác định restriction nào liên quan. 402 cần kiểm tra credits/budget; 403 cần
kiểm tra permission/policy; 404 cần xác minh slug/endpoint availability; 429 chờ theo
rate limits thay vì chạy lặp. Đây là hướng kiểm tra theo status, chưa phải kết luận
nguyên nhân cho case hiện tại.

Nếu minimal 401 vẫn xuất hiện dù cùng tiến trình GET `/key` 200, cung cấp OpenRouter
support thời gian, endpoint/model, HTTP status và report đã che; giữ thông tin account/
request identifiers trong trao đổi riêng với operator. Không gửi key, Authorization
header, `.env`, raw body hoặc tài liệu y khoa vào issue/report công khai.

## Files, regression và bảo toàn

Chỉ thêm CLI `app/rag/cloud/diagnostics.py`, tests `tests/test_rag_cloud_diagnostics.py`
và tài liệu này. Không sửa production transport/embedding/runtime/config/service,
không refactor RAG hoặc thêm feature retrieval/generation mới. Không có bản sửa 401
giả định nên không có test giả nhận là tái hiện root cause; MockTransport tests xác
minh invariants và hành vi diagnostic, không chứng nhận live inference.

| Kiểm tra | Kết quả |
| --- | --- |
| Diagnostic unit tests lần đầu | 35 passed, 0.65s; sau đó thêm test CLI exit cho A-only |
| Diagnostic tests cuối | **36 passed**, 0.44s |
| Cloud contracts/parsing, Dense/Hybrid, rights/revocation, secrets, RAGService/ingestion, CV registry/startup | **332 passed, 6 skipped**, 92.85s |
| `compileall`, `pip check`, `git diff --check` | PASS |
| Protected SHA-256 snapshot | **113/113 không đổi** |
| Live embedding A/B | NOT RUN / 0 paid requests |

Regression chạy trước bổ sung test CLI exit cuối; diagnostic đã chạy lại 36 ca sau
điều chỉnh đó. Các lượt có ca trùng nhau, không cộng pass. Sáu skip: E5 local thiếu
weights, quyền symlink Windows thiếu và bốn ca actual E1/E2/E3/rollback chưa bật
`WP_CV4_REAL`. CV registry/startup regression đạt; lượt này không tuyên bố đã chạy lại
CV inference thật. Hai server 7860/7861 giữ PID 34584/27432.

Logs gitignored tại `runtime/wp6b-401/`: safe JSON, diagnostic tests, regression text/XML,
protected hashes. Basetemp riêng trong runtime; không chiếm server 7860/7861 hoặc chạy
training. Giữ nguyên notebook đã sửa và thư mục Faster R-CNN chưa commit của operator;
không stage các tài sản đó. `.env` tiếp tục ignored, không commit secrets.

## Trạng thái WP6B.2/WP6B.3

Giữ nguyên implementation và regression của reranking/grounded generation. Không
chạy lại acceptance toàn pipeline trong lần này. Chưa có embedding live PASS mới nên
chưa đủ evidence để đề xuất xác nhận live hai bước sau. Nếu A/B live PASS, operator
có thể cấp **một opt-in mới** cho full pipeline acceptance; quyền đó không được suy ra
từ opt-in diagnostic. Không push/merge nếu chưa có yêu cầu mới.

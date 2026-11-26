# Hệ thống hỗ trợ phân tích tổn thương da

WP-CV4 (26/11/2026): đã xác minh các inference export E1/E2/E3 bằng checksum, metadata,
strict state_dict và inference CPU thật. Registry chọn E2
`E2-a2301ed5919a4b9089dc3cc73af65e70`; E1 mới giữ inactive để rollback, E3 inactive.
Kết quả kiểm thử và giới hạn provenance: [WP-CV4](docs/WP_CV4_E123_INTEGRATION.md).

Demo acceptance 25/11/2026: **hai mô hình thật và Chromium đã đạt trên CPU** (160 pass, 1 optional pilot skip). Xem [runbook demo](docs/DEMO_RUNBOOK.md) và [hồ sơ release acceptance](docs/DEMO_RELEASE_ACCEPTANCE.md). Pilot inactive không chặn release này; CUDA chưa kiểm chứng.

Ứng dụng nghiên cứu dành cho nhân viên y tế đã được đào tạo. Mục tiêu của đồ án là hỗ trợ quan sát tổn thương trên ảnh soi da, tra cứu kiến thức da liễu và chuẩn bị hội chẩn. **CV MVP tích hợp SegFormer MiT-B2 và Faster R-CNN qua adapter, registry và orchestrator độc lập.** Nguyên mẫu này chưa được thẩm định để sử dụng lâm sàng và không tự động chẩn đoán bệnh da.

| Mô-đun | Trạng thái CV MVP |
| --- | --- |
| Tổng quan | Thông tin ứng dụng và trạng thái mô-đun |
| Computer Vision | Tải JPEG/PNG; phân đoạn và detection độc lập; bốn thẻ xem ảnh, opacity, xuất Overlay và Localization PNG ở kích thước gốc. |
| Medical RAG | BM25 / Dense E5 Small / Hybrid RRF, citation document ID và trang PDF, retrieval-only local |
| Report | Báo cáo tổng hợp MVP và trạng thái kiểm thử |

Nút **Phân tích** khả dụng khi ít nhất một mô hình đã nạp thành công và phiên có ảnh hợp lệ. **Localization** phân biệt box từ mask SegFormer (vàng) và detection Faster R-CNN (đỏ, có điểm). Một mô hình lỗi vẫn giữ kết quả thành công của mô hình kia. Điểm detection không phải xác suất ung thư. [Kiến trúc, artifact, kiểm chứng và giới hạn cuối sprint](docs/FINAL_CV_MVP.md).

PNG Localization có detection dùng tên `cv-localization-<run>.png`; ảnh chỉ từ mask và Overlay giữ tiền tố `segformer-`. Nếu phân tích thất bại, giao diện ghi lỗi riêng của từng mô hình.

Artifact pilot step 7200 đúng hash trong brief **chưa có**. Mốc demo 25/11 dùng inference E1 từ run hoàn tất; WP-CV4 hiện dùng E2 và giữ export E1 mới làm rollback, không gán nhãn pilot thành final.

## Cài đặt và chạy cục bộ

Dùng Python 3.12. Các lệnh dưới đây chạy tại thư mục gốc của kho mã.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\Activate.ps1
python main.py
```

Trên Linux hoặc macOS, thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`, kích hoạt bằng `source .venv/bin/activate`. Sau khi cài CV và chuẩn bị artifact cục bộ, chỉ cần **`python main.py`**. Lệnh `python -m app.main` vẫn hoạt động qua cùng hàm khởi động. Ứng dụng tự đọc `config/models.json` từ gốc dự án, nạp đúng các model `active=true` (SegFormer `checkpoint-v1` và detection `E2-a2301ed5919a4b9089dc3cc73af65e70`), rồi mở Gradio tại `http://127.0.0.1:7860` với `share=False`. Thiết bị mặc định `auto`: CUDA khi PyTorch hỗ trợ, còn lại CPU.

Terminal ghi registry đã chọn, trạng thái READY / FAILED / UNAVAILABLE, version, device thực tế của model đã nạp, checkpoint dự kiến và nguyên nhân nếu lỗi. Không tự tải weights. Một model lỗi không chặn model kia; nếu cả hai chưa sẵn sàng, ứng dụng vẫn cho tải/xem ảnh nhưng khóa **Phân tích** và hiển thị lý do.

### Bật phân đoạn SegFormer

Ví dụ cài đặt CPU cho Windows hoặc Linux. Nếu môi trường đã có bộ PyTorch/Torchvision đang hoạt động, kiểm tra phiên bản trước và dùng môi trường ảo riêng nếu cần một cặp phiên bản khác. Hướng dẫn CUDA và macOS có trong [tài liệu WP2](docs/WP2_SEGFORMER_INTEGRATION.md).

```powershell
.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-cv.txt
.venv\Scripts\python.exe -m app.cv.checkpoint
.venv\Scripts\python.exe main.py
```

Lệnh tải là thao tác chuẩn bị chủ động, xác minh SHA-256 và dùng lại tệp hợp lệ nếu đã có; startup không gọi downloader. Để dùng bản sao cục bộ của SegFormer, sửa `checkpoint_path` của entry segmentation trong registry; có thể xác minh riêng bằng `LESION_SEGFORMER_CHECKPOINT` và `python -m app.cv.checkpoint --verify-only`. Trong **Computer Vision**, tải ảnh soi da có giấy phép phù hợp rồi chọn **Phân tích**. Sau khi hoàn tất, chọn độ mờ và bấm **Tải Overlay PNG**. Khi có box, dùng **Tải Localization PNG** để tải ảnh định vị.

### Chạy đủ hai mô hình qua registry

Sau khi cài CV và chuẩn bị checkpoint SegFormer như trên, đặt ba artifact inference tại đường dẫn trong `config/models.json`. Máy hiện tại có E1/E2/E3 theo run ID; đường dẫn cũ `E1-final-inference.pt` không còn tồn tại và đã được thay bằng E1 mới, không gán lại checksum cũ. Kiểm chứng độc lập cả ba (CPU, tuần tự, không tải weights):

```powershell
.venv\Scripts\python.exe -m app.cv.verify_detection --image model_store/wp2-verification/dermoscopy.jpg
.venv\Scripts\python.exe main.py
```

Mở `http://127.0.0.1:7860` → **Computer Vision** → tải ảnh → **Phân tích**. Không cần đặt `LESION_MODEL_REGISTRY`; nếu có, biến này ưu tiên hơn registry mặc định. Registry có ưu tiên hơn `LESION_SEGFORMER_CHECKPOINT`. Chỉ khi **registry mặc định không tồn tại và không có override**, ứng dụng cảnh báo rồi giữ chế độ SegFormer cũ qua `LESION_SEGFORMER_CHECKPOINT`; detection không khả dụng. Registry đã được chọn mà thiếu, hỏng hoặc cấu hình sai sẽ báo lỗi, không tự chuyển sang checkpoint khác.

E2 đã được chọn từ Validation; không dùng Official Test hoặc smoke image để chọn lại model/ngưỡng. Rollback: dừng ứng dụng, đặt `active=false` cho E2 và `active=true` cho `E1-93e17545c5b84d78b46a9040cc7a8942`, giữ E3/pilot inactive rồi khởi động lại. Giữ tối đa một detection active; không sửa weights, checksum, threshold hoặc preprocessing. Chi tiết và lệnh rollback an toàn trong [WP-CV4](docs/WP_CV4_E123_INTEGRATION.md).

## Cấu hình

Các biến môi trường sau ghi đè cấu hình mặc định. Đường dẫn tương đối được tính từ thư mục dự án.

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LESION_APP_NAME` | Hệ thống hỗ trợ phân tích tổn thương da | Tên hiển thị |
| `LESION_HOST` / `LESION_PORT` | `127.0.0.1` / `7860` | Địa chỉ IP hoặc `localhost` và cổng lắng nghe |
| `LESION_RUNTIME_DIR` | `runtime/` | Dữ liệu tạm. Dùng thư mục dành riêng cho một tiến trình ứng dụng |
| `LESION_MODEL_STORE_DIR` | `model_store/` | Nơi CLI lưu checkpoint |
| `LESION_SEGFORMER_CHECKPOINT` | Chưa cấu hình | Đường dẫn SegFormer cho CLI xác minh và chế độ cũ khi không có registry mặc định |
| `LESION_SEGFORMER_DEVICE` | `auto` | Chọn CUDA nếu PyTorch nhận diện được CUDA, CPU nếu không. Có thể chọn cố định `cpu` hoặc `cuda` |
| `LESION_MODEL_REGISTRY` | `config/models.json` | Override JSON chọn artifact/version; mặc định tự tìm từ project root |
| `LESION_DETECTION_DEVICE` | `auto` | `auto`, `cpu`, `cuda`; CUDA chưa được kiểm chứng trong sprint này |
| `LESION_DETECTION_PILOT_MODE` | `false` | Pilot chỉ được phép khi bật rõ ràng và entry có `research_pilot=true` |
| `LESION_MAX_UPLOAD_MIB` | `10` | Dung lượng tệp, từ 1 đến 64 MiB |
| `LESION_MAX_IMAGE_PIXELS` | `16000000` | Tổng số pixel, tối đa 36 triệu |
| `LESION_MAX_IMAGE_EDGE` | `8192` | Số pixel mỗi cạnh, tối đa 16.384 |
| `LESION_SESSION_TTL_SECONDS` | `3600` | Thời gian giữ tệp tạm của ứng dụng sau lần sử dụng cuối, từ 1 giây đến 7 ngày |
| `LESION_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` hoặc `CRITICAL` |
| `LESION_RESEARCH_MODE` | `true` | WP1 yêu cầu `true` hoặc `1` |

`runtime/` và `model_store/` nằm trong `.gitignore`. Nếu đặt lại vị trí trong kho mã, giữ chúng dưới hai thư mục tương ứng. Hai vị trí phải tách biệt. Cấu hình host ngoài loopback làm cổng lắng nghe có thể truy cập từ mạng, tùy cấu hình máy.

## Phiên và dữ liệu tạm

Mỗi phiên Gradio giữ một ảnh hiện tại và kết quả của từng mô hình tương ứng. Ảnh được chuẩn hóa hướng EXIF và màu RGB. Khi thay ảnh hoặc đặt lại, ứng dụng hủy hiệu lực của lượt phân tích và yêu cầu xóa ảnh cùng mask trước đó. Dữ liệu của ứng dụng nằm trong `runtime/sessions/`. Một lượt phân tích đang đọc hoặc ghi có thể giữ tệp đến khi thao tác hoàn tất. Kết quả của lượt đã hết hiệu lực bị loại bỏ.

Ứng dụng mã hóa PNG xuất trong bộ nhớ khi người dùng yêu cầu. Máy chủ kiểm tra phiên trình duyệt, ảnh và lượt phân tích hiện tại trước khi trả PNG. Quyền xuất hết hạn theo TTL của phiên.

Gradio quản lý riêng tệp tải lên trong `runtime/gradio-cache/` theo tuổi tệp. Đặt lại ảnh không xóa ngay mọi bản sao thuộc cache của Gradio. Bộ nhớ trình duyệt, tệp còn lại sau khi tiến trình dừng đột ngột và chi tiết vòng đời được mô tả trong tài liệu nền tảng. [Kiến trúc và giới hạn WP1](docs/CV_MVP_FOUNDATION.md)

## Medical RAG

Tám tài liệu được nạp từ corpus local đã đồng ý indexing, dùng multilingual E5 Small 384 chiều trên CPU và ChromaDB hiện có. Cấu hình local không bật reranking hay generation outbound. Không gửi corpus lên cloud chỉ vì đã có quyền indexing local.

```powershell
$env:RAG_CONFIG = (Resolve-Path 'config/rag.local.json').Path
.venv\Scripts\python.exe main.py
```

Trong tab **Medical RAG**, chọn BM25, Dense hoặc Hybrid, đặt số nguồn và chọn một chế độ:

- **Retrieve Only** hiển thị đoạn truy xuất, score, tiêu đề, trang PDF, chunk ID và URL nguồn.
- **Evidence Summary** chọn các câu nguyên văn dựa trên trùng lặp từ khóa. Đây là trích đoạn có dẫn chứng, không phải câu trả lời do LLM tạo. Nếu không chọn được câu phù hợp, chế độ này abstain.
- **Ask with LLM** chạy retrieval trước. Với cấu hình corpus hiện tại, generation bị tắt bởi quyền xử lý dữ liệu. Ollama có trên máy nhưng chưa có model được cài và không tự tải model.

Markdown được render bằng component Gradio có sanitize HTML. Mỗi nguồn đánh dấu riêng trạng thái `needs_human_review`; kiểm tra metadata không đồng nghĩa với xác minh nội dung. Mở accordion văn bản gốc để đọc chunk nguyên trạng. Không nhập thông tin nhận dạng bệnh nhân.

WP6E giữ 40 câu hỏi. Khi chưa có chunk liên quan được con người duyệt, Precision@5, Recall@5, MRR@10 và nDCG@5 được báo `N/A`. Chạy lại bằng:

```powershell
.venv\Scripts\python.exe scripts\rag_evaluate.py --repeats 1
```

Kết quả kiểm chứng hiện tại và giới hạn nằm trong [Main System v1.1 acceptance](docs/MAIN_SYSTEM_V1_1_FINAL_ACCEPTANCE.md). Cài đặt Windows và demo CV hiện hữu vẫn theo hướng dẫn ở trên và [demo runbook](docs/DEMO_RUNBOOK.md).

## Kiểm thử

Medical RAG local hiện dùng multilingual E5 Small, ChromaDB, BM25 / Dense / Hybrid và ba chế độ hiển thị. Generation không được bật với corpus chỉ có quyền local indexing. Nội dung citation vẫn cần human review theo trạng thái nguồn.
Thiết kế adapter OpenRouter, quyền outbound, cache, cấu hình key và acceptance WP6B lịch sử:
[WP6B Cloud OpenRouter](docs/WP6B_CLOUD_OPENROUTER.md).
Hướng dẫn dữ liệu, quyền AI, demo tổng hợp và kết quả kiểm thử: [WP6A RAG Foundation](docs/WP6A_RAG_FOUNDATION.md).
Cấu hình JSON có validation, tham số mặc định và trạng thái kiểm chứng E5 thật: [WP6A.1 Verification](docs/WP6A_1_VERIFICATION.md).

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
$env:RAG_SYNTHETIC_DEMO = '1'
.venv\Scripts\python.exe main.py
```

Chọn **Medical RAG** rồi Search. Corpus demo luôn có nhãn **SYNTHETIC DEMO** và tách biệt với corpus tám tài liệu.
BM25 dùng được ngay. Cấu hình synthetic có thể dùng OpenRouter cho Dense/Hybrid; cần API key và quyền outbound riêng. Không dùng synthetic demo để suy ra quyền xử lý corpus thật. JSON WP6A cũ vẫn dùng backend local đã cấu hình.
Tắt demo bằng `Remove-Item Env:RAG_SYNTHETIC_DEMO`.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q tests --basetemp runtime/wp3-regression
```

Bộ kiểm thử thông thường không tải checkpoint. Để chạy đủ ca đơn vị CV, cài thêm `requirements-cv.txt`. Lệnh kiểm chứng cả hai mô hình thật và Chromium có trong [báo cáo CV MVP](docs/FINAL_CV_MVP.md). Tài liệu WP2/WP3 ghi nhận các mốc trước sprint này.

Dùng thư mục riêng cho `--basetemp` vì pytest sẽ dọn thư mục đó khi chạy lại.

Pytest đã cấu hình mặc định `runtime/pytest-tmp`, phù hợp Windows và tránh lỗi quyền trên thư mục temp hệ thống.

Đánh giá WP6E chạy bằng:

```powershell
.venv\Scripts\python.exe scripts\rag_evaluate.py --repeats 1
```

Kết quả nằm trong `runtime/rag/wp6e/`. Metric không có ground truth được ghi `N/A`.

Để pytest chạy các ca trình duyệt, cài Playwright và Chromium trong môi trường kiểm thử.

```powershell
.venv\Scripts\python.exe -m pip install playwright==1.65.0
.venv\Scripts\python.exe -m playwright install chromium
```

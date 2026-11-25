# CV MVP — báo cáo cuối sprint 25/11/2026

> Đây là báo cáo lịch sử của sprint trước acceptance cuối. Kết quả mới: [DEMO_RELEASE_ACCEPTANCE.md](DEMO_RELEASE_ACCEPTANCE.md), hai mô hình thật và Chromium đạt trên CPU; [runbook](DEMO_RUNBOOK.md). Theo phạm vi release hiện tại, pilot inactive không bắt buộc và không chặn demo E1 hoàn tất. Các số đo/trạng thái bên dưới được giữ làm lịch sử.

**CV MVP CONDITIONALLY READY — BLOCKERS REMAIN**

Ứng dụng đã chạy hai checkpoint thật trên CPU, có giao diện và export PNG được kiểm chứng bằng Chromium. Artifact pilot chính xác trong brief chưa được cung cấp; không coi ca pilot là đạt. CUDA chưa được kiểm chứng. Chỉ dùng cho nghiên cứu, không chẩn đoán hoặc đề xuất điều trị.

Lượt tiếp tục bắt đầu từ HEAD `10e02697a98c2b6239a3cb93d2e5129f2fe2f04d`, đã chứa phần triển khai trên base `a1839a8`. Commit `602332a` sửa việc quy nguồn tên PNG: Localization có detection dùng `cv-localization-<run>.png`; Overlay và Localization chỉ từ mask giữ tên WP3. Lỗi phân tích được hiển thị theo từng mô hình, kể cả khi chỉ detector được cấu hình hoặc cả hai cùng thất bại. Mô tả ảnh và trang Tổng quan đã bổ sung detection.

## Kiến trúc và tệp chính

| Tệp | Trách nhiệm |
| --- | --- |
| `app/cv/adapters.py` | Protocol `SegmentationAdapter`, `DetectionAdapter`, prediction không phụ thuộc tensor |
| `app/cv/artifacts.py`, `registry.py` | Metadata artifact bất biến, SHA-256, lựa chọn phiên bản và cô lập cấu hình lỗi theo mô-đun |
| `app/cv/detection.py` | Faster R-CNN ResNet50-FPN V2; kiểm tra payload, strict weights, preprocessing, forward và validation box/score |
| `app/cv/export_detection.py` | Lệnh offline xuất inference từ best/last có checksum và run đã hoàn tất; không chạy trong ứng dụng |
| `app/cv/orchestrator.py`, `analysis.py` | Chạy tuần tự, bắt lỗi riêng từng mô hình, ghép `CVResult`, chỉ nhận kết quả có RunToken hiện tại |
| `app/cv/schemas.py`, `availability.py` | Trạng thái tổng `partial`, lỗi/thời gian/provenance riêng từng mô hình |
| `app/cv/segformer.py` | Giữ loader, checksum, preprocessing và ngưỡng 0,7; chuyển kiểu prediction dùng chung sang contracts |
| `app/cv/visualization.py`, `app/ui/cv_page.py` | Bốn thẻ cũ; mask box vàng, detection đỏ và điểm; hỗ trợ chỉ detector hoặc chỉ segmenter |
| `app/ui/shell.py`, `png_export.py` | UI gọi application layer; export Overlay/Localization theo browser cookie, proof, session, action, run và TTL |
| `app/config.py`, `config/models.json` | Cấu hình device, pilot mode và phiên bản active |
| `tests/test_detection.py`, `tests/test_cv_mvp.py`, `pytest.ini` | Kiểm tra adapter, cô lập lỗi, registry/rollback, hai mô hình thật và trình duyệt; thu thập mặc định trong `tests/` |

Luồng: Gradio → CVOrchestrator → adapters → CVResult → visualization/export. UI không khởi tạo kiến trúc cụ thể. Registry chỉ nạp các entry active; thay file cấu hình không thay mô hình đang chạy. Khởi động lại là ranh giới đổi phiên bản. Một orchestrator khóa thực thi toàn bộ lượt phân tích, tránh hai forward lớn đồng thời.

`segmentation`, `detection` và `localization.source=derived_from_mask` là ba trường riêng. Khi một mô hình lỗi và mô hình kia thành công, aggregate là `partial`; chỉ một mô hình cấu hình và chạy thành công thì aggregate là `completed`. Hai mô hình không cấu hình trả `not_available`; hai lỗi trả `failed`. Không có fallback sang trọng số ngẫu nhiên.

## Artifact thực tế

Thư mục được người dùng chỉ định: `model_store/detection/E1`. Kiểm tra best/last bằng `torch.load(weights_only=True)`, checksum sidecar, run ID, vị trí huấn luyện, config, manifest và `completed.json`:

- Run `93e17545c5b84d78b46a9040cc7a8942`, hoàn tất bằng early stopping tại step 7265; best epoch lưu dưới chỉ số 9.
- Best SHA-256: `07e05c6f6f949a6a17f88da5cc282566acb18c7ba83e2ef7e3cc3a676c307937`.
- Last SHA-256: `f3d023e94c893bcedf9e8e535013365cdf30ef30172ee859367a71d212214825`.
- Artifact ứng dụng: `model_store/detection/E1-final-inference.pt`, version `E1-best-completed`, format `isic-fasterrcnn-inference-v1`.
- SHA-256 artifact: `eed8c28c2777005a2c6033692137f3cac8df8c68251ecb342ee4b8edf3068c9f`.

Payload inference chỉ chứa model, labels, model_config, preprocessing, operating threshold, selection và environment/evaluation metadata; không có optimizer/scheduler/scaler/RNG. Labels đúng pipeline: `0=__background__`, `1=skin_lesion`. Min/max size 800/1333; score threshold 0,5; RGB float32 CHW [0,1], resize/normalize nội bộ Torchvision. Constructor đặt `weights=None`, `weights_backbone=None`; `load_state_dict(strict=True)`, `eval()` và `torch.inference_mode()`.

Pilot được yêu cầu: `E1_step07200_PILOT_UNFINISHED.pt`, SHA-256 `327c2281b69a9591939c1abe883cb4337fecc9983fdf8746d26c7b5a32011141`, format `isic-fasterrcnn-pilot-v1`: **chưa có**. Snapshot training `step-00007200.ckpt` có format số `1` và optimizer state, không phải artifact pilot đó. Không đổi tên, không gán hash hoặc provenance của pilot cho snapshot này.

Entry pilot mặc định inactive. Loader chỉ cho pilot khi `LESION_DETECTION_PILOT_MODE=true`, entry `research_pilot=true`, đúng hash và metadata inference tương thích. Do chưa đọc được artifact pilot thực tế, đường pilot chỉ kiểm chứng bằng unit test hợp đồng; compatibility và real forward của pilot vẫn bị chặn. Nếu payload thực tế khác, phải kiểm tra và bổ sung mapping rõ ràng; không bỏ key hoặc nới strict loading.

## Chuẩn bị và chạy Windows PowerShell

Chạy tại thư mục gốc repository; dùng Python 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-cv.txt -r requirements-dev.txt
.venv\Scripts\python.exe -m app.cv.checkpoint
# Chỉ khi chưa có artifact inference; không ghi đè bản đã đăng ký:
.venv\Scripts\python.exe -m app.cv.export_detection model_store/detection/E1 model_store/detection/E1-final-inference.pt
# Ghi SHA-256 của file thực tế vào config/models.json nếu vừa xuất lại ở máy khác.
$env:LESION_MODEL_REGISTRY = 'config/models.json'
$env:LESION_SEGFORMER_DEVICE = 'cpu'
$env:LESION_DETECTION_DEVICE = 'cpu'
.venv\Scripts\python.exe -m app.main
```

Máy hiện tại đã có cả checkpoint và artifact inference, nên chỉ cần bốn lệnh cuối từ `$env:LESION_MODEL_REGISTRY`. Mở `http://127.0.0.1:7860` → Computer Vision → tải ảnh được phép sử dụng → Phân tích. Overlay tự mở khi có mask; chỉ detector thì mở Localization. Chỉnh opacity; xem Localization và Chi tiết mô hình; tải hai PNG; thay ảnh và Đặt lại. PNG xuất giữ kích thước ảnh gốc sau hiệu chỉnh EXIF.

## Đăng ký E2/E3 và rollback

1. Chép artifact inference đã xác minh vào một đường dẫn mới dưới `model_store/detection/`; không ghi đè E1.
2. Sao chép entry detection trong `config/models.json`; điền version, path, SHA-256, architecture, format, threshold, min/max size và preprocessing khớp metadata. Dùng `isic-fasterrcnn-inference-v1`, `research_pilot=false` cho export chuẩn.
3. Đặt entry cũ `active=false`, entry mới `active=true`; chỉ một active cho mỗi mô-đun. Dừng ứng dụng và khởi động lại với cùng biến `LESION_MODEL_REGISTRY`.
4. Rollback: đảo hai cờ active và khởi động lại. Giữ nguyên file và SHA-256 của entry trước. Không sửa callback, viewer hoặc schema.

Registry ưu tiên hơn biến checkpoint SegFormer cũ. Muốn chạy chỉ một mô-đun, tắt mọi entry active của mô-đun còn lại. Không đặt registry thì giữ cấu hình SegFormer WP2/WP3 qua `LESION_SEGFORMER_CHECKPOINT`. Lỗi entry detection không ngăn segmentation nạp; JSON toàn cục không đọc được thì cả hai được báo unavailable/failed an toàn.

## Kiểm chứng

Ảnh mẫu đã có từ WP2: `model_store/wp2-verification/dermoscopy.jpg` (330 × 244). Không crawl dữ liệu, huấn luyện lại hoặc benchmark test set. Lượt CPU mới nhất đo được SegFormer 415,36 ms, Faster R-CNN 2412,99 ms; detection `[134.668, 87.285, 183.778, 126.461]`, score `0.803465`. Đây là thời gian forward một lượt, không phải chỉ tiêu hiệu năng. Box dùng pixel ảnh gốc, cạnh phải/dưới loại trừ; không diễn giải score thành xác suất ung thư.

Môi trường kiểm chứng: Python 3.12, Torch `2.10.0+cpu`, Torchvision `0.25.0+cpu`, Gradio `6.32.0`; `torch.cuda.is_available() == False`. Hash hai artifact đã được tính lại và khớp registry; cả hai loader và real forward đều đạt.

```powershell
.venv\Scripts\python.exe -m pip install playwright==1.65.0
.venv\Scripts\python.exe -m playwright install chromium
$env:WP2_SMOKE_CHECKPOINT = (Resolve-Path 'model_store/segformer/real_only/best.pt').Path
$env:WP2_SMOKE_IMAGE = (Resolve-Path 'model_store/wp2-verification/dermoscopy.jpg').Path
$env:WP2_SMOKE_DEVICE = 'cpu'
$env:CV_SMOKE_REGISTRY = (Resolve-Path 'config/models.json').Path
$testRun = Join-Path (Get-Location) ('runtime/pytest-release-' + [guid]::NewGuid().ToString('N'))
.venv\Scripts\python.exe -m pytest -q -ra --tb=short --capture=tee-sys --basetemp "$testRun" --junitxml runtime/cv-continuation-results.xml *> runtime/cv-continuation-tests.log
$LASTEXITCODE # 0 khi đạt; đọc log bên dưới để kiểm tra số pass/skip
Get-Content runtime/cv-continuation-tests.log -Tail 12
```

Kết quả kiểm chứng mới nhất được ghi trong `runtime/cv-continuation-tests.log` và `runtime/cv-continuation-results.xml` (Git-ignored), exit code 0. Ca real pilot bị skip với lý do BLOCKED. Bỏ biến môi trường smoke thì các ca real khác cũng skip, không được coi là chứng cứ inference thật.

| Phạm vi kiểm chứng CPU | Pass | Fail/error | Skip |
| --- | ---: | ---: | ---: |
| Suite ứng dụng đầy đủ, 90,93 giây | 160 | 0 | 1 |
| Hồi quy WP1–WP3 có sẵn (nằm trong dòng trên) | 112 | 0 | 0 |
| Adapter và tích hợp CV mới (nằm trong dòng trên) | 48 | 0 | 1 |

Năm ca dùng checkpoint thật đã đạt: SegFormer callback, SegFormer browser, WP3 browser/export, CVResult hai mô hình và browser hai mô hình. Các unit test dùng dữ liệu tổng hợp không được dùng làm bằng chứng inference thật. `python -m compileall -q app` và `git diff --check` cũng đạt.

Các kiểm tra gồm: hồi quy WP1–WP3 với checkpoint SegFormer thật; hai mô hình cùng CVResult; ma trận thành công/lỗi/thiếu mô hình; output detection không hợp lệ và ghi mask thất bại vẫn giữ mô hình kia; Reset/upload khi detector đang chạy; rollback; chỉ nạp active; detector-only Gradio; PNG đủ độ phân giải, cookie/proof/TTL, truy cập chéo phiên và đường dẫn trực tiếp; Chromium hai mô hình, opacity, download và thu hồi export sau đổi ảnh/Reset.

Lượt tiếp tục kiểm tra thêm tên PNG ở HTTP response và download Chromium, tương thích tên WP3, thông báo detector-only failure và hai mô hình cùng thất bại. Bốn tệp mã/test được sửa: `app/ui/cv_page.py`, `app/ui/png_export.py`, `app/ui/overview.py`, `tests/test_cv_mvp.py`; tài liệu cập nhật gồm `README.md` và báo cáo này. Không thay đổi weights, threshold, training workspace hoặc cơ chế cấp quyền PNG.

Lần pytest ban đầu thu thập nhầm test/thư viện trong `finetune_detection/runs` (50 lỗi collection), và lần chạy không chỉ định basetemp vướng quyền thư mục Temp Windows (48 setup errors). `pytest.ini` giới hạn suite ứng dụng; lệnh cuối dùng basetemp riêng. Một lỗi giữ tệp tạm trong export đã được kiểm thử cũ phát hiện và sửa; không bỏ các ca bảo mật.

## Giới hạn còn lại

- Chưa xác minh artifact pilot đúng hash; CUDA chưa chạy trong môi trường PyTorch CPU hiện tại.
- E1 best-completed là mô hình nghiên cứu của run hoàn tất, không phải chứng nhận lâm sàng hoặc mô hình cuối cùng của toàn dự án.
- Localization kết hợp hai nguồn với màu/nhãn khác nhau; chưa có công tắc ẩn từng nguồn. Overlay và các nút PNG hiện có được giữ.
- Phiên browser giữ một ảnh hiện tại; không có hồ sơ bệnh nhân, RAG, chẩn đoán, đề xuất điều trị hoặc cloud deployment.
- `finetune_detection/` không được sửa trong sprint. Notebook có thay đổi của người dùng từ trước; giữ nguyên, không stage/commit.
- Không commit weights, dữ liệu, runtime, ảnh xuất hoặc môi trường ảo; không push.

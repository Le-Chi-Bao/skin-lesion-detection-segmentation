# Khởi động CV bằng một lệnh — 25/11/2026

Base HEAD: `80627b12cdb1443419db483c387c1e01952e5339`.

Sau khi kích hoạt môi trường có các dependency và chuẩn bị checkpoint cục bộ:

```powershell
python main.py
```

Trước đây `AppConfig.from_env()` để `model_registry=None` khi không có biến môi trường, nên `create_orchestrator()` dùng luồng SegFormer cũ và không tạo detector. Nay cấu hình khởi động tự tìm `config/models.json` từ project root, độc lập thư mục làm việc. `LESION_MODEL_REGISTRY` vẫn có ưu tiên cao nhất. `AppConfig()` trực tiếp vẫn giữ chế độ cũ cho các caller nhúng/test đã truyền cấu hình rõ ràng.

Chỉ khi registry mặc định không tồn tại và không có override mới cảnh báo rồi dùng chế độ SegFormer cũ. Registry đã chọn nhưng lỗi không được thay bằng registry/checkpoint khác. Adapter chỉ được tạo/nạp cho entry active, tối đa một entry mỗi module. `config/models.json` không thay đổi: SegFormer `checkpoint-v1`, detection `E1-best-completed`, pilot inactive. Không sửa adapter inference, contracts, weights, preprocessing hoặc threshold.

## File thay đổi

| File | Vai trò |
| --- | --- |
| `main.py` | Entry ở gốc, chỉ gọi `app.main.main()` và trả exit code. |
| `app/config.py` | Tìm registry mặc định; ưu tiên override; cảnh báo và giữ chế độ cũ nếu thiếu mặc định. |
| `app/main.py` | Ghi URL ứng dụng trước khi Gradio khởi động. |
| `app/cv/registry.py` | Ghi nguyên nhân lỗi đọc/validate registry; giải thích trường hợp không có model active. |
| `app/cv/orchestrator.py` | Diagnostics registry và từng model: trạng thái, version, device thực tế khi READY, checkpoint và lỗi. |
| `app/cv/availability.py` | Dùng “Không khả dụng” thay cho “Chưa tích hợp”. |
| `app/core/errors.py` | Hướng dẫn thiếu checkpoint phù hợp cả SegFormer và Faster R-CNN. |
| `app/ui/cv_page.py` | Hiện nguyên nhân từng model và thông báo Phân tích bị vô hiệu hóa khi không model nào sẵn sàng. |
| `tests/test_startup.py` | 19 ca startup/selection/failure, gồm đường dẫn lỗi và đăng ký E3 tương lai. |
| `tests/test_app_smoke.py` | Kiểm tra trạng thái mới và Analyze không có callback khi cả hai model chưa sẵn sàng. |
| `tests/test_cv_mvp.py` | Giữ browser smoke qua module với override; thêm smoke thật qua `main.py` không override registry/device và kiểm tra diagnostics. |
| `tests/test_wp2_application.py` | Fixture browser WP2/WP3 chọn registry chỉ SegFormer để giữ nguyên toàn bộ assertion pixel/PNG riêng của luồng đó. |
| `README.md`, `docs/DEMO_RUNBOOK.md` | Hướng dẫn một lệnh, fallback, override và cách thay phiên bản detection. |
| `docs/ONE_COMMAND_ACCEPTANCE.md` | Bằng chứng và cách tái chạy kiểm thử cho thay đổi này. |

## Kiểm thử thực tế

Môi trường: Windows, Python 3.12.5, PyTorch 2.10.0+cpu, Gradio 6.32.0, Chromium qua Playwright. CUDA không có trong môi trường này; chưa kiểm chứng real inference trên CUDA.

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:CV_SMOKE_REGISTRY = (Resolve-Path 'config/models.json').Path
$env:WP2_SMOKE_CHECKPOINT = (Resolve-Path 'model_store/segformer/real_only/best.pt').Path
$env:WP2_SMOKE_IMAGE = (Resolve-Path 'model_store/wp2-verification/dermoscopy.jpg').Path
.venv\Scripts\python.exe -m pytest -q tests --basetemp runtime/one-command-regression --tb=short -r s
```

Kết quả toàn bộ regression: **179 passed, 1 skipped trong 127,28 giây**. Skip duy nhất là artifact pilot `E1_step07200_PILOT_UNFINISHED.pt` đúng hash chưa có. Không xóa/nới assertion để đạt kiểm thử.

Sau khi bổ sung ca đường dẫn checkpoint không hợp lệ và hoàn thiện diagnostics, chạy lại các test chịu ảnh hưởng:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_startup.py tests/test_cv_mvp.py -k 'not real' --basetemp runtime/one-command-final-unit --tb=short
```

Kết quả: **46 passed, 4 deselected**, gồm đủ **19 test startup**. Đây là unit tests với loader/đầu ra tổng hợp khi cần, không dùng để tuyên bố real inference.

Hai checkpoint thật đều qua kiểm tra SHA-256 và nạp thành công trên CPU:

| Model/version | Checkpoint | SHA-256 |
| --- | --- | --- |
| SegFormer / `checkpoint-v1` | `model_store/segformer/real_only/best.pt` | `d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87` |
| Faster R-CNN / `E1-best-completed` | `model_store/detection/E1-final-inference.pt` | `eed8c28c2777005a2c6033692137f3cac8df8c68251ecb342ee4b8edf3068c9f` |

Real-model tests chạy inference qua validation → orchestrator → CVResult → visualization → PNG. Chromium kiểm tra cả `python -m app.main` có override và `python main.py` không có biến `LESION_MODEL_REGISTRY`/device, dùng device `auto` → CPU. Đã kiểm tra detection và mask thật, bốn tab, opacity, PNG đúng kích thước gốc, phiên khác không lấy được kết quả/PNG, đổi ảnh hủy export cũ, Reset dọn kết quả và khóa Analyze/export.

Bằng chứng cục bộ, không commit ảnh hay weights:

- `runtime/one-command-regression.log`
- `runtime/one-command-regression/test_real_dual_model_browser_u1/server.log`
- `runtime/one-command-regression/test_real_dual_model_browser_u1/dual-model-localization.png`
- `runtime/one-command-regression/test_real_dual_model_browser_u1/dual-model-reset.png`

## Đăng ký E3

Chỉ sau khi có artifact inference E3 đã xác minh: thêm entry detection trong `config/models.json` với version, đường dẫn và SHA-256 thật, `architecture=fasterrcnn_resnet50_fpn_v2`, `artifact_format=isic-fasterrcnn-inference-v1`, `research_pilot=false`; threshold, min/max size và preprocessing phải khớp metadata artifact. Tắt active của E1, bật active của E3 rồi khởi động lại. Loader vẫn kiểm SHA-256, metadata và strict state dict. Không có artifact được xác minh thì giữ E1; không tự bật E3/pilot. Rollback bằng active E1.

Không sửa `finetune_detection/`, không thay checkpoint hoặc thêm tính năng RAG. Notebook có thay đổi sẵn của người dùng được giữ ngoài commit. Không push hoặc merge.

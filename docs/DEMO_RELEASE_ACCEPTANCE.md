# Final demo acceptance — 25/11/2026, Asia/Bangkok

**FINAL DEMO VERIFIED — RELEASE READY**

Hai mô hình thật chạy CPU; workflow Chromium và các gate bảo mật đã đạt trên máy demo. Không huấn luyện, benchmark test set, đổi checkpoint/ngưỡng, sửa UI hoặc thêm tính năng. Pilot inactive không thuộc gate release hiện tại. [Runbook trình bày](DEMO_RUNBOOK.md).

## A. Repository và phạm vi freeze

- Repository: `https://github.com/Le-Chi-Bao/skin-lesion-detection-segmentation`, branch `main`.
- HEAD mã ứng dụng được kiểm chứng: **`c2e2f48e66745bfe9be97fdc5d1f4cda1982cf5e`**, đúng starting HEAD yêu cầu. Acceptance chỉ bổ sung tài liệu; mã ứng dụng, tests, active registry và weights giữ nguyên so với commit này.
- Notebook `finetune_detection/notebooks/Faster_RCNN_ISIC2018_Colab.ipynb` đã dirty từ trước; giữ nguyên, không stage/commit. Freeze không bao gồm thay đổi này.
- Trong lúc kiểm tra, `config/models.json` và `.gitignore` biến mất khỏi gốc; xuất hiện bản tương ứng ở `app/config/models.json` và `tests/.gitignore`. Git blob của từng cặp khớp HEAD (`7569dbe0…`, `cc121f83…`). Không tìm thấy thao tác di chuyển này trong mã/test; không kết luận nguyên nhân khi chưa có bằng chứng. Đã sao chép lại vị trí chuẩn mà không ghi đè tệp đang tồn tại.
- Bộ duyệt tự động chặn thao tác xóa bản dư (“blocked by policy”). Cách xử lý thay thế đã hoàn tất: so SHA-256 từng cặp rồi **lưu trữ không xóa** hai bản dư trong `runtime/acceptance-file-recovery/` (Git-ignored). Không còn bản dư trong app/tests, không tham gia runtime hoặc commit release.
- Commit tài liệu freeze được ghi trong báo cáo trả người dùng; tra lại bằng `git log -1 --format=fuller -- docs/DEMO_RELEASE_ACCEPTANCE.md`. Không push, không tạo public release hoặc tag.

## B. Artifact và loader

| Mô hình / phiên bản | Tệp thực tế | Bytes | SHA-256 thực tế | Kết quả |
| --- | --- | ---: | --- | --- |
| SegFormer MiT-B2 / `checkpoint-v1` | `model_store/segformer/real_only/best.pt` | 99002337 | `d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87` | Khớp; load/forward CPU đạt |
| Faster R-CNN ResNet50-FPN V2 / `E1-best-completed` | `model_store/detection/E1-final-inference.pt` | 173406019 | `eed8c28c2777005a2c6033692137f3cac8df8c68251ecb342ee4b8edf3068c9f` | Khớp; load/forward CPU đạt |

`Get-FileHash -Algorithm SHA256` đã chạy trên cả hai file thật. Registry gốc chọn đúng hai entry active, threshold 0,7/0,5; pilot inactive. Loader kiểm checksum trước deserialize, dùng `weights_only=True`, constructor không tải pretrained weights, `load_state_dict(strict=True)`, eval/inference mode; các hợp đồng này cũng được suite kiểm tra. Không random/fallback inference. UI có provenance/hash/format/device riêng từng mô hình.

## C. Inference và tài nguyên thực tế

Ảnh được dùng: fixture demo đã có từ WP2 **`model_store/wp2-verification/dermoscopy.jpg`**, 330×244, SHA-256 `30d963f7ee6112d6bace3edb96cb46298a57090191c67f7badb4c4827285a242`. Phiên browser thứ hai dùng bản xoay 90° trong bộ nhớ từ chính fixture này, 244×330. Không lấy ảnh bệnh nhân mới hoặc dữ liệu test-set.

| Ảnh | SegFormer thật | Faster R-CNN thật |
| --- | --- | --- |
| Gốc 330×244 | Mask mode L, giá trị {0,255}, **1.016 foreground pixel**, box `[138,89,181,124]` | **1 box** `[134.667618,87.285263,183.777649,126.461304]`, score **0.803465188** |
| Xoay 244×330 | Mask nhị phân, **1.021 foreground pixel**, box `[90,148,125,191]` | **1 box** `[86.071175,147.081329,126.474220,194.872177]`, score **0.938004792** |

Tất cả tọa độ nằm trong ảnh gốc tương ứng. Box mask và detection là hai nguồn riêng. Không dùng score làm xác suất bệnh.

- Lượt suite hai mô hình: **868,64 ms SegFormer**, **5218,56 ms Faster R-CNN**.
- Browser bổ sung: **5 forward thật mỗi mô hình**, gồm các lượt kết quả bị hủy bởi Reset/upload. SegFormer **789,69–925,37 ms**, detector **5158,56–5385,87 ms**. Lượt đầu từ click đến kết quả UI: **7,016 giây**. Không đặt latency target hoặc coi đây là benchmark.
- CPU: Intel Core i5-12500H, 12 core / 16 logical processor; `OMP_NUM_THREADS=4`; RAM nhìn thấy 16.388.972 KiB (~15,63 GiB). RAM trống dao động thấp, khoảng 0,5–1,6 GiB trong các lần quan sát; không OOM trong các gate đã chạy.
- Một mẫu đọc tiến trình server trong lượt browser: working set **1.213.100.032 bytes**, peak working set đến thời điểm đo **1.346.560.000 bytes**, private bytes **1.720.950.784**. Đây không phải đo leak dài hạn hoặc cực đại toàn bộ vòng đời.
- Có RTX 3050 Laptop GPU và Intel Iris Xe nhưng Torch **2.10.0+cpu**, `torch.cuda.is_available() == False`, `torch.version.cuda is None`. **CPU VERIFIED; CUDA NOT VERIFIED**. Không thay môi trường để bật CUDA.
- Server HTTP khởi động thành công; nhiều lần phân tích liên tiếp, hai phiên độc lập và hủy kết quả đang pending đều đạt. Tắt bằng đường `demo.close()` của ứng dụng: exit **0**, event `shutdown`, cổng 7860 không còn listener.

## D. Gradio end-to-end trên Chromium

Suite đã chạy app qua `python -m app.main`; kiểm tra bổ sung dùng cùng `app.main.main()`, registry thật và `launch_options`, chỉ bọc loader callbacks để ghi event và điều khiển kết thúc server. Không thay prediction bằng mock, không bỏ verification.

| Gate | Bằng chứng / kết quả |
| --- | --- |
| HTTP / research disclaimer | HTTP 200 tại `http://127.0.0.1:7860`; disclaimer nghiên cứu và không tự động chẩn đoán hiển thị. Không có claim chẩn đoán unsupported trong workflow được xem. |
| Upload / Original | Ảnh đúng, kích thước 330×244; phiên khác 244×330; Overlay base dùng cùng source của Original. |
| Analyze / Mask | Hai forward thật hoàn tất; mask mode L, {0,255}, kích thước gốc. |
| Overlay / opacity | 0%, 40%, 100% đúng CSS; không thêm `/queue/join` hoặc `predict` event. Suite còn so screenshot giữa các opacity. |
| Localization / metadata | Box vàng mask và đỏ detector hiển thị, score `0.803`; phiên bản, hai hash, format, device và inference time riêng hiển thị. Đã xem screenshot thực tế. |
| Overlay PNG | Download Chromium thành công; PNG 330×244; **mọi pixel** khớp công thức blend source RGB với mask tại opacity 40%. |
| Localization PNG | PNG 330×244; **mọi byte pixel** khớp ảnh Localization hiển thị; có cả màu vàng và đỏ. Phiên xoay xuất đúng 244×330. |
| Thay ảnh | Thay bằng ảnh xoay khác kích thước xóa mask/overlay/box cũ, vô hiệu hai nút và thu hồi export cũ HTTP 404. |
| Reset | Original/Mask/Overlay/Localization trống, Analyze/export vô hiệu; không hồi sinh kết quả cũ. |
| Pending / responsiveness | Reset sau khi real predict bắt đầu: khoảng **0,110 giây**; upload replacement: **0,187 giây**. Chờ cả hai forward cũ xong vẫn không nhận kết quả hết hiệu lực. |
| Phiên riêng biệt | Hai browser context, hai ảnh khác nhau, session/run khác; export chéo phiên HTTP 404. |
| Cleanup | Sau Reset cả hai phiên: không còn PNG dưới runtime sessions. Cache Gradio vẫn có entry chờ cơ chế TTL; không khẳng định Reset xóa ngay cache này. Không có pageerror trong lượt browser bổ sung. |

## E. Regression, security và lệnh đã chạy

Môi trường hiện có `.venv\Scripts\python.exe`, Python **3.12.5**; không tạo venv hay cài lại dependency.

| Dependency | Phiên bản thực tế |
| --- | --- |
| Torch / Torchvision | 2.10.0+cpu / 0.25.0+cpu |
| Gradio / Pillow | 6.32.0 / 12.3.0 |
| segmentation-models-pytorch / timm | 0.5.0 / 1.0.31 |
| NumPy / opencv-python-headless | 2.5.5 / 4.13.0.92 |
| pytest / Playwright / Chromium | 9.2.0 / 1.65.0 / 158.0.8127.18 |
| Starlette / Pydantic / filelock | 1.10.0 / 2.14.4 / 3.32.7 |

Lệnh regression cuối (PowerShell, tại gốc repo):

```powershell
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:LESION_MODEL_REGISTRY = 'config/models.json'
$env:LESION_SEGFORMER_DEVICE = 'cpu'
$env:LESION_DETECTION_DEVICE = 'cpu'
$env:WP2_SMOKE_CHECKPOINT = (Resolve-Path 'model_store/segformer/real_only/best.pt').Path
$env:WP2_SMOKE_IMAGE = (Resolve-Path 'model_store/wp2-verification/dermoscopy.jpg').Path
$env:WP2_SMOKE_DEVICE = 'cpu'
$env:CV_SMOKE_REGISTRY = (Resolve-Path 'config/models.json').Path
$testRun = Join-Path (Get-Location) ('runtime/pytest-demo-' + [guid]::NewGuid().ToString('N'))
.venv\Scripts\python.exe -m pytest -q -ra --tb=short --capture=tee-sys --basetemp "$testRun" --junitxml runtime/final-demo-results.xml tests 2>&1 | Tee-Object runtime/final-demo-tests.log
if ($LASTEXITCODE -ne 0) { throw 'Final demo regression tests failed' }
```

Kết quả **160 passed, 0 failed, 1 skipped**, **101,83 giây**, exit **0**. Skip duy nhất: thiếu pilot `E1_step07200_PILOT_UNFINISHED.pt`, SHA-256 `327c2281b69a9591939c1abe883cb4337fecc9983fdf8746d26c7b5a32011141`; entry inactive và không phải release requirement. Không skip gate real-model hoặc browser bắt buộc. Không nới assertion/xóa test.

Lượt đầu trong khi registry bị chuyển: **153 pass, 6 fail, 2 skip** trong 69,84 giây. Các fail là thiếu `config/models.json`; hai ca dual-model chưa thực thi do Resolve-Path không lấy được registry. Lượt này **không được coi đạt**; log/XML được giữ riêng `runtime/final-demo-initial-tests.log`, `runtime/final-demo-initial-results.xml`. Sau khôi phục chạy lại đầy đủ như trên, không che lỗi lượt đầu.

Kiểm tra bổ sung tại máy demo (script và bằng chứng cục bộ Git-ignored):

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
.venv\Scripts\python.exe -u runtime/demo_acceptance_browser.py 2>&1 | Tee-Object runtime/final-demo-browser.log
.venv\Scripts\python.exe -u runtime/demo_acceptance_modes.py 2>&1 | Tee-Object runtime/final-demo-modes.log
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m compileall -q app
git diff --check
```

Browser bổ sung: **12 nhóm kiểm tra đạt**, exit 0. Bằng chứng `runtime/demo-browser-20261125-181707/evidence.json`, `inference-events.jsonl`, `server.log`, screenshot chỉ chứa fixture demo. Script này là công cụ đo cục bộ, không phải dependency của bản release; khi checkout máy khác, dùng suite tracked phía trên và kịch bản runbook.

Modes thật: **4/4 đạt**, registry tạm, không sửa registry active:

| Chế độ | Aggregate | Forward thành công |
| --- | --- | --- |
| SegFormer-only | completed | SegFormer 927,69 ms |
| Detection-only | completed | Faster R-CNN 5828,29 ms |
| Detection trỏ file thiếu | partial | SegFormer 904,81 ms; lỗi detector được cô lập |
| Segmentation trỏ file thiếu | partial | Faster R-CNN 5495,47 ms; lỗi segmenter được cô lập |

Bằng chứng modes: `runtime/demo-modes-20261125-181825/evidence.json`. Suite cũng kiểm tra lỗi inference độc lập, output không hợp lệ, empty-mask/zero-detection contracts, reset/upload pending, TTL, proof/cookie/run/action, stale export, chọn active và rollback qua cấu hình. Selection E2/rollback thử bằng registry tạm, không khẳng định đã chạy E2/E3 thật.

Cả `/gradio_api/file=` và `/file=` trả **403** cho trọng số; export chéo phiên/cũ trả **404**. Cơ chế private-session và checksum giữ nguyên. `pip check`: **No broken requirements found**; compileall và diff whitespace check đạt. Không chạy tests trong training workspace.

## F. Sửa lỗi và commit

Sửa lỗi filesystem đã chứng minh: phục hồi registry và ignore file đúng vị trí bằng bản sao nguyên nội dung; lưu hai bản dư vào runtime để giữ bằng chứng, không cần sửa production code. HEAD code vẫn là `c2e2f48e66745bfe9be97fdc5d1f4cda1982cf5e`. Commit freeze chỉ gồm runbook, báo cáo acceptance và liên kết/chú thích tài liệu. Notebook, weights, runtime, screenshot và export không được stage.

## G. Vận hành demo

Dùng [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md): lệnh CPU, URL, kịch bản năm phút, phục hồi cổng/checkpoint/hash/RAM/CUDA/download/initialization và cách đăng ký E2/E3/rollback. Không tải lại artifact khi file hiện có đúng hash. Bản release dùng E1 hoàn tất; việc thiếu pilot không chặn demo này.

## H. Giới hạn, checklist và đề xuất

- CPU và workflow browser đã được kiểm chứng trên máy này; CUDA **NOT VERIFIED**, không là gate bắt buộc vì môi trường hiện tại là CPU.
- Mô hình nghiên cứu, chưa thẩm định lâm sàng. Không đo độ chính xác/benchmark test set, không suy luận hiệu quả y khoa từ hai ảnh demo.
- Inference CPU mất vài giây; chưa stress test dài hạn hoặc chứng minh không leak. Máy có ít RAM trống; đóng ứng dụng nặng trước buổi trình bày.
- Reset dọn dữ liệu application session; cache upload Gradio có vòng đời TTL riêng. Export tải về do browser quản lý.
- Nguồn của thao tác chuyển hai tệp chưa xác định; vị trí chuẩn và nội dung đã phục hồi, có rerun chứng minh. Bản dư được lưu dưới runtime, không thuộc release; working tree ngoài tài liệu freeze chỉ còn notebook đã dirty từ trước.

Checklist freeze:

- [x] HEAD mã ứng dụng được ghi chính xác; chỉ stage tài liệu cần thiết.
- [x] Hai artifact/hash/threshold active xác minh, không thay weights.
- [x] Real CPU dual/single modes, failure isolation, consecutive runs đạt.
- [x] Chromium upload → Analyze → views → opacity → PNG → đổi ảnh → Reset đạt.
- [x] Hai phiên, export authorization, file routes, pending edits, cleanup và shutdown đạt.
- [x] Suite đầy đủ 160 pass / 0 fail / 1 optional skip; dependency/compile/whitespace checks đạt.
- [x] Runbook, dependency versions, timing và giới hạn được ghi.
- [x] Không commit artifact/runtime/private data; không push/public release.

Đề xuất tag sau khi xem commit tài liệu: **`cv-mvp-demo-v1.0`**. Chưa tạo tag. Không còn P0 blocker cho demo CPU với hai artifact đã xác minh.

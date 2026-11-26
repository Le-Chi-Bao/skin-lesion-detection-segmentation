# WP-CV4 — E1/E2/E3 verification and E2 integration

Ngày thực hiện: 26/11/2026 (Asia/Bangkok). Phạm vi: inference exports thật,
registry, Main System Gradio, hồi quy CV và Medical RAG WP6A/WP6A.1.

## 1. Mục tiêu và nguồn Git

Nhánh `feat/cv-e123-e2-integration` bắt đầu tại `2dea7ea`, trùng `origin/main`
và HEAD của `feat/rag-wp6a1`; local `main` còn ở `b8600b4`. Không reset/rebase
thay đổi người dùng. Notebook `finetune_detection/notebooks/Faster_RCNN_ISIC2018_Colab.ipynb`
đã modified và thư mục `finetune_detection/FasterRCNN_ISIC2018/` đã untracked
trước tác vụ: giữ nguyên, không stage/commit. Không push/merge.

E2 được chọn theo Validation trước khi tích hợp. Không training, không chạy lại
Official Test, không tải ảnh hoặc dataset, không chỉnh threshold/preprocessing.

## 2. Artifact thực tế và phương pháp xác minh

Cả ba file tồn tại, đọc được, mỗi file **173,413,181 bytes**. SHA-256 streaming:

| Artifact/version | SHA-256 | min/max size | Best epoch | Kết quả CPU |
| --- | --- | --- | --- | --- |
| E1-93e17545c5b84d78b46a9040cc7a8942 | `9232b4cbb8051c8057e08c28f277b9c219f814e71ed0af7353859b74d4fdabfc` | 800 / 1333 | 9 | PASS |
| E2-a2301ed5919a4b9089dc3cc73af65e70 | `9c083aadba4b47cf4b5d5ff794cd23856977ffa65dc35bf5217cfa1e27b4a78c` | 800 / 1333 | 6 | PASS |
| E3-c3e92b0e29394ae3a4f9e3108b6ece20 | `5068b816d5d1381ef4601ff622da8101b2fd9d09985cdd693c26e77a073e14e3` | 1024 / 1600 | 6 | PASS |

Đường dẫn: `model_store/detection/<version>.pt`. Run ID là phần sau `E1-`,
`E2-`, `E3-`. Không đổi tên, xóa, ghi đè hoặc commit weights.

`app.cv.verify_detection` thực hiện:

1. Streaming SHA-256, đối chiếu registry trước deserialization.
2. `torch.load(weights_only=True, map_location="cpu")`; không có unsafe fallback.
3. Kiểm tra format `isic-fasterrcnn-inference-v1`, architecture
   `fasterrcnn_resnet50_fpn_v2`, labels `{0: __background__, 1: skin_lesion}`,
   đủ model_config và preprocessing khớp loader Main System.
4. Đối chiếu experiment, run ID theo filename, best epoch 9/6/6,
   selection metric `validation map_50_95`, manifest hash và giá trị AP hữu hạn.
5. Kiểm tra **404 tensors**/artifact đều hữu hạn; dựng model không tải pretrained
   weights và nạp `strict=True`, eval mode, đúng runtime transform min/max.
6. Forward CPU thật; kiểm tra raw boxes/labels/scores, dtype, shape, hữu hạn,
   label=1, tọa độ trong ảnh gốc và kết quả lọc đúng `score >= 0.5`.
7. Băm lại file và kiểm tra size/mtime trong `finally`, cả khi kiểm tra thất bại.

Cả ba khai báo cùng training manifest:
`b1d84af32aa58a89e3566c21e888fdfd38df59c940325e182b6dd2bdd16b8c08`.
Đây là đối chiếu giá trị provenance; không có nội dung manifest để tái tạo hash.

Preprocessing giữ nguyên: RGB float32 CHW / [0,1], ImageNet mean/std,
GeneralizedRCNNTransform, box score threshold nội bộ 0.0, NMS 0.5,
maxDets=100, box xyxy source pixels với cạnh phải/dưới loại trừ.
`trainable_backbone_layers=3`; operating threshold **0.5** cho cả ba.

Colab khai báo PyTorch 2.14.1+cu130 / Torchvision 0.29.1+cu130 trên Tesla T4;
máy local dùng PyTorch **2.10.0+cpu**, Torchvision **0.25.0+cpu**, Gradio **6.32.0**.
Strict load và forward local đạt dù môi trường khác nhau; CUDA local không khả dụng.

## 3. Validation và quyết định chọn E2

Các số sau là **Validation Colab**, đối chiếu với `E*/logs/history.json` tại
best epoch và `E*/completed.json`; không phải accuracy được đo lại trên Main System.

| Metric | E1 | E2 | E3 |
| --- | ---: | ---: | ---: |
| Best epoch | 9 | 6 | 6 |
| mAP@50 | 0.9561 | 0.9665 | 0.8327 |
| mAP@50:95 | 0.6893 | 0.7022 | 0.5363 |
| Recall@0.5 | 0.9461 | 0.9422 | 0.8266 |
| Mean matched IoU | 0.8601 | 0.8602 | 0.8296 |
| FP/image | 0.2023 | 0.1464 | 0.4509 |
| ms/image (Colab) | 161.7133 | 175.8862 | 230.9590 |

E2 có validation mAP@50:95 cao nhất và FP/image thấp nhất, recall gần E1
(E1 vẫn có recall cao hơn). Giữ quyết định đã khóa, không dùng Test hoặc ảnh
smoke để chọn lại checkpoint, điều chỉnh threshold hoặc suy ra độ chính xác lâm sàng.

## 4. Model Registry

Schema version **1**, không đổi schema. Các entry có đủ module/model_id/version,
checkpoint_path/SHA-256, architecture/format, pilot/active, threshold, min/max,
preprocessing. Registry từ chối nhiều hơn một active cho mỗi module.

| Entry | Active | Threshold | Vai trò |
| --- | --- | ---: | --- |
| SegFormer checkpoint-v1 | true | 0.7 | Segmentation hiện tại |
| E2-a2301ed5919a4b9089dc3cc73af65e70 | true | 0.5 | Detection triển khai |
| E1-93e17545c5b84d78b46a9040cc7a8942 | false | 0.5 | Rollback đã kiểm tra |
| E3-c3e92b0e29394ae3a4f9e3108b6ece20 | false | 0.5 | Thí nghiệm độc lập |
| E1-step7200-pilot | false | 0.5 | Giữ lịch sử; artifact thiếu, không thuộc gate WP-CV4 |

SegFormer giữ nguyên đường dẫn `model_store/segformer/real_only/best.pt`, hash
`d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87`,
weights, preprocessing và threshold. E2 chỉ được activate sau khi candidate
registry trong runtime đã qua xác minh và CPU forward thật.

Reference `model_store/detection/E1-final-inference.pt` cũ không còn tồn tại.
Entry E1 cũ được thay bằng version theo run ID và hash thực tế mới, không gán
checksum `eed8c28c...` cũ cho export mới. File cũ không bị tạo giả hoặc ghi đè.
Startup log bổ sung SHA-256 đã chọn bên cạnh version/device/checkpoint.

## 5. Real inference độc lập

Ảnh fixture đã có: `model_store/wp2-verification/dermoscopy.jpg`, **330×244**,
SHA-256 `30d963f7ee6112d6bace3edb96cb46298a57090191c67f7badb4c4827285a242`.
Không dùng Official Test cho smoke. Chỉ nạp một detector mỗi lần.

Lượt CLI xác minh trước activation (CPU):

| Artifact | Raw predictions | Boxes >=0.5 | Box xyxy trên ảnh gốc | Score | Forward ms |
| --- | ---: | ---: | --- | ---: | ---: |
| E1 | 100 | 1 | [134.668, 87.285, 183.778, 126.461] | 0.803465 | 5324.31 |
| E2 | 100 | 1 | [133.888, 86.300, 184.329, 127.086] | 0.652013 | 5040.93 |
| E3 | 100 | 0 | — | — | 6904.00 |

Không NaN/Inf, incompatible state_dict hoặc box vượt ảnh. E3 zero-detection
ở ngưỡng đã khóa là output hợp lệ. Không bắt buộc phát hiện được trên mọi ảnh.
Thời gian một lượt CPU không phải benchmark, không so trực tiếp với Colab GPU.
Evidence ban đầu: `runtime/wp-cv4/artifact-verification.json` (ignored).

## 6. Main System và Gradio

Chromium đã chạy thành công cả `python main.py` và `python -m app.main`, không
thay UI. Log xác nhận SegFormer READY và E2 READY trên CPU, đúng SHA/version.
Kiểm tra trực quan screenshot thấy box E2 đỏ score 0.652 và box từ mask vàng.

- Tải ảnh hợp lệ và Phân tích chạy cả hai model; Original/Mask/Overlay/Localization
  đều có ảnh, opacity 65% hoạt động, details đúng version/hash/format E2.
- Overlay và Localization tải PNG đúng kích thước ảnh gốc và tên theo run hiện tại;
  PNG Localization có pixel màu detection thật.
- Hai context độc lập phân tích ảnh 330×244 và bản xoay 244×330. PNG từng context
  giữ đúng kích thước của phiên đó, run khác nhau; export chéo phiên trả 404.
- Thay ảnh/Reset thu hồi export cũ, xóa views và khóa nút phù hợp. Truy cập trực tiếp
  các checkpoint qua Gradio bị chặn 403.
- Ma trận orchestrator lỗi detection/segmentation độc lập, mask persistence,
  stale inference và export proof/TTL đều đạt trong full regression.

Evidence ignored: `runtime/wp-cv4/full-regression/test_real_dual_model_browser_u{0,1}/`
và `runtime/wp-cv4/final-acceptance/test_real_dual_model_browser_u{0,1}/` có
`server.log`, `dual-model-localization.png`, `dual-model-reset.png`.

```powershell
.venv\Scripts\Activate.ps1
$env:LESION_SEGFORMER_DEVICE = 'cpu'
$env:LESION_DETECTION_DEVICE = 'cpu'
python main.py
```

Mở `http://127.0.0.1:7860` → Computer Vision → tải JPEG/PNG hợp lệ → Phân tích.
Xem Original/Mask/Overlay/Localization, Chi tiết mô hình và tải Overlay/Localization
PNG. Runtime của mỗi tiến trình phải riêng. Startup không tải weights.

## 7. Regression và tái kiểm chứng

Lệnh dùng lại các artifact hiện có, không tải dữ liệu:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUTF8 = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CV_SMOKE_REGISTRY = (Resolve-Path config/models.json).Path
$env:WP2_SMOKE_CHECKPOINT = (Resolve-Path model_store/segformer/real_only/best.pt).Path
$env:WP2_SMOKE_IMAGE = (Resolve-Path model_store/wp2-verification/dermoscopy.jpg).Path
$env:WP2_SMOKE_DEVICE = 'cpu'
$env:WP_CV4_REAL = '1'
.venv\Scripts\python.exe -m pytest tests -q -ra --basetemp runtime/wp-cv4-rerun --junitxml runtime/wp-cv4-rerun.xml
.venv\Scripts\python.exe -m app.cv.verify_detection --image model_store/wp2-verification/dermoscopy.jpg
```

Chọn basetemp mới vì pytest dọn thư mục basetemp khi chạy lại.
Nếu bỏ `WP_CV4_REAL`/các biến smoke, ca real-model sẽ skip; không coi đó là đã xác minh.

Lượt ban đầu: **7 failed, 52 passed, 7 skipped**. Bảy lỗi đều là assertion/fixture
mặc định cũ hardcode `E1-best-completed` trong startup/selection tests.
Giữ nguyên các tình huống và assertion E1 đó bằng fixture lịch sử
`tests/fixtures/cv_registry_e1.json` lấy nguyên từ Git base. Không xóa test,
xfail, nới threshold/output contract hoặc sửa code để báo nhầm E1.
Browser tests lấy version từ registry thực sự và bổ sung assertion hash.
Các test WP-CV4 mới khóa version/hash E2, kiểm tra provenance, resize,
checksum trước deserialization, registry chỉ một active, dữ liệu rollback giữ nguyên,
và real CPU load/inference E1/E2/E3. RAG source/tests không sửa.

Kết quả trên cây code bàn giao (full suite và các kiểm tra bổ sung sau đó):

| Lượt kiểm thử | PASS | FAIL | SKIP | Thời gian |
| --- | ---: | ---: | ---: | ---: |
| Full `tests/`, real E1/E2/E3, SegFormer và Chromium bật | 371 | 0 | 3 | 198.17 s |
| Bổ sung rollback E2→E1 thật + browser hai context có kết quả, cả hai entrypoint | 3 | 0 | 0 | 83.38 s |
| Verifier unit regression, gồm phát hiện file bị writer khác đổi khi load thất bại | 14 | 0 | 0 | 2.08 s |
| Medical RAG trong full suite (tập con, không cộng vào tổng) | 175 | 0 | 2 | — |

Các lượt bổ sung có test chạy trùng full suite; không cộng số PASS thành số ca duy nhất.
Log/JUnit: `runtime/wp-cv4/{full-regression,final-acceptance,verifier-regression}.{log,xml}`.
`compileall` app/new tests, `git diff --check`, `pip check` đều đạt.
Browser supplemental có hai Pillow DeprecationWarning trong assertion kiểm tra pixel;
không lỗi pipeline.

Ba skip của full suite:

1. Artifact pilot E1 step7200 thiếu; entry inactive, không thuộc artifacts E1/E2/E3 mới.
2. Ca symlink ingestion thiếu quyền tạo symlink trên Windows.
3. Real `multilingual-e5-base` thiếu weights local: `embedding_model_unavailable_local`.
   **BLOCKED / NOT VERIFIED** cho real E5; không suy ra Dense/Hybrid thật đã được xác minh.

WP6A/WP6A.1 BM25, chunking, ingestion, config validation, retrieval contracts,
hybrid/RRF unit tests, quyền corpus/citation và Medical RAG browser/session isolation
đều đạt. Không sửa RAG, Cross-Encoder, OpenRouter hoặc tải thêm model.

## 8. Rollback E2 → E1

Dừng server. Bản E1 mới đã strict-load và forward CPU thật, không cần export lại.
Chỉ đổi active flags, không đổi hash/path/threshold. Script sau giữ nguyên mọi
entry và kiểm tra version đích tồn tại trước khi ghi:

```powershell
@'
import json
from pathlib import Path
p = Path('config/models.json')
data = json.loads(p.read_text(encoding='utf-8'))
target = 'E1-93e17545c5b84d78b46a9040cc7a8942'
matches = [r for r in data['models'] if r['module'] == 'detection' and r['version'] == target and not r['research_pilot']]
assert len(matches) == 1
for row in data['models']:
    if row['module'] == 'detection':
        row['active'] = row['version'] == target
assert sum(r['active'] for r in data['models'] if r['module'] == 'detection') == 1
p.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
'@ | .venv\Scripts\python.exe -
.venv\Scripts\python.exe -m app.cv.verify_detection --experiment E1 --image model_store/wp2-verification/dermoscopy.jpg
.venv\Scripts\python.exe main.py
```

Kiểm tra log READY và hash `9232b4...dabfc`. Quay lại E2: dùng cùng script,
đổi target thành `E2-a2301ed5919a4b9089dc3cc73af65e70`, xác minh E2 rồi restart.
Không dùng `git checkout` cả registry vì có thể mất cấu hình người dùng khác.

Ca rollback bổ sung đã thật sự chọn E2 qua `ModelRegistry.create`, nạp CPU và
forward, sau đó chuyển một registry tạm sang E1, nạp và forward lại; mỗi lần
chỉ giữ một detector. Cả hai trả một box đúng như bảng smoke. Production registry
không đổi trong thử nghiệm và vẫn chọn E2. Evidence:
`runtime/wp-cv4/final-acceptance/test_real_registry_e2_to_e1_ro0/rollback-evidence.json`.

## 9. Official Test — báo cáo có sẵn, không dùng model selection

`runtime/experiments/FasterRCNN_ISIC2018/` **không tồn tại**. Các đường dẫn yêu cầu
`comparison.md`, `deployment_selection.json`, `train_manifest.json` và
`official_test/{E1,E2,E3}/official_test_metrics.json` đều thiếu ở vị trí đó.

Nguồn thay thế có sẵn, được đọc nhưng không sửa/commit:
`finetune_detection/FasterRCNN_ISIC2018/official_test/{E1,E2,E3}/` có cả
`official_test_metrics.json` và `official_test_protocol.json`.
`comparison.md`, `deployment_selection.json`, `train_manifest.json` vẫn thiếu
ở thư mục thay thế; không có nội dung train/test manifest để kiểm tra dataset
membership, leakage hoặc tái tạo digest. Không tuyên bố tái xác minh accuracy.

Protocol của ba metrics file khớp sidecar protocol tương ứng; artifact hash,
selection metadata và operating threshold khớp artifacts/registry. Tất cả ghi
split `ISIC 2018 Task 1 official test`, **1000 images**, chung test manifest hash:
`1ce64e8409d10049f68570f973a0566f2e38712869d8b0755c1bac730a2eae28`.
Evaluation: COCO bbox all areas, maxDets=100, score-ranked AP; operating IoU=0.50.

| Official Test metric (reported Colab) | E1 | E2 | E3 |
| --- | ---: | ---: | ---: |
| mAP@50 | 0.9590 | 0.9646 | 0.8598 |
| mAP@50:95 | 0.5986 | 0.6199 | 0.5291 |
| Recall@0.5 | 0.9500 | 0.9470 | 0.8440 |
| Mean matched IoU | 0.8235 | 0.8272 | 0.8149 |
| FP/image | 0.1730 | 0.1230 | 0.3150 |
| ms/image (Colab Tesla T4) | 152.2969 | 150.9279 | 215.3324 |

SHA-256 các metrics report đã đọc, phục vụ nhận diện đúng snapshot:

- E1: `42c8e23c8be4988b31b7f8e6625d585d05cab87d9501fcbd8d4d7ec6eb074f93`
- E2: `c7cebb2a8a7dcb4336fe450a3445ac26be3ae4e4e288efc398013bba91c2ac0f`
- E3: `f70e84943f10f956f7ac4d667554377dc648a722da4216718837fc42f96b6b7a`

Evidence kiểm tra nhất quán: `runtime/wp-cv4/results-inspection.json` (ignored).
Không chạy Test tự động và không dùng các số Test trên để đổi E2/ngưỡng.

## 10. Giới hạn và kết quả cuối

Ảnh smoke chỉ xác minh khả năng nạp/inference, không chứng minh accuracy,
generalization hoặc an toàn lâm sàng. Missing dataset manifests giới hạn kiểm tra
provenance báo cáo. CUDA không khả dụng; pilot cũ không được cung cấp.
Medical RAG real E5 phụ thuộc local assets như đã ghi trong WP6A.1; không tải
weights hoặc thay đổi RAG, Cross-Encoder, OpenRouter trong tác vụ này.

**CV integration acceptance: PASS** — ba artifacts đã kiểm chứng, E2 active,
SegFormer+E2 chạy qua Main System, một detection active, E1 rollback thực tế đạt.
Các checkpoint và fixture ảnh giữ nguyên hash sau toàn bộ kiểm tra.

**Bàn giao tổng thể: PARTIAL về kiểm chứng toàn bộ môi trường**, do real E5 còn
BLOCKED và ca Windows symlink SKIPPED. Không tuyên bố toàn bộ CV+Medical RAG
real-model stack VERIFIED. Các thiếu sót này không chặn E2 inference integration;
accuracy Official Test vẫn chỉ là kết quả báo cáo Colab với giới hạn manifest ở mục 9.

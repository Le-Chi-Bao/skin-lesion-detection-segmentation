# Phase 2C — Đưa phân vùng thử nghiệm lên Gradio

Ngày kiểm chứng: **16/11/2026**, múi giờ Asia/Bangkok.

## Mục tiêu và phạm vi

Đưa segmentation đã chạy thật ở Phase 2B lên Gradio, giữ chức năng tải ảnh và
xem trước. Kết quả được gọi là **Phân vùng thử nghiệm**, không phải chẩn đoán.
Đầu vào triển khai là [báo cáo 2B](../outputs/phase2b/report.md), `app.py` và
`teledermoscopy/vision/segmentation.py` hiện có.

Không train, không thêm detection, classification hoặc RAG, không sửa crawler,
không tải lại checkpoint/ảnh và không push trong Phase 2C.

## Các file thay đổi

| File | Thay đổi |
|---|---|
| `app.py` | Giữ `preview_image`; thêm nút Phân vùng, vùng mask/overlay, thiết bị và xử lý đổi/xóa ảnh |
| `teledermoscopy/gradio_segmentation.py` | Controller nối Gradio với predictor 2B, cache mô hình, khóa inference, xử lý lỗi và bỏ kết quả cũ |
| `tests/test_gradio_segmentation.py` | 9 unit test mới cho callback, cache, lỗi và trạng thái ảnh |
| `README.md` | Cập nhật luồng UI, môi trường cần dùng và dẫn tới tài liệu này |
| `docs/phase-2c.md` | Mục tiêu, cách chạy, bằng chứng và giới hạn kiểm chứng |

Logic CLI và `teledermoscopy/vision/segmentation.py` của 2B giữ nguyên.
Các ảnh, screenshot, JSON kiểm chứng và script thao tác trình duyệt được lưu local
trong `outputs/phase2c/`; chúng không thuộc kết quả chẩn đoán hoặc đánh giá mô hình.

## Luồng từ nút bấm đến kết quả

1. Mỗi phiên Gradio có `gr.State` chứa revision của ảnh. Đổi/xóa file làm tăng
   revision và xóa ảnh đang hiển thị, mask, overlay và thiết bị cũ.
2. **Xem ảnh** gọi lại `preview_image` để hiển thị ảnh gốc. Nút này cũng xóa kết
   quả phân vùng cũ, không gọi predictor và không nạp weights.
3. **Phân vùng** gọi generator `SegmentationController.segment`. Callback trả
   trạng thái rỗng trước khi chạy, vì vậy kết quả trước không tồn tại trong lúc
   chờ kết quả mới. Chưa chọn ảnh hoặc ảnh hỏng sẽ trả lỗi tiếng Việt.
4. Controller dùng `read_rgb_image` của 2B, kiểm tra checkpoint local, rồi vào khóa
   dùng chung. Chỉ khi chưa có predictor mới tạo `SegmentationPredictor(...,
   device="auto")`. Predictor được giữ lại cho các lượt tiếp theo trong tiến
   trình app. Mỗi app có một cache; kết quả và revision thuộc từng phiên.
5. `SegmentationPredictor.predict(rgb)` thực hiện toàn bộ inference 2B. Controller
   chỉ chuyển các mảng ảnh trả về thành PIL để Gradio hiển thị, gọi lại
   `result.overlay()` của 2B; không sao chép preprocessing/postprocessing.
6. UI hiển thị ảnh gốc, mask PNG một kênh, overlay PNG RGB và thiết bị thực tế.
   Nếu revision đã đổi trong khi xử lý, kết quả đó bị bỏ. Event đổi ảnh/xem trước
   cũng hủy generator đang chờ; inference đang chạy có thể hoàn tất ở backend
   nhưng kết quả của ảnh cũ không được trả lại UI.
7. Lỗi mới luôn trả mask/overlay/device rỗng. CUDA hết VRAM có thông báo riêng;
   không tự chuyển sang CPU hoặc tạo mask thay thế. Checkpoint thiếu vẫn được
   báo lỗi kể cả khi predictor đã được cache.

Gradio giới hạn event phân vùng ở concurrency 1; controller có thêm khóa bảo vệ
cache và model khi được gọi từ các phiên khác nhau. Mở app chỉ import thư viện và
dựng giao diện, không khởi tạo model hoặc tải weights.

## Cấu hình được giữ nguyên từ 2B

- Local checkpoint: `checkpoints/segformer-gate2/real_only/best.pt`.
- SMPWrapper bọc SegFormer `mit_b2`, 3 kênh vào, 1 kênh ra, `encoder_weights=None`.
- Nạp checkpoint trên CPU với `weights_only=True`, `load_state_dict(strict=True)`.
- RGB, resize 512×512 bằng INTER_AREA; chuẩn hóa ImageNet theo 2B.
- `eval()` và `inference_mode()`, float32; sigmoid logits.
- Resize **xác suất** về kích thước ảnh gốc bằng INTER_LINEAR, rồi threshold **>= 0.5**.
- `auto` chọn CUDA khi khả dụng, nếu không chọn CPU ngay từ đầu. Lỗi inference
  không kích hoạt fallback hoặc thay đổi ngưỡng.
- App chính chỉ lắng nghe `127.0.0.1:7860`, `share=False`, analytics tắt.

Môi trường tái sử dụng: Python 3.12.5, Gradio 6.32.0, torch 2.11.0+cu126,
torchvision 0.26.0+cu126, SMP 0.5.0, timm 1.0.31, NumPy 2.5.4,
Pillow 13.0.0, opencv-python-headless 4.13.0.92. Không cài thêm dependency ở 2C.

## Lệnh chạy lại trên Windows PowerShell

Chạy tại thư mục gốc repo; dùng `.venv` và tài nguyên local của Phase 2B:

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở http://127.0.0.1:7860. Chọn `data\samples\ISIC_0012169.jpg`, nhấn **Xem ảnh**
hoặc **Phân vùng**. Khi ô upload đã có file, bấm **Clear** (dấu ×) trước khi chọn
file khác. Nhấn `Ctrl+C` trong terminal để dừng app.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Nếu dựng môi trường mới thay vì dùng lại 2B:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

Không chạy lệnh cài lại khi môi trường 2B vẫn dùng được. Dependency không kèm
checkpoint/ảnh; cần đặt lại các tài nguyên đã xác minh đúng đường dẫn local.

## Kết quả kiểm thử tự động

**18/18 test đạt**, gồm toàn bộ 9 test của Bước 1/Phase 2B và 9 test controller mới.

- Chưa chọn ảnh, ảnh không hợp lệ và checkpoint thiếu: báo lỗi, trả output rỗng.
- Thiếu checkpoint dù predictor đã cache: vẫn báo lỗi, không dùng model để che lỗi.
- Hết VRAM: **giả lập bằng mock** `torch.cuda.OutOfMemoryError`, kiểm tra thông báo
  và không khởi tạo predictor CPU thay thế.
- Mở app không gọi constructor model; hai callback phân vùng dùng một predictor.
- Đổi/xóa ảnh hoặc xem trước lại làm mất hiệu lực revision và xóa kết quả cũ.
- Đổi revision trong lúc predict chạy làm callback bỏ kết quả.
- Các test CLI, RGB, normalization, threshold và preview cũ vẫn đạt.

Các kết quả mock chỉ dùng trong unit test và không ghi ra mask/overlay để trình
bày như inference thật. Lần chạy unittest trên Windows có `ResourceWarning:
unclosed event loop` khi dựng Gradio Blocks; tất cả test vẫn hoàn tất với `OK`.

## Inference thật và so sánh với CLI 2B

Trong phiên UI kiểm chứng cuối, đã hoàn tất **3 lượt phân vùng thật** bằng cách
bấm nút trên Edge headless qua Playwright. Ứng dụng dùng đúng `build_demo` và
controller thật, không mock model. Hai lượt đầu được tải mask/overlay từ URL ảnh
Gradio thực sự đang hiển thị để so sánh với output CLI 2B đã lưu.

| Hạng mục | Kết quả đo thực tế |
|---|---|
| Ảnh | `ISIC_0012169.jpg` |
| Thiết bị | `cuda:0 — NVIDIA GeForce RTX 3050 Laptop GPU` |
| Kích thước ảnh gốc/mask/overlay | 3008×2000 |
| Giá trị mask | 0 và 255, PNG mode L |
| Pixel thuộc mask | 801.770 / 6.016.000 = 13,327293882978725% |
| Mask UI lượt 1 so với CLI 2B | 0 pixel khác; sai khác giá trị lớn nhất 0 |
| Mask UI lượt 2 so với CLI 2B | 0 pixel khác; sai khác giá trị lớn nhất 0 |
| Overlay UI so với CLI, cả hai lượt | 0 pixel khác; sai khác giá trị lớn nhất 0 |
| Nạp model | 0 lần sau mở app/xem trước; 1 lần sau phân vùng; cùng predictor ở lượt tiếp theo |

SHA-256 hai mask UI tải về đều khớp CLI:

```text
1276e69204efe607b13083589bd416f2b15fb455f7842b9ce3d172889b1a0d4e
```

Không có sai khác cần điều chỉnh pipeline. Kết quả này xác minh việc tích hợp,
không chứng minh độ chính xác phân đoạn. Không tính Dice/IoU hoặc dùng nhãn bệnh
để diễn giải mask. Giấy phép và giới hạn split ảnh mẫu giữ như báo cáo 2A.1/2B.

## Các thao tác UI đã kiểm tra

Đã thực sự thao tác trên DOM của Edge headless, không chỉ gọi callback Python:

- Mở trang, xem trước và phân vùng khi chưa chọn file.
- Tải file `.png` hỏng rồi nhấn Phân vùng: lỗi rõ, không nạp model.
- Tải ảnh mẫu, xem trước: ảnh gốc hiển thị, cache model vẫn chưa nạp.
- Phân vùng nhiều lần, xem đủ ảnh gốc/mask/overlay/device; kiểm tra dùng lại model.
- Đổi ảnh rồi xem trước: mask, overlay và thiết bị cũ bị xóa.
- Xóa file sau một lượt thành công rồi bấm Phân vùng: không còn kết quả cũ.
- Đổi file khi event phân vùng đang chờ/chạy: kết quả cũ không xuất hiện trở lại.
- Thiếu checkpoint: mở app phụ ở `127.0.0.1:7861`, dùng cùng code nhưng controller
  trỏ tới `outputs/phase2c/missing-best.pt` không tồn tại. UI báo lỗi và không có
  mask/overlay; không di chuyển hoặc sửa checkpoint đã xác minh.
- Không ghi nhận lỗi JavaScript trong phiên kiểm chứng cuối.

Bằng chứng local:

- `outputs/phase2c/ui_verification.json`: checklist UI và telemetry cache.
- `outputs/phase2c/mask_comparison.json`: sai khác pixel và checksum.
- `outputs/phase2c/ui_mask_1.png`, `ui_mask_2.png`, `ui_overlay_1.png`, `ui_overlay_2.png`.
- `outputs/phase2c/ui_success.png`, `ui_missing_checkpoint.png`: screenshot UI.
- `outputs/phase2c/serve_ui_verification.py`, `browser_verify.py`: launcher/driver kiểm chứng.

## Phần chưa xác minh

- Không cố tình gây hết VRAM thật; nhánh lỗi này chỉ được kiểm tra bằng mock.
- Chưa chạy inference UI trên CPU vì `auto` đã chọn CUDA trên máy hiện tại.
- Chưa kiểm thử tải nhiều người dùng; kiểm chứng này là smoke test local một ảnh.

Đã dừng sau Phase 2C. Không train, không bổ sung chức năng ngoài phạm vi hoặc push.

# System architecture — Phase 1

## Phạm vi và quyết định thiết kế

Đây là nền tảng cho đồ án sinh viên: module nhỏ có type hints, YAML + dataclass,
NumPy/OpenCV cho ảnh, `Protocol` cho hợp đồng model. Không thêm service layer,
dependency injection framework hoặc model registry. Giữ các thư mục yêu cầu;
data loader hoàn chỉnh và metrics được triển khai ở phase sau.

`src` là package Python được cài editable từ root qua `pyproject.toml`. `app` và
`scripts` cũng là package. Các command dùng `python -m ...` để tránh chỉnh
`sys.path` và tránh phụ thuộc đường dẫn tuyệt đối. Cấu hình và tài liệu nằm trong
source checkout; bản đóng gói wheel độc lập chưa phải mục tiêu Phase 1.

## Luồng dữ liệu dự kiến

```text
original RGB uint8 H×W×3
  │
  ├─ Detector adapter [future]
  │    └─ resize/normalize → Faster R-CNN → invert resize → original xyxy boxes
  │
  ├─ choose lesion box [future; no-detection policy not implemented]
  │    └─ crop_roi(original, original_box, padding)
  │         ├─ exact clipped integer window + original H,W (CropMetadata)
  │         └─ RGB crop → resize_image → U-Net [future]
  │              └─ probability/logits → explicit threshold [future]
  │                   └─ restore_roi_mask(binary_mask, metadata)
  │                        └─ original-sized bool mask
  │
  └─ full-image U-Net baseline [future]
       └─ resize full image → segmentation → resize binary mask to original H,W
```

Phase 1 chạy được I/O, resize, ground truth mask → box, crop, map box theo resize
và restore. `prepare_roi` nối crop + RGB resize và giữ metadata, không gọi model.
Streamlit chỉ xem ảnh; không lấy box từ ground truth để trình bày thành detection.

## Trách nhiệm từng module

| Module | Trách nhiệm hiện tại | Chưa triển khai |
| --- | --- | --- |
| `configs/` | Cấu hình chung và 4 thí nghiệm kế thừa | Hyperparameter search |
| `src/utils/config.py` | YAML safe load, merge, validation, path resolution | Remote config |
| `src/utils/device.py` | Auto CPU/CUDA, kiểm tra CUDA yêu cầu tường minh | Multi-GPU |
| `src/utils/seed.py` | Seed Python/NumPy/PyTorch | Worker seed trong DataLoader |
| `src/data/image_io.py` | RGB và binary mask từ path/file stream | ISIC Dataset |
| `src/data/geometry.py` | Resize, box, crop, mask crop/restore | Augmentation |
| `src/models/detection/` | `Detection` và `Detector` Protocol | Faster R-CNN, loss, weights |
| `src/models/segmentation/` | `Segmenter` Protocol cho ROI/full-image | U-Net, loss, weights |
| `src/pipeline/spatial.py` | Crop + resize RGB, giữ metadata | End-to-end inference |
| `src/evaluation/` | Package dành chỗ, import an toàn | Dice/IoU/AP |
| `app/streamlit_app.py` | Upload, xem ảnh, thông báo checkpoint | Mask/box/overlay prediction |
| `scripts/smoke_test.py` | Import, config, device, geometry, AppTest | Training |
| `tests/` | Dữ liệu synthetic và UI harness | Accuracy trên ISIC |
| `notebooks/` | Setup và smoke check cho Colab | Training notebook |

Dependency đi từ app/script → pipeline/data/utils; pipeline → data. Data không
import model hoặc Streamlit. Model contracts dùng kiểu dữ liệu geometry, không
phụ thuộc app. Protocol chỉ mô tả giao diện; không được khởi tạo để suy luận.

## Hợp đồng dữ liệu và tọa độ

- Ảnh là `numpy.ndarray`, RGB `uint8`, `(H, W, 3)`; OpenCV chỉ dùng resize nên
  không có đổi RGB/BGR ngầm. I/O giữ orientation pixel lưu trong tệp.
- Mask nội bộ là boolean `(H, W)`. Utility chấp nhận 0/1/255 và reject giá trị
  xác suất/NaN. `read_binary_mask` riêng cho ground truth dùng pixel khác 0.
- `image_size` và `original_size` luôn `(height, width)`. OpenCV nhận `(width,
  height)` và sự chuyển thứ tự được gói trong utility.
- Box là xyxy nửa mở, cạnh phải/dưới có thể bằng W/H. Mask chỉ có pixel `(x,y)`
  cho box `(x,y,x+1,y+1)`. Mask rỗng trả về `None`; caller xử lý rõ ràng.
- Mask nhiều vùng foreground tạo box bao toàn bộ các vùng. Không suy ra
  connected components hoặc số đối tượng từ mask ở Phase 1.
- Box liên tục từ detector được làm tròn floor cạnh đầu, ceil cạnh cuối khi
  crop. Padding tỉ lệ độc lập theo width/height, sau đó clip vào ảnh gốc.
- Box không giao ảnh bị reject; mask sai chiều hoặc metadata vượt biên bị reject.
- Nếu detector chạy trên ảnh resize, gọi
  `scale_bbox(box, detector_size, original_size)` trước crop. Không áp dụng hàm
  này cho letterbox/rotate; transform tương ứng phải lưu và nghịch đảo riêng.
- `CropMetadata` lưu kích thước ảnh gốc và crop window **thực tế** sau padding,
  làm tròn và clip. Restore resize bằng nearest đến crop size rồi paste đúng
  window trên canvas False. Không tự scale box bằng kích thước U-Net.
- Mask resize xuống có thể mất pixel/boundary; restore đúng hệ tọa độ không có
  nghĩa đảo ngược được thông tin đã mất. Test round-trip tuyệt đối chỉ áp dụng
  khi không có resize gây mất thông tin; test khác kiểm tra phạm vi paste sau resize.

## Hợp đồng mô hình tương lai

`Detector.predict` nhận ảnh RGB gốc và trả `list[Detection]`, box quy về ảnh gốc.
Adapter chịu trách nhiệm preprocessing/transform của Torchvision; không resize
hai lần vô tình. ISIC segmentation mask không chứa annotation detection riêng;
Phase 2 sẽ tạo bounding box ground truth, Phase 3 mới train detector.

`Segmenter.predict` nhận RGB và trả binary mask cùng H,W với ảnh nó nhận. Cả
ROI lẫn full-image dùng cùng contract; adapter giữ normalization, resize và
threshold trong một nơi. Training API/loss chưa được cố định bằng Protocol này.

Phase sau phải quy định chọn box theo score, ngưỡng confidence, và hành vi khi
không có detection. Phase 1 không âm thầm dùng full-image làm một detection giả.
Checkpoint tồn tại không chứng minh đúng kiến trúc, đủ an toàn hay đã huấn luyện;
UI không load nó trong phase này.

## Cấu hình, đường dẫn và tính tái lập

`extends` resolve theo thư mục YAML đang đọc; circular inheritance báo lỗi.
Các path bên trong YAML resolve theo project root, không theo thư mục config.
Load config không kiểm tra dataset tồn tại và không tạo output: test/UI hoạt
động khi chưa có ISIC. Có thể cấu hình external dataset bằng path người dùng tự
chọn; source code không chứa path máy cá nhân.

Các thí nghiệm ROI 256/512 và full-image 512 phải dùng chung split ở phase sau.
ROI ground truth và ROI detector là hai điều kiện thí nghiệm khác nhau: test
bằng box ground truth không đo được sai số detector. Seed hiện kiểm soát random
generators cơ bản; deterministic kernels và DataLoader workers là công việc sau.

## Kiểm chứng và giới hạn

Unit tests dùng ảnh/mask synthetic, kiểm tra cả pixel cuối, box một pixel, mask
rỗng, padding sát biên, box float, size không vuông, resize nhãn và crop metadata.
Device tests mô phỏng nhánh CUDA selection và cấp phát tensor CPU thật; smoke
test cấp phát trên thiết bị thực tế mà auto chọn. CUDA giả lập không chứng minh
GPU kernels chạy được. UI tests dùng `AppTest`, tách biệt server HTTP smoke.
Kết quả thực thi được ghi ở `verification.md`, không suy ra từ mã test tồn tại.

Chưa có huấn luyện, download dataset/weights, full data pipeline, metrics,
inference end-to-end hoặc kiểm chứng GPU Colab trong Phase 1.

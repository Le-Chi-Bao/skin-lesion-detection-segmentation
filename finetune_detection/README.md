# Fine-tuning Faster R-CNN trên Google Colab

Workspace dùng `fasterrcnn_resnet50_fpn_v2` với backbone ResNet50-FPN, khởi tạo từ COCO và thay `FastRCNNPredictor` bằng head hai lớp. Nhãn 0 là nền và nhãn 1 là `skin_lesion`. RPN đề xuất vùng nghi ngờ, sau đó head phân loại vùng và hồi quy tọa độ box. Ba tầng cuối của backbone ResNet có gradient. Đầu vào là ảnh soi da RGB, đầu ra là box, nhãn và điểm tin cậy. [API Faster R-CNN của Torchvision](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)

## Chạy trên Colab

1. Từ thư mục gốc của repo, dùng lệnh bên dưới để tạo ZIP chỉ gồm các tệp đã commit trong `finetune_detection/`. ZIP đặt thư mục `finetune_detection/` ở cấp cao nhất. Có thể dùng ZIP đi kèm trong `finetune_detection/runs/`.
   ```powershell
   git archive --format=zip -o finetune_detection/runs/finetune_detection.zip HEAD finetune_detection
   ```
2. Tạo thư mục `MyDrive/FasterRCNN_ISIC2018/` và tải `finetune_detection.zip` lên. Nếu có ZIP chính thức của ISIC, đặt chúng trong `MyDrive/FasterRCNN_ISIC2018/archives/` với tên bên dưới. Chuẩn bị khoảng 30 GiB trống trên ổ đĩa Colab cho ZIP và dữ liệu đã giải nén. Dành khoảng 8 GiB Drive cho ba run mặc định và các artifact xuất model, cộng thêm dung lượng ZIP dữ liệu nếu lưu trên Drive.
3. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** và tải `notebooks/Faster_RCNN_ISIC2018_Colab.ipynb`. Chọn **Runtime → Change runtime type → GPU**. Chạy ô 1 đến ô 6 và cấp quyền gắn Drive ở ô 3. Ô 5 ưu tiên ZIP từ Drive và mặc định tải tệp còn thiếu từ ISIC. Đổi `DRIVE_PATH` ở ô 3 nếu dùng thư mục Drive khác.
4. Chọn `E1`, `E2`, `E3` hoặc `ALL` ở ô 7. Chạy ô 8 để huấn luyện và theo dõi ngay bên dưới ô. Dùng một phiên huấn luyện cho mỗi thư mục run. `ALL` chạy tuần tự theo thứ tự E1, E2, E3 và bỏ qua những run đã hoàn tất, đã xác minh.
5. Chọn `VIEW_EXPERIMENT` ở ô 9 để xem validation tại best epoch. Ô 10 lưu `comparison.md` vào Drive khi có kết quả. Sau khi chọn run qua validation, đặt `EXPORT_EXPERIMENT` tại ô 11. Artifact lưu ở `exports/<experiment>-<run_id>.pt`.

Ô 2 kiểm tra CUDA, AMP, toán tử NMS của Torchvision và backend COCO trước khi huấn luyện. Cặp wheel dự phòng là PyTorch 2.9.1 và Torchvision 0.24.1 với CUDA 12.6. Bật `INSTALL_FALLBACK` khi kiểm tra cặp wheel sẵn có thất bại, rồi chạy lại ô. Nếu Colab yêu cầu khởi động lại session sau khi cài wheel, khởi động lại và chạy từ ô 1. Khi resume, giữ các phiên bản Torch, Torchvision, NumPy và Pillow của checkpoint, cùng bản build CUDA và loại thiết bị. `environment.json` lưu thông tin Python, GPU, bộ nhớ GPU, driver, CUDA, cuDNN và các thư viện chính. [Cặp wheel chính thức của PyTorch](https://pytorch.org/get-started/previous-versions/) · [Google Colab Runtime Versions](https://research.google.com/colaboratory/runtime-version-faq.html)

## Dữ liệu ISIC 2018 Task 1

| Mục đích | ZIP ảnh | ZIP ground truth | Số ảnh |
| --- | --- | --- | ---: |
| Train và validation nội bộ | `ISIC2018_Task1-2_Training_Input.zip` | `ISIC2018_Task1_Training_GroundTruth.zip` | 2.594 |
| Official test sau khi chốt model | `ISIC2018_Task1-2_Test_Input.zip` | `ISIC2018_Task1_Test_GroundTruth.zip` | 1.000 |

Pipeline kiểm tra dung lượng, sao chép hoặc tải ZIP vào ổ đĩa Colab, rồi giải nén cục bộ và tái sử dụng cache trong cùng runtime. Cache Colab sẽ mất khi runtime mới được cấp. Giữ ZIP trên Drive nếu cần dùng lại qua nhiều session. Các ZIP ảnh được chia sẻ giữa Task 1 và Task 2, dùng ZIP ground truth **Task 1** để lấy mask tổn thương. [Tệp và giấy phép chính thức của ISIC](https://challenge.isic-archive.com/data/)

Ảnh và mask được ghép theo image ID. Hàm chuyển đổi dùng mọi pixel foreground trong mask nhị phân, kiểm tra kích thước và lưu box `[x1, y1, x2, y2]` trong hệ tọa độ ảnh gốc với cạnh phải và dưới loại trừ. Ví dụ pixel cột 7 nằm trong box có `x2 = 8`. Manifest lưu đường dẫn tương đối, kích thước, diện tích box, SHA-256 của ảnh và mask, SHA-256 của ZIP, nguồn dữ liệu và danh sách split. Trùng ID là lỗi. Mask rỗng, file hỏng, sai kích thước hoặc ghép thiếu được ghi vào `invalid_samples.json`. Mặc định `fail` dừng để kiểm tra. `exclude` chỉ dùng sau khi rà soát report và vẫn yêu cầu số file nguồn đúng với bộ chính thức.

Split train/validation xấp xỉ 80/20 với seed 42. Một manifest cố định dùng cho cả ba experiment, lưu trên Drive tại `train_manifest.json`. Ảnh trùng byte được xếp cùng split. Có thể cung cấp CSV có cột `image,lesion_id` ở ô 5. Các ảnh có cùng lesion ID được giữ cùng split. CSV chính thức của Task 3 có 0 ID trùng với 2.594 mask của Task 1 trong bản kiểm tra. Khi thiếu định danh tin cậy, chia theo image ID và SHA-256 của ảnh. Cách này chưa xác nhận độc lập theo bệnh nhân, hoặc nhận ra ảnh khác byte của cùng tổn thương. Tỷ lệ thực có thể thay đổi nhẹ khi giữ cả nhóm cùng nhau. [ISIC 2018 và metadata theo task](https://challenge.isic-archive.com/data/#2018)

Bộ dữ liệu gồm ảnh soi da từ challenge và mỗi mẫu hợp lệ có một box tổn thương. FP/Image ở đây phản ánh số phát hiện dư trên các ảnh có tổn thương. Đánh giá trên ảnh không có tổn thương hoặc ảnh từ nguồn, thiết bị hay bệnh nhân khác cần dữ liệu kiểm định phù hợp. Trang ISIC ghi CC0 ở dòng Task 1 và có hướng dẫn ghi công cho tập huấn luyện ISIC 2018 tổng hợp. Giữ các tệp `LICENSE.txt` và `ATTRIBUTION.txt` trong ZIP gốc khi phân phối dữ liệu, đối chiếu điều khoản tương ứng và ghi nguồn khi công bố. Ghi công HAM10000 cho ViDIR Group, Department of Dermatology, Medical University of Vienna và dùng trích dẫn MSK theo hướng dẫn ISIC nếu áp dụng. [Giấy phép và trích dẫn ISIC](https://challenge.isic-archive.com/data/) · [Codella và cộng sự](https://arxiv.org/abs/1902.03368) · [Tschandl và cộng sự](https://doi.org/10.1038/sdata.2018.161)

## Cấu hình và chỉ số

| Experiment | Min/max size | Microbatch × tích lũy | Augmentation |
| --- | --- | --- | --- |
| E1 | 800/1333 | 2 × 2 | Lật ngang và dọc |
| E2 | 800/1333 | 2 × 2 | Lật, xoay 90° và biến đổi sáng/tương phản |
| E3 | 1024/1600 | 1 × 4 | Như E2 |

Mỗi chiều lật có xác suất 0,5. Xoay lấy đều một góc trong `{0°, 90°, 180°, 270°}`. Sáng và tương phản được áp dụng cùng nhau với xác suất 0,5, hệ số mỗi phép lấy đều độc lập trong `[0,90; 1,10]`. Box được cập nhật với mỗi phép biến đổi hình học. Validation giữ ảnh gốc, chỉ chuyển RGB sang tensor; resize và chuẩn hóa nằm trong model.

SGD dùng LR 0,0025, momentum 0,9, weight decay 0,0001 và tối đa 25 epoch. LR tăng tuyến tính từ 0,1% của LR cấu hình trong 500 optimizer step đầu, rồi giảm cosine tới 1% của LR cấu hình. Scheduler chỉ tiến khi optimizer cập nhật tham số. Batch tích lũy mục tiêu là 4 ảnh và nhóm cuối epoch có thể nhỏ hơn. AMP FP16 dùng `torch.amp` trên GPU hỗ trợ. Early stopping kiểm tra từ epoch 10 và dừng nếu mAP@50:95 không tăng trong 5 lượt validation liên tiếp. Nếu GPU hết bộ nhớ, run dừng với cấu hình đã ghi. Muốn thử batch hoặc resize khác, sao chép YAML và đặt `CONFIG_PATH` cùng `RUN_NAME` mới ở ô 7.

| Chỉ số | Cách tính |
| --- | --- |
| mAP@50 | COCO bbox AP với IoU 0,50, một lớp lesion, mọi kích thước, tối đa 100 phát hiện mỗi ảnh. |
| mAP@50:95 | Trung bình COCO AP trên 10 mức IoU từ 0,50 đến 0,95, bước 0,05. AP xếp hạng theo score từ đầu ra sau NMS. |
| Recall@IoU 0,5 | Tỷ lệ ground truth khớp với dự đoán có score ≥ 0,50. Ghép theo score giảm dần, mỗi ground truth được dùng một lần. |
| Mean matched IoU | IoU trung bình của các cặp match ở ngưỡng trên. Hiện `N/A` nếu không có cặp khớp. |
| FP/Image | Số dự đoán đạt ngưỡng score nhưng không khớp, chia cho toàn bộ ảnh. Dự đoán trùng sau match được tính là FP. |
| ms/image | Thời gian trung bình của model forward FP32 và đồng bộ CUDA với batch validation 1. |
| Peak MiB | Bộ nhớ GPU được PyTorch cấp phát cao nhất trong train và validation của epoch. |

Nếu validation không có phát hiện, AP và recall bằng 0. Chọn best checkpoint bằng mAP@50:95 lớn nhất, giữ epoch xuất hiện trước nếu hòa. So sánh recall và FP/image trước khi xác định model triển khai. Ô 11 ghi riêng ứng viên đứng đầu theo mAP và model được chọn trong `deployment_selection.json`. Nếu hai lựa chọn khác nhau, có thể điền lý do vào `EXPORT_REASON`. Báo cáo dùng số đo thật từ checkpoint. [COCO API dùng cho AP](https://github.com/cocodataset/cocoapi)

## Tiến độ và phục hồi

Trong ô 8, thanh Training hiện tên experiment, epoch, batch, %, bốn thành phần loss, tổng loss, LR, VRAM đang cấp phát, thời gian, ETA và tốc độ ảnh/giây. Thanh Validation đếm số ảnh. Cuối epoch có bản tóm tắt chỉ số, thời gian, VRAM và trạng thái backup. Thời gian epoch và tổng cộng dồn train/validation qua các phiên đã checkpoint, bỏ qua thời gian chờ giữa phiên và lưu file. `logs/` có `metrics.csv`, `history.json`, `run_config.yaml`, `environment.json`, `training.log` và `training_curves.png`.

Với mỗi experiment, Drive có `last.ckpt`, `best.ckpt`, checkpoint trước đó, tối đa hai snapshot định kỳ trong `checkpoints/`, cùng log. Pipeline lưu sau mỗi epoch, mỗi 200 lần cập nhật optimizer và khi validation mAP@50:95 đạt mức cao mới. Checkpoint được ghi và đọc thử trên ổ cục bộ, sao chép vào tệp staging trên Drive, kiểm tra byte và SHA-256, rồi thay tệp chính trong khi giữ tệp trước đó để phục hồi. Nếu Drive lỗi, ô 8 in cảnh báo và giữ bản cục bộ tới khi runtime kết thúc.

Khi mất kết nối, chọn GPU và chạy lại các ô 1–8 với cùng thư mục Drive, manifest và cấu hình. `resume="auto"` kiểm tra checksum và tính tương thích, phục hồi vị trí batch, optimizer, scheduler, AMP scaler và RNG. Sampler dùng thứ tự xác định theo epoch và `num_workers=0` để tránh tiêu thụ RNG từ các worker tải trước. Bản CPU đã được kiểm thử đối chiếu model và optimizer sau gián đoạn. Sai khác phần cứng, driver hoặc kernel CUDA có thể làm kết quả số khác. Run sẽ tính lại batch sau checkpoint gần nhất. Khi mọi bản checkpoint của run đều lỗi hoặc sai cấu hình, xem thông báo rồi khôi phục backup hợp lệ hoặc dùng một tên run riêng. Run đã hoàn tất có bảo vệ chống ghi đè.

## CLI và official test

Sau khi cài `requirements.txt` và cặp Torch/Torchvision phù hợp, chạy các lệnh sau từ thư mục `finetune_detection/`.

```bash
python -m scripts.prepare_data --download
python -m scripts.verify_data
python -m scripts.train E1 --drive-root /content/drive/MyDrive/FasterRCNN_ISIC2018
python -m scripts.compare_experiments --drive-root /content/drive/MyDrive/FasterRCNN_ISIC2018
python -m pytest -q tests
```

CLI mặc định dùng `data/training/manifest.json` và `runs/` dưới workspace. Notebook dùng ổ cục bộ `/content/faster_rcnn_isic2018/` cho các thư mục này. `src.model.load_inference(path, device)` nạp artifact đã xuất, dựng đúng kiến trúc, resize và head. Metadata mô tả ảnh RGB float32 CHW trong [0, 1], chuẩn hóa nội bộ, hệ tọa độ box và ngưỡng score.

Official test chỉ chạy **sau khi khóa lựa chọn từ validation**. Giữ một thư mục output Drive cố định để lưu khóa SHA-256 của artifact, manifest, ngưỡng và giao thức chấm điểm. Script kiểm tra trùng image ID và ảnh trùng byte với tập train/validation. Kết quả test lưu riêng trong `official_test_metrics.json` và dùng cho báo cáo cuối. Trên Colab, thay tên artifact bằng đường dẫn hiện ở ô 11 trước khi chạy lệnh dưới đây.

```bash
DRIVE=/content/drive/MyDrive/FasterRCNN_ISIC2018
python -m scripts.evaluate official-test \
  --artifact "$DRIVE/exports/E1-RUN_ID.pt" \
  --training-manifest /content/faster_rcnn_isic2018/data/training/manifest.json \
  --data /content/faster_rcnn_isic2018/data/official-test \
  --archives "$DRIVE/archives" --download \
  --output "$DRIVE/official_test" --confirm-final-selection
```

Các bài kiểm thử tự động dùng ảnh tổng hợp. Bộ kiểm thử kiểm tra cả forward/backward của Faster R-CNN thật và tính nhất quán khi tiếp tục một vòng train nhỏ. Cặp wheel PyTorch 2.9.1 và Torchvision 0.24.1 với CUDA 12.6 đã chạy thử trên RTX 3050. Bài kiểm tra Jupyter cục bộ xác nhận thanh Training, Validation và tóm tắt epoch ngay dưới ô huấn luyện. Widget Colab, Google Drive và VRAM ở độ phân giải đầy đủ cần xác minh trên phiên Colab thực.

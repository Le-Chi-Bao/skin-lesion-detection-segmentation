# WP2 · Tích hợp SegFormer MiT-B2

Ứng dụng phân đoạn một vùng tiền cảnh tổn thương trên ảnh soi da. Mask và box minh họa vùng dự đoán của mô hình, phục vụ nghiên cứu. Checkpoint được tác giả công bố dưới dạng hiện vật nghiên cứu và không được dùng để ra quyết định lâm sàng. Ứng dụng không phân loại bệnh, ước tính nguy cơ ác tính, chẩn đoán hay đề xuất điều trị. [Thẻ mô hình](https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/blob/19ca826598cb59a6ac5b18b6a5636ba9298851b5/README.md)

## Checkpoint và môi trường

| Thuộc tính | Giá trị |
| --- | --- |
| Kho Hugging Face | `mokshhere/skin-lesion-segformer-gate2` |
| Phiên bản ghim | `19ca826598cb59a6ac5b18b6a5636ba9298851b5` |
| Tệp | `real_only/best.pt` |
| Kích thước | 99.002.337 byte |
| SHA-256 | `d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87` |
| Kiến trúc | `smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=1)` |
| Cấu trúc thực tế | `dict` với các trường `model`, `cfg`, `epoch`, `val_dice`, `which`. `model` là `OrderedDict` gồm 348 mục, cùng tiền tố `model.`. Checkpoint ghi epoch 21 và `which=raw`. |

Thông tin được đối chiếu với tệp checkpoint, registry, mã huấn luyện và bản tóm tắt của tác giả. [Checkpoint và LFS SHA-256](https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/blob/19ca826598cb59a6ac5b18b6a5636ba9298851b5/real_only/best.pt), [registry và SMPWrapper](https://github.com/ishaannk/skin-lesion-seg/blob/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/src/models/registry.py), [trình huấn luyện](https://github.com/ishaannk/skin-lesion-seg/blob/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/src/engine/trainer.py), [tóm tắt huấn luyện](https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/blob/19ca826598cb59a6ac5b18b6a5636ba9298851b5/real_only/summary.json)

WP2 dùng Python 3.12, Gradio 6.32.0, PyTorch 2.10.0, Torchvision 0.25.0, segmentation-models-pytorch 0.5.0, OpenCV headless 4.13.0.92 và NumPy 2.x. Torchvision là phụ thuộc của SMP qua timm. Trên Windows và Linux, chọn index CPU hoặc CUDA phù hợp với trình điều khiển. Trên macOS dùng cặp PyTorch/Torchvision trên PyPI và đặt thiết bị là `cpu`. Nếu đã có môi trường PyTorch hoạt động với phiên bản khác, dùng môi trường ảo riêng cho WP2. [Bảng cặp phiên bản và wheel PyTorch](https://pytorch.org/get-started/previous-versions/#v2100)

```powershell
# Dùng một trong hai lệnh cài PyTorch, tùy môi trường.
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-cv.txt
```

## Tải, xác minh và chạy

```powershell
python -m app.cv.checkpoint
$env:LESION_SEGFORMER_CHECKPOINT = 'model_store/segformer/real_only/best.pt'
python -m app.cv.checkpoint --verify-only
$env:LESION_SEGFORMER_DEVICE = 'auto'
python -m app.main
```

CLI tải đúng revision vào `model_store/segformer/real_only/best.pt` hoặc thư mục do `LESION_MODEL_STORE_DIR` đặt. Tải vào tệp tạm cùng thư mục đích, đối chiếu kích thước và SHA-256 rồi mới đặt vào vị trí cuối. CLI in metadata xác minh dưới dạng JSON và dùng lại tệp hợp lệ khi chạy lại. Khi dùng bản sao cục bộ, đặt `LESION_SEGFORMER_CHECKPOINT` thành đường dẫn của bản sao đó. Mọi bản sao phải khớp kích thước và SHA-256 trong bảng.

Máy chủ khởi động ở `http://127.0.0.1:7860` theo mặc định. Cấu hình checkpoint tường minh bật SegFormer. Khi chưa cấu hình hoặc tệp đã cấu hình bị thiếu, Gradio vẫn phục vụ tải và xem ảnh, với thông báo tiếng Việt và nút **Phân tích** vô hiệu hóa. Một tệp sai SHA-256, lỗi thư viện hoặc lỗi thiết bị đưa mô hình sang `failed`. Cấu hình thiết bị `auto` chọn CUDA khi PyTorch nhận diện được CUDA và chọn CPU trong trường hợp còn lại. `cpu` ép chạy trên CPU. `cuda` yêu cầu CUDA và báo lỗi nếu thiết bị không dùng được.

Mỗi tiến trình ứng dụng xác minh kích thước và SHA-256 rồi nạp mô hình một lần trước khi phục vụ. Loader dùng `torch.load(..., map_location="cpu", weights_only=True)`, trích `model` nếu có, bỏ đúng một tiền tố `model.` nếu toàn bộ khóa có tiền tố đó rồi kiểm tra tập khóa và nạp với `strict=True`. Mô hình dùng `eval()`, khóa gradient và không yêu cầu tải trọng số encoder. Lỗi đọc an toàn dừng nạp với thông báo cụ thể. Khi đổi cấu hình hoặc sửa tệp, khởi động lại ứng dụng để nạp lại.

## Đường đi của ảnh và mask

1. Lấy PNG RGB đầy đủ đã áp dụng EXIF từ phiên. Resize thẳng sang 512 × 512 bằng OpenCV `INTER_AREA` như bước tạo cache ISIC của tác giả. Ảnh chữ nhật được ánh xạ theo từng trục riêng, với kích thước gốc lưu trong `CVResult`. [Mã tạo cache](https://github.com/ishaannk/skin-lesion-seg/blob/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/src/data/preprocess.py)
2. Chia pixel cho 255 và chuẩn hóa từng kênh RGB với mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. Đầu vào float32 có dạng `[1, 3, 512, 512]`. Tập đánh giá của tác giả lấy ảnh trực tiếp từ cache khi kích thước cache và kích thước mô hình đều là 512. [Dataset và transform đánh giá](https://github.com/ishaannk/skin-lesion-seg/blob/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/src/data/dataset.py), [cấu hình checkpoint](https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/blob/19ca826598cb59a6ac5b18b6a5636ba9298851b5/real_only/config.json)
3. Chạy một lượt forward float32 trong `torch.inference_mode()`. Kiểm tra logits dạng `[1, 1, 512, 512]` và mọi giá trị hữu hạn. Áp dụng sigmoid và giữ pixel có xác suất `>= 0.7`. Ngưỡng `0.7` do tác giả chọn trên tập validation. Tệp đánh giá của tác giả có thêm kết quả ở ngưỡng `0.5` và ghi `tta=false`. [Metadata chọn ngưỡng](https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/blob/19ca826598cb59a6ac5b18b6a5636ba9298851b5/real_only/eval_test.json)
4. Resize mask đã nhị phân bằng nearest-neighbor của Pillow về kích thước gốc. PNG có giá trị 0 và 255. Từ mask cuối, lấy box nhỏ nhất chứa tiền cảnh theo `[x1, y1, x2, y2]` với cạnh phải và cạnh dưới loại trừ. Mask rỗng có cảnh báo và box để trống. `CVResult` lưu tham chiếu mask theo phiên, nguồn `derived_from_mask`, checkpoint, SHA-256, thiết bị, ngưỡng và thời gian forward. Thời gian CUDA có đồng bộ thiết bị trước và sau forward.

WP2 áp dụng ngưỡng ở lưới 512 trước khi resize để giữ mask nhị phân trong bước ánh xạ về ảnh gốc. Giao thức `native` của tác giả resize bản đồ xác suất bằng `INTER_LINEAR` trước khi áp dụng ngưỡng. Điểm Dice và IoU của tác giả thuộc giao thức đánh giá đó. WP2 không chạy tập test ISIC hay hiệu chỉnh ngưỡng và các ca smoke chỉ kiểm tra kỹ thuật. [Mã đánh giá của tác giả](https://github.com/ishaannk/skin-lesion-seg/blob/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/src/engine/evaluate.py)

## Giao diện, phiên và giới hạn

- Trong **Computer Vision**, tải ảnh, bấm **Phân tích** và mở **Mask**. Trạng thái bên cạnh nút hiển thị tiến trình theo lượt và metadata khi hoàn tất. **Localization** chỉ hiển thị tọa độ box suy ra từ mask. Faster R-CNN có trạng thái `not_available`.
- Tác vụ suy luận dùng hàng đợi riêng với một lượt forward tại một thời điểm. Tải ảnh, Reset và hiển thị kết quả được áp dụng theo thứ tự trên hàng đợi sửa ảnh của WP1. Một lượt dùng `RunToken` và chỉ được nhận kết quả một lần. Reset và thay ảnh làm token cũ mất hiệu lực, còn tác vụ nền hoàn tất thì dọn mask đã lỗi thời.
- Ảnh và mask được lưu riêng theo UUID phiên trong `runtime/sessions/`, với tên ngẫu nhiên. Vòng đời tệp giữ tài nguyên đang được đọc hoặc ghi. Mask được chuyển tới giao diện dưới dạng dữ liệu PNG nội tuyến trong phản hồi của phiên. Tuyến HTTP tải tệp trực tiếp bị chặn. Suy luận chạy cục bộ và không chuyển ảnh lên Hugging Face.
- Cạnh nhỏ và chi tiết biên có thể mất khi ảnh bị đưa về lưới 512. Mask trống chỉ cho biết không có pixel vượt ngưỡng sau khi xử lý. Overlay, điều chỉnh độ mờ, vẽ box trên ảnh và xuất PNG thuộc WP3.

## Kiểm thử và tái lập

```powershell
python -m pip install -r requirements-dev.txt playwright==1.65.0
python -m playwright install chromium
python -m pytest -q tests --basetemp runtime/wp2-regression
```

Bộ thông thường bao gồm hồi quy WP1, các ca đơn vị với đầu vào kiểm soát và kiểm thử callback Gradio cô lập. Hai ca checkpoint thật được bật tường minh. Dùng một thư mục `--basetemp` dành riêng cho mỗi lệnh.

Mẫu là bản thu nhỏ 330 × 244 của ảnh soi da *Dermatoscopy Nevus Reticular.jpg* do Philipp Tschandl công bố, giấy phép CC BY-SA 4.0. SHA-256 của bản dùng trong kiểm thử là `30d963f7ee6112d6bace3edb96cb46298a57090191c67f7badb4c4827285a242`. [Nguồn và giấy phép ảnh](https://commons.wikimedia.org/wiki/File:Dermatoscopy_Nevus_Reticular.jpg)

```powershell
New-Item -ItemType Directory -Force model_store/wp2-verification | Out-Null
curl.exe -fLsS 'https://thumb.wikimedia.org/wikipedia/commons/thumb/9/9f/Dermatoscopy_Nevus_Reticular.jpg/330px-Dermatoscopy_Nevus_Reticular.jpg' -o model_store/wp2-verification/dermoscopy.jpg
$env:WP2_SMOKE_CHECKPOINT = 'model_store/segformer/real_only/best.pt'
$env:WP2_SMOKE_IMAGE = 'model_store/wp2-verification/dermoscopy.jpg'
$env:WP2_SMOKE_DEVICE = 'cpu'
python -m pytest -q -s tests/test_wp2_application.py -k real_checkpoint --basetemp runtime/wp2-cpu-smoke
```

Smoke test kiểm tra logits thật, mask nhị phân và box, tuần tự hóa `CVResult` và nội dung PNG được Gradio trả về. Ca Chromium khởi động máy chủ bằng `app.main` và kiểm tra tải ảnh → **Phân tích** → **Mask** → **Đặt lại**, cùng sự cô lập giữa hai phiên trình duyệt. Để kiểm tra lại trên GPU tương thích, dùng môi trường CUDA, đặt `WP2_SMOKE_DEVICE=cuda` và chọn một `--basetemp` riêng.

### Kết quả ngày 24/11/2026

| Lệnh kiểm tra | Kết quả |
| --- | --- |
| `python -m pytest -q tests --basetemp runtime/wp2-wp1-baseline` tại commit WP1 | 54 đạt |
| `python -m pytest -q tests --basetemp runtime/wp2-regression-final` | 74 đạt, 2 ca checkpoint thật bỏ qua theo cấu hình |
| `python -m pytest -q -s tests --basetemp runtime/wp2-full-cpu` với hai biến đường dẫn smoke test và thiết bị `cpu` | 76 đạt, gồm cả callback và Chromium với checkpoint thật |
| `python -m pytest -q -s tests/test_wp2_application.py -k real_checkpoint --basetemp runtime/wp2-cuda-smoke` trong môi trường CUDA với thiết bị `cuda` | 2 đạt, 5 ca ngoài bộ lọc |

PyTorch 2.10.0 và Torchvision 0.25.0 được kiểm tra bằng hai bộ wheel `+cpu` và `+cu126`. Cả hai dùng NumPy 2.5.5 và Pillow 12.3.0. Forward float32 đo được **548,72 ms** trên Intel Core i5-12500H và **1.220,03 ms** trên NVIDIA GeForce RTX 3050 Laptop GPU 4 GiB, driver 566.07, runtime CUDA 12.6. Mỗi số đo là một lượt forward đầu tiên trong bài kiểm thử.

Cả hai môi trường trả logits hữu hạn dạng `[1, 1, 512, 512]`, mask **330 × 244** với 1.016 pixel tiền cảnh và box **[138, 89, 181, 124]**. Trên Chromium, nút Phân tích bật sau khi tải ảnh hợp lệ, mask thật xuất hiện trong phiên thực hiện lượt phân tích và Reset xóa kết quả hiển thị. Ca kiểm thử callback giữ suy luận đang chạy trong khi thay ảnh hoặc Reset, sau đó xác nhận kết quả cũ bị bỏ và tệp tạm được dọn.

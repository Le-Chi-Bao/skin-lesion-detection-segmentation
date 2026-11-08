# Phase 1 — kết quả kiểm chứng thực tế

Ngày thực hiện: **2026-11-08**, theo múi giờ người dùng Asia/Bangkok.
Các kết quả dưới đây đến từ lệnh đã chạy, không phải dự đoán từ source code.

## Môi trường

Windows 64-bit. Lúc kiểm tra ban đầu, `python` không có trong PATH và `py -0p`
không tìm thấy Python đã cài. Đã chuẩn bị Python 3.11.9 embeddable chính thức
trong `.tools/python311/`, bootstrap pip, cài cặp torch/torchvision CPU rồi cài
editable project bằng `pip install -r requirements.txt` thành công.
Sau khi điều chỉnh dependency tối thiểu của Streamlit, metadata editable đã
được cài lại bằng `pip install --no-deps --no-build-isolation -e .`.

| Thành phần | Phiên bản thực tế |
| --- | --- |
| Python | 3.11.9 |
| torch | 2.9.1+cpu |
| torchvision | 0.24.1+cpu |
| NumPy | 2.4.6 |
| opencv-python-headless | 4.14.0.94 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| Streamlit | 1.67.0 |
| pytest | 9.2.0 |
| Ruff | 0.17.4 |

`torch.cuda.is_available()` trả về `False`. Import PyTorch/Torchvision và cấp phát
tensor CPU thật thành công. CUDA selection được kiểm tra bằng monkeypatch trong
unit tests; **chưa chạy GPU kernels, T4 hoặc Colab thực tế**. Không suy ra tình
trạng GPU phần cứng từ kết quả của bản PyTorch CPU này.

## Các lệnh và kết quả cuối cùng

Chạy tại project root bằng `.\.tools\python311\python.exe`:

| Command suffix | Kết quả thực tế |
| --- | --- |
| `-m pytest -q` | **74 passed in 5.60s**, exit code 0 |
| `-m scripts.smoke_test` | Import, cấu hình, CPU allocation, geometry và AppTest đạt; exit code 0 |
| `-m scripts.check_streamlit_server` | Health HTTP 200 và frontend HTML HTTP 200; exit code 0 |
| `-m pip check` | `No broken requirements found.`, exit code 0 |
| `-m ruff check .` | `All checks passed!`, exit code 0 |
| `-m ruff format --check .` | `30 files already formatted` ở lần kiểm tra cuối, exit code 0 |

Output smoke script:

```json
{
  "imports": "passed",
  "configuration": "configs/base.yaml",
  "device": "cpu",
  "cuda_available": false,
  "coordinate_round_trip": "passed (synthetic ground truth, no AI inference)",
  "streamlit_app": "passed (AppTest)"
}
```

## Phạm vi tests

- Ảnh RGB, grayscale → RGB, giữ channel order, đường dẫn Unicode, ảnh hỏng,
  tệp không tồn tại; binary mask 0/1 và 0/255.
- Kích thước H/W, nearest-neighbor labels, box nửa mở, pixel cuối ảnh, box một
  pixel, mask rỗng, vùng foreground rời nhau, box float và resize không vuông.
- ROI padding theo mỗi trục, clip vào biên, crop mask cùng metadata, restore về
  canvas gốc, lỗi box/mask/size/metadata và round-trip không mất tọa độ.
- Load cả 5 config, kế thừa path, project root tùy chọn, current directory
  khác root, circular inheritance, giá trị sai và typo keys.
- CPU tensor thật, hai nhánh auto CPU/CUDA giả lập, CUDA yêu cầu nhưng không
  khả dụng, và seed tái lập Python/NumPy/PyTorch CPU.
- AppTest render title/uploader/result sections, checkpoint chưa cấu hình,
  checkpoint thiếu và tệp checkpoint có tồn tại nhưng không bao giờ được load.
- Luồng upload được kiểm tra bằng BytesIO ảnh synthetic/tệp hỏng và monkeypatch
  uploader; logic đọc/hiển thị/lỗi chạy trong AppTest. Đây không phải thao tác
  upload tự động bằng trình duyệt thật.
- Server Streamlit thật khởi động headless trên localhost, phục vụ health và
  HTML, sau đó script dừng đúng process con do nó tạo. Health/HTML không chứng
  minh riêng việc thực thi script UI; AppTest kiểm chứng phần UI Python.

Kiểm tra bổ sung đã chạy: compile cú pháp cho 27 tệp Python; parse
`pyproject.toml` bằng `tomllib`; parse notebook JSON, kiểm tra cell IDs duy nhất
và compile các code cell. Notebook chưa được thực thi trong runtime Colab.

Trong lần pytest đầu có 1 lỗi test vì tên phần tử ảnh trong AppTest thay đổi
từ `imgs` sang `image`; đã điều chỉnh assertion cho cả hai dạng. Cũng đã thay
`use_container_width` deprecated bằng `width="stretch"` và yêu cầu Streamlit
>= 1.49. Lần chạy đầy đủ sau sửa đạt kết quả nêu trên.
AppTest có thể ghi warning `missing ScriptRunContext` khi khởi tạo test harness
ở bare mode; smoke script vẫn hoàn tất với exit code 0, không có app exception.

## Các phần chưa được triển khai/kiểm chứng

Không có ISIC thực tế, download dataset, pretrained weights, mô hình Faster
R-CNN/U-Net, training loop, data loader đầy đủ, augmentation, metrics accuracy,
suy luận end-to-end hoặc prediction AI. Project hiện là nền tảng kỹ thuật phục
vụ đồ án.

Test binary checkpoint dùng tệp synthetic không phải model thật và không gọi
`torch.load`. Chưa kiểm chứng pretrained/checkpoint compatibility. Resize mask
có thể mất chi tiết: tests không tuyên bố mọi phép resize đều đảo ngược chính xác.

## Tài liệu chính thức đã đối chiếu

- [PyTorch — cặp phiên bản và CPU/CUDA wheels](https://pytorch.org/get-started/previous-versions/).
- [Streamlit — AppTest](https://docs.streamlit.io/develop/api-reference/app-testing).
- [Streamlit 1.49 — st.image width](https://docs.streamlit.io/1.49.0/develop/api-reference/media/st.image).
- [ISIC — dữ liệu challenge](https://challenge.isic-archive.com/data/#2018), chỉ đọc
  hướng dẫn và nguồn dữ liệu, không tải archive.

# Báo cáo Phase 2A.1 — 2026-11-16 (Asia/Bangkok)

| Hạng mục | Kết quả | Bằng chứng |
|---|---|---|
| CUDA | PASS | RTX 3050 Laptop GPU, CUDA 12.6, tensor trên cuda:0, đồng bộ GPU, CPU trả [[7, 10], [15, 22]] |
| Strict load | PASS | weights_only=True, map_location=cpu, strict=True; 348 entries, missing/unexpected đều rỗng; kiểm tra mọi tensor đã sao chép khớp weights |
| Ảnh mẫu | PASS cho smoke test | ISIC_0012169, ảnh dermoscopic JPEG RGB 3008x2000; metadata công khai CC-0, attribution Anonymous |

Python: `3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)]`.
Executable: `E:\Các môn học năm 3\Thị giác máy tính\course work\.venv\Scripts\python.exe`.
Pip nằm trong `.venv/Lib/site-packages`, không cài global.
Driver NVIDIA giữ nguyên 566.07; GPU 4096 MiB, compute capability 8.6.

## Phiên bản thực tế

- torch: `2.11.0+cu126`
- torchvision: `0.26.0+cu126`
- segmentation-models-pytorch: `0.5.0`
- timm: `1.0.31`
- numpy: `2.5.4`
- Pillow: `13.0.0`
- Gradio: 6.32.0 (giữ nguyên)
- `pip check`: No broken requirements found.
- Danh sách đầy đủ: `pip-freeze.txt`.

## Nguồn SMP và load weights

Upstream: https://github.com/ishaannk/skin-lesion-seg/tree/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7
Không có manifest/lockfile ghim SMP chính xác. README yêu cầu SMP >=0.5.
Phiên bản dùng thử: SMP 0.5.0; không khẳng định đây là phiên bản huấn luyện gốc.
Đã tạo SMPWrapper có self.model bọc smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=1).
Đã thực sự giải tuần tự checkpoint bằng torch.load(..., map_location="cpu", weights_only=True),
nạp checkpoint["model"] bằng load_state_dict(..., strict=True) và đối chiếu mọi tensor.
348 state-dict entries, 24722369 parameters. Không bỏ qua key hoặc sửa tên key.
Không tải encoder weights riêng. Không gọi forward/inference.

## Checkpoint

- Local: `E:\Các môn học năm 3\Thị giác máy tính\course work\checkpoints\segformer-gate2\real_only\best.pt`
- URL: https://huggingface.co/mokshhere/skin-lesion-segformer-gate2/resolve/19ca826598cb59a6ac5b18b6a5636ba9298851b5/real_only/best.pt
- Revision: `19ca826598cb59a6ac5b18b6a5636ba9298851b5`
- SHA-256: `d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87`
- Size: 99002337 bytes. SHA-256 khớp LFS metadata upstream.
- Provenance: `E:\Các môn học năm 3\Thị giác máy tính\course work\checkpoints\segformer-gate2\real_only\provenance.json`
- Epoch 21, weights raw; input working resolution 512x512.

## Ảnh mẫu

- ISIC ID: ISIC_0012169
- Local: `E:\Các môn học năm 3\Thị giác máy tính\course work\data\samples\ISIC_0012169.jpg`
- Trang metadata: https://api.isic-archive.com/images/ISIC_0012169/
- API metadata: https://api.isic-archive.com/api/v2/images/ISIC_0012169/
- URL tải ảnh: https://isic-archive.s3.us-east-1.amazonaws.com/images/ISIC_0012169.jpg
- Giấy phép: CC-0 / CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/
- Ghi công: ISIC_0012169 - Anonymous, via ISIC Archive, CC0 1.0
- SHA-256: `45ec44071e288dfe84b889189968ea3a343933d05451219ca4e345c54f1a1653`
- Nguồn upstream liệt kê ảnh trong tập test: https://raw.githubusercontent.com/ishaannk/skin-lesion-seg/d3d3195f13b02366cbc1332cbb57d7ca9234b7a7/experiments/segformer_real_s1337/perimage_test_t0.5_native.csv
- Chưa độc lập đối chiếu split với danh sách challenge chính thức; chỉ dùng smoke test, không dùng đánh giá mô hình.
- API báo 573464 bytes nhưng S3 trả 576614 bytes.
  Đã xác minh Content-Length và MD5 `798570f65d5251c8f2e2a5fb3274be60` khớp ETag S3, Pillow verify hợp lệ và kích thước khớp metadata.
- Lưu metadata và ghi công bên cạnh ảnh trong data/samples/ISIC_0012169.metadata.json và ISIC_0012169.provenance.json.

## Lệnh đã thực hiện

```powershell
# Run in the repository root, Windows PowerShell.
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe outputs\phase2a1\check_cuda.py
.\.venv\Scripts\python.exe -m pip install segmentation-models-pytorch==0.5.0 -c requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe outputs\phase2a1\download_assets.py
.\.venv\Scripts\python.exe outputs\phase2a1\check_checkpoint.py
```

Các script trong outputs/phase2a1 chỉ phục vụ tải tài sản và xác minh local, không phải CLI segmentation.
Các đường dẫn .venv/, checkpoints/, data/, outputs/ đều đã nằm trong .gitignore.
Không sửa app.py, Gradio, crawler, RAG hay requirements.txt; không push.
Đã dừng ở 2A.1, chưa chạy inference. Báo cáo này không phải đánh giá chất lượng mô hình.
